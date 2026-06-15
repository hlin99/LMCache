// SPDX-License-Identifier: Apache-2.0

//
// SYCL implementation of multi_layer_block_kv_transfer for Intel XPU
// (PVC / Arc / Battlemage).
//
// Design notes:
//
// 1. Independence from CUDA: this file must compile with icpx alone.
//    It includes only SYCL and PyTorch/ATen headers.
//
// 2. Vectorization: uses int64_t (8 bytes) as the maximum copy
//    granularity, with fallback to int32_t and int16_t based on
//    head_bytes alignment.  No uint4 or other CUDA-specific types.
//
// 3. Kernel topology: nd_range<3> with group dimensions
//    (kv_size, total_blocks, nl) and a 1-D local range (wg_size).
//    Work-group size is rounded up to sub-group size 16 and capped at 256.
//    Each work-item strides across all bs * scalars_per_token elements in
//    the block using a stride loop.
//
// 4. Supported GPUKVFormat values (NHD + MLA only):
//      NB_NL_TWO_BS_NH_HS   — cross-layer single tensor
//      NL_X_TWO_NB_BS_NH_HS — vLLM flash attention
//      NL_X_NB_TWO_BS_NH_HS — vLLM flash infer
//      NL_X_NB_BS_HS        — vLLM MLA
//      NL_X_NBBS_ONE_HS     — SGLang MLA
//    All other formats throw std::runtime_error.

// sycl/accessor.hpp references the deprecated 'host_buffer' internally
// even when user code only uses USM pointers; suppress the noise.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <sycl/sycl.hpp>
#pragma GCC diagnostic pop

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include "mp_mem_kernels_sycl.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tuning constants (mirrored from mem_kernels_sycl.cpp)
// ---------------------------------------------------------------------------
namespace {

constexpr int INTEL_SUB_GROUP_SIZE = 16;
constexpr int MAX_WG_SIZE = 256;

inline int round_up_to_sg(int n) {
  return ((n + INTEL_SUB_GROUP_SIZE - 1) / INTEL_SUB_GROUP_SIZE) *
         INTEL_SUB_GROUP_SIZE;
}

// ---------------------------------------------------------------------------
// Kernel submit helper
// ---------------------------------------------------------------------------
//
// Template parameters:
//   scalar_t           — working scalar type (int64_t / int32_t / int16_t)
//   LMCACHE_TO_ENGINE  — true = H2D (LMCache → engine), false = D2H
//   format             — compile-time GPUKVFormat (eliminates branches)
//
// All five supported NHD / MLA formats have the same intra-block memory
// layout: [BS, NH, HS] (or [BS, HS] for MLA with NH==1).  This means the
// flat index ``i`` within a block is identical for both the engine and the
// LMCache side, so a single stride loop suffices.
//
// paged_buffer_ptrs — scalar_t** pointing to XPU memory that holds the
//   per-layer (or single cross-layer) tensor data pointers.  Accessed via
//   USM device dereference inside the kernel.
//
// obj0..obj3 — individual scalar_t* values captured by value so the SYCL
//   lambda receives device-accessible copies without dangling pointer risk.
//
template <typename scalar_t, bool LMCACHE_TO_ENGINE, GPUKVFormat format>
void submit_block_kv_transfer_kernel(
    sycl::queue& queue,
    scalar_t** paged_buffer_ptrs,
    const int64_t* block_ids_ptr,
    scalar_t* obj0,
    scalar_t* obj1,
    scalar_t* obj2,
    scalar_t* obj3,
    int total_blocks,
    int num_blocks_per_object,
    int skip_prefix_n_blocks,
    PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size,
    int wg_size) {
  const int kv_size = shape_desc.kv_size;
  const int nl = shape_desc.nl;
  const int bs = shape_desc.bs;
  const int nb = shape_desc.nb;

  // Pre-compute scalar counts outside the kernel to avoid repeated
  // template method calls from within device code.
  const size_t spt = shape_desc.scalars_per_token<scalar_t>();
  const size_t spb = shape_desc.scalars_per_block<scalar_t>();
  const size_t total_elements = static_cast<size_t>(bs) * spt;

  // Grid: (kv_size, total_blocks, nl) groups; each group has wg_size threads.
  sycl::range<3> global_range(
      static_cast<size_t>(kv_size),
      static_cast<size_t>(total_blocks),
      static_cast<size_t>(nl) * static_cast<size_t>(wg_size));
  sycl::range<3> local_range(1, 1, static_cast<size_t>(wg_size));

  queue.parallel_for(
      sycl::nd_range<3>(global_range, local_range),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        const int k_or_v = static_cast<int>(item.get_group(0));
        const int flat_block_idx = static_cast<int>(item.get_group(1));
        const int layer_idx = static_cast<int>(item.get_group(2));
        const int tid = static_cast<int>(item.get_local_id(2));
        const int num_threads = static_cast<int>(item.get_local_range(2));

        // Skip leading prefix blocks.
        if (flat_block_idx < skip_prefix_n_blocks) return;

        const int obj_idx = flat_block_idx / num_blocks_per_object;
        const int block_idx_in_object = flat_block_idx % num_blocks_per_object;
        const int engine_block_idx =
            static_cast<int>(block_ids_ptr[flat_block_idx]);

        // ----------------------------------------------------------------
        // Engine global offset: start of this block within the engine
        // tensor, in scalar_t units.  Formula depends on format.
        // ----------------------------------------------------------------
        size_t engine_global;
        if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
          // Single tensor [NB, NL, 2, BS, NH, HS]
          engine_global =
              static_cast<size_t>(k_or_v) * spb +
              static_cast<size_t>(layer_idx) * static_cast<size_t>(kv_size) *
                  spb +
              static_cast<size_t>(engine_block_idx) *
                  static_cast<size_t>(kv_size) * spb *
                  static_cast<size_t>(nl);
        } else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
          // Per-layer tensor [2, NB, BS, NH, HS]
          engine_global = static_cast<size_t>(engine_block_idx) * spb +
                          static_cast<size_t>(k_or_v) *
                              static_cast<size_t>(nb) * spb;
        } else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
          // Per-layer tensor [NB, 2, BS, NH, HS]
          engine_global =
              static_cast<size_t>(engine_block_idx) *
                  static_cast<size_t>(kv_size) * spb +
              static_cast<size_t>(k_or_v) * spb;
        } else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS ||
                              format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
          // Per-layer MLA tensor ([NB, BS, HS] or [NBBS, 1, HS])
          engine_global = static_cast<size_t>(engine_block_idx) * spb;
        }

        // ----------------------------------------------------------------
        // LMCache global offset: 2LTD layout [2, L, chunk_size, NH*HS].
        // token_offset_in_object = block_idx_in_object * bs (tokens).
        // ----------------------------------------------------------------
        const size_t token_offset_in_object =
            static_cast<size_t>(block_idx_in_object) *
            static_cast<size_t>(bs);
        const size_t lmcache_global =
            static_cast<size_t>(k_or_v) *
                static_cast<size_t>(nl) *
                static_cast<size_t>(lmcache_chunk_size) * spt +
            static_cast<size_t>(layer_idx) *
                static_cast<size_t>(lmcache_chunk_size) * spt +
            token_offset_in_object * spt;

        // ----------------------------------------------------------------
        // Select engine layer pointer.
        // NB_NL_TWO_BS_NH_HS stores all layers in a single tensor (idx 0).
        // All other formats store one tensor per layer.
        // ----------------------------------------------------------------
        scalar_t* engine_layer_ptr;
        if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
          engine_layer_ptr = paged_buffer_ptrs[0];
        } else {
          engine_layer_ptr = paged_buffer_ptrs[layer_idx];
        }

        // ----------------------------------------------------------------
        // Select LMCache object pointer (up to 4 objects).
        // Individual variables are captured by value to avoid dangling
        // pointer risk from capturing a local array by pointer.
        // ----------------------------------------------------------------
        scalar_t* lmcache_ptr;
        if (obj_idx == 0) {
          lmcache_ptr = obj0;
        } else if (obj_idx == 1) {
          lmcache_ptr = obj1;
        } else if (obj_idx == 2) {
          lmcache_ptr = obj2;
        } else {
          lmcache_ptr = obj3;
        }

        // ----------------------------------------------------------------
        // Stride loop: copy total_elements = bs * spt scalars.
        //
        // Because all supported formats use NHD intra-block layout
        // ([BS, NH, HS]), the flat index i maps directly to the same
        // offset in both the engine and the LMCache buffer.
        // ----------------------------------------------------------------
        for (size_t i = static_cast<size_t>(tid); i < total_elements;
             i += static_cast<size_t>(num_threads)) {
          if constexpr (LMCACHE_TO_ENGINE) {
            engine_layer_ptr[engine_global + i] =
                lmcache_ptr[lmcache_global + i];
          } else {
            lmcache_ptr[lmcache_global + i] =
                engine_layer_ptr[engine_global + i];
          }
        }
      });
}

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

#define LAUNCH_BLOCK_KERNEL(DIRECTION, FORMAT)                               \
  submit_block_kv_transfer_kernel<scalar_t, DIRECTION, FORMAT>(             \
      queue, paged_buffer_ptrs, block_ids_ptr,                              \
      obj0, obj1, obj2, obj3,                                               \
      total_blocks, num_blocks_per_object, skip_prefix_n_blocks,            \
      shape_desc, lmcache_chunk_size, wg_size);

#define DISPATCH_FORMAT(DIRECTION)                                           \
  switch (gpu_kv_format) {                                                   \
    case GPUKVFormat::NB_NL_TWO_BS_NH_HS:                                   \
      LAUNCH_BLOCK_KERNEL(DIRECTION, GPUKVFormat::NB_NL_TWO_BS_NH_HS);      \
      break;                                                                 \
    case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:                                 \
      LAUNCH_BLOCK_KERNEL(DIRECTION, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);    \
      break;                                                                 \
    case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:                                 \
      LAUNCH_BLOCK_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);    \
      break;                                                                 \
    case GPUKVFormat::NL_X_NB_BS_HS:                                        \
      LAUNCH_BLOCK_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_BS_HS);           \
      break;                                                                 \
    case GPUKVFormat::NL_X_NBBS_ONE_HS:                                     \
      LAUNCH_BLOCK_KERNEL(DIRECTION, GPUKVFormat::NL_X_NBBS_ONE_HS);        \
      break;                                                                 \
    default:                                                                 \
      throw std::runtime_error(                                              \
          "multi_layer_block_kv_transfer (SYCL): unsupported GPUKVFormat " + \
          std::to_string(static_cast<int>(gpu_kv_format)));                  \
  }

// ---------------------------------------------------------------------------
// Templated implementation
// ---------------------------------------------------------------------------
template <typename scalar_t>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs,
    const torch::Tensor& block_ids,
    const torch::Device& device,
    TransferDirection direction,
    PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks) {
  // --- Validation ---
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1–4 LMCache objects, got ", num_objects);

  const int total_blocks = static_cast<int>(block_ids.size(0));
  TORCH_CHECK(total_blocks % num_objects == 0,
              "block_ids length (", total_blocks,
              ") must be divisible by num_objects (", num_objects, ")");
  const int num_blocks_per_object = total_blocks / num_objects;

  TORCH_CHECK(
      num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
      "blocks_per_object * block_size (",
      num_blocks_per_object * shape_desc.bs,
      ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  // --- Build typed object pointers (captured by value in kernel) ---
  scalar_t* obj0 = (num_objects > 0)
                       ? reinterpret_cast<scalar_t*>(lmcache_objects_ptrs[0])
                       : nullptr;
  scalar_t* obj1 = (num_objects > 1)
                       ? reinterpret_cast<scalar_t*>(lmcache_objects_ptrs[1])
                       : nullptr;
  scalar_t* obj2 = (num_objects > 2)
                       ? reinterpret_cast<scalar_t*>(lmcache_objects_ptrs[2])
                       : nullptr;
  scalar_t* obj3 = (num_objects > 3)
                       ? reinterpret_cast<scalar_t*>(lmcache_objects_ptrs[3])
                       : nullptr;

  // --- Paged-buffer pointer array (XPU USM device memory) ---
  scalar_t** paged_buffer_ptrs =
      reinterpret_cast<scalar_t**>(paged_buffer_ptrs_tensor.data_ptr());

  // --- Block IDs (XPU int64 tensor) ---
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  // --- Work-group size: round up scalars_per_token to sub-group boundary ---
  const size_t spt = shape_desc.scalars_per_token<scalar_t>();
  const int wg_size = round_up_to_sg(
      std::min(static_cast<int>(spt), MAX_WG_SIZE));

  // --- Acquire the XPU stream for the target device ---
  const c10::OptionalDeviceGuard device_guard(device);
  sycl::queue& queue =
      c10::xpu::getCurrentXPUStream(device.index()).queue();

  if (direction == TransferDirection::H2D) {
    DISPATCH_FORMAT(true);
  } else {
    DISPATCH_FORMAT(false);
  }
}

#undef DISPATCH_FORMAT
#undef LAUNCH_BLOCK_KERNEL

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs,
    const torch::Tensor& block_ids,
    const torch::Device& device,
    TransferDirection direction,
    PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks) {
  const int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(int16_t) == 0,
              "head_size * element_size (", head_bytes,
              ") must be divisible by 2 for vectorized access");

#define LAUNCH_TEMPLATED(type)                                        \
  do {                                                                \
    multi_layer_block_kv_transfer_templated<type>(                    \
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids,   \
        device, direction, shape_desc, lmcache_chunk_size,           \
        gpu_kv_format, skip_prefix_n_blocks);                        \
  } while (0)

  if (head_bytes % sizeof(int64_t) == 0) {
    LAUNCH_TEMPLATED(int64_t);   // 8 bytes per transfer
  } else if (head_bytes % sizeof(int32_t) == 0) {
    LAUNCH_TEMPLATED(int32_t);   // 4 bytes per transfer
  } else {
    LAUNCH_TEMPLATED(int16_t);   // 2 bytes per transfer (minimum)
  }

#undef LAUNCH_TEMPLATED
}
