// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header is intentionally independent of CUDA. It must compile without
// any CUDA headers present. The SYCL build uses only the headers below.

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <vector>

#include "mem_kernels_sycl.h"  // TransferDirection, GPUKVFormat

/**
 * Shape descriptor for a vLLM paged KV buffer.
 *
 * Mirrors the CUDA ``PageBufferShapeDesc`` in ``csrc/mp_mem_kernels.cuh``
 * but uses plain ``inline`` instead of ``__host__ __device__`` so it
 * compiles without any CUDA headers.
 */
struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size (tokens per block)
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes per scalar (1 or 2)
  // Physical per-block stride in source-dtype element units, used by
  // formats whose dim-0 is the block axis to step over padding bytes.
  // 0 means "unset — fall back to the format-specific tight stride".
  //
  // CONTRACT: pass ``tensor.stride(0)`` verbatim; do NOT pre-multiply
  // by any inner dimension.
  //
  // Honoured today only by NL_X_NB_BS_HS (MLA). All other formats
  // ignore this field.
  int block_stride_elems;

  /**
   * Number of ScalarType elements per attention head.
   *
   * @tparam ScalarType  The working scalar type (e.g. int64_t, int32_t,
   *                     int16_t).
   * @return  hs * element_size / sizeof(ScalarType)
   */
  template <typename ScalarType>
  inline size_t scalars_per_head() const {
    return static_cast<size_t>(hs) * element_size / sizeof(ScalarType);
  }

  /**
   * Number of ScalarType elements per token (all heads).
   *
   * @tparam ScalarType  The working scalar type.
   * @return  nh * hs * element_size / sizeof(ScalarType)
   */
  template <typename ScalarType>
  inline size_t scalars_per_token() const {
    return static_cast<size_t>(nh) * hs * element_size / sizeof(ScalarType);
  }

  /**
   * Physical per-block stride in ScalarType element units.
   *
   * Returns the tight ``bs * nh * hs`` stride by default, or the
   * physical ``block_stride_elems`` stride when dim-0 carries padding
   * (today only NL_X_NB_BS_HS / MLA).
   *
   * @tparam ScalarType  The working scalar type.
   * @return  padded-or-tight stride in ScalarType units
   */
  template <typename ScalarType>
  inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

/**
 * Holds up to 4 typed pointers to LMCache memory objects.
 *
 * @tparam ScalarType  The working scalar type.
 */
template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0–4
};

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects (SYCL / XPU implementation).
 *
 * @param paged_buffer_ptrs_tensor  XPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor).
 * @param lmcache_objects_ptrs      Raw pointer values for up to 4 LMCache
 *                                  memory objects (XPU USM device pointers).
 * @param block_ids                 XPU int64 tensor of vLLM block indices,
 *                                  one entry per block across all objects.
 * @param device                    XPU device of the vLLM tensors.
 * @param direction                 H2D (LMCache → vLLM) or D2H (vLLM →
 *                                  LMCache).
 * @param shape_desc                Shape descriptor for the paged buffer.
 * @param lmcache_chunk_size        Tokens per LMCache memory object.
 * @param gpu_kv_format             GPUKVFormat identifier.  Only the 5
 *                                  NHD / MLA formats supported by the SYCL
 *                                  backend are accepted; others throw
 *                                  std::runtime_error.
 * @param skip_prefix_n_blocks      Number of leading blocks (by flat index)
 *                                  to leave untouched.
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs,
    const torch::Tensor& block_ids,
    const torch::Device& device,
    TransferDirection direction,
    PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks);
