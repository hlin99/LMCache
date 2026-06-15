# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import pytest
import torch

pytest.importorskip(
    "lmcache.xpu_ops",
    reason="Requires SYCL extension lmcache.xpu_ops",
)

# First Party
import lmcache.xpu_ops as xpu_ops

# Skip all tests if an XPU device is unavailable.
pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or torch.xpu.device_count() == 0,
    reason="No Intel XPU present",
)

# ---------------------------------------------------------------------------
# Supported formats (5 NHD + MLA formats covered by the SYCL backend)
# ---------------------------------------------------------------------------
FMT_CROSS_LAYER = xpu_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
FMT_NORMAL = xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
FMT_FLASH_INFER = xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
FMT_MLA = xpu_ops.GPUKVFormat.NL_X_NB_BS_HS
FMT_SGLANG_MLA = xpu_ops.GPUKVFormat.NL_X_NBBS_ONE_HS

# (gpu_kv_format, nl, nh, hs, is_mla)
FORMAT_PARAMS = [
    (FMT_CROSS_LAYER, 4, 8, 128, False),
    (FMT_NORMAL, 4, 8, 128, False),
    (FMT_FLASH_INFER, 4, 8, 128, False),
    (FMT_MLA, 4, 1, 576, True),
    (FMT_SGLANG_MLA, 4, 1, 576, True),
]


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


def _create_random_tensor(shape, dtype, device):
    return torch.rand(shape, dtype=dtype, device=device)


def _create_zero_tensor(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)


def create_vllm_tensors(gpu_kv_format, nl, nb, bs, nh, hs, dtype, device):
    """Create random vLLM paged-buffer tensors for the given format."""
    nbbs = nb * bs
    if gpu_kv_format == FMT_NORMAL:
        return [
            _create_random_tensor([2, nb, bs, nh, hs], dtype, device) for _ in range(nl)
        ]
    elif gpu_kv_format == FMT_CROSS_LAYER:
        return [_create_random_tensor([nb, nl, 2, bs, nh, hs], dtype, device)]
    elif gpu_kv_format == FMT_FLASH_INFER:
        return [
            _create_random_tensor([nb, 2, bs, nh, hs], dtype, device) for _ in range(nl)
        ]
    elif gpu_kv_format == FMT_MLA:
        return [_create_random_tensor([nb, bs, hs], dtype, device) for _ in range(nl)]
    elif gpu_kv_format == FMT_SGLANG_MLA:
        return [_create_random_tensor([nbbs, 1, hs], dtype, device) for _ in range(nl)]
    raise ValueError(f"Unknown format: {gpu_kv_format}")


def create_zero_vllm_tensors(gpu_kv_format, nl, nb, bs, nh, hs, dtype, device):
    """Create zero-filled vLLM paged-buffer tensors for the given format."""
    nbbs = nb * bs
    if gpu_kv_format == FMT_NORMAL:
        return [
            _create_zero_tensor([2, nb, bs, nh, hs], dtype, device) for _ in range(nl)
        ]
    elif gpu_kv_format == FMT_CROSS_LAYER:
        return [_create_zero_tensor([nb, nl, 2, bs, nh, hs], dtype, device)]
    elif gpu_kv_format == FMT_FLASH_INFER:
        return [
            _create_zero_tensor([nb, 2, bs, nh, hs], dtype, device) for _ in range(nl)
        ]
    elif gpu_kv_format == FMT_MLA:
        return [_create_zero_tensor([nb, bs, hs], dtype, device) for _ in range(nl)]
    elif gpu_kv_format == FMT_SGLANG_MLA:
        return [_create_zero_tensor([nbbs, 1, hs], dtype, device) for _ in range(nl)]
    raise ValueError(f"Unknown format: {gpu_kv_format}")


def create_memory_objects(
    kv_dim, nl, tokens_per_object, hidden_dim, num_objects, dtype, device
):
    """Create zero-filled LMCache memory objects [2, L, T, NH*HS]."""
    shape = [kv_dim, nl, tokens_per_object, hidden_dim]
    return [_create_zero_tensor(shape, dtype, device) for _ in range(num_objects)]


def get_block_data(vllm_tensors, gpu_kv_format, nl, bs, block_idx):
    """Extract all-layer data for *block_idx* as a list of layer tensors."""
    results = []
    for layer_idx in range(nl):
        if gpu_kv_format == FMT_NORMAL:
            results.append(vllm_tensors[layer_idx][:, block_idx, :, :, :].clone())
        elif gpu_kv_format == FMT_CROSS_LAYER:
            results.append(vllm_tensors[0][block_idx, layer_idx, :, :, :, :].clone())
        elif gpu_kv_format == FMT_FLASH_INFER:
            results.append(vllm_tensors[layer_idx][block_idx, :, :, :, :].clone())
        elif gpu_kv_format == FMT_MLA:
            results.append(vllm_tensors[layer_idx][block_idx, :, :].clone())
        elif gpu_kv_format == FMT_SGLANG_MLA:
            ts, ed = block_idx * bs, (block_idx + 1) * bs
            results.append(vllm_tensors[layer_idx][ts:ed, 0, :].clone())
    return results


# ---------------------------------------------------------------------------
# Kernel call helper
# ---------------------------------------------------------------------------


def call_block_kernel(
    vllm_tensors,
    mem_objects,
    block_ids,
    gpu_kv_format,
    direction,
    nl,
    nb,
    bs,
    nh,
    hs,
    is_mla,
    tokens_per_object,
    skip_prefix_n_blocks=0,
):
    """Call xpu_ops.multi_layer_block_kv_transfer with the given arguments."""
    device = vllm_tensors[0].device

    shape_desc = xpu_ops.PageBufferShapeDesc()
    shape_desc.kv_size = 1 if is_mla else 2
    shape_desc.nl = nl
    shape_desc.nb = nb
    shape_desc.bs = bs
    shape_desc.nh = nh
    shape_desc.hs = hs
    shape_desc.element_size = vllm_tensors[0].element_size()
    shape_desc.block_stride_elems = 0

    ptrs = [t.data_ptr() for t in vllm_tensors]
    paged_buffer_ptrs_tensor = torch.tensor(ptrs, dtype=torch.int64, device=device)
    lmcache_objects_ptrs = [m.data_ptr() for m in mem_objects]

    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    xpu_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs_tensor,
        lmcache_objects_ptrs,
        block_ids_gpu,
        device,
        direction,
        shape_desc,
        tokens_per_object,
        gpu_kv_format,
        skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
NB = 200  # Must be >= 2 * TOTAL_BLOCKS so disjoint D2H and H2D block ID sets fit
BS = 16
NUM_MEMORY_OBJECTS = 4
TOKENS_PER_OBJECT = 256
BLOCKS_PER_OBJECT = TOKENS_PER_OBJECT // BS  # 16
TOTAL_BLOCKS = NUM_MEMORY_OBJECTS * BLOCKS_PER_OBJECT  # 64


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gpu_kv_format,nl,nh,hs,is_mla",
    FORMAT_PARAMS,
    ids=["cross_layer", "normal", "flash_infer", "mla", "sglang_mla"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_block_transfer_roundtrip(gpu_kv_format, nl, nh, hs, is_mla, dtype):
    """
    D2H → H2D roundtrip: data written via D2H must be recoverable via H2D.

    Uses disjoint source and target block IDs so the result is unambiguous.
    """
    device = torch.device("xpu")
    kv_dim = 1 if is_mla else 2
    hidden_dim = nh * hs

    source_vllm = create_vllm_tensors(gpu_kv_format, nl, NB, BS, nh, hs, dtype, device)
    target_vllm = create_zero_vllm_tensors(
        gpu_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    mem_objects = create_memory_objects(
        kv_dim, nl, TOKENS_PER_OBJECT, hidden_dim, NUM_MEMORY_OBJECTS, dtype, device
    )

    rng_d2h = random.Random(42)
    block_ids_d2h = rng_d2h.sample(range(NB), TOTAL_BLOCKS)
    excluded = set(block_ids_d2h)
    available = [i for i in range(NB) if i not in excluded]
    rng_h2d = random.Random(123)
    block_ids_h2d = rng_h2d.sample(available, TOTAL_BLOCKS)

    # D2H: source vLLM → LMCache memory objects
    call_block_kernel(
        source_vllm,
        mem_objects,
        block_ids_d2h,
        gpu_kv_format,
        xpu_ops.TransferDirection.D2H,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
    )
    torch.xpu.synchronize()

    # H2D: LMCache memory objects → target vLLM
    call_block_kernel(
        target_vllm,
        mem_objects,
        block_ids_h2d,
        gpu_kv_format,
        xpu_ops.TransferDirection.H2D,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
    )
    torch.xpu.synchronize()

    # Verify: target[h2d_block_i] == source[d2h_block_i]
    for i in range(TOTAL_BLOCKS):
        src_data = get_block_data(source_vllm, gpu_kv_format, nl, BS, block_ids_d2h[i])
        tgt_data = get_block_data(target_vllm, gpu_kv_format, nl, BS, block_ids_h2d[i])
        for layer_idx in range(nl):
            assert torch.equal(src_data[layer_idx], tgt_data[layer_idx]), (
                f"Mismatch at block {i}, layer {layer_idx}"
            )


@pytest.mark.parametrize(
    "gpu_kv_format,nl,nh,hs,is_mla",
    FORMAT_PARAMS,
    ids=["cross_layer", "normal", "flash_infer", "mla", "sglang_mla"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_block_transfer_skip_prefix(gpu_kv_format, nl, nh, hs, is_mla, dtype):
    """Verify skip_prefix_n_blocks skips the first N blocks globally."""
    device = torch.device("xpu")
    kv_dim = 1 if is_mla else 2
    hidden_dim = nh * hs
    skip = 4

    source_vllm = create_vllm_tensors(gpu_kv_format, nl, NB, BS, nh, hs, dtype, device)
    target_vllm = create_zero_vllm_tensors(
        gpu_kv_format, nl, NB, BS, nh, hs, dtype, device
    )
    mem_objects = create_memory_objects(
        kv_dim, nl, TOKENS_PER_OBJECT, hidden_dim, NUM_MEMORY_OBJECTS, dtype, device
    )

    rng_d2h = random.Random(42)
    block_ids_d2h = rng_d2h.sample(range(NB), TOTAL_BLOCKS)
    excluded = set(block_ids_d2h)
    available = [i for i in range(NB) if i not in excluded]
    rng_h2d = random.Random(123)
    block_ids_h2d = rng_h2d.sample(available, TOTAL_BLOCKS)

    # D2H with prefix skip
    call_block_kernel(
        source_vllm,
        mem_objects,
        block_ids_d2h,
        gpu_kv_format,
        xpu_ops.TransferDirection.D2H,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
        skip_prefix_n_blocks=skip,
    )
    torch.xpu.synchronize()

    # H2D with prefix skip
    call_block_kernel(
        target_vllm,
        mem_objects,
        block_ids_h2d,
        gpu_kv_format,
        xpu_ops.TransferDirection.H2D,
        nl,
        NB,
        BS,
        nh,
        hs,
        is_mla,
        TOKENS_PER_OBJECT,
        skip_prefix_n_blocks=skip,
    )
    torch.xpu.synchronize()

    # Non-skipped blocks [skip, TOTAL_BLOCKS) must match.
    for i in range(skip, TOTAL_BLOCKS):
        src_data = get_block_data(source_vllm, gpu_kv_format, nl, BS, block_ids_d2h[i])
        tgt_data = get_block_data(target_vllm, gpu_kv_format, nl, BS, block_ids_h2d[i])
        for layer_idx in range(nl):
            assert torch.equal(src_data[layer_idx], tgt_data[layer_idx]), (
                f"Mismatch at block {i}, layer {layer_idx}"
            )

    # Skipped blocks in target must remain zero.
    for i in range(skip):
        tgt_data = get_block_data(target_vllm, gpu_kv_format, nl, BS, block_ids_h2d[i])
        for layer_idx in range(nl):
            block = tgt_data[layer_idx].to(torch.float32)
            assert block.abs().sum().item() == 0, (
                f"Skipped block {i}, layer {layer_idx} is not zero"
            )
