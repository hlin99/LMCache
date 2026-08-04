# SPDX-License-Identifier: Apache-2.0
"""Parity tests: torch_ops.execute_cb_retrieve_plan_flat vs blend_v3 ABC loop.

These tests verify that the portable torch-based executor in
``lmcache/v1/platform/torch_ops.py`` produces bit-identical paged KV results
to the former per-wave A→B→C Python fallback that was inlined in blend_v3.py.

Coverage includes:
  - Shifted chunks (old_st != cur_st) and non-shifted (old_st == cur_st).
  - Mixed shifted/non-shifted chunks in the same request.
  - The split KV layout (NL_X_TWO_NB_NH_BS_HS).
  - The fused-packed HND layout (NL_X_NB_NH_BS_TWO_HS).
  - Partial chunk (fewer tokens than slot capacity).
  - Multiple waves (double-buffer slot reuse).
"""

# Standard
from dataclasses import dataclass

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform import torch_ops as py_ops

if torch_dev is None or not torch_dev.is_available():
    pytest.skip(
        "torch device not available, skipping parity test",
        allow_module_level=True,
    )

_DEV = torch.device(torch_device_type)

_NL = 4
_SPC = 8  # tokens per chunk / slot
_NH = 2
_HS = 16
_NB = 128  # blocks in the paged buffer
_BS = 4  # block size (tokens per block)


@dataclass(frozen=True)
class _FmtCase:
    """Layout geometry needed for both reference scatter and plan executor."""

    fmt: "py_ops.EngineKVFormat"
    kv_size: int  # chunk leading planes (1 fused, 2 split)
    hidden: int  # per-plane scalars per token
    head_stride: int  # rope stride per K head
    paged_shape: tuple  # shape of one per-layer paged tensor


_SPLIT = _FmtCase(
    fmt=py_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
    kv_size=2,
    hidden=_NH * _HS,
    head_stride=_HS,
    paged_shape=(2, _NB, _NH, _BS, _HS),
)

_PACKED = _FmtCase(
    fmt=py_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
    kv_size=1,
    hidden=_NH * 2 * _HS,
    head_stride=2 * _HS,
    paged_shape=(_NB, _NH, _BS, 2 * _HS),
)


def _build_slot_mapping(n_chunks: int, dev: torch.device) -> torch.Tensor:
    """Build a simple invertible slot mapping for n_chunks * _SPC tokens."""
    pos = torch.arange(n_chunks * _SPC, device=dev, dtype=torch.long)
    block_ids = torch.arange(_NB, device=dev, dtype=torch.long).flip(0)
    return block_ids[pos // _BS] * _BS + pos % _BS


def _reference_abc_scatter(
    case: _FmtCase,
    host_chunks: "list[torch.Tensor]",
    paged_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    old_sts: "list[int]",
    cur_sts: "list[int]",
    cos_sin: torch.Tensor,
) -> None:
    """Former blend_v3 per-wave A→B→C reference executed sequentially.

    Mirrors the old inline fallback in cb_retrieve_pre_computed:
      A. Copy host chunk to GPU temp slot.
      B. Re-apply RoPE only if old_st != cur_st.
      C. Scatter slot to paged KV cache.
    """
    dev = slot_mapping.device
    ramp = torch.arange(_SPC, device=dev, dtype=torch.long).repeat(_NL)
    for i, host in enumerate(host_chunks):
        # A: host → GPU temp slot
        buf = host.to(dev)
        # B: re-RoPE (only when position changes)
        if old_sts[i] != cur_sts[i]:
            k_view = buf[0].reshape(_NL * _SPC, _NH, case.head_stride)
            py_ops.rotary_embedding_k_fused_strided(
                old_sts[i] + ramp,
                cur_sts[i] + ramp,
                k_view,
                _HS,
                case.head_stride,
                cos_sin,
                True,
            )
        # C: scatter temp slot → paged KV
        py_ops.multi_layer_kv_transfer(
            buf,
            paged_ptrs,
            slot_mapping[i * _SPC : (i + 1) * _SPC],
            dev,
            _NB * _BS,
            py_ops.TransferDirection.H2D,
            case.fmt,
            block_size=_BS,
            head_size=_HS,
        )
    if torch_device_type == "cuda":
        torch.cuda.synchronize()


def _run_torch_ops_plan(
    case: _FmtCase,
    n_chunks: int,
    max_batch: int,
    host_chunks: "list[torch.Tensor]",
    paged_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    old_sts: "list[int]",
    cur_sts: "list[int]",
    cos_sin: torch.Tensor,
    slots: "list[torch.Tensor]",
) -> None:
    """Drive py_ops.execute_cb_retrieve_plan_flat with a double-buffer plan.

    Builds the same flat int64 table layout as _build_cb_retrieve_plan_flat
    in blend_v3.py and invokes the torch-ops executor.
    """
    chunk_bytes = case.kv_size * _NL * _SPC * case.hidden * torch.bfloat16.itemsize
    spec = py_ops.CBGroupSpec(
        paged_ptrs.data_ptr(),
        [s.data_ptr() for s in slots],
        _NL,
        _SPC,
        case.hidden,
        torch.bfloat16.itemsize,
        case.fmt,
        _NB * _BS,
        _BS,
        _HS,
        slot_mapping.data_ptr(),
        slot_mapping.numel(),
        cos_sin.data_ptr(),
        _HS,
        _NH,
        case.head_stride,
        15,  # at::ScalarType::BFloat16
        True,  # is_neox
    )
    wave = max_batch // 2
    staging, ropes, scatters, step_offsets = [], [], [], []
    for w0 in range(0, n_chunks, wave):
        step_idx = w0 // wave
        base = (step_idx % 2) * wave
        for j in range(min(wave, n_chunks - w0)):
            ci = w0 + j
            slot = base + j
            staging.append(
                (slots[slot].data_ptr(), host_chunks[ci].data_ptr(), chunk_bytes, 0)
            )
            ropes.append((0, slot, old_sts[ci], cur_sts[ci]))
            scatters.append((0, slot, ci * _SPC, _SPC))
        step_offsets.append((len(staging), len(ropes), len(scatters)))
    py_ops.execute_cb_retrieve_plan_flat(
        _DEV,
        1 << 26,
        [spec],
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.asarray(scatters, dtype=np.int64),
        np.asarray(step_offsets, dtype=np.int64),
    )
    if torch_device_type == "cuda":
        torch.cuda.synchronize()


def _paged_tensors(
    case: _FmtCase,
) -> "tuple[list[torch.Tensor], torch.Tensor]":
    """Allocate per-layer paged tensors and a pointer tensor."""
    layers = [torch.zeros(*case.paged_shape, dtype=torch.bfloat16, device=_DEV)]
    ptrs = torch.tensor([t.data_ptr() for t in layers], dtype=torch.long, device=_DEV)
    return layers, ptrs


@pytest.mark.parametrize(
    "fmt_key,case",
    [("split", _SPLIT), ("packed", _PACKED)],
    ids=["split", "packed"],
)
@pytest.mark.parametrize(
    "n_chunks,max_batch,shift_mask",
    [
        # All shifted
        (4, 4, "all"),
        # No shifted (pure prefix copy)
        (4, 4, "none"),
        # Mixed: alternating shifted/non-shifted
        (6, 4, "alternating"),
        # Many waves → double-buffer slot reuse
        (12, 4, "all"),
    ],
    ids=["all_shifted", "no_shift", "mixed", "multiwave"],
)
def test_torch_ops_parity_with_blend_v3_abc(
    n_chunks: int, max_batch: int, shift_mask: str, fmt_key: str, case: _FmtCase
) -> None:
    """torch_ops executor must produce identical paged KV as blend_v3 ABC loop.

    Validates correctness for shifted/non-shifted chunks, layout variants, and
    multi-wave slot reuse. The reference is the former inline A→B→C fallback.
    """
    torch.manual_seed(n_chunks + max_batch)
    dtype = torch.bfloat16

    host_chunks = [
        torch.randn(
            case.kv_size, _NL, _SPC, case.hidden, dtype=dtype
        ).pin_memory()
        for _ in range(n_chunks)
    ]
    cos_sin = torch.randn(8192, _HS, dtype=dtype, device=_DEV)
    slot_mapping = _build_slot_mapping(n_chunks, _DEV)

    # Build old_st / cur_st per chunk according to shift_mask
    old_sts: list[int] = []
    cur_sts: list[int] = []
    for i in range(n_chunks):
        cur = i * _SPC
        if shift_mask == "all":
            old = cur + 512
        elif shift_mask == "none":
            old = cur
        else:  # alternating
            old = cur + 512 if i % 2 == 0 else cur
        old_sts.append(old)
        cur_sts.append(cur)

    # Reference: old blend_v3 A→B→C loop (single layer for simplicity)
    ref_layers, ref_ptrs = _paged_tensors(case)
    _reference_abc_scatter(
        case, host_chunks, ref_ptrs, slot_mapping, old_sts, cur_sts, cos_sin
    )

    # Under test: torch_ops flat-plan executor
    new_layers, new_ptrs = _paged_tensors(case)
    slots = [
        torch.zeros(
            case.kv_size, _NL, _SPC, case.hidden, dtype=dtype, device=_DEV
        )
        for _ in range(max_batch)
    ]
    _run_torch_ops_plan(
        case,
        n_chunks,
        max_batch,
        host_chunks,
        new_ptrs,
        slot_mapping,
        old_sts,
        cur_sts,
        cos_sin,
        slots,
    )

    for layer_idx in range(len(ref_layers)):
        assert torch.equal(ref_layers[layer_idx], new_layers[layer_idx]), (
            f"layer {layer_idx} mismatch "
            f"(fmt={fmt_key}, n_chunks={n_chunks}, shift_mask={shift_mask})"
        )


@pytest.mark.parametrize(
    "fmt_key,case",
    [("split", _SPLIT), ("packed", _PACKED)],
    ids=["split", "packed"],
)
def test_torch_ops_partial_chunk_parity(fmt_key: str, case: _FmtCase) -> None:
    """Partial chunk (fewer tokens than slot capacity) must match reference."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    n_chunks = 2

    host_chunks = [
        torch.randn(
            case.kv_size, _NL, _SPC, case.hidden, dtype=dtype
        ).pin_memory()
        for _ in range(n_chunks)
    ]
    cos_sin = torch.randn(8192, _HS, dtype=dtype, device=_DEV)
    # Only use half the slot tokens for the slot_mapping to simulate partial chunk
    n_tok_used = _SPC // 2
    pos = torch.arange(n_chunks * n_tok_used, device=_DEV, dtype=torch.long)
    block_ids = torch.arange(_NB, device=_DEV, dtype=torch.long).flip(0)
    slot_mapping = block_ids[pos // _BS] * _BS + pos % _BS

    old_sts = [i * _SPC + 256 for i in range(n_chunks)]
    cur_sts = [i * _SPC for i in range(n_chunks)]

    # Reference: A→B→C per chunk using only n_tok_used tokens for scatter
    ref_layers, ref_ptrs = _paged_tensors(case)
    dev = _DEV
    ramp = torch.arange(_SPC, device=dev, dtype=torch.long).repeat(_NL)
    for i, host in enumerate(host_chunks):
        buf = host.to(dev)
        if old_sts[i] != cur_sts[i]:
            k_view = buf[0].reshape(_NL * _SPC, _NH, case.head_stride)
            py_ops.rotary_embedding_k_fused_strided(
                old_sts[i] + ramp,
                cur_sts[i] + ramp,
                k_view,
                _HS,
                case.head_stride,
                cos_sin,
                True,
            )
        py_ops.multi_layer_kv_transfer(
            buf[:, :, :n_tok_used, :].contiguous(),
            ref_ptrs,
            slot_mapping[i * n_tok_used : (i + 1) * n_tok_used],
            dev,
            _NB * _BS,
            py_ops.TransferDirection.H2D,
            case.fmt,
            block_size=_BS,
            head_size=_HS,
        )
    if torch_device_type == "cuda":
        torch.cuda.synchronize()

    # Under test: plan with partial n_tok
    new_layers, new_ptrs = _paged_tensors(case)
    max_batch = 4
    wave = max_batch // 2
    slots = [
        torch.zeros(
            case.kv_size, _NL, _SPC, case.hidden, dtype=dtype, device=_DEV
        )
        for _ in range(max_batch)
    ]
    chunk_bytes = case.kv_size * _NL * _SPC * case.hidden * dtype.itemsize
    spec = py_ops.CBGroupSpec(
        new_ptrs.data_ptr(),
        [s.data_ptr() for s in slots],
        _NL,
        _SPC,
        case.hidden,
        dtype.itemsize,
        case.fmt,
        _NB * _BS,
        _BS,
        _HS,
        slot_mapping.data_ptr(),
        slot_mapping.numel(),
        cos_sin.data_ptr(),
        _HS,
        _NH,
        case.head_stride,
        15,
        True,
    )
    staging, ropes, scatters, step_offsets = [], [], [], []
    for w0 in range(0, n_chunks, wave):
        step_idx = w0 // wave
        base = (step_idx % 2) * wave
        for j in range(min(wave, n_chunks - w0)):
            ci = w0 + j
            slot = base + j
            staging.append(
                (slots[slot].data_ptr(), host_chunks[ci].data_ptr(), chunk_bytes, 0)
            )
            ropes.append((0, slot, old_sts[ci], cur_sts[ci]))
            scatters.append((0, slot, ci * n_tok_used, n_tok_used))
        step_offsets.append((len(staging), len(ropes), len(scatters)))
    py_ops.execute_cb_retrieve_plan_flat(
        _DEV,
        1 << 26,
        [spec],
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.asarray(scatters, dtype=np.int64),
        np.asarray(step_offsets, dtype=np.int64),
    )
    if torch_device_type == "cuda":
        torch.cuda.synchronize()

    for layer_idx in range(len(ref_layers)):
        assert torch.equal(ref_layers[layer_idx], new_layers[layer_idx]), (
            f"layer {layer_idx} partial-chunk mismatch (fmt={fmt_key})"
        )
