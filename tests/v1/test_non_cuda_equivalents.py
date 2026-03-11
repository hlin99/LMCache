# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python fallbacks in lmcache.non_cuda_equivalents.

Two parameterised backends are exercised:
  - ``cuda_c_ops``      – uses the compiled CUDA C-extension (lmcache.c_ops)
  - ``cuda_cuda_py_ops``– uses the pure-Python fallback
                          (lmcache.non_cuda_equivalents)

Each test scenario is a function registered in SCENARIO_REGISTRY.  The
TestScenarios class runs every (backend, scenario) combination.
"""

# Standard
from typing import Any
import random

# Third Party
import pytest
import torch

# First Party
import lmcache.non_cuda_equivalents as py_ops

_CUDA_AVAILABLE = torch.cuda.is_available()

if _CUDA_AVAILABLE:
    try:
        # First Party
        import lmcache.c_ops as lmc_ops

        _HAS_C_OPS = True
    except ImportError:
        lmc_ops = None  # type: ignore[assignment]
        _HAS_C_OPS = False
else:
    lmc_ops = None  # type: ignore[assignment]
    _HAS_C_OPS = False

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Each backend is a tuple (name: str, ops: module).
# ``ops`` must expose: multi_layer_kv_transfer, TransferDirection, GPUKVFormat.
BACKENDS = [
    pytest.param(
        ("cuda_c_ops", lmc_ops),
        id="cuda_c_ops",
        marks=[
            pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available"),
            pytest.mark.skipif(not _HAS_C_OPS, reason="lmcache.c_ops not available"),
        ],
    ),
    pytest.param(
        ("cuda_cuda_py_ops", py_ops),
        id="cuda_cuda_py_ops",
        marks=[
            pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available"),
        ],
    ),
]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> tuple:
    """Provide a (name, ops) backend tuple to each test."""
    return request.param


# ---------------------------------------------------------------------------
# Helper: generate per-layer paged KV-cache tensors on *device*
# ---------------------------------------------------------------------------


def _make_paged_kv_caches(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    gpu_kv_format: int,
    device: str,
    dtype: torch.dtype,
) -> list:
    """Return a list of per-layer paged KV-cache tensors.

    :param int num_layers: Number of transformer layers.
    :param int num_blocks: Number of paged blocks per layer.
    :param int block_size: Tokens per block.
    :param int num_heads: Number of KV heads (ignored for MLA).
    :param int head_size: Head dimension.
    :param int gpu_kv_format: One of the :class:`py_ops.GPUKVFormat` values.
    :param str device: Target device string (e.g. ``"cuda"``).
    :param torch.dtype dtype: Element dtype.
    :returns: List of ``num_layers`` tensors in the requested layout.
    """
    if gpu_kv_format == py_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        shape = [2, num_blocks, block_size, num_heads, head_size]
    elif gpu_kv_format == py_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        shape = [num_blocks, 2, block_size, num_heads, head_size]
    elif gpu_kv_format == py_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # vLLM MLA
        shape = [num_blocks, block_size, head_size]
    else:
        raise ValueError(
            f"Unsupported format for _make_paged_kv_caches: {gpu_kv_format}"
        )

    return [torch.rand(shape, dtype=dtype, device=device) for _ in range(num_layers)]


def _compute_expected_d2h(
    kv_caches: list,
    slot_mapping_cpu: torch.Tensor,
    num_layers: int,
    num_tokens: int,
    hidden_size: int,
    gpu_kv_format: int,
    page_buffer_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the expected CPU key_value tensor for a D2H transfer.

    The result is computed by directly indexing the paged tensors on CPU,
    providing a format-agnostic ground truth independent of any transfer
    function.

    :param list kv_caches: Per-layer paged tensors (may be on CUDA or CPU).
    :param torch.Tensor slot_mapping_cpu: CPU int64 tensor of shape
        ``[num_tokens]`` mapping token positions to flat paged slots.
    :param int num_layers: Number of layers.
    :param int num_tokens: Number of tokens.
    :param int hidden_size: ``num_heads * head_size`` (or just ``head_size``
        for MLA).
    :param int gpu_kv_format: One of the :class:`py_ops.GPUKVFormat` values.
    :param int page_buffer_size: ``num_blocks * block_size``.
    :param int block_size: Tokens per page block.
    :param torch.dtype dtype: Element dtype.
    :returns: CPU tensor of shape ``[kv_dim, num_layers, num_tokens, hidden]``
        where ``kv_dim`` is 1 for MLA formats and 2 for non-MLA formats.
    """
    is_mla = gpu_kv_format in (
        py_ops.GPUKVFormat.NL_X_NB_BS_HS,
        py_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    kv_dim = 1 if is_mla else 2
    expected = torch.zeros(kv_dim, num_layers, num_tokens, hidden_size, dtype=dtype)

    # Valid token positions and their paged slot indices
    valid_mask = slot_mapping_cpu >= 0
    valid_slots_cpu = slot_mapping_cpu[valid_mask]

    for layer_id in range(num_layers):
        paged = kv_caches[layer_id].cpu()

        if is_mla:
            paged_flat = paged.reshape(page_buffer_size, hidden_size)
            expected[0, layer_id, valid_mask, :] = paged_flat[valid_slots_cpu]
        elif gpu_kv_format == py_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
            num_blocks = page_buffer_size // block_size
            paged_r = paged.reshape(num_blocks, 2, block_size, hidden_size)
            block_idx = (valid_slots_cpu // block_size).long()
            block_off = (valid_slots_cpu % block_size).long()
            # gathered: [num_valid, 2, hidden]
            gathered = paged_r[block_idx, :, block_off, :]
            expected[:, layer_id, valid_mask, :] = gathered.transpose(0, 1)
        else:
            # NL_X_TWO_NB_BS_NH_HS (and NB_NL_TWO_BS_NH_HS)
            paged_flat = paged.reshape(2, page_buffer_size, hidden_size)
            expected[:, layer_id, valid_mask, :] = paged_flat[:, valid_slots_cpu, :]

    return expected


# ---------------------------------------------------------------------------
# Scenario functions
# ---------------------------------------------------------------------------


def scenario_multi_layer_kv_transfer(backend: tuple) -> None:
    """Verify D2H (paged→key_value) and H2D (key_value→paged) transfers.

    For D2H: paged CUDA tensors are created with random data; the transfer
    function is called to copy them to a CPU key_value buffer; the result is
    compared against values obtained by direct indexing.

    For H2D: the CPU key_value is written first, then copied to fresh paged
    tensors; the result is verified by comparing the paged data at the
    transferred slots against the original CPU buffer.

    :param tuple backend: ``(name, ops)`` pair where *name* is a string ID
        and *ops* is the module exposing ``multi_layer_kv_transfer``,
        ``TransferDirection``, and ``GPUKVFormat``.
    """
    _name, ops = backend
    device = "cuda"
    dtype = torch.bfloat16

    num_layers = 4
    num_blocks = 50
    block_size = 16
    num_heads = 8
    head_size = 128
    num_tokens = 32
    hidden_size = num_heads * head_size
    page_buffer_size = num_blocks * block_size

    formats_and_block_sizes = [
        (py_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, 0),
        (py_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, block_size),
        (py_ops.GPUKVFormat.NL_X_NB_BS_HS, 0),
    ]

    for gpu_kv_format, bs in formats_and_block_sizes:
        is_mla = gpu_kv_format == py_ops.GPUKVFormat.NL_X_NB_BS_HS
        kv_dim = 1 if is_mla else 2
        eff_hidden = head_size if is_mla else hidden_size

        # Create paged CUDA tensors with random data.
        kv_caches = _make_paged_kv_caches(
            num_layers,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            gpu_kv_format,
            device,
            dtype,
        )

        # Register paged tensors for the Python fallback (no-op for C ops).
        for t in kv_caches:
            py_ops.register_tensor(t)

        # Build pointer tensor (pinned CPU for C-ops compatibility).
        kv_ptrs = torch.empty(num_layers, dtype=torch.int64, pin_memory=True)
        for i, t in enumerate(kv_caches):
            kv_ptrs[i] = t.data_ptr()

        # Random slot mapping on CUDA (matches C-ops calling convention).
        slots = random.sample(range(page_buffer_size), num_tokens)
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
        slot_mapping_cpu = slot_mapping.cpu()

        # ------------------------------------------------------------------
        # D2H test: paged CUDA → CPU key_value
        # ------------------------------------------------------------------
        key_value = torch.zeros(kv_dim, num_layers, num_tokens, eff_hidden, dtype=dtype)

        ops.multi_layer_kv_transfer(
            key_value,
            kv_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            ops.TransferDirection.D2H,
            gpu_kv_format,
            bs,
        )

        expected = _compute_expected_d2h(
            kv_caches,
            slot_mapping_cpu,
            num_layers,
            num_tokens,
            eff_hidden,
            gpu_kv_format,
            page_buffer_size,
            block_size,
            dtype,
        )

        # Detailed assertion message to aid debugging.
        for kv in range(kv_dim):
            for layer in range(num_layers):
                for token in range(num_tokens):
                    if slot_mapping_cpu[token] < 0:
                        continue
                    ok = torch.equal(
                        key_value[kv, layer, token],
                        expected[kv, layer, token],
                    )
                    assert ok, (
                        f"Mismatch: {kv} paged2lmc (tensor_list=False), "
                        f"kv={kv}, layer={layer}, token={token}"
                    )

        # ------------------------------------------------------------------
        # H2D test: CPU key_value → fresh paged CUDA tensors
        # ------------------------------------------------------------------
        key_value_src = torch.rand(
            kv_dim, num_layers, num_tokens, eff_hidden, dtype=dtype
        )

        kv_caches_new = _make_paged_kv_caches(
            num_layers,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            gpu_kv_format,
            device,
            dtype,
        )
        for t in kv_caches_new:
            py_ops.register_tensor(t)

        kv_ptrs_new = torch.empty(num_layers, dtype=torch.int64, pin_memory=True)
        for i, t in enumerate(kv_caches_new):
            kv_ptrs_new[i] = t.data_ptr()

        ops.multi_layer_kv_transfer(
            key_value_src,
            kv_ptrs_new,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            ops.TransferDirection.H2D,
            gpu_kv_format,
            bs,
        )

        # Verify H2D by reading back with D2H (using Python fallback as oracle).
        key_value_readback = torch.zeros(
            kv_dim, num_layers, num_tokens, eff_hidden, dtype=dtype
        )
        py_ops.multi_layer_kv_transfer(
            key_value_readback,
            kv_ptrs_new,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            py_ops.TransferDirection.D2H,
            gpu_kv_format,
            bs,
        )

        for kv in range(kv_dim):
            for layer in range(num_layers):
                for token in range(num_tokens):
                    if slot_mapping_cpu[token] < 0:
                        continue
                    ok = torch.equal(
                        key_value_src[kv, layer, token],
                        key_value_readback[kv, layer, token],
                    )
                    assert ok, (
                        f"Mismatch: {kv} lmc2paged (tensor_list=False), "
                        f"kv={kv}, layer={layer}, token={token}"
                    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: dict[str, Any] = {
    "multi_layer_kv_transfer": scenario_multi_layer_kv_transfer,
}

# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestScenarios:
    """Runs every registered scenario against every configured backend."""

    @pytest.mark.parametrize(
        "name,fn",
        [
            pytest.param(name, fn, id=f"{name}-{fn.__name__}")
            for name, fn in SCENARIO_REGISTRY.items()
        ],
    )
    def test_1_scenario(
        self,
        backend: tuple,
        name: str,
        fn: Any,
    ) -> None:
        """Run a single scenario with a specific backend configuration.

        Each (scenario, backend) pair is a separate pytest test case, giving
        fine-grained visibility into which combinations pass or fail.

        :param tuple backend: ``(name, ops)`` backend descriptor.
        :param str name: Scenario name from :data:`SCENARIO_REGISTRY`.
        :param fn: Scenario callable; receives *backend* and asserts results.
        """
        fn(backend)
