# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the opt-in SGLang fused-content transfer buffers."""

# Future
from __future__ import annotations

# Standard
from types import SimpleNamespace
from typing import TYPE_CHECKING

# Third Party
import pytest
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter


def make_adapter(
    dtype: torch.dtype, *, fused: bool
) -> tuple[SGLangUnifiedKVAdapter, list[torch.Tensor], list[torch.Tensor]]:
    """Create a two-layer CPU MHA pool and its real public adapter."""
    components = pytest.importorskip("sglang.srt.mem_cache.unified_cache.components")
    # First Party
    from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter

    keys = [
        torch.arange(64, dtype=dtype).reshape(8, 2, 4) + layer * 100
        for layer in range(2)
    ]
    values = [key + 1000 for key in keys]
    pool = SimpleNamespace(k_buffer=keys, v_buffer=values)
    adapter = SGLangUnifiedKVAdapter(
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        req_to_token_pool=object(),
        tree_components=(components.ComponentType.FULL,),
        mamba_component=None,
        page_size=1,
        sliding_window_size=None,
        vllm_fused_kv=fused,
    )
    return adapter, keys, values


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_store_packs_kv_per_head_without_changing_native_pool(
    dtype: torch.dtype,
) -> None:
    adapter, keys, values = make_adapter(dtype, fused=True)
    group = adapter.resolve_registered_groups()[0]
    slots = torch.tensor([6, 1, 4])
    originals = [tensor.clone() for tensor in (*keys, *values)]
    for tensor in group.kv_tensors:
        tensor.zero_()

    adapter.pack_for_store(slots)

    assert len(group.kv_tensors) == 2
    for key, value, fused in zip(keys, values, group.kv_tensors, strict=True):
        assert fused.shape == (8, 2, 8)
        assert torch.equal(fused[slots], torch.cat((key[slots], value[slots]), dim=-1))
        assert torch.count_nonzero(fused[[0, 2, 3, 5, 7]]) == 0
    for actual, expected in zip((*keys, *values), originals, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_load_unpacks_vllm_content_only_into_selected_slots(dtype: torch.dtype) -> None:
    adapter, keys, values = make_adapter(dtype, fused=True)
    group = adapter.resolve_registered_groups()[0]
    slots = torch.tensor([6, 1, 4])
    originals = [tensor.clone() for tensor in (*keys, *values)]
    for key, value, fused in zip(keys, values, group.kv_tensors, strict=True):
        fused.copy_(torch.cat((key, value), dim=-1))
        key[slots] = 0
        value[slots] = 0

    adapter.unpack_after_load(slots)

    for actual, expected in zip((*keys, *values), originals, strict=True):
        assert torch.equal(actual, expected)


def test_default_registration_preserves_original_components() -> None:
    adapter, keys, values = make_adapter(torch.bfloat16, fused=False)
    tensors = adapter.resolve_registered_groups()[0].kv_tensors

    assert [tensor.data_ptr() for tensor in tensors] == [
        tensor.data_ptr() for tensor in (*keys, *values)
    ]


def test_fused_registration_rejects_recurrent_hybrid() -> None:
    components = pytest.importorskip("sglang.srt.mem_cache.unified_cache.components")
    # First Party
    from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter

    with pytest.raises(ValueError, match="FULL attention only"):
        SGLangUnifiedKVAdapter(
            token_to_kv_pool_allocator=object(),
            req_to_token_pool=object(),
            tree_components=(
                components.ComponentType.FULL,
                components.ComponentType.MAMBA,
            ),
            mamba_component=None,
            page_size=1,
            sliding_window_size=None,
            vllm_fused_kv=True,
        )


def test_hybrid_packs_grouped_conv_and_recurrent_rows_and_restores_slots() -> None:
    components = pytest.importorskip("sglang.srt.mem_cache.unified_cache.components")
    # First Party
    from lmcache.integration.sglang.unified_kv_adapter import SGLangUnifiedKVAdapter

    conv = torch.arange(6 * 4 * 6 * 3, dtype=torch.bfloat16).reshape(6, 4, 6, 3)
    temporal = torch.arange(6 * 4 * 1 * 2 * 2, dtype=torch.float32).reshape(
        6, 4, 1, 2, 2
    )
    original_conv, original_temporal = conv.clone(), temporal.clone()
    pool = SimpleNamespace(
        k_buffer=[torch.zeros(16, 1, 2, dtype=torch.bfloat16) for _ in range(2)],
        v_buffer=[torch.zeros(16, 1, 2, dtype=torch.bfloat16) for _ in range(2)],
    )
    allocator = SimpleNamespace(
        get_kvcache=lambda: pool,
        translate_kv_indices_for_transfer=lambda indices: indices,
    )
    state_pool = SimpleNamespace(
        mamba_cache=SimpleNamespace(conv=[conv], temporal=temporal),
        mamba_layer_ids=[0, 1, 2, 4, 5, 6],
    )
    adapter = SGLangUnifiedKVAdapter(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=SimpleNamespace(
            mamba_pool=state_pool, translate_mamba_indices=lambda indices: indices
        ),
        tree_components=(components.ComponentType.FULL, components.ComponentType.MAMBA),
        mamba_component=SimpleNamespace(mamba_checkpoint_grid=4),
        page_size=1,
        sliding_window_size=None,
        vllm_fused_kv=True,
        vllm_mamba_block_size=4,
    )
    groups = adapter.resolve_registered_groups()
    assert [group.name for group in groups] == ["mamba", "full"]
    assert groups[0].tokens_per_block == 4
    assert groups[0].slots_per_block == 1
    assert all(tensor.shape == (4, 1, 64) for tensor in groups[0].kv_tensors)
    indices = adapter.device_indices_by_group(
        torch.arange(8), mamba_value=torch.tensor([2])
    )
    assert indices[0].tolist() == [0, 2]
    assert indices[1].tolist() == list(range(8))

    adapter.pack_recurrent_for_store(indices[0])

    for packed, source_layer in zip(
        groups[0].kv_tensors, [0, 3, 1, 4, 2, 5], strict=True
    ):
        raw = packed[2].view(torch.uint8).flatten()
        expected = torch.cat(
            (
                original_conv[source_layer, 2].view(torch.uint8).flatten(),
                original_temporal[source_layer, 2].view(torch.uint8).flatten(),
            )
        )
        assert torch.equal(raw[:52], expected)
        assert torch.count_nonzero(raw[52:]) == 0
    conv[:, 2] = 0
    temporal[:, 2] = 0

    adapter.unpack_recurrent_after_load(indices[0])

    assert torch.equal(conv, original_conv)
    assert torch.equal(temporal, original_temporal)
    for transfer in (
        adapter.pack_recurrent_for_store,
        adapter.unpack_recurrent_after_load,
    ):
        with pytest.raises(ValueError, match="nonzero checkpoint"):
            transfer(torch.zeros(2, dtype=torch.int64))

    state_pool.mamba_cache.conv[0] = conv.transpose(-1, -2).contiguous()
    with pytest.raises(ValueError, match="DS QKV conv state"):
        adapter.resolve_registered_groups()
