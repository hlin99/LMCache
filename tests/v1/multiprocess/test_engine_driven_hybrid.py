# SPDX-License-Identifier: Apache-2.0
"""Tests for Engine-driven hybrid/HMA KV-cache multi-group support.

Coverage
--------
- ``group_copy`` planning utilities (no GPU required)
- ``EngineDrivenContextMetadata`` per-group helpers
- ``GroupLayoutSpec`` / ``RegisterEngineDrivenContextPayload`` encoding
- ``server_transfer`` multi-group reserve helpers
- Single-group backward compatibility
- Full-attention + sliding-window hybrid plan / flatten / unflatten round-trip
- ``skip_first_n_tokens`` propagated through scatter
- Block-ID underflow fail-closed behaviour
- ``gather_engine_groups`` / ``scatter_engine_groups`` with mocked primitives
"""

# Standard
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.group_copy import (
    GroupCopyPlan,
    build_group_kv_subset,
    compute_group_blocks_in_chunk,
    flatten_chunks_group_major,
    gather_engine_groups,
    plan_group_copy,
    scatter_engine_groups,
    unflatten_chunks_group_major,
)


# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------


def _make_kv(
    num_layers: int = 4,
    num_blocks: int = 8,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Create a simple NHD KV-cache dict for testing."""
    return {
        f"layer_{i}": torch.randn(2, num_blocks, block_size, num_heads, head_size)
        for i in range(num_layers)
    }


def _make_obj_key(
    chunk_hash: str = "abc",
    object_group_id: int = 0,
    world_size: int = 1,
) -> ObjectKey:
    """Create a minimal ObjectKey for testing."""
    return ObjectKey(
        chunk_hash=chunk_hash,
        object_group_id=object_group_id,
        world_size=world_size,
        model_name="m",
    )


# ---------------------------------------------------------------------------
# group_copy — pure planning utilities
# ---------------------------------------------------------------------------


class TestFlattenUnflatten:
    """Round-trip tests for flatten/unflatten group-major wire helpers."""

    def test_single_group_round_trip(self) -> None:
        """flatten then unflatten is identity for a single group."""
        g0 = [torch.zeros(3), torch.ones(3), torch.full((3,), 2.0)]
        flat = flatten_chunks_group_major([g0])
        assert len(flat) == 3
        recovered = unflatten_chunks_group_major(flat, [3])
        assert len(recovered) == 1
        assert len(recovered[0]) == 3
        for a, b in zip(g0, recovered[0]):
            assert torch.equal(a, b)

    def test_two_group_round_trip(self) -> None:
        """flatten then unflatten is identity for two groups of unequal size."""
        g0 = [torch.zeros(2), torch.ones(2)]
        g1 = [torch.full((2,), 3.0), torch.full((2,), 4.0), torch.full((2,), 5.0)]
        flat = flatten_chunks_group_major([g0, g1])
        assert len(flat) == 5
        recovered = unflatten_chunks_group_major(flat, [2, 3])
        assert all(torch.equal(a, b) for a, b in zip(g0, recovered[0]))
        assert all(torch.equal(a, b) for a, b in zip(g1, recovered[1]))

    def test_unflatten_raises_on_count_mismatch(self) -> None:
        """unflatten raises ValueError when group_counts sum != len(flat)."""
        flat = [torch.zeros(1)] * 4
        with pytest.raises(ValueError, match="group_counts sum"):
            unflatten_chunks_group_major(flat, [2, 1])  # sum=3 != 4

    def test_empty_groups(self) -> None:
        """Handles a group with zero chunks."""
        g0 = [torch.zeros(4)]
        g1: list[torch.Tensor] = []
        flat = flatten_chunks_group_major([g0, g1])
        assert len(flat) == 1
        recovered = unflatten_chunks_group_major(flat, [1, 0])
        assert len(recovered) == 2
        assert len(recovered[1]) == 0


class TestComputeGroupBlocksInChunk:
    """compute_group_blocks_in_chunk derives per-group blocks-per-chunk."""

    def test_same_block_size(self) -> None:
        """Returns ref_blocks_in_chunk when group_block_size == ref_block_size."""
        result = compute_group_blocks_in_chunk(4, 16, 16)
        assert result == 4

    def test_larger_block_size(self) -> None:
        """Group with 2× block size gets half as many blocks per chunk."""
        # chunk_tokens = 4*16 = 64; new group block_size=32 → 64/32=2
        result = compute_group_blocks_in_chunk(4, 16, 32)
        assert result == 2

    def test_smaller_block_size(self) -> None:
        """Group with ½ block size gets twice as many blocks per chunk."""
        result = compute_group_blocks_in_chunk(4, 16, 8)
        assert result == 8

    def test_tokens_per_block_override(self) -> None:
        """tokens_per_block overrides group_block_size when non-zero."""
        # chunk_tokens=64, tokens_per_block=16 → 4 (ignores group_block_size=32)
        result = compute_group_blocks_in_chunk(4, 16, 32, group_tokens_per_block=16)
        assert result == 4

    def test_raises_on_non_divisible(self) -> None:
        """Raises ValueError when chunk_tokens is not divisible by block_size."""
        with pytest.raises(ValueError, match="not divisible"):
            compute_group_blocks_in_chunk(3, 16, 7)


class TestBuildGroupKvSubset:
    """build_group_kv_subset extracts the right layers."""

    def test_subset_single_index(self) -> None:
        """Single layer index extracts exactly one entry."""
        kv = _make_kv(num_layers=4)
        sub = build_group_kv_subset(kv, [2])
        assert list(sub.keys()) == ["layer_2"]

    def test_subset_multiple_indices(self) -> None:
        """Multiple indices preserve insertion order."""
        kv = _make_kv(num_layers=4)
        sub = build_group_kv_subset(kv, [0, 2])
        assert list(sub.keys()) == ["layer_0", "layer_2"]

    def test_all_layers(self) -> None:
        """Selecting all layers returns the whole dict."""
        kv = _make_kv(num_layers=4)
        sub = build_group_kv_subset(kv, [0, 1, 2, 3])
        assert list(sub.keys()) == list(kv.keys())

    def test_out_of_range_raises(self) -> None:
        """Out-of-range index raises ValueError."""
        kv = _make_kv(num_layers=2)
        with pytest.raises(ValueError, match="out of range"):
            build_group_kv_subset(kv, [5])


class TestPlanGroupCopy:
    """plan_group_copy builds correct GroupCopyPlan entries."""

    def test_empty_infos_returns_empty(self) -> None:
        """Empty engine_group_infos returns empty list (legacy fallback)."""
        kv = _make_kv(num_layers=4)
        result = plan_group_copy(kv, [[0, 1, 2, 3]], 2, [], [4])
        assert result == []

    def test_single_group_same_block_size(self) -> None:
        """Single group with matching block sizes produces one plan."""
        kv = _make_kv(num_layers=2)
        info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
        plans = plan_group_copy(
            kv,
            [[0, 1, 2, 3]],  # 4 block IDs
            blocks_in_chunk=2,
            engine_group_infos=[info],
            group_block_sizes=[4],
        )
        assert len(plans) == 1
        plan = plans[0]
        assert plan.lmcache_group_idx == 0
        assert plan.engine_group_id == 0
        assert plan.flat_block_ids == [0, 1, 2, 3]
        assert plan.blocks_in_chunk == 2
        assert plan.num_chunks == 2  # 4 blocks / 2 per chunk

    def test_two_groups_different_block_sizes(self) -> None:
        """Two groups with different block sizes produce two plans with adjusted
        blocks_in_chunk.
        """
        kv = _make_kv(num_layers=4)
        infos = [
            # Group 0: layers 0-1, block_size=4
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1)),
            # Group 1: layers 2-3, block_size=8 (2× group 0)
            EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3)),
        ]
        # Reference: 2 blocks/chunk at block_size=4 → chunk_tokens=8
        # Group 1: chunk_tokens=8, block_size=8 → 1 block/chunk
        plans = plan_group_copy(
            kv,
            [[0, 1, 2, 3], [10, 11]],
            blocks_in_chunk=2,
            engine_group_infos=infos,
            group_block_sizes=[4, 8],
        )
        assert len(plans) == 2
        # Group 0
        assert plans[0].blocks_in_chunk == 2
        assert plans[0].num_chunks == 2
        # Group 1: 8 chunk_tokens / 8 block_size = 1 block/chunk; 2 blocks → 2 chunks
        assert plans[1].blocks_in_chunk == 1
        assert plans[1].num_chunks == 2

    def test_kv_subset_assigned_by_layer_indices(self) -> None:
        """Each plan's kv_subset contains only the layers in its EngineGroupInfo."""
        kv = _make_kv(num_layers=4)
        infos = [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1)),
            EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3)),
        ]
        plans = plan_group_copy(
            kv, [[0, 1], [0, 1]], 2, infos, [4, 4]
        )
        assert set(plans[0].kv_subset.keys()) == {"layer_0", "layer_1"}
        assert set(plans[1].kv_subset.keys()) == {"layer_2", "layer_3"}

    def test_sliding_window_group_and_full_attention_group(self) -> None:
        """Groups with sw_size_tokens set are handled by plan_group_copy.

        plan_group_copy is agnostic to sw_size_tokens (the engine context
        handles skipping); only num_chunks is affected by blocks_in_chunk.
        """
        kv = _make_kv(num_layers=4)
        infos = [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), sw_size_tokens=-1),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(2, 3), sw_size_tokens=16
            ),
        ]
        plans = plan_group_copy(
            kv, [[0, 1, 2, 3], [0, 1, 2, 3]], 2, infos, [4, 4]
        )
        assert len(plans) == 2
        assert plans[0].num_chunks == 2
        assert plans[1].num_chunks == 2


# ---------------------------------------------------------------------------
# gather_engine_groups / scatter_engine_groups with mocked primitives
# ---------------------------------------------------------------------------


class TestGatherScatterEngineGroups:
    """gather/scatter round-trip using mocked GPU primitives."""

    def _make_plans(
        self,
        kv: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
        block_sizes: list[int],
    ) -> list[GroupCopyPlan]:
        """Build GroupCopyPlans for tests without a real registration."""
        infos = [
            EngineGroupInfo(engine_group_id=g, layer_indices=tuple(range(len(kv))))
            for g in range(len(block_ids))
        ]
        return plan_group_copy(kv, block_ids, blocks_in_chunk, infos, block_sizes)

    def test_gather_calls_primitive_per_group(self) -> None:
        """gather_engine_groups calls gather_paged_kv_to_cpu once per group."""
        kv = _make_kv(num_layers=2)
        plans = self._make_plans(kv, [[0, 1], [2, 3]], blocks_in_chunk=1, block_sizes=[4, 4])
        call_count = 0

        def fake_gather(
            kv_caches: dict,
            block_ids: list,
            blocks_in_chunk: int,
            **kwargs: Any,
        ) -> list[torch.Tensor]:
            nonlocal call_count
            call_count += 1
            return [torch.zeros(1)] * (len(block_ids) // blocks_in_chunk)

        with patch(
            "lmcache.v1.multiprocess.transfer_context.group_copy.gather_paged_kv_to_cpu",
            side_effect=fake_gather,
        ):
            result = gather_engine_groups(plans)

        assert call_count == 2
        assert len(result) == 2

    def test_scatter_calls_primitive_per_group(self) -> None:
        """scatter_engine_groups calls scatter_cpu_to_paged_kv once per group."""
        kv = _make_kv(num_layers=2)
        plans = self._make_plans(kv, [[0, 1], [2, 3]], blocks_in_chunk=1, block_sizes=[4, 4])
        chunks_per_group = [
            [torch.zeros(1), torch.zeros(1)],
            [torch.zeros(1), torch.zeros(1)],
        ]
        call_count = 0

        def fake_scatter(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1

        with patch(
            "lmcache.v1.multiprocess.transfer_context.group_copy.scatter_cpu_to_paged_kv",
            side_effect=fake_scatter,
        ):
            scatter_engine_groups(plans, chunks_per_group)

        assert call_count == 2

    def test_scatter_propagates_skip_first_n_tokens(self) -> None:
        """skip_first_n_tokens is forwarded to every scatter_cpu_to_paged_kv call."""
        kv = _make_kv(num_layers=2)
        plans = self._make_plans(kv, [[0, 1], [2, 3]], blocks_in_chunk=1, block_sizes=[4, 4])
        chunks_per_group = [[torch.zeros(1)] * 2, [torch.zeros(1)] * 2]
        captured: list[int] = []

        def fake_scatter(*args: Any, skip_first_n_tokens: int = 0, **kwargs: Any) -> None:
            captured.append(skip_first_n_tokens)

        with patch(
            "lmcache.v1.multiprocess.transfer_context.group_copy.scatter_cpu_to_paged_kv",
            side_effect=fake_scatter,
        ):
            scatter_engine_groups(plans, chunks_per_group, skip_first_n_tokens=8)

        assert captured == [8, 8]

    def test_scatter_raises_on_group_count_mismatch(self) -> None:
        """scatter_engine_groups raises ValueError on len mismatch."""
        kv = _make_kv(num_layers=2)
        plans = self._make_plans(kv, [[0, 1], [2, 3]], blocks_in_chunk=1, block_sizes=[4, 4])
        # Only provide 1 group of chunks but plans has 2 groups
        with pytest.raises(ValueError, match="chunks_per_group"):
            scatter_engine_groups(plans, [[torch.zeros(1)]])

    def test_gather_returns_empty_for_empty_plans(self) -> None:
        """gather_engine_groups returns [] when plans is empty."""
        assert gather_engine_groups([]) == []

    def test_scatter_no_ops_for_empty_plans(self) -> None:
        """scatter_engine_groups is a no-op when plans is empty."""
        scatter_engine_groups([], [])  # Should not raise


# ---------------------------------------------------------------------------
# EngineDrivenContextMetadata per-group helpers
# ---------------------------------------------------------------------------


class TestEngineDrivenContextMetadataPerGroup:
    """EngineDrivenContextMetadata.block_size_for_group / use_mla_for_group."""

    def _make_meta(
        self,
        group_block_sizes: list[int] | None = None,
        group_use_mla: list[bool] | None = None,
    ) -> "Any":
        """Import and instantiate EngineDrivenContextMetadata for testing."""
        from lmcache.v1.multiprocess.transfer_context.base import (
            EngineDrivenContextMetadata,
        )

        return EngineDrivenContextMetadata(
            instance_id=1,
            chunk_size=8,
            blocks_in_chunk=2,
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2, 2, 8])],
                dtypes=["float32"],
            ),
            attn_desc=MagicMock(),
            group_block_sizes=group_block_sizes or [],
            group_use_mla=group_use_mla or [],
        )

    def test_block_size_for_group_single(self) -> None:
        """Returns the per-group block size when registered."""
        meta = self._make_meta(group_block_sizes=[4, 8])
        assert meta.block_size_for_group(0) == 4
        assert meta.block_size_for_group(1) == 8

    def test_block_size_for_group_fallback(self) -> None:
        """Falls back to 0 when group_block_sizes is empty."""
        meta = self._make_meta(group_block_sizes=[])
        assert meta.block_size_for_group(0) == 0

    def test_use_mla_for_group(self) -> None:
        """Returns per-group MLA flag when registered."""
        meta = self._make_meta(group_use_mla=[False, True])
        assert meta.use_mla_for_group(0) is False
        assert meta.use_mla_for_group(1) is True

    def test_use_mla_for_group_fallback(self) -> None:
        """Falls back to False when group_use_mla is empty."""
        meta = self._make_meta(group_use_mla=[])
        assert meta.use_mla_for_group(0) is False

    def test_num_object_groups_single(self) -> None:
        """num_object_groups returns 1 when no multi-group layout."""
        meta = self._make_meta()
        assert meta.num_object_groups == 1

    def test_num_object_groups_multi(self) -> None:
        """num_object_groups returns the number of shapes in layout_desc."""
        from lmcache.v1.multiprocess.transfer_context.base import (
            EngineDrivenContextMetadata,
        )

        meta = EngineDrivenContextMetadata(
            instance_id=1,
            chunk_size=8,
            blocks_in_chunk=2,
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([2, 2, 8]), torch.Size([2, 4, 8])],
                dtypes=["float32", "float16"],
            ),
            attn_desc=MagicMock(),
            group_block_sizes=[4, 4],
            group_use_mla=[False, False],
        )
        assert meta.num_object_groups == 2


# ---------------------------------------------------------------------------
# GroupLayoutSpec / RegisterEngineDrivenContextPayload encoding
# ---------------------------------------------------------------------------


class TestGroupLayoutSpecEncoding:
    """GroupLayoutSpec and RegisterEngineDrivenContextPayload serialise correctly."""

    def test_group_layout_spec_fields(self) -> None:
        """GroupLayoutSpec stores all fields correctly."""
        import msgspec

        from lmcache.v1.multiprocess.custom_types import GroupLayoutSpec

        spec = GroupLayoutSpec(
            block_size=16,
            num_layers=4,
            hidden_dim_size=128,
            dtype_str="float16",
            kv_size=2,
            sw_size_tokens=64,
            engine_group_id=1,
            layer_indices=(0, 1, 2, 3),
        )
        assert spec.block_size == 16
        assert spec.sw_size_tokens == 64
        assert spec.engine_group_id == 1
        assert spec.layer_indices == (0, 1, 2, 3)

        # Should be msgspec-encodable.
        encoded = msgspec.json.encode(spec)
        decoded = msgspec.json.decode(encoded, type=GroupLayoutSpec)
        assert decoded == spec

    def test_register_payload_no_group_layouts(self) -> None:
        """RegisterEngineDrivenContextPayload with empty group_layouts encodes OK."""
        import msgspec

        from lmcache.v1.multiprocess.custom_types import (
            RegisterEngineDrivenContextPayload,
        )

        payload = RegisterEngineDrivenContextPayload(
            instance_id=1,
            chunk_size=16,
            num_layers=2,
            hidden_dim_size=64,
            dtype_str="float32",
            kv_size=2,
            block_size=4,
            model_name="m",
            world_size=1,
            shm_name="",
            pool_size=0,
        )
        assert payload.group_layouts == []
        encoded = msgspec.json.encode(payload)
        decoded = msgspec.json.decode(
            encoded, type=RegisterEngineDrivenContextPayload
        )
        assert decoded.group_layouts == []

    def test_register_payload_with_group_layouts(self) -> None:
        """RegisterEngineDrivenContextPayload with group_layouts encodes OK."""
        import msgspec

        from lmcache.v1.multiprocess.custom_types import (
            GroupLayoutSpec,
            RegisterEngineDrivenContextPayload,
        )

        gl = [
            GroupLayoutSpec(
                block_size=4,
                num_layers=2,
                hidden_dim_size=64,
                dtype_str="float32",
                kv_size=2,
                sw_size_tokens=-1,
                engine_group_id=0,
                layer_indices=(0, 1),
            ),
            GroupLayoutSpec(
                block_size=8,
                num_layers=2,
                hidden_dim_size=128,
                dtype_str="float16",
                kv_size=1,
                sw_size_tokens=32,
                engine_group_id=1,
                layer_indices=(2, 3),
            ),
        ]
        payload = RegisterEngineDrivenContextPayload(
            instance_id=1,
            chunk_size=16,
            num_layers=4,
            hidden_dim_size=64,
            dtype_str="float32",
            kv_size=2,
            block_size=4,
            model_name="m",
            world_size=1,
            shm_name="",
            pool_size=0,
            group_layouts=gl,
        )
        encoded = msgspec.json.encode(payload)
        decoded = msgspec.json.decode(
            encoded, type=RegisterEngineDrivenContextPayload
        )
        assert len(decoded.group_layouts) == 2
        assert decoded.group_layouts[0].block_size == 4
        assert decoded.group_layouts[1].block_size == 8
        assert decoded.group_layouts[1].sw_size_tokens == 32


# ---------------------------------------------------------------------------
# server_transfer multi-group reserve helpers
# ---------------------------------------------------------------------------


class TestServerTransferMultiGroupHelpers:
    """Unit tests for _split_obj_keys_by_group and _per_group_layout_desc."""

    def test_split_obj_keys_by_group_two_groups(self) -> None:
        """_split_obj_keys_by_group correctly separates keys by object_group_id."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _split_obj_keys_by_group,
        )

        keys = [
            _make_obj_key("h0", 0),
            _make_obj_key("h1", 0),
            _make_obj_key("h2", 1),
            _make_obj_key("h3", 1),
            _make_obj_key("h4", 1),
        ]
        result = _split_obj_keys_by_group(keys, num_groups=2)
        assert len(result[0]) == 2
        assert len(result[1]) == 3
        # Original indices should be preserved
        assert result[0][0][0] == 0
        assert result[1][0][0] == 2

    def test_split_obj_keys_single_group(self) -> None:
        """Single group: all keys go to group 0."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _split_obj_keys_by_group,
        )

        keys = [_make_obj_key("h0", 0), _make_obj_key("h1", 0)]
        result = _split_obj_keys_by_group(keys, num_groups=1)
        assert len(result[0]) == 2

    def test_per_group_layout_desc_selects_correct_shape(self) -> None:
        """_per_group_layout_desc returns a single-entry layout for the given group."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _per_group_layout_desc,
        )

        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8]), torch.Size([2, 8, 16])],
            dtypes=["float32", "float16"],
        )
        g0 = _per_group_layout_desc(layout, 0)
        assert g0.shapes == [torch.Size([2, 4, 8])]
        assert g0.dtypes == ["float32"]

        g1 = _per_group_layout_desc(layout, 1)
        assert g1.shapes == [torch.Size([2, 8, 16])]
        assert g1.dtypes == ["float16"]

    def test_per_group_layout_desc_out_of_range_fallback(self) -> None:
        """_per_group_layout_desc falls back to index 0 for out-of-range group."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _per_group_layout_desc,
        )

        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8])],
            dtypes=["float32"],
        )
        g5 = _per_group_layout_desc(layout, 5)
        assert g5.shapes == [torch.Size([2, 4, 8])]

    def test_reserve_multi_group_calls_reserve_per_group(self) -> None:
        """_reserve_multi_group calls reserve_write once per non-empty group."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _reserve_multi_group,
        )

        mock_storage = MagicMock()
        mock_storage.reserve_write.return_value = {}

        keys = [
            _make_obj_key("h0", 0),
            _make_obj_key("h1", 0),
            _make_obj_key("h2", 1),
        ]
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8]), torch.Size([2, 8, 16])],
            dtypes=["float32", "float16"],
        )
        _reserve_multi_group(mock_storage, keys, layout, num_groups=2)

        # Should call reserve_write twice: once per group
        assert mock_storage.reserve_write.call_count == 2
        # Group 0 call uses shapes[0]
        g0_layout = mock_storage.reserve_write.call_args_list[0][0][1]
        assert g0_layout.shapes == [torch.Size([2, 4, 8])]
        # Group 1 call uses shapes[1]
        g1_layout = mock_storage.reserve_write.call_args_list[1][0][1]
        assert g1_layout.shapes == [torch.Size([2, 8, 16])]

    def test_reserve_multi_group_skips_empty_groups(self) -> None:
        """_reserve_multi_group skips groups with no keys (sparse stores)."""
        from lmcache.v1.multiprocess.modules.server_transfer import (
            _reserve_multi_group,
        )

        mock_storage = MagicMock()
        mock_storage.reserve_write.return_value = {}

        # Only group-0 keys, no group-1 keys
        keys = [_make_obj_key("h0", 0), _make_obj_key("h1", 0)]
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8]), torch.Size([2, 8, 16])],
            dtypes=["float32", "float16"],
        )
        _reserve_multi_group(mock_storage, keys, layout, num_groups=2)
        # Only 1 call (group 1 has no keys)
        assert mock_storage.reserve_write.call_count == 1


# ---------------------------------------------------------------------------
# Single-group backward compatibility
# ---------------------------------------------------------------------------


class TestSingleGroupBackwardCompat:
    """Single-group (legacy) behaviour must not regress."""

    def test_plan_group_copy_single_group(self) -> None:
        """Single-group plan_group_copy with matching block sizes behaves like legacy."""
        kv = _make_kv(num_layers=2)
        info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
        plans = plan_group_copy(
            kv, [[0, 1, 2, 3]], blocks_in_chunk=2, engine_group_infos=[info], group_block_sizes=[4]
        )
        assert len(plans) == 1
        plan = plans[0]
        assert plan.num_chunks == 2
        assert list(plan.kv_subset.keys()) == ["layer_0", "layer_1"]

    def test_flatten_single_group_is_identity(self) -> None:
        """Single-group flatten returns the group's chunks unchanged."""
        chunks = [torch.full((5,), float(i)) for i in range(3)]
        flat = flatten_chunks_group_major([chunks])
        assert len(flat) == 3
        assert all(torch.equal(flat[i], chunks[i]) for i in range(3))

    def test_unflatten_single_group_is_identity(self) -> None:
        """Single-group unflatten recovers the original."""
        chunks = [torch.ones(4) * i for i in range(5)]
        recovered = unflatten_chunks_group_major(chunks, [5])
        assert len(recovered) == 1
        assert all(torch.equal(recovered[0][i], chunks[i]) for i in range(5))


# ---------------------------------------------------------------------------
# Block-ID underflow / insufficient block IDs
# ---------------------------------------------------------------------------


class TestBlockIdUnderflow:
    """Engine-driven path must not silently succeed with too few block IDs."""

    def test_plan_with_zero_blocks_produces_zero_chunks(self) -> None:
        """Zero block IDs for a group results in num_chunks=0 (fail-closed)."""
        kv = _make_kv(num_layers=2)
        info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
        plans = plan_group_copy(
            kv, [[]], blocks_in_chunk=2, engine_group_infos=[info], group_block_sizes=[4]
        )
        assert plans[0].num_chunks == 0

    def test_plan_with_insufficient_blocks_truncates(self) -> None:
        """Fewer block IDs than blocks_in_chunk yields 0 full chunks (no partial)."""
        kv = _make_kv(num_layers=2)
        info = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1))
        plans = plan_group_copy(
            kv, [[0]],  # 1 block ID, need 2 for one chunk
            blocks_in_chunk=2,
            engine_group_infos=[info],
            group_block_sizes=[4],
        )
        # 1 // 2 == 0
        assert plans[0].num_chunks == 0
