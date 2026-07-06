# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for TransferPlanBuilder.

All tests are CPU-friendly and do not require CUDA or lmc_ops native extensions.
They verify the public interface and docstring contract of TransferPlanBuilder:
what groups, chunks, block IDs, layouts, and skip offsets the builder produces.
"""

# Standard
from unittest.mock import MagicMock, PropertyMock

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.transfer_plan import (
    TransferDirection,
    TransferPlan,
    TransferPlanBuilder,
    recalculate_blocks_to_skip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_object_key(chunk_idx: int, group_id: int = 0) -> ObjectKey:
    """Create a minimal ObjectKey for testing."""
    return MagicMock(spec=ObjectKey)


def _make_obj_keys(num_chunks: int, num_groups: int = 1) -> list[list[ObjectKey]]:
    """Create a list of lists of object keys for testing."""
    return [
        [_make_object_key(i, g) for i in range(num_chunks)] for g in range(num_groups)
    ]


def _mock_manager(
    kernel_groups: list[dict],
    object_groups: list[list[int]] | None = None,
    sw_sizes_chunks: list[int] | None = None,
    chunk_size: int = 16,
) -> MagicMock:
    """Build a minimal KVLayerGroupsManager mock for testing.

    Args:
        kernel_groups: List of per-kernel-group dicts with keys:
            ``blocks_per_chunk`` (int), ``sw_size_tokens`` (int, default -1).
        object_groups: Grouping of kernel-group indices into object groups.
            Defaults to one group containing all kernel groups.
        sw_sizes_chunks: Per-object-group sw_size_chunks. ``-1`` = full attention.
        chunk_size: LMCache chunk size in tokens.

    Returns:
        A MagicMock that satisfies the KVLayerGroupsManager interface used by
        :class:`TransferPlanBuilder`.
    """
    if object_groups is None:
        object_groups = [list(range(len(kernel_groups)))]
    if sw_sizes_chunks is None:
        sw_sizes_chunks = [-1] * len(object_groups)

    manager = MagicMock()
    manager.num_kernel_groups = len(kernel_groups)
    manager.num_object_groups = len(object_groups)

    # calculate_num_blocks(kg_id, num_tokens) -> int
    def calc_blocks(kg_id: int, num_tokens: int) -> int:
        bpc_full = kernel_groups[kg_id]["blocks_per_chunk"]
        sw_size_tokens = kernel_groups[kg_id].get("sw_size_tokens", -1)
        if sw_size_tokens != -1 and num_tokens < chunk_size:
            # sub-chunk: proportional
            return max(1, bpc_full * num_tokens // chunk_size)
        return bpc_full * num_tokens // chunk_size

    manager.calculate_num_blocks.side_effect = calc_blocks

    # get_subchunk_sw_size_tokens(kg_id) -> int
    def get_sw_size(kg_id: int) -> int:
        sw = kernel_groups[kg_id].get("sw_size_tokens", -1)
        if sw == -1 or sw >= chunk_size:
            return chunk_size
        return sw

    manager.get_subchunk_sw_size_tokens.side_effect = get_sw_size

    # object_groups property
    og_mocks = []
    for kg_indices in object_groups:
        og_mock = MagicMock()
        og_mock.kernel_group_indices = kg_indices
        og_mocks.append(og_mock)
    type(manager).object_groups = PropertyMock(return_value=og_mocks)

    # get_attn_desc()
    attn_desc = MagicMock()
    attn_desc.num_chunks_in_sw = sw_sizes_chunks

    def is_full_attention(og_id: int) -> bool:
        return sw_sizes_chunks[og_id] == -1

    attn_desc.is_full_attention.side_effect = is_full_attention
    manager.get_attn_desc.return_value = attn_desc

    return manager


def _flat_shape(num_layers: int = 2, tokens: int = 16, hidden: int = 64) -> torch.Size:
    return torch.Size([2, num_layers, tokens, hidden])


# ---------------------------------------------------------------------------
# recalculate_blocks_to_skip helper
# ---------------------------------------------------------------------------


class TestRecalculateBlocksToSkip:
    """Tests for the recalculate_blocks_to_skip() helper."""

    def test_full_attention_group_unchanged(self) -> None:
        assert recalculate_blocks_to_skip(4, 4, 8) == 8

    def test_zero_skip_stays_zero(self) -> None:
        assert recalculate_blocks_to_skip(4, 2, 0) == 0

    def test_subchunk_swa_full_windows(self) -> None:
        # 4 blocks per chunk, 2 blocks per window.
        # 8 blocks to skip = 2 full chunks worth of full blocks.
        # In downsampled space: 2 * 2 = 4 blocks.
        assert recalculate_blocks_to_skip(4, 2, 8) == 4

    def test_subchunk_swa_partial_tail(self) -> None:
        # 4 per chunk, 2 per window.
        # 6 blocks to skip = 1 full chunk (4 blocks) + 2 tail blocks.
        # tail_blocks - (4-2) = 2 - 2 = 0 → 1*2 + 0 = 2
        assert recalculate_blocks_to_skip(4, 2, 6) == 2

    def test_subchunk_swa_tail_negative_clamped(self) -> None:
        # 4 per chunk, 2 per window.
        # 5 blocks = 1 full chunk + 1 tail.
        # 1 - (4-2) = -1 → clamped to 0 → 1*2 + 0 = 2
        assert recalculate_blocks_to_skip(4, 2, 5) == 2


# ---------------------------------------------------------------------------
# Single dense group
# ---------------------------------------------------------------------------


class TestSingleDenseGroup:
    """TransferPlanBuilder with a single full-attention kernel group."""

    def _build(
        self,
        num_chunks: int = 3,
        blocks_per_chunk: int = 2,
        direction: TransferDirection = TransferDirection.STORE,
    ) -> TransferPlan:
        chunk_size = 16
        total_blocks = num_chunks * blocks_per_chunk
        block_ids = [list(range(total_blocks))]
        obj_keys = _make_obj_keys(num_chunks, num_groups=1)
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": blocks_per_chunk}],
            chunk_size=chunk_size,
        )
        shape_dtypes = [(_flat_shape(), torch.float16)]
        return TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=block_ids,
            obj_keys_per_obj_group=obj_keys,
            direction=direction,
            kernel_group_shape_dtypes=shape_dtypes,
        )

    def test_no_underflow(self) -> None:
        plan = self._build()
        assert not plan.underflow

    def test_direction_preserved(self) -> None:
        plan = self._build(direction=TransferDirection.RETRIEVE)
        assert plan.direction is TransferDirection.RETRIEVE

    def test_chunk_count(self) -> None:
        plan = self._build(num_chunks=4)
        assert plan.num_chunks == 4

    def test_kernel_group_plan(self) -> None:
        plan = self._build(num_chunks=2, blocks_per_chunk=3)
        assert len(plan.kernel_groups) == 1
        kg_plan = plan.kernel_groups[0]
        assert kg_plan.kernel_group_id == 0
        assert kg_plan.blocks_per_chunk == 3
        assert kg_plan.blocks_per_window == 3  # full-attention → no downsampling
        assert len(kg_plan.selected_block_ids) == 2 * 3

    def test_object_group_plan(self) -> None:
        plan = self._build(num_chunks=2)
        assert len(plan.object_groups) == 1
        og = plan.object_groups[0]
        assert og.object_group_id == 0
        assert len(og.object_keys) == 2
        assert og.num_objects_to_skip == 0  # store, so always 0
        assert og.kernel_group_ids == [0]

    def test_layout_desc_built(self) -> None:
        plan = self._build()
        og = plan.object_groups[0]
        assert isinstance(og.layout_desc, MemoryLayoutDesc)
        assert len(og.layout_desc.shapes) == 1

    def test_selected_block_ids_by_group(self) -> None:
        plan = self._build(num_chunks=2, blocks_per_chunk=2)
        by_group = plan.selected_block_ids_by_group()
        assert len(by_group) == 1
        assert len(by_group[0]) == 4  # 2 chunks * 2 blocks

    def test_underflow_when_short_block_ids(self) -> None:
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 4}], chunk_size=chunk_size
        )
        # Only 3 blocks supplied, need 4 (1 chunk * 4 bpc)
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[[0, 1, 2]],  # one short
            obj_keys_per_obj_group=_make_obj_keys(1),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )
        assert plan.underflow
        assert plan.kernel_groups == []
        assert plan.object_groups == []


# ---------------------------------------------------------------------------
# Multiple kernel groups
# ---------------------------------------------------------------------------


class TestMultipleKernelGroups:
    """TransferPlanBuilder with multiple kernel groups in one object group."""

    def _build(
        self,
        num_chunks: int = 2,
        blocks_per_chunks: tuple[int, ...] = (2, 4),
        direction: TransferDirection = TransferDirection.STORE,
    ) -> TransferPlan:
        chunk_size = 16
        kg_defs = [{"blocks_per_chunk": bpc} for bpc in blocks_per_chunks]
        block_ids = [list(range(num_chunks * bpc)) for bpc in blocks_per_chunks]
        # All kernel groups in one object group.
        object_groups = [list(range(len(blocks_per_chunks)))]
        manager = _mock_manager(
            kernel_groups=kg_defs,
            object_groups=object_groups,
            chunk_size=chunk_size,
        )
        shape_dtypes = [(_flat_shape(), torch.float16)] * len(blocks_per_chunks)
        obj_keys = _make_obj_keys(num_chunks, num_groups=1)
        return TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=block_ids,
            obj_keys_per_obj_group=obj_keys,
            direction=direction,
            kernel_group_shape_dtypes=shape_dtypes,
        )

    def test_no_underflow(self) -> None:
        plan = self._build()
        assert not plan.underflow

    def test_kernel_group_count(self) -> None:
        plan = self._build(blocks_per_chunks=(2, 4, 6))
        assert len(plan.kernel_groups) == 3

    def test_per_group_blocks_per_chunk(self) -> None:
        plan = self._build(num_chunks=2, blocks_per_chunks=(2, 4))
        assert plan.kernel_groups[0].blocks_per_chunk == 2
        assert plan.kernel_groups[1].blocks_per_chunk == 4

    def test_selected_block_ids_match_counts(self) -> None:
        plan = self._build(num_chunks=2, blocks_per_chunks=(2, 4))
        assert len(plan.kernel_groups[0].selected_block_ids) == 4  # 2*2
        assert len(plan.kernel_groups[1].selected_block_ids) == 8  # 2*4

    def test_underflow_on_any_group_short(self) -> None:
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}, {"blocks_per_chunk": 4}],
            object_groups=[[0, 1]],
            chunk_size=chunk_size,
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[[0, 1, 2, 3], [0, 1]],  # group 1 too short for 1 chunk
            obj_keys_per_obj_group=_make_obj_keys(1),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)] * 2,
        )
        assert plan.underflow


# ---------------------------------------------------------------------------
# Multiple object groups
# ---------------------------------------------------------------------------


class TestMultipleObjectGroups:
    """TransferPlanBuilder with multiple object groups (hybrid model)."""

    def _build(
        self,
        num_chunks: int = 3,
        blocks_per_chunks: tuple[int, ...] = (2, 2),
        sw_sizes_chunks: tuple[int, ...] = (-1, 2),
        direction: TransferDirection = TransferDirection.RETRIEVE,
    ) -> TransferPlan:
        chunk_size = 16
        kg_defs = [{"blocks_per_chunk": bpc} for bpc in blocks_per_chunks]
        block_ids = [list(range(num_chunks * bpc)) for bpc in blocks_per_chunks]
        # Each kernel group in its own object group.
        object_groups = [[i] for i in range(len(blocks_per_chunks))]
        manager = _mock_manager(
            kernel_groups=kg_defs,
            object_groups=object_groups,
            sw_sizes_chunks=list(sw_sizes_chunks),
            chunk_size=chunk_size,
        )
        shape_dtypes = [(_flat_shape(), torch.float16)] * len(blocks_per_chunks)
        obj_keys = _make_obj_keys(num_chunks, num_groups=len(blocks_per_chunks))
        return TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=block_ids,
            obj_keys_per_obj_group=obj_keys,
            direction=direction,
            kernel_group_shape_dtypes=shape_dtypes,
        )

    def test_object_group_count(self) -> None:
        plan = self._build()
        assert len(plan.object_groups) == 2

    def test_full_attention_group_zero_skip(self) -> None:
        plan = self._build(direction=TransferDirection.RETRIEVE)
        full_attn_group = plan.object_groups[0]  # sw_size_chunks=-1
        assert full_attn_group.num_objects_to_skip == 0

    def test_swa_group_correct_skip(self) -> None:
        # num_chunks=3, sw_size_chunks=2 → skip = max(0, 3-2) = 1
        plan = self._build(
            num_chunks=3,
            sw_sizes_chunks=(-1, 2),
            direction=TransferDirection.RETRIEVE,
        )
        swa_group = plan.object_groups[1]  # sw_size_chunks=2
        assert swa_group.num_objects_to_skip == 1

    def test_store_has_zero_skip(self) -> None:
        # Even for SWA groups, skip should be 0 for store.
        plan = self._build(direction=TransferDirection.STORE)
        for og in plan.object_groups:
            assert og.num_objects_to_skip == 0

    def test_kernel_group_ids_per_object_group(self) -> None:
        plan = self._build()
        assert plan.object_groups[0].kernel_group_ids == [0]
        assert plan.object_groups[1].kernel_group_ids == [1]

    def test_layout_desc_per_group(self) -> None:
        plan = self._build()
        for og in plan.object_groups:
            assert isinstance(og.layout_desc, MemoryLayoutDesc)


# ---------------------------------------------------------------------------
# Sliding-window / subchunk downsampling
# ---------------------------------------------------------------------------


class TestSlidingWindowDownsampling:
    """Block-ID downsampling for subchunk sliding-window groups."""

    def _build_subchunk_swa(
        self,
        num_chunks: int = 2,
        blocks_per_chunk: int = 4,
        sw_size_tokens: int = 8,
        chunk_size: int = 16,
    ) -> TransferPlan:
        """Build a plan for a single subchunk-SWA kernel group."""
        # blocks_per_window = blocks_per_chunk * sw_size_tokens / chunk_size
        # = 4 * 8 / 16 = 2
        manager = _mock_manager(
            kernel_groups=[
                {"blocks_per_chunk": blocks_per_chunk, "sw_size_tokens": sw_size_tokens}
            ],
            chunk_size=chunk_size,
        )
        block_ids = [list(range(num_chunks * blocks_per_chunk))]
        return TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=block_ids,
            obj_keys_per_obj_group=_make_obj_keys(num_chunks),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )

    def test_downsampled_block_count(self) -> None:
        plan = self._build_subchunk_swa(
            num_chunks=2, blocks_per_chunk=4, sw_size_tokens=8, chunk_size=16
        )
        kg = plan.kernel_groups[0]
        assert kg.blocks_per_window == 2
        # 2 chunks * 2 blocks per window = 4 selected blocks
        assert len(kg.selected_block_ids) == 4

    def test_trailing_blocks_kept(self) -> None:
        # Original blocks: [0,1,2,3, 4,5,6,7]
        # 4 per chunk, 2 per window → keep last 2 of each chunk:
        # chunk 0: keep [2,3]; chunk 1: keep [6,7]
        plan = self._build_subchunk_swa(
            num_chunks=2, blocks_per_chunk=4, sw_size_tokens=8, chunk_size=16
        )
        kg = plan.kernel_groups[0]
        assert kg.selected_block_ids == [2, 3, 6, 7]

    def test_full_attention_no_downsampling(self) -> None:
        # sw_size_tokens=-1 → full attention → keep all blocks
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 4, "sw_size_tokens": -1}],
            chunk_size=16,
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=16,
            block_ids=[list(range(8))],
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )
        assert len(plan.kernel_groups[0].selected_block_ids) == 8

    def test_downsampling_preserves_order(self) -> None:
        # Original: [10,20,30,40]; 4 per chunk, 2 per window → keep [30,40]
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 4, "sw_size_tokens": 8}],
            chunk_size=16,
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=16,
            block_ids=[[10, 20, 30, 40]],
            obj_keys_per_obj_group=_make_obj_keys(1),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )
        assert plan.kernel_groups[0].selected_block_ids == [30, 40]


# ---------------------------------------------------------------------------
# skip_first_n_tokens (passed through but not used by builder)
# ---------------------------------------------------------------------------


class TestSkipFirstNTokens:
    """Plans build correctly regardless of skip_first_n_tokens value."""

    def test_skip_tokens_does_not_affect_store(self) -> None:
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}], chunk_size=chunk_size
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[list(range(4))],
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
            skip_first_n_tokens=8,
        )
        assert not plan.underflow
        assert plan.num_chunks == 2


# ---------------------------------------------------------------------------
# Object key resolution
# ---------------------------------------------------------------------------


class TestObjectKeyResolution:
    """Object keys are resolved per object group and carried in the plan."""

    def test_keys_per_object_group(self) -> None:
        num_chunks = 3
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}, {"blocks_per_chunk": 2}],
            object_groups=[[0], [1]],
            chunk_size=chunk_size,
        )
        og0_keys = [_make_object_key(i, 0) for i in range(num_chunks)]
        og1_keys = [_make_object_key(i, 1) for i in range(num_chunks)]
        obj_keys = [og0_keys, og1_keys]
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[list(range(6)), list(range(6))],
            obj_keys_per_obj_group=obj_keys,
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)] * 2,
        )
        assert plan.object_groups[0].object_keys == og0_keys
        assert plan.object_groups[1].object_keys == og1_keys


# ---------------------------------------------------------------------------
# Underflow validation
# ---------------------------------------------------------------------------


class TestUnderflowValidation:
    """TransferPlan.underflow is set correctly and clears object groups."""

    def test_partial_underflow_fails_closed(self) -> None:
        """If ANY group is short, the whole plan is marked underflow."""
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}, {"blocks_per_chunk": 2}],
            object_groups=[[0, 1]],
            chunk_size=chunk_size,
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[list(range(4)), [0]],  # group 1 only 1 block, needs 2
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)] * 2,
        )
        assert plan.underflow
        assert plan.object_groups == []
        assert plan.kernel_groups == []

    def test_exact_coverage_not_underflow(self) -> None:
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 3}], chunk_size=chunk_size
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[list(range(3))],  # exactly 1 chunk * 3 bpc
            obj_keys_per_obj_group=_make_obj_keys(1),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )
        assert not plan.underflow

    def test_extra_block_ids_not_underflow(self) -> None:
        chunk_size = 16
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}], chunk_size=chunk_size
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=[list(range(10))],  # more than needed
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(_flat_shape(), torch.float16)],
        )
        assert not plan.underflow


# ---------------------------------------------------------------------------
# Layout descriptor generation
# ---------------------------------------------------------------------------


class TestLayoutDescGeneration:
    """Per-object-group MemoryLayoutDesc is built correctly."""

    def test_single_kernel_group_layout(self) -> None:
        shape = _flat_shape(num_layers=4, tokens=16, hidden=32)
        dtype = torch.bfloat16
        manager = _mock_manager(kernel_groups=[{"blocks_per_chunk": 2}], chunk_size=16)
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=16,
            block_ids=[list(range(4))],
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(shape, dtype)],
        )
        ld = plan.object_groups[0].layout_desc
        assert ld.shapes == [shape]
        assert ld.dtypes == [dtype]

    def test_two_kernel_groups_in_one_object_group(self) -> None:
        s0 = _flat_shape(num_layers=2, tokens=16, hidden=64)
        s1 = _flat_shape(num_layers=1, tokens=16, hidden=32)
        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": 2}, {"blocks_per_chunk": 2}],
            object_groups=[[0, 1]],
            chunk_size=16,
        )
        plan = TransferPlanBuilder.build(
            kv_groups_manager=manager,
            lmcache_tokens_per_chunk=16,
            block_ids=[list(range(4)), list(range(4))],
            obj_keys_per_obj_group=_make_obj_keys(2),
            direction=TransferDirection.STORE,
            kernel_group_shape_dtypes=[(s0, torch.float16), (s1, torch.bfloat16)],
        )
        ld = plan.object_groups[0].layout_desc
        assert len(ld.shapes) == 2
        assert ld.shapes[0] == s0
        assert ld.shapes[1] == s1
        assert ld.dtypes == [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# build_from_cache_context convenience wrapper
# ---------------------------------------------------------------------------


class TestBuildFromCacheContext:
    """build_from_cache_context delegates to build() correctly."""

    def test_delegates_to_build(self) -> None:
        chunk_size = 16
        blocks_per_chunk = 2
        num_chunks = 2

        manager = _mock_manager(
            kernel_groups=[{"blocks_per_chunk": blocks_per_chunk}],
            chunk_size=chunk_size,
        )
        shape = _flat_shape()
        dtype = torch.float16

        cache_context = MagicMock()
        cache_context.kv_layer_groups_manager = manager
        cache_context.lmcache_tokens_per_chunk = chunk_size
        cache_context.get_kernel_group_shape_dtype.return_value = (shape, dtype)

        block_ids = [list(range(num_chunks * blocks_per_chunk))]
        obj_keys = _make_obj_keys(num_chunks)

        plan = TransferPlanBuilder.build_from_cache_context(
            cache_context=cache_context,
            block_ids=block_ids,
            obj_keys_per_obj_group=obj_keys,
            direction=TransferDirection.STORE,
        )

        assert not plan.underflow
        assert plan.num_chunks == num_chunks
        cache_context.get_kernel_group_shape_dtype.assert_called_once_with(
            chunk_size, 0
        )
