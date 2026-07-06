# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the path-agnostic TransferPlanBuilder and planning helpers.

These tests verify the pure planning logic in
``lmcache.v1.multiprocess.transfer_plan`` without requiring GPU, CUDA
extensions, or any multiprocess server infrastructure.
"""

# Standard
from dataclasses import dataclass
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupBatchTransferPlan,
    ObjectBatchTransferPlan,
    TransferDirection,
    TransferPlanBuilder,
    build_object_group_layout_desc,
    downsample_block_ids,
    recalculate_blocks_to_skip,
    validate_block_ids,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_object_key(request_id: str, chunk_idx: int, obj_group: int = 0) -> ObjectKey:
    """Create a minimal ObjectKey for testing."""
    key = MagicMock(spec=ObjectKey)
    key.request_id = request_id
    key.chunk_idx = chunk_idx
    key.obj_group = obj_group
    return key


@dataclass
class _KernelGroupSpec:
    """Simple spec for building mock BaseCacheContext."""

    block_size: int  # tokens per block for this group
    sw_size_tokens: int = -1  # -1 => full attention; positive => sliding window


def _make_mock_cache_context(
    chunk_size: int,
    kernel_groups: list[_KernelGroupSpec],
    object_group_kernel_indices: list[list[int]] | None = None,
    sw_size_chunks_per_object_group: list[int] | None = None,
    max_batch_size: int = 4,
) -> MagicMock:
    """Build a MagicMock BaseCacheContext for testing.

    Args:
        chunk_size: LMCache chunk size in tokens.
        kernel_groups: One entry per kernel group.  ``block_size`` is the
            number of tokens per paged block; ``sw_size_tokens`` is the
            subchunk sliding-window size in tokens (``-1`` for full attn).
        object_group_kernel_indices: Which kernel groups belong to each
            object group.  Defaults to ``[[0], [1], ...]``.
        sw_size_chunks_per_object_group: Cross-chunk SW sizes for each
            object group.  ``-1`` means full attention.  Defaults to all
            ``-1`` (full attention).
        max_batch_size: Maximum number of concurrent batches returned by
            ``cache_context.max_batch_size``.  Defaults to 4.

    Returns:
        A fully configured MagicMock that mirrors the public interface of
        ``BaseCacheContext`` used by the planning helpers.
    """
    num_kg = len(kernel_groups)
    if object_group_kernel_indices is None:
        object_group_kernel_indices = [[i] for i in range(num_kg)]
    num_og = len(object_group_kernel_indices)
    if sw_size_chunks_per_object_group is None:
        sw_size_chunks_per_object_group = [-1] * num_og

    # ---- KVLayerGroupsManager mock ----------------------------------------
    kv_mgr = MagicMock()
    kv_mgr.num_kernel_groups = num_kg
    kv_mgr.num_object_groups = num_og

    # object_groups[i].kernel_group_indices
    object_group_mocks: list[MagicMock] = []
    for kg_indices in object_group_kernel_indices:
        og = MagicMock()
        og.kernel_group_indices = list(kg_indices)
        object_group_mocks.append(og)
    kv_mgr.object_groups = object_group_mocks

    def _get_subchunk_sw_size_tokens(kg_id: int) -> int:
        sw = kernel_groups[kg_id].sw_size_tokens
        # When not a subchunk-sw group, return a very large number (> chunk_size)
        # so min(chunk_size, sw) == chunk_size.
        return sw if sw > 0 else chunk_size * 1000

    kv_mgr.get_subchunk_sw_size_tokens.side_effect = _get_subchunk_sw_size_tokens

    attn_desc = AttnWindowDesc(num_chunks_in_sw=list(sw_size_chunks_per_object_group))
    kv_mgr.get_attn_desc.return_value = attn_desc

    # ---- BaseCacheContext mock -----------------------------------------------
    ctx = MagicMock()
    ctx.lmcache_tokens_per_chunk = chunk_size
    ctx.kv_layer_groups_manager = kv_mgr

    def _calculate_num_blocks(num_tokens: int, kg_id: int) -> int:
        bs = kernel_groups[kg_id].block_size
        return max(0, (num_tokens + bs - 1) // bs) if bs > 0 else 0

    ctx.calculate_num_blocks.side_effect = _calculate_num_blocks

    def _get_kernel_group_shape_dtype(num_tokens: int, kg_id: int):
        # Third Party
        import torch

        bs = kernel_groups[kg_id].block_size
        num_blocks = _calculate_num_blocks(num_tokens, kg_id)
        return (torch.Size([num_blocks, bs]), torch.float16)

    ctx.get_kernel_group_shape_dtype.side_effect = _get_kernel_group_shape_dtype

    # Shape / format / slot helpers (provide plausible stubs)
    ctx.get_shape_desc.return_value = MagicMock(name="PageBufferShapeDesc")
    ctx.get_engine_kv_format.return_value = MagicMock(name="EngineKVFormat")
    ctx.get_slots_per_chunk_in_sw.return_value = chunk_size
    ctx.max_batch_size = max_batch_size

    return ctx


def _make_object_keys(request_id: str, num_chunks: int) -> list[ObjectKey]:
    return [_make_object_key(request_id, i) for i in range(num_chunks)]


# ---------------------------------------------------------------------------
# recalculate_blocks_to_skip
# ---------------------------------------------------------------------------


class TestRecalculateBlocksToSkip:
    """Tests for ``recalculate_blocks_to_skip``."""

    def test_full_attention_identity(self):
        """When blocks_per_chunk == blocks_per_window the result is unchanged."""
        assert recalculate_blocks_to_skip(8, 8, 4) == 4

    def test_zero_skip(self):
        """No skip required remains 0 regardless of window size."""
        assert recalculate_blocks_to_skip(8, 4, 0) == 0

    def test_sliding_window_skip_within_first_chunk(self):
        """Skip that fits within one full chunk is clamped to the window."""
        # 8 blocks per chunk, 4 per window  →  window is last 4 blocks
        # blocks_to_skip = 2 (within the non-window prefix) → adjusted = 0
        result = recalculate_blocks_to_skip(8, 4, 2)
        assert result == 0

    def test_sliding_window_skip_spanning_tail(self):
        """Skip that reaches into the window frame is kept."""
        # blocks_per_chunk=8, blocks_per_window=4, skip=6
        # 6 // 8 = 0 full windows; tail = 6; tail_to_skip = 6 - (8-4) = 2
        result = recalculate_blocks_to_skip(8, 4, 6)
        assert result == 2

    def test_sliding_window_skip_full_chunks(self):
        """Skipping whole chunks converts correctly."""
        # skip = 16 = 2 full chunks of 8; 2 * 4 = 8 window blocks skipped
        result = recalculate_blocks_to_skip(8, 4, 16)
        assert result == 8

    def test_sliding_window_large_skip_clamps(self):
        """Skip beyond window tail clamps to 0 within partial chunk."""
        # tail = 3, tail_to_skip = 3 - (8-4) = -1  →  max(0, -1) = 0
        result = recalculate_blocks_to_skip(8, 4, 3)
        assert result == 0


# ---------------------------------------------------------------------------
# validate_block_ids
# ---------------------------------------------------------------------------


class TestValidateBlockIds:
    """Tests for ``validate_block_ids``."""

    def test_exact_coverage_passes(self):
        block_ids = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 4 ids / group, bpc=2, 2 chunks
        assert validate_block_ids(block_ids, [2, 2], num_chunks=2) is True

    def test_excess_coverage_passes(self):
        block_ids = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert validate_block_ids(block_ids, [2, 2], num_chunks=2) is True

    def test_underflow_group_zero(self):
        block_ids = [[0], [4, 5, 6, 7]]  # group 0 has only 1 block (needs 4)
        assert validate_block_ids(block_ids, [2, 2], num_chunks=2) is False

    def test_underflow_group_one(self):
        block_ids = [[0, 1, 2, 3], [4]]
        assert validate_block_ids(block_ids, [2, 2], num_chunks=2) is False

    def test_empty_block_ids_zero_chunks(self):
        """Empty block IDs are valid when num_chunks == 0."""
        assert validate_block_ids([[], []], [2, 2], num_chunks=0) is True

    def test_single_group(self):
        block_ids = [[0, 1]]
        assert validate_block_ids(block_ids, [2], num_chunks=1) is True


# ---------------------------------------------------------------------------
# downsample_block_ids
# ---------------------------------------------------------------------------


class TestDownsampleBlockIds:
    """Tests for ``downsample_block_ids``."""

    def test_full_attention_group_unchanged(self):
        """Full-attention groups (sw == chunk_size) keep all blocks."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],  # full attn
        )
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]  # 2 chunks × 4 blocks each
        result = downsample_block_ids(ctx, block_ids)
        assert result[0] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_sliding_window_keeps_tail_blocks(self):
        """Subchunk-SW groups keep only the last blocks_per_window per chunk."""
        # chunk_size=4, block_size=1, sw_size_tokens=2
        # → keep_blocks_per_chunk=2, total_blocks_per_chunk=4
        # input: [a, b, c, d, e, f, g, h]  (2 chunks)
        # expected: [c, d, g, h]  (last 2 per chunk)
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=2)],
        )
        block_ids = [[10, 11, 12, 13, 20, 21, 22, 23]]
        result = downsample_block_ids(ctx, block_ids)
        assert result[0] == [12, 13, 22, 23]

    def test_full_window_keeps_all(self):
        """When sw_size_tokens == chunk_size the behaviour equals full attention."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=4)],
        )
        block_ids = [[0, 1, 2, 3]]
        result = downsample_block_ids(ctx, block_ids)
        assert result[0] == [0, 1, 2, 3]

    def test_mixed_groups(self):
        """Full-attention and SW groups in the same call are handled per-group."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[
                _KernelGroupSpec(block_size=1),  # full attn
                _KernelGroupSpec(block_size=1, sw_size_tokens=2),  # sw
            ],
            object_group_kernel_indices=[[0, 1]],
        )
        block_ids = [
            [0, 1, 2, 3, 4, 5, 6, 7],  # group 0: unchanged
            [10, 11, 12, 13, 20, 21, 22, 23],  # group 1: downsampled
        ]
        result = downsample_block_ids(ctx, block_ids)
        assert result[0] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert result[1] == [12, 13, 22, 23]

    def test_returns_new_list(self):
        """downsample_block_ids must not modify the original block_ids in-place."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=2)],
        )
        original = [[10, 11, 12, 13, 20, 21, 22, 23]]
        original_copy = [list(g) for g in original]
        downsample_block_ids(ctx, original)
        # Original must be unchanged.
        assert original == original_copy

    def test_misaligned_block_ids_raises(self):
        """Block IDs not a multiple of blocks_per_chunk raise ValueError."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        # 5 ids, blocks_per_chunk=4 → not a multiple
        block_ids = [[0, 1, 2, 3, 4]]
        with pytest.raises(ValueError):
            downsample_block_ids(ctx, block_ids)


# ---------------------------------------------------------------------------
# build_object_group_layout_desc
# ---------------------------------------------------------------------------


class TestBuildObjectGroupLayoutDesc:
    """Tests for ``build_object_group_layout_desc``."""

    def test_single_kernel_group(self):
        # Third Party
        import torch

        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        layout = build_object_group_layout_desc(ctx, num_tokens=4, object_group_id=0)
        assert isinstance(layout, MemoryLayoutDesc)
        assert len(layout.shapes) == 1
        assert len(layout.dtypes) == 1
        assert layout.dtypes[0] == torch.float16

    def test_multi_kernel_group_object_group(self):
        """Two kernel groups in one object group produce two shape entries."""

        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[
                _KernelGroupSpec(block_size=1),
                _KernelGroupSpec(block_size=2),
            ],
            object_group_kernel_indices=[[0, 1]],
        )
        layout = build_object_group_layout_desc(ctx, num_tokens=4, object_group_id=0)
        assert len(layout.shapes) == 2
        assert len(layout.dtypes) == 2

    def test_second_object_group(self):
        """Object group 1 uses only its own kernel groups."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[
                _KernelGroupSpec(block_size=1),
                _KernelGroupSpec(block_size=1),
            ],
            # two object groups: each has one kernel group
        )
        layout0 = build_object_group_layout_desc(ctx, num_tokens=4, object_group_id=0)
        layout1 = build_object_group_layout_desc(ctx, num_tokens=4, object_group_id=1)
        assert len(layout0.shapes) == 1
        assert len(layout1.shapes) == 1


# ---------------------------------------------------------------------------
# TransferPlanBuilder.build_store_plan
# ---------------------------------------------------------------------------


class TestTransferPlanBuilderStore:
    """Tests for ``TransferPlanBuilder.build_store_plan``."""

    def _make_ctx_and_keys(self, num_chunks: int = 2, chunk_size: int = 4):
        ctx = _make_mock_cache_context(
            chunk_size=chunk_size,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        keys = [_make_object_keys("req-store", num_chunks)]  # one object group
        return ctx, keys

    def test_returns_transfer_plan_on_success(self):
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=2, chunk_size=4)
        # 2 chunks × 4 blocks_per_chunk = 8 block ids
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        builder = TransferPlanBuilder(ctx)
        plan = builder.build_store_plan("req-store", obj_keys, block_ids)
        assert plan is not None
        assert plan.direction == TransferDirection.STORE
        assert plan.request_id == "req-store"
        assert plan.chunk_size == 4
        assert len(plan.object_groups) == 1

    def test_returns_none_on_underflow(self):
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=2, chunk_size=4)
        block_ids = [[0, 1, 2]]  # only 3 ids, need 8
        builder = TransferPlanBuilder(ctx)
        plan = builder.build_store_plan("req-store", obj_keys, block_ids)
        assert plan is None

    def test_object_keys_preserved(self):
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=2, chunk_size=4)
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        assert plan.object_groups[0].object_keys == obj_keys[0]

    def test_skip_blocks_is_zero_for_store(self):
        """Store operations always have skip_blocks == 0 in all batch plans."""
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=2, chunk_size=4)
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        for og in plan.object_groups:
            for batch in og.batches:
                for kg_batch in batch.kernel_groups:
                    assert kg_batch.skip_blocks == 0

    def test_num_objects_to_skip_is_zero_for_store(self):
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=2, chunk_size=4)
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        for og in plan.object_groups:
            assert og.num_objects_to_skip == 0

    def test_selected_block_ids_per_kernel_group_matches_downsampled(self):
        """selected_block_ids_per_kernel_group is the flat convenience copy."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=2)],
        )
        # SW group: keep last 2 per chunk
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        obj_keys = [_make_object_keys("req", 2)]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        expected = [2, 3, 6, 7]
        assert plan.selected_block_ids_per_kernel_group[0] == expected
        assert plan.object_groups[0].kernel_groups[0].selected_block_ids == expected

    def test_layout_desc_present(self):
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=1, chunk_size=4)
        block_ids = [[0, 1, 2, 3]]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        layout = plan.object_groups[0].layout_desc
        assert isinstance(layout, MemoryLayoutDesc)

    def test_multi_object_group(self):
        """Store plan covers all object groups."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[
                _KernelGroupSpec(block_size=1),  # kg 0 → og 0
                _KernelGroupSpec(block_size=1),  # kg 1 → og 1
            ],
        )
        # 2 object groups, each with 2 chunks × 4 blocks = 8 block ids
        obj_keys = [_make_object_keys("req", 2), _make_object_keys("req", 2)]
        block_ids = [list(range(8)), list(range(8, 16))]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        assert len(plan.object_groups) == 2
        assert plan.object_groups[0].object_group_id == 0
        assert plan.object_groups[1].object_group_id == 1


# ---------------------------------------------------------------------------
# TransferPlanBuilder.build_retrieve_plan
# ---------------------------------------------------------------------------


class TestTransferPlanBuilderRetrieve:
    """Tests for ``TransferPlanBuilder.build_retrieve_plan``."""

    def _make_ctx_and_keys(self, num_chunks: int = 2, chunk_size: int = 4):
        ctx = _make_mock_cache_context(
            chunk_size=chunk_size,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        keys = [_make_object_keys("req-ret", num_chunks)]
        return ctx, keys

    def test_returns_transfer_plan_on_success(self):
        ctx, obj_keys = self._make_ctx_and_keys()
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req-ret", obj_keys, block_ids
        )
        assert plan is not None
        assert plan.direction == TransferDirection.RETRIEVE

    def test_returns_none_on_underflow(self):
        ctx, obj_keys = self._make_ctx_and_keys()
        block_ids = [[0]]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req-ret", obj_keys, block_ids
        )
        assert plan is None

    def test_skip_first_n_tokens_produces_skip_blocks(self):
        """Non-zero skip_first_n_tokens sets skip_blocks on the first batch."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        obj_keys = [_make_object_keys("req", 2)]
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        # Skip 2 tokens → skip_blocks = 2 (with bpc=4, bpw=4, recalculate=identity)
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req", obj_keys, block_ids, skip_first_n_tokens=2
        )
        assert plan is not None
        og = plan.object_groups[0]
        assert len(og.batches) > 0
        first_batch = og.batches[0]
        # First batch has skip_blocks=2; subsequent batches (if any) have 0.
        assert first_batch.kernel_groups[0].skip_blocks == 2
        for batch in og.batches[1:]:
            for kg_batch in batch.kernel_groups:
                assert kg_batch.skip_blocks == 0

    def test_zero_skip_produces_zero_skip_blocks(self):
        ctx, obj_keys = self._make_ctx_and_keys()
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req-ret", obj_keys, block_ids, skip_first_n_tokens=0
        )
        assert plan is not None
        for og in plan.object_groups:
            for batch in og.batches:
                for kg_batch in batch.kernel_groups:
                    assert kg_batch.skip_blocks == 0

    def test_full_attention_num_objects_to_skip_zero(self):
        """Full-attention groups have num_objects_to_skip == 0."""
        ctx, obj_keys = self._make_ctx_and_keys(num_chunks=4)
        block_ids = [list(range(16))]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req-ret", obj_keys, block_ids
        )
        assert plan is not None
        assert plan.object_groups[0].num_objects_to_skip == 0

    def test_sliding_window_object_group_skips_prefix(self):
        """SW object groups skip leading chunks that fall outside the window."""
        # 4 chunks, SW size = 2 chunks → skip first 2 chunks
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            object_group_kernel_indices=[[0]],
            sw_size_chunks_per_object_group=[2],  # window = 2 chunks
        )
        obj_keys = [_make_object_keys("req", 4)]
        block_ids = [list(range(16))]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        # 4 chunks, window covers last 2 → skip first 2
        assert og.num_objects_to_skip == 2

    def test_selected_block_ids_reflect_downsampling(self):
        """SW downsampling is reflected in the plan's selected block IDs."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=2)],
        )
        obj_keys = [_make_object_keys("req", 2)]
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7]]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        # SW → last 2 per chunk: [2, 3, 6, 7]
        assert plan.selected_block_ids_per_kernel_group[0] == [2, 3, 6, 7]


# ---------------------------------------------------------------------------
# Batch-level plan: ObjectBatchTransferPlan / KernelGroupBatchTransferPlan
# ---------------------------------------------------------------------------


class TestBatchLevelPlan:
    """Tests verifying that TransferPlanBuilder pre-computes correct batch-level
    geometry so the executor can iterate plan batches without recalculation."""

    # ---- Store batches -------------------------------------------------------

    def test_store_batches_one_per_chunk(self):
        """Store operations produce one batch per chunk (batch_size=1)."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        obj_keys = [_make_object_keys("req", 3)]
        block_ids = [list(range(12))]  # 3 chunks × 4 blocks
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        assert len(og.batches) == 3
        for idx, batch in enumerate(og.batches):
            assert batch.start_object_idx == idx
            assert batch.batch_len == 1
            assert len(batch.kernel_groups) == 1

    def test_store_batch_block_positions(self):
        """Store batches carry correct start_block_pos and block_count."""
        # chunk_size=4, block_size=1 → bpc=bpw=4
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        obj_keys = [_make_object_keys("req", 2)]
        block_ids = [list(range(8))]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        batches = plan.object_groups[0].batches
        # batch 0: object 0, bpw=4 → start=0, count=4
        assert batches[0].kernel_groups[0].start_block_pos == 0
        assert batches[0].kernel_groups[0].block_count == 4
        # batch 1: object 1 → start=4, count=4
        assert batches[1].kernel_groups[0].start_block_pos == 4
        assert batches[1].kernel_groups[0].block_count == 4

    def test_store_sw_batch_block_positions(self):
        """Store batches with SW groups use blocks_per_window for positions."""
        # chunk_size=4, block_size=1, sw=2 → bpc=4, bpw=2
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1, sw_size_tokens=2)],
        )
        obj_keys = [_make_object_keys("req", 3)]
        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # 3 chunks × 4
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        batches = plan.object_groups[0].batches
        assert len(batches) == 3
        # block positions indexed by window-frame (bpw=2)
        assert batches[0].kernel_groups[0].start_block_pos == 0
        assert batches[0].kernel_groups[0].block_count == 2
        assert batches[1].kernel_groups[0].start_block_pos == 2
        assert batches[1].kernel_groups[0].block_count == 2
        assert batches[2].kernel_groups[0].start_block_pos == 4
        assert batches[2].kernel_groups[0].block_count == 2

    # ---- Retrieve batches ---------------------------------------------------

    def test_retrieve_batches_limited_by_max_batch_size(self):
        """Retrieve groups batches by max_batch_size chunks."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            max_batch_size=2,
        )
        obj_keys = [_make_object_keys("req", 4)]
        block_ids = [list(range(16))]  # 4 chunks × 4 blocks
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        # 4 chunks / batch_size=2 → 2 batches
        assert len(og.batches) == 2
        assert og.batches[0].start_object_idx == 0
        assert og.batches[0].batch_len == 2
        assert og.batches[1].start_object_idx == 2
        assert og.batches[1].batch_len == 2

    def test_retrieve_sw_skip_reflected_in_start_object_idx(self):
        """SW retrieve: first batch's start_object_idx equals num_objects_to_skip."""
        # 4 chunks, SW=2 → skip first 2
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            sw_size_chunks_per_object_group=[2],
            max_batch_size=4,
        )
        obj_keys = [_make_object_keys("req", 4)]
        block_ids = [list(range(16))]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        assert og.num_objects_to_skip == 2
        assert len(og.batches) == 1  # 2 remaining chunks fit in one batch (max=4)
        assert og.batches[0].start_object_idx == 2
        assert og.batches[0].batch_len == 2

    def test_retrieve_sw_batch_block_positions(self):
        """Block positions in SW retrieve batches are offset by skipped objects."""
        # 4 chunks, SW=2 chunks, bpw=bpc=4 (full-attn kernel group)
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            sw_size_chunks_per_object_group=[2],
            max_batch_size=4,
        )
        obj_keys = [_make_object_keys("req", 4)]
        block_ids = [list(range(16))]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        batch = og.batches[0]
        kg_batch = batch.kernel_groups[0]
        # start_object_idx=2, bpw=4 → start_block_pos=8; batch_len=2, block_count=8
        assert kg_batch.start_block_pos == 8
        assert kg_batch.block_count == 8

    def test_retrieve_skip_first_n_tokens_first_batch_skip_blocks(self):
        """skip_first_n_tokens produces correct skip_blocks only on first batch."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            max_batch_size=1,
        )
        obj_keys = [_make_object_keys("req", 3)]
        block_ids = [list(range(12))]
        # Skip 2 tokens → first batch skip_blocks=2; subsequent batches skip_blocks=0
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req", obj_keys, block_ids, skip_first_n_tokens=2
        )
        assert plan is not None
        og = plan.object_groups[0]
        assert og.batches[0].kernel_groups[0].skip_blocks == 2
        for batch in og.batches[1:]:
            for kg_batch in batch.kernel_groups:
                assert kg_batch.skip_blocks == 0

    def test_retrieve_skip_that_covers_entire_first_batch_drops_batch(self):
        """A batch whose full token range is below skip_first_n_tokens is dropped."""
        # chunk_size=4, max_batch_size=1 → each batch = 1 chunk = 4 tokens
        # skip_first_n_tokens=5 → first batch [0,4) is fully covered → dropped
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
            max_batch_size=1,
        )
        obj_keys = [_make_object_keys("req", 3)]
        block_ids = [list(range(12))]
        plan = TransferPlanBuilder(ctx).build_retrieve_plan(
            "req", obj_keys, block_ids, skip_first_n_tokens=5
        )
        assert plan is not None
        og = plan.object_groups[0]
        # Batch 0 (tokens 0-4) is dropped; remaining: batch 1 (tokens 4-8) and 2 (8-12)
        assert len(og.batches) == 2
        assert og.batches[0].start_object_idx == 1

    def test_batch_plan_types(self):
        """Plan batches use ObjectBatchTransferPlan / KernelGroupBatchTransferPlan."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[_KernelGroupSpec(block_size=1)],
        )
        obj_keys = [_make_object_keys("req", 1)]
        block_ids = [list(range(4))]
        plan = TransferPlanBuilder(ctx).build_store_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        assert isinstance(og.batches[0], ObjectBatchTransferPlan)
        assert isinstance(og.batches[0].kernel_groups[0], KernelGroupBatchTransferPlan)

    def test_retrieve_multi_kernel_group_batch_has_entry_per_kg(self):
        """Each batch includes one KernelGroupBatchTransferPlan per kernel group."""
        ctx = _make_mock_cache_context(
            chunk_size=4,
            kernel_groups=[
                _KernelGroupSpec(block_size=1),
                _KernelGroupSpec(block_size=2),
            ],
            object_group_kernel_indices=[[0, 1]],
        )
        obj_keys = [_make_object_keys("req", 2)]
        block_ids = [list(range(8)), list(range(4))]  # kg0: bpc=4; kg1: bpc=2
        plan = TransferPlanBuilder(ctx).build_retrieve_plan("req", obj_keys, block_ids)
        assert plan is not None
        og = plan.object_groups[0]
        for batch in og.batches:
            # One entry per kernel group in the object group
            assert len(batch.kernel_groups) == 2
            kg_ids = {kgb.kernel_group_id for kgb in batch.kernel_groups}
            assert kg_ids == {0, 1}
