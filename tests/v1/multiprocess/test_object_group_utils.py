# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``object_group_utils``.

These tests cover:

* Pure geometry helpers (CPU-only, no GPU/native-extension required):
  ``has_sufficient_block_ids``, ``select_block_ids_for_window``,
  ``recalculate_blocks_to_skip``, ``compute_num_objects_to_skip``, and
  ``batched_iteration_with_skip``.
* ``prepare_object_group_transfer`` — verified by mocking the native
  ``lmc_ops`` types so the test does not require a GPU.
* ``execute_prepared_object_group_transfer`` — verified by mocking
  ``lmc_ops.execute_object_group_transfer``.
"""

# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_attn_desc(num_chunks_in_sw: list[int]) -> Any:
    """Return a real AttnWindowDesc without importing anything GPU-bound."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc

    return AttnWindowDesc(num_chunks_in_sw=num_chunks_in_sw)


# ---------------------------------------------------------------------------
# has_sufficient_block_ids
# ---------------------------------------------------------------------------


class TestHasSufficientBlockIds:
    """Tests for :func:`has_sufficient_block_ids`."""

    def test_returns_true_when_all_groups_cover_all_chunks(self) -> None:
        """Every group with at least num_chunks * bpc block IDs passes."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            has_sufficient_block_ids,
        )

        assert has_sufficient_block_ids(
            block_ids=[[0, 1, 2, 3], [10, 11, 12, 13, 14, 15]],
            blocks_per_chunk=[2, 3],
            num_chunks=2,
        )

    def test_returns_false_when_any_group_is_short(self) -> None:
        """A single underfilled group fails the validation."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            has_sufficient_block_ids,
        )

        assert not has_sufficient_block_ids(
            block_ids=[[0, 1, 2, 3], [10, 11, 12, 13, 14]],
            blocks_per_chunk=[2, 3],
            num_chunks=2,
        )

    def test_extra_block_ids_are_allowed(self) -> None:
        """Groups may contain more than the minimum required raw block IDs."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            has_sufficient_block_ids,
        )

        assert has_sufficient_block_ids(
            block_ids=[[0, 1, 2, 3, 4]],
            blocks_per_chunk=[2],
            num_chunks=2,
        )

    def test_zero_chunks_with_empty_block_ids_returns_true(self) -> None:
        """Zero num_chunks is trivially satisfied by any block-ID list."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            has_sufficient_block_ids,
        )

        assert has_sufficient_block_ids(
            block_ids=[[]],
            blocks_per_chunk=[4],
            num_chunks=0,
        )


# ---------------------------------------------------------------------------
# select_block_ids_for_window
# ---------------------------------------------------------------------------


class TestSelectBlockIdsForWindow:
    """Tests for :func:`select_block_ids_for_window`."""

    def test_full_window_returns_all_blocks(self) -> None:
        """When keep == total, every block is returned unchanged."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        result = select_block_ids_for_window(
            ids, total_blocks_per_chunk=4, keep_blocks_per_chunk=4
        )
        assert result == ids

    def test_keep_trailing_two_per_chunk(self) -> None:
        """Keep last 2 of every 4-block chunk."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        ids = [10, 11, 12, 13, 20, 21, 22, 23]
        result = select_block_ids_for_window(
            ids, total_blocks_per_chunk=4, keep_blocks_per_chunk=2
        )
        assert result == [12, 13, 22, 23]

    def test_keep_one_per_chunk(self) -> None:
        """Keep only the last block of each chunk."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        ids = [1, 2, 3, 4]
        result = select_block_ids_for_window(
            ids, total_blocks_per_chunk=4, keep_blocks_per_chunk=1
        )
        assert result == [4]

    def test_single_chunk(self) -> None:
        """Single-chunk input with partial keep."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        ids = [10, 11, 12, 13, 14, 15, 16, 17]
        result = select_block_ids_for_window(
            ids, total_blocks_per_chunk=8, keep_blocks_per_chunk=3
        )
        assert result == [15, 16, 17]

    def test_empty_input(self) -> None:
        """Empty input returns empty output."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        result = select_block_ids_for_window(
            [], total_blocks_per_chunk=4, keep_blocks_per_chunk=2
        )
        assert result == []

    def test_invalid_total_blocks(self) -> None:
        """total_blocks_per_chunk < 1 raises ValueError."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        with pytest.raises(ValueError, match="total_blocks_per_chunk must be >= 1"):
            select_block_ids_for_window(
                [1, 2], total_blocks_per_chunk=0, keep_blocks_per_chunk=1
            )

    def test_keep_exceeds_total_raises(self) -> None:
        """keep_blocks_per_chunk > total_blocks_per_chunk raises ValueError."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        with pytest.raises(ValueError, match="keep_blocks_per_chunk must be in"):
            select_block_ids_for_window(
                [1, 2, 3, 4], total_blocks_per_chunk=4, keep_blocks_per_chunk=5
            )

    def test_keep_zero_raises(self) -> None:
        """keep_blocks_per_chunk < 1 raises ValueError."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        with pytest.raises(ValueError, match="keep_blocks_per_chunk must be in"):
            select_block_ids_for_window(
                [1, 2, 3, 4], total_blocks_per_chunk=4, keep_blocks_per_chunk=0
            )

    def test_len_not_multiple_raises(self) -> None:
        """len(block_ids) not a multiple of total_blocks_per_chunk raises ValueError."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        with pytest.raises(
            ValueError, match="must be a multiple of total_blocks_per_chunk"
        ):
            select_block_ids_for_window(
                [1, 2, 3], total_blocks_per_chunk=4, keep_blocks_per_chunk=2
            )

    @pytest.mark.parametrize(
        "total,keep,ids,expected",
        [
            (2, 1, [10, 11, 20, 21, 30, 31], [11, 21, 31]),
            (3, 2, [1, 2, 3, 4, 5, 6], [2, 3, 5, 6]),
            (1, 1, [7, 8, 9], [7, 8, 9]),
        ],
    )
    def test_parametrized_cases(
        self, total: int, keep: int, ids: list, expected: list
    ) -> None:
        """Parametrized spot-checks for various (total, keep) combinations."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_window,
        )

        assert select_block_ids_for_window(ids, total, keep) == expected


class TestSelectBlockIdsForCacheContext:
    """Tests for :func:`select_block_ids_for_cache_context`."""

    def test_selects_per_kernel_group_windows_without_mutating_input(self) -> None:
        """Each kernel group uses its own subchunk window geometry."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            select_block_ids_for_cache_context,
        )

        class _FakeGroupsManager:
            num_kernel_groups = 2

            def get_subchunk_sw_size_tokens(self, kernel_group_id: int) -> int:
                return [8, 4][kernel_group_id]

        class _FakeCacheContext:
            lmcache_tokens_per_chunk = 8
            kv_layer_groups_manager = _FakeGroupsManager()

            def calculate_num_blocks(
                self, num_tokens: int, _kernel_group_id: int
            ) -> int:
                return num_tokens // 2

        block_ids = [[0, 1, 2, 3, 4, 5, 6, 7], [10, 11, 12, 13, 20, 21, 22, 23]]

        result = select_block_ids_for_cache_context(_FakeCacheContext(), block_ids)

        assert result == [[0, 1, 2, 3, 4, 5, 6, 7], [12, 13, 22, 23]]
        assert block_ids == [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 20, 21, 22, 23],
        ]


# ---------------------------------------------------------------------------
# recalculate_blocks_to_skip
# ---------------------------------------------------------------------------


class TestRecalculateBlocksToSkip:
    """Tests for :func:`recalculate_blocks_to_skip`."""

    def test_equal_window_and_chunk_is_identity(self) -> None:
        """When blocks_per_window == blocks_per_chunk, return unchanged."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        assert recalculate_blocks_to_skip(4, 4, 6) == 6
        assert recalculate_blocks_to_skip(1, 1, 0) == 0

    def test_zero_skip_always_zero(self) -> None:
        """Zero blocks_to_skip always returns zero."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        assert recalculate_blocks_to_skip(4, 2, 0) == 0

    def test_skip_less_than_chunk_tail(self) -> None:
        """Skip falls entirely within the discarded prefix of the first chunk."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        # blocks_per_chunk=4, blocks_per_window=2, skip=1
        # tail = 1, tail_to_skip = 1 - (4-2) = -1 → 0
        # result = 0 * 2 + 0 = 0
        assert recalculate_blocks_to_skip(4, 2, 1) == 0

    def test_skip_exactly_one_full_chunk(self) -> None:
        """Skip exactly one full-chunk's worth maps to one window."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        # blocks_per_chunk=4, blocks_per_window=2, skip=4
        # full_windows = 1, tail = 0, result = 1 * 2 = 2
        assert recalculate_blocks_to_skip(4, 2, 4) == 2

    def test_skip_extends_into_window_region(self) -> None:
        """Skip that extends into the kept window region is mapped correctly."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        # blocks_per_chunk=4, blocks_per_window=2, skip=3
        # tail = 3, tail_to_skip = 3 - (4-2) = 1, result = 0*2 + 1 = 1
        assert recalculate_blocks_to_skip(4, 2, 3) == 1

    def test_example_from_docstring(self) -> None:
        """Verify the example in the docstring."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        # blocks_per_chunk=4, blocks_per_window=2, skip=6
        # full_windows = 1, tail = 2, tail_to_skip = 2-(4-2) = 0, result = 2
        assert recalculate_blocks_to_skip(4, 2, 6) == 2

    @pytest.mark.parametrize("skip", [0, 1, 2, 3, 4, 5, 8, 12])
    def test_result_never_negative(self, skip: int) -> None:
        """Result is always >= 0."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            recalculate_blocks_to_skip,
        )

        assert recalculate_blocks_to_skip(4, 2, skip) >= 0


# ---------------------------------------------------------------------------
# compute_num_objects_to_skip
# ---------------------------------------------------------------------------


class TestComputeNumObjectsToSkip:
    """Tests for :func:`compute_num_objects_to_skip`."""

    def test_d2h_always_zero(self) -> None:
        """D2H (store) never skips any objects."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        attn_desc = _make_attn_desc(num_chunks_in_sw=[3])
        assert compute_num_objects_to_skip(attn_desc, 0, 10, is_h2d=False) == 0

    def test_full_attention_h2d_zero(self) -> None:
        """Full-attention groups never skip objects, even for H2D."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        attn_desc = _make_attn_desc(num_chunks_in_sw=[-1])  # -1 means full attention
        assert compute_num_objects_to_skip(attn_desc, 0, 10, is_h2d=True) == 0

    def test_sliding_window_h2d_skips_prefix(self) -> None:
        """H2D with a sliding window skips objects before the window."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        # sw = 3, num_objects = 7 → skip 4 objects
        attn_desc = _make_attn_desc(num_chunks_in_sw=[3])
        assert compute_num_objects_to_skip(attn_desc, 0, 7, is_h2d=True) == 4

    def test_sliding_window_num_objects_less_than_sw_skip_zero(self) -> None:
        """When fewer objects than the window size, skip nothing."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        attn_desc = _make_attn_desc(num_chunks_in_sw=[5])
        assert compute_num_objects_to_skip(attn_desc, 0, 3, is_h2d=True) == 0

    def test_sliding_window_num_objects_equals_sw_skip_zero(self) -> None:
        """Exactly sw_size objects → skip zero."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        attn_desc = _make_attn_desc(num_chunks_in_sw=[4])
        assert compute_num_objects_to_skip(attn_desc, 0, 4, is_h2d=True) == 0

    def test_multiple_object_groups_correct_group_selected(self) -> None:
        """The correct group's window size is used."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            compute_num_objects_to_skip,
        )

        # group 0 = full attention, group 1 = sw=2
        attn_desc = _make_attn_desc(num_chunks_in_sw=[-1, 2])
        assert compute_num_objects_to_skip(attn_desc, 0, 10, is_h2d=True) == 0
        assert compute_num_objects_to_skip(attn_desc, 1, 10, is_h2d=True) == 8


# ---------------------------------------------------------------------------
# batched_iteration_with_skip (re-exported by the helpers module)
# ---------------------------------------------------------------------------


class TestBatchedIterationWithSkipViaHelpers:
    """Smoke tests verifying the helpers re-export is wired correctly."""

    def test_skip_and_batch(self) -> None:
        """Import from helpers module returns correct results."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            batched_iteration_with_skip,
        )

        result = list(batched_iteration_with_skip(range(10), 3, skip_count=2))
        assert result == [
            (2, (2, 3, 4)),
            (5, (5, 6, 7)),
            (8, (8, 9)),
        ]


# ---------------------------------------------------------------------------
# execute_prepared_object_group_transfer
# ---------------------------------------------------------------------------


class TestExecutePreparedObjectGroupTransfer:
    """Tests for :func:`execute_prepared_object_group_transfer`."""

    def test_noop_on_empty_batch_steps(self) -> None:
        """No call to lmc_ops when batch_steps is empty."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        direction = lmc_ops.TransferDirection.D2H
        with patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec:
            execute_prepared_object_group_transfer(direction, "cuda:0", [], [])
            mock_exec.assert_not_called()

    def test_calls_execute_with_correct_args(self) -> None:
        """When batch_steps is non-empty, execute_object_group_transfer is called."""
        # First Party
        from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        direction = lmc_ops.TransferDirection.H2D
        fake_specs = [MagicMock(name="spec0")]
        fake_steps = [MagicMock(name="step0")]
        device = "cuda:0"

        with patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec:
            execute_prepared_object_group_transfer(
                direction, device, fake_specs, fake_steps
            )
            mock_exec.assert_called_once_with(
                direction,
                device,
                LazyMemoryAllocator.PIN_CHUNK_SIZE,
                fake_specs,
                fake_steps,
            )

    def test_multiple_batch_steps_calls_execute_once(self) -> None:
        """Multiple batch steps still result in a single execute call."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        direction = lmc_ops.TransferDirection.D2H
        fake_specs = [MagicMock(), MagicMock()]
        fake_steps = [MagicMock(), MagicMock(), MagicMock()]

        with patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec:
            execute_prepared_object_group_transfer(
                direction, "cuda:1", fake_specs, fake_steps
            )
            assert mock_exec.call_count == 1

    def test_custom_host_buffer_alignment_is_forwarded(self) -> None:
        """Caller-provided host buffer alignment is forwarded to lmc_ops."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        direction = lmc_ops.TransferDirection.H2D
        fake_specs = [MagicMock(name="spec0")]
        fake_steps = [MagicMock(name="step0")]
        host_buffer_alignment = 4096

        with patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec:
            execute_prepared_object_group_transfer(
                direction,
                "cuda:0",
                fake_specs,
                fake_steps,
                host_buffer_alignment=host_buffer_alignment,
            )
            mock_exec.assert_called_once_with(
                direction,
                "cuda:0",
                host_buffer_alignment,
                fake_specs,
                fake_steps,
            )


# ---------------------------------------------------------------------------
# prepare_object_group_transfer — mock-based integration tests
# ---------------------------------------------------------------------------


def _make_mock_cache_context(
    *,
    lmcache_tokens_per_chunk: int = 8,
    max_batch_size: int = 2,
    num_kernel_groups: int = 1,
    object_group_kernel_indices: list[int] | None = None,
    subchunk_sw_size_tokens: int = 8,
    blocks_per_chunk: int = 2,
    blocks_per_window: int = 2,
    num_chunks_in_sw: list[int] | None = None,
) -> MagicMock:
    """Build a minimal mock BaseCacheContext for prepare_object_group_transfer tests.

    Args:
        lmcache_tokens_per_chunk: Chunk size in tokens.
        max_batch_size: Maximum batch size.
        num_kernel_groups: Number of kernel groups in the object group.
        object_group_kernel_indices: Kernel group IDs for object group 0.
        subchunk_sw_size_tokens: Sub-chunk sliding-window size.
        blocks_per_chunk: Value returned by ``calculate_num_blocks`` when called
            with ``lmcache_tokens_per_chunk`` tokens.  Must be consistent with
            the chunk geometry used in each test (i.e. the block IDs tensor
            must contain exactly ``num_chunks * blocks_per_chunk`` entries).
        blocks_per_window: Value returned by ``calculate_num_blocks`` for any
            other token count (i.e. the sliding-window size in blocks).
        num_chunks_in_sw: Per-object-group SW chunk count for AttnWindowDesc.
            Use ``[-1]`` (the default) to simulate full-attention (no skip).

    Returns:
        Configured MagicMock acting as BaseCacheContext.
    """
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc

    if object_group_kernel_indices is None:
        object_group_kernel_indices = list(range(num_kernel_groups))
    if num_chunks_in_sw is None:
        num_chunks_in_sw = [-1]  # full attention

    ctx = MagicMock(name="cache_context")
    ctx.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
    ctx.max_batch_size = max_batch_size
    ctx.device = "cpu"

    # KV groups manager
    kv_mgr = MagicMock(name="kv_layer_groups_manager")
    object_group = MagicMock(name="object_group")
    object_group.kernel_group_indices = object_group_kernel_indices
    kv_mgr.object_groups = {0: object_group}
    kv_mgr.get_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=num_chunks_in_sw
    )
    kv_mgr.get_subchunk_sw_size_tokens.return_value = subchunk_sw_size_tokens
    ctx.kv_layer_groups_manager = kv_mgr

    # Block calculations: return blocks_per_chunk for chunk-size tokens,
    # blocks_per_window for anything else (window-size tokens).
    def _calculate_num_blocks(tokens: int, kernel_group_id: int) -> int:
        if tokens == lmcache_tokens_per_chunk:
            return blocks_per_chunk
        return blocks_per_window

    ctx.calculate_num_blocks.side_effect = _calculate_num_blocks

    # GPU tensors: use small CPU tensors so data_ptr() calls don't crash.
    paged_ptrs = torch.zeros(1, dtype=torch.int64)
    ctx.get_kernel_group_kv_pointers.return_value = paged_ptrs

    def _get_temp_kg_buffer(slot: int, kernel_group_id: int) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.int64)

    ctx.get_temp_kernel_group_buffer.side_effect = _get_temp_kg_buffer

    def _get_temp_og_buffer(slot: int, object_group_id: int) -> torch.Tensor:
        # Object-group buffer used for staging; nbytes must match memory_obj.
        return torch.zeros(4, dtype=torch.uint8)

    ctx.get_temp_object_group_buffer.side_effect = _get_temp_og_buffer

    ctx.get_shape_desc.return_value = MagicMock(name="shape_desc")
    ctx.get_slots_per_chunk_in_sw.return_value = lmcache_tokens_per_chunk
    ctx.get_engine_kv_format.return_value = MagicMock(name="engine_kv_format")

    return ctx


def _make_mock_memory_obj(size: int = 4) -> MagicMock:
    """Create a MemoryObj-like mock with fields used by the staging builder.

    Args:
        size: Byte size returned by ``get_size()`` and ``meta.address``.

    Returns:
        Configured MagicMock with ``raw_tensor``, ``get_size()``,
        ``data_ptr``, and ``meta.address``.
    """
    mo = MagicMock(name="memory_obj")
    mo.raw_tensor = torch.zeros(size, dtype=torch.uint8)
    mo.get_size.return_value = size
    mo.data_ptr = 0xDEADBEEF
    mo.meta.address = 0
    return mo


@contextmanager
def _patch_native_types_and_staging():
    """Patch native lmc_ops plan types and provide fake staging for CPU-only runs.

    Patches ``lmc_ops.KernelGroupSpec``, ``.LaunchVar``, and ``.BatchStep`` so
    tests can run without the compiled C extension.  The yielded
    ``staging`` mock is passed directly to ``prepare_object_group_transfer`` as
    its caller-provided staging builder.

    Yields:
        A dict with ``"spec"`` (patched ``KernelGroupSpec``), ``"step"``
        (patched ``BatchStep``), ``"lv"`` (patched ``LaunchVar``), and
        ``"staging"`` (fake staging builder) mocks.
    """
    # First Party
    import lmcache.c_ops as lmc_ops

    mock_staging = MagicMock(return_value=[MagicMock(name="staging_copy")])
    with (
        patch.object(lmc_ops, "KernelGroupSpec") as mock_spec_cls,
        patch.object(lmc_ops, "BatchStep") as mock_step_cls,
        patch.object(lmc_ops, "LaunchVar") as mock_lv_cls,
    ):
        mock_spec_cls.return_value = MagicMock(name="spec")
        mock_step_cls.return_value = MagicMock(name="step")
        mock_lv_cls.return_value = MagicMock(name="lv")
        yield {
            "spec": mock_spec_cls,
            "step": mock_step_cls,
            "lv": mock_lv_cls,
            "staging": mock_staging,
        }


class TestPrepareObjectGroupTransfer:
    """Tests for :func:`prepare_object_group_transfer`."""

    def test_builds_one_kernel_group_spec(self) -> None:
        """prepare_object_group_transfer creates one KernelGroupSpec per group."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(num_kernel_groups=1, blocks_per_chunk=2)
        block_ids_gpu = [torch.tensor([0, 1, 2, 3], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj()]

        with _patch_native_types_and_staging() as mocks:
            specs, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
                staging_builder=mocks["staging"],
            )

        assert len(specs) == 1
        mocks["spec"].assert_called_once()

    def test_builds_batch_steps(self) -> None:
        """One batch step is created when memory_objs fits in one batch."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj() for _ in range(3)]

        with _patch_native_types_and_staging() as mocks:
            specs, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
                staging_builder=mocks["staging"],
            )

        # All 3 objects fit in one batch.
        assert len(steps) == 1

    def test_batches_objects_by_batch_size(self) -> None:
        """Multiple batch steps are created when mem_objs exceed batch_size."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=2,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2, 3], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj() for _ in range(4)]

        with _patch_native_types_and_staging() as mocks:
            _, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=2,  # force 2 batches
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
                staging_builder=mocks["staging"],
            )

        assert len(steps) == 2

    def test_sliding_window_h2d_skips_leading_objects(self) -> None:
        """Sliding-window H2D skips objects outside the window."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        # sw = 2, 5 objects → skip 3
        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=5,
            blocks_per_chunk=1,
            blocks_per_window=1,
            num_chunks_in_sw=[2],
        )
        block_ids_gpu = [torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj() for _ in range(5)]

        with _patch_native_types_and_staging() as mocks:
            _, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=5,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.H2D,
                staging_builder=mocks["staging"],
            )

        # Only 2 objects (the window) are transferred → 1 batch step.
        assert len(steps) == 1

    def test_none_in_d2h_batch_is_skipped(self) -> None:
        """None entries in D2H batches are silently skipped."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        # Second object is None (D2H skip); must not raise.
        mem_objs: list = [_make_mock_memory_obj(), None, _make_mock_memory_obj()]

        with _patch_native_types_and_staging() as mocks:
            # Must not raise for D2H with None entries.
            prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
                staging_builder=mocks["staging"],
            )

    def test_none_in_h2d_batch_raises(self) -> None:
        """None entries in H2D batches raise ValueError."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        mem_objs: list = [_make_mock_memory_obj(), None, _make_mock_memory_obj()]

        with _patch_native_types_and_staging() as mocks:
            with pytest.raises(
                ValueError,
                match=r"Object is None.*cannot perform H2D copy",
            ):
                prepare_object_group_transfer(
                    ctx,
                    block_ids_gpu,
                    mem_objs,
                    object_group_id=0,
                    batch_size=4,
                    skip_first_n_tokens=0,
                    direction=lmc_ops.TransferDirection.H2D,
                    staging_builder=mocks["staging"],
                )

    def test_skip_first_n_tokens_skips_all_batches(self) -> None:
        """skip_first_n_tokens >= total tokens produces no batch steps."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        chunk_size = 4
        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=chunk_size,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        # 3 chunks, skip the first chunk entirely
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj() for _ in range(3)]

        with _patch_native_types_and_staging() as mocks:
            _, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=chunk_size * 3,  # skip all
                direction=lmc_ops.TransferDirection.H2D,
                staging_builder=mocks["staging"],
            )

        # All batches fully skipped → no steps
        assert len(steps) == 0

    def test_empty_batch_steps_when_all_none_d2h(self) -> None:
        """All-None D2H batch produces no batch steps."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        mem_objs: list = [None, None, None]

        with _patch_native_types_and_staging() as mocks:
            _, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                object_group_id=0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=lmc_ops.TransferDirection.D2H,
                staging_builder=mocks["staging"],
            )

        assert len(steps) == 0


# ---------------------------------------------------------------------------
# End-to-end: prepare → execute path calls lmc_ops.execute_object_group_transfer
# ---------------------------------------------------------------------------


class TestPrepareAndExecutePipeline:
    """End-to-end pipeline: prepare_object_group_transfer then
    execute_prepared_object_group_transfer.
    """

    def test_execute_called_when_batch_steps_non_empty(self) -> None:
        """When prepare returns non-empty batch_steps, execute forwards them."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=2,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1], dtype=torch.int32)]
        mem_objs = [_make_mock_memory_obj() for _ in range(2)]
        direction = lmc_ops.TransferDirection.D2H

        with (
            _patch_native_types_and_staging() as mocks,
            patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec,
        ):
            specs, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=direction,
                staging_builder=mocks["staging"],
            )
            assert len(steps) > 0, (
                "expected non-empty batch_steps for valid memory objects"
            )
            execute_prepared_object_group_transfer(direction, ctx.device, specs, steps)
            mock_exec.assert_called_once()

    def test_execute_not_called_when_batch_steps_empty(self) -> None:
        """Empty batch_steps (all-None D2H) → execute is never invoked."""
        # First Party
        from lmcache.v1.multiprocess.object_group_utils import (
            execute_prepared_object_group_transfer,
            prepare_object_group_transfer,
        )
        import lmcache.c_ops as lmc_ops

        ctx = _make_mock_cache_context(
            lmcache_tokens_per_chunk=4,
            max_batch_size=4,
            blocks_per_chunk=1,
            blocks_per_window=1,
        )
        block_ids_gpu = [torch.tensor([0, 1, 2], dtype=torch.int32)]
        mem_objs: list = [None, None, None]
        direction = lmc_ops.TransferDirection.D2H

        with (
            _patch_native_types_and_staging() as mocks,
            patch.object(lmc_ops, "execute_object_group_transfer") as mock_exec,
        ):
            specs, steps = prepare_object_group_transfer(
                ctx,
                block_ids_gpu,
                mem_objs,
                0,
                batch_size=4,
                skip_first_n_tokens=0,
                direction=direction,
                staging_builder=mocks["staging"],
            )
            execute_prepared_object_group_transfer(direction, ctx.device, specs, steps)

            mock_exec.assert_not_called()
