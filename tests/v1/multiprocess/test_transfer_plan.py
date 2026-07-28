# SPDX-License-Identifier: Apache-2.0
"""Unit tests for path-agnostic multiprocess transfer-plan helpers."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transfer_plan import (
    batched_iteration_with_skip,
    compute_num_objects_to_skip,
    downsample_block_ids,
    has_sufficient_block_ids,
    recalculate_blocks_to_skip,
    select_block_ids_for_window,
)


def test_has_sufficient_block_ids_variants() -> None:
    """Sufficient, insufficient, and extra block-ID coverage is detected."""
    blocks_per_chunk = [2, 3]
    num_chunks = 2

    assert has_sufficient_block_ids(
        [[1, 2, 3, 4], [9, 8, 7, 6, 5, 4]], blocks_per_chunk, num_chunks
    )
    assert not has_sufficient_block_ids(
        [[1, 2, 3], [9, 8, 7, 6, 5, 4]], blocks_per_chunk, num_chunks
    )
    assert has_sufficient_block_ids(
        [[1, 2, 3, 4, 10], [9, 8, 7, 6, 5, 4, 3]], blocks_per_chunk, num_chunks
    )


def test_has_sufficient_block_ids_group_mismatch_raises() -> None:
    """Mismatched group counts are rejected via strict zip semantics."""
    with pytest.raises(ValueError, match="zip"):
        has_sufficient_block_ids([[1, 2]], [2, 3], num_chunks=1)


@pytest.mark.parametrize(
    ("block_ids", "total_blocks_per_chunk", "keep_blocks_per_chunk", "expected"),
    [
        ([1, 2, 3, 4, 5, 6], 3, 3, [1, 2, 3, 4, 5, 6]),
        ([1, 2, 3, 4, 5, 6], 3, 2, [2, 3, 5, 6]),
    ],
)
def test_select_block_ids_for_window(
    block_ids, total_blocks_per_chunk, keep_blocks_per_chunk, expected
) -> None:
    """Selection keeps either full windows or each chunk's trailing window."""
    original = list(block_ids)
    assert (
        select_block_ids_for_window(
            block_ids, total_blocks_per_chunk, keep_blocks_per_chunk
        )
        == expected
    )
    assert block_ids == original


def test_select_block_ids_for_window_invalid_geometry_and_incomplete_chunks() -> None:
    """Invalid geometry and incomplete chunks raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=0, keep_blocks_per_chunk=1
        )
    with pytest.raises(ValueError, match="at least one"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=2, keep_blocks_per_chunk=0
        )
    with pytest.raises(ValueError, match="less than or equal"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=1, keep_blocks_per_chunk=2
        )
    with pytest.raises(ValueError, match="multiple"):
        select_block_ids_for_window(
            [1, 2, 3], total_blocks_per_chunk=2, keep_blocks_per_chunk=1
        )


def test_downsample_block_ids_multi_group_and_no_input_mutation() -> None:
    """Downsampling preserves group ordering and leaves input untouched."""
    source = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ]
    source_snapshot = [list(group) for group in source]

    downsampled = downsample_block_ids(
        source,
        blocks_per_chunk=[4, 2],
        blocks_per_window=[4, 1],
    )

    assert downsampled == [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [12, 14, 16, 18],
    ]
    assert source == source_snapshot


def test_compute_num_objects_to_skip_cases() -> None:
    """Skip count covers store/full-attention/sliding-window retrieve cases."""
    assert (
        compute_num_objects_to_skip(sw_size_chunks=4, num_objects=7, is_retrieve=False)
        == 0
    )
    assert (
        compute_num_objects_to_skip(sw_size_chunks=8, num_objects=7, is_retrieve=True)
        == 0
    )
    assert (
        compute_num_objects_to_skip(sw_size_chunks=3, num_objects=7, is_retrieve=True)
        == 4
    )


def test_batched_iteration_with_skip_preserves_original_indices() -> None:
    """Batch start indices remain in original sequence coordinate space."""
    result = list(
        batched_iteration_with_skip(
            list(range(10)),
            batch_size=3,
            skip_count=4,
        )
    )
    assert result == [
        (4, (4, 5, 6)),
        (7, (7, 8, 9)),
    ]


@pytest.mark.parametrize(
    ("batch_size", "skip_count", "message"),
    [
        (0, 0, "batch size must be at least one"),
        (1, -1, "skip_count must be non-negative"),
    ],
)
def test_batched_iteration_with_skip_invalid_arguments(
    batch_size, skip_count, message
) -> None:
    """Public argument validation raises ValueError."""
    with pytest.raises(ValueError, match=message):
        list(batched_iteration_with_skip([1, 2, 3], batch_size, skip_count))


def test_recalculate_blocks_to_skip_cases() -> None:
    """Identity, prefix drop, window retain, full chunk, and multi-chunk cases."""
    assert recalculate_blocks_to_skip(4, 4, 3) == 3
    assert recalculate_blocks_to_skip(4, 2, 1) == 0
    assert recalculate_blocks_to_skip(4, 2, 3) == 1
    assert recalculate_blocks_to_skip(4, 2, 4) == 2
    assert recalculate_blocks_to_skip(4, 2, 9) == 4
