# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the path-neutral launch planner and executor."""

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context.common_copy import (
    RegisteredGroup,
    build_group_transfer_plans,
)
from lmcache.v1.multiprocess.transfer_context.common_exec import (
    CopyBatch,
    CopyEndpoint,
    GroupLaunch,
    execute_copy_batches,
    plan_copy_batches,
)
from lmcache.v1.platform.ops_types import TransferDirection


def _group(
    kernel_group_id: int = 0,
    *,
    tokens_per_block: int = 4,
    blocks_per_chunk: int = 2,
    copy_blocks_per_chunk: int = 2,
) -> RegisteredGroup:
    """Build one transfer group with eight logical tokens per chunk.

    Args:
        kernel_group_id: Registration order of the group.
        tokens_per_block: Logical tokens represented by one block.
        blocks_per_chunk: Full engine blocks per LMCache chunk.
        copy_blocks_per_chunk: Trailing blocks copied after window trimming.

    Returns:
        A transfer group suitable for launch-planning tests.
    """
    return RegisteredGroup(
        kernel_group_id=kernel_group_id,
        object_group_id=0,
        engine_group_id=kernel_group_id,
        layer_indices=(kernel_group_id,),
        tokens_per_block=tokens_per_block,
        slots_per_block=tokens_per_block,
        blocks_per_chunk=blocks_per_chunk,
        copy_blocks_per_chunk=copy_blocks_per_chunk,
        chunk_tokens=8,
        shape=torch.Size([1, copy_blocks_per_chunk * tokens_per_block, 1]),
        dtype=torch.float32,
        engine_kv_format=0,
        sw_size_tokens=-1,
    )


class _RecordingEndpoint(CopyEndpoint):
    """Endpoint that records the call order instead of copying anything."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def stage_objects_to_device(self, batch: CopyBatch) -> None:
        """Record an H2D staging step."""
        self.calls.append(("to_device", batch.object_start, batch.num_objects))

    def stage_objects_from_device(self, batch: CopyBatch) -> None:
        """Record a D2H staging step."""
        self.calls.append(("from_device", batch.object_start, batch.num_objects))

    def launch_group_copy(self, batch: CopyBatch, launch: GroupLaunch) -> None:
        """Record one group launch."""
        self.calls.append(
            (
                "launch",
                batch.object_start,
                launch.group_index,
                launch.block_offset,
                launch.num_blocks,
                launch.skip_blocks,
            )
        )

    def end_batch(self, batch: CopyBatch) -> None:
        """Record the end of a batch."""
        self.calls.append(("end", batch.object_start))

    def flush(self) -> None:
        """Record the single flush."""
        self.calls.append(("flush",))


def test_batches_cover_every_object_in_order() -> None:
    """Objects are split into full batches plus a trailing partial batch."""
    plans = build_group_transfer_plans([_group()], [list(range(10))])

    batches = plan_copy_batches(plans, max_objects_per_batch=2)

    assert [(b.object_start, b.num_objects) for b in batches] == [
        (0, 2),
        (2, 2),
        (4, 1),
    ]
    assert [b.launches[0].block_offset for b in batches] == [0, 4, 8]
    assert [b.launches[0].num_blocks for b in batches] == [4, 4, 2]
    assert all(b.launches[0].skip_blocks == 0 for b in batches)


def test_skip_is_placed_in_the_batch_that_contains_it() -> None:
    """A prefix guard drops fully covered batches and trims the partial one."""
    # Three objects of eight tokens; guard the first twelve tokens.
    plans = build_group_transfer_plans(
        [_group()], [list(range(6))], skip_first_n_tokens=12
    )

    batches = plan_copy_batches(plans, max_objects_per_batch=1)

    # Object 0 is fully guarded, object 1 keeps its second block, object 2 is
    # copied whole.
    assert [(b.object_start, b.launches[0].skip_blocks) for b in batches] == [
        (1, 1),
        (2, 0),
    ]


def test_unavailable_objects_drop_only_their_batch() -> None:
    """A batch containing an object the path cannot copy is skipped."""
    plans = build_group_transfer_plans([_group()], [list(range(8))])

    batches = plan_copy_batches(
        plans,
        max_objects_per_batch=2,
        available_objects=[True, False, True, True],
    )

    assert [b.object_start for b in batches] == [2]


def test_groups_of_one_object_group_share_the_batch_split() -> None:
    """Every group of a batch is launched with its own block geometry."""
    plans = build_group_transfer_plans(
        [
            _group(0),
            _group(1, tokens_per_block=2, blocks_per_chunk=4, copy_blocks_per_chunk=2),
        ],
        [list(range(4)), list(range(10, 18))],
    )

    batches = plan_copy_batches(plans, max_objects_per_batch=2)

    assert len(batches) == 1
    assert [(lch.group_index, lch.num_blocks) for lch in batches[0].launches] == [
        (0, 4),
        (1, 4),
    ]


def test_plan_rejects_invalid_inputs() -> None:
    """Bad batch sizes, mismatched geometry, and wrong flag counts are errors."""
    plans = build_group_transfer_plans([_group()], [list(range(4))])

    with pytest.raises(ValueError, match="at least one"):
        plan_copy_batches(plans, max_objects_per_batch=0)
    with pytest.raises(ValueError, match="available_objects has"):
        plan_copy_batches(
            plans, max_objects_per_batch=1, available_objects=[True, True, True]
        )
    assert plan_copy_batches([], max_objects_per_batch=1) == []

    mismatched = build_group_transfer_plans([_group(0)], [list(range(4))])
    mismatched += build_group_transfer_plans([_group(1)], [list(range(8))])
    with pytest.raises(ValueError, match="must agree on object geometry"):
        plan_copy_batches(mismatched, max_objects_per_batch=1)


def test_execute_orders_staging_launches_and_flush() -> None:
    """Staging happens on the object side the direction requires."""
    plans = build_group_transfer_plans([_group()], [list(range(4))])
    batches = plan_copy_batches(plans, max_objects_per_batch=1)

    store = _RecordingEndpoint()
    execute_copy_batches(batches, store, direction=TransferDirection.D2H)
    assert store.calls == [
        ("launch", 0, 0, 0, 2, 0),
        ("from_device", 0, 1),
        ("end", 0),
        ("launch", 1, 0, 2, 2, 0),
        ("from_device", 1, 1),
        ("end", 1),
        ("flush",),
    ]

    retrieve = _RecordingEndpoint()
    execute_copy_batches(batches, retrieve, direction=TransferDirection.H2D)
    assert [call[0] for call in retrieve.calls] == [
        "to_device",
        "launch",
        "end",
        "to_device",
        "launch",
        "end",
        "flush",
    ]


def test_execute_flushes_even_without_batches() -> None:
    """Endpoints that defer work always get their single issue point."""
    endpoint = _RecordingEndpoint()

    execute_copy_batches([], endpoint, direction=TransferDirection.D2H)

    assert endpoint.calls == [("flush",)]


def test_execute_rejects_an_unsupported_direction() -> None:
    """Only H2D and D2H are meaningful for a KV transfer."""
    with pytest.raises(ValueError, match="Unsupported transfer direction"):
        execute_copy_batches(
            [], _RecordingEndpoint(), direction=cast(TransferDirection, 7)
        )
