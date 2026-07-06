# SPDX-License-Identifier: Apache-2.0
"""Unit tests for execute_transfer_plan_copy traversal function.

These tests verify that the shared executor function in
``lmcache.v1.multiprocess.transfer_plan_executor`` correctly iterates
batches, resolves kernel-group plans by ID, and dispatches callbacks in
the expected order — all without requiring GPU, CUDA extensions, or any
multiprocess server infrastructure.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupBatchTransferPlan,
    KernelGroupTransferPlan,
    ObjectBatchTransferPlan,
    ObjectGroupTransferPlan,
    TransferDirection,
)
from lmcache.v1.multiprocess.transfer_plan_executor import (
    execute_transfer_plan_copy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_object_key(idx: int) -> ObjectKey:
    """Create a minimal mock ObjectKey for testing."""
    key = MagicMock(spec=ObjectKey)
    key.chunk_idx = idx
    return key


def _make_kg_plan(kg_id: int, og_id: int = 0) -> KernelGroupTransferPlan:
    """Create a minimal KernelGroupTransferPlan for testing."""
    return KernelGroupTransferPlan(
        kernel_group_id=kg_id,
        object_group_id=og_id,
        blocks_per_chunk=4,
        blocks_per_window=4,
        selected_block_ids=list(range(16)),
        slots_per_chunk=64,
        shape_desc=MagicMock(),
        dtype=MagicMock(),
        engine_kv_format=MagicMock(),
    )


def _make_kg_batch(
    kg_id: int, start: int = 0, count: int = 4, skip: int = 0
) -> KernelGroupBatchTransferPlan:
    """Create a KernelGroupBatchTransferPlan for testing."""
    return KernelGroupBatchTransferPlan(
        kernel_group_id=kg_id,
        start_block_pos=start,
        block_count=count,
        skip_blocks=skip,
    )


def _make_object_group_plan(
    kg_ids: list[int],
    num_batches: int = 2,
    og_id: int = 0,
) -> ObjectGroupTransferPlan:
    """Create an ObjectGroupTransferPlan with the given kernel groups."""
    kernel_groups = [_make_kg_plan(kg_id, og_id) for kg_id in kg_ids]
    batches = []
    for batch_idx in range(num_batches):
        kg_batches = [
            _make_kg_batch(kg_id, start=batch_idx * 4, count=4) for kg_id in kg_ids
        ]
        batches.append(
            ObjectBatchTransferPlan(
                start_object_idx=batch_idx,
                batch_len=1,
                kernel_groups=kg_batches,
            )
        )
    return ObjectGroupTransferPlan(
        object_group_id=og_id,
        object_keys=[_make_object_key(i) for i in range(num_batches)],
        layout_desc=MemoryLayoutDesc(shapes=[], dtypes=[]),
        kernel_groups=kernel_groups,
        num_chunks=num_batches,
        num_objects_to_skip=0,
        batches=batches,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteTransferPlanCopy:
    """Tests for execute_transfer_plan_copy."""

    def test_callback_order_two_batches_two_kernel_groups(self) -> None:
        """Callbacks are called in expected order: before/copy/copy/after."""
        og_plan = _make_object_group_plan(kg_ids=[0, 1], num_batches=2)
        call_log: list[str] = []

        def before(og, batch, direction):
            call_log.append(f"before batch {batch.start_object_idx}")

        def copy_kg(og, batch, kg_plan, kg_batch, direction):
            call_log.append(
                f"copy kg {kg_batch.kernel_group_id} batch {batch.start_object_idx}"
            )

        def after(og, batch, direction):
            call_log.append(f"after batch {batch.start_object_idx}")

        execute_transfer_plan_copy(
            og_plan,
            TransferDirection.STORE,
            before_object_batch=before,
            copy_kernel_group_batch=copy_kg,
            after_object_batch=after,
        )

        expected = [
            "before batch 0",
            "copy kg 0 batch 0",
            "copy kg 1 batch 0",
            "after batch 0",
            "before batch 1",
            "copy kg 0 batch 1",
            "copy kg 1 batch 1",
            "after batch 1",
        ]
        assert call_log == expected

    def test_resolves_kernel_group_plan_by_id_not_position(self) -> None:
        """KernelGroupTransferPlan is resolved by kernel_group_id, not index."""
        # Kernel groups declared in non-sequential order: [2, 5]
        kg_ids = [2, 5]
        kernel_groups = [_make_kg_plan(kg_id) for kg_id in kg_ids]
        kg_batches = [_make_kg_batch(5, start=0, count=4)]
        batches = [
            ObjectBatchTransferPlan(
                start_object_idx=0,
                batch_len=1,
                kernel_groups=kg_batches,
            )
        ]
        og_plan = ObjectGroupTransferPlan(
            object_group_id=0,
            object_keys=[_make_object_key(0)],
            layout_desc=MemoryLayoutDesc(shapes=[], dtypes=[]),
            kernel_groups=kernel_groups,
            num_chunks=1,
            num_objects_to_skip=0,
            batches=batches,
        )

        received_kg_plans: list[KernelGroupTransferPlan] = []

        def copy_kg(og, batch, kg_plan, kg_batch, direction):
            received_kg_plans.append(kg_plan)

        execute_transfer_plan_copy(
            og_plan,
            TransferDirection.RETRIEVE,
            copy_kernel_group_batch=copy_kg,
        )

        assert len(received_kg_plans) == 1
        assert received_kg_plans[0].kernel_group_id == 5

    def test_before_and_after_are_optional(self) -> None:
        """Executor works without before_object_batch and after_object_batch."""
        og_plan = _make_object_group_plan(kg_ids=[0], num_batches=1)
        copy_count = []

        def copy_kg(og, batch, kg_plan, kg_batch, direction):
            copy_count.append(1)

        # Should not raise
        execute_transfer_plan_copy(
            og_plan,
            TransferDirection.STORE,
            copy_kernel_group_batch=copy_kg,
        )

        assert len(copy_count) == 1

    def test_passes_same_plan_objects_to_callbacks(self) -> None:
        """Callbacks receive the exact same plan objects (identity check)."""
        og_plan = _make_object_group_plan(kg_ids=[0, 1], num_batches=1)

        received_og_plans: list[ObjectGroupTransferPlan] = []
        received_batch_plans: list[ObjectBatchTransferPlan] = []
        received_directions: list[TransferDirection] = []

        def before(og, batch, direction):
            received_og_plans.append(og)
            received_batch_plans.append(batch)
            received_directions.append(direction)

        def copy_kg(og, batch, kg_plan, kg_batch, direction):
            received_og_plans.append(og)
            received_batch_plans.append(batch)
            received_directions.append(direction)

        def after(og, batch, direction):
            received_og_plans.append(og)
            received_batch_plans.append(batch)
            received_directions.append(direction)

        direction = TransferDirection.RETRIEVE
        execute_transfer_plan_copy(
            og_plan,
            direction,
            before_object_batch=before,
            copy_kernel_group_batch=copy_kg,
            after_object_batch=after,
        )

        # before + 2 copy_kg + after = 4 calls
        assert all(og is og_plan for og in received_og_plans)
        assert all(b is og_plan.batches[0] for b in received_batch_plans)
        assert all(d is direction for d in received_directions)

    def test_empty_batches_no_callbacks(self) -> None:
        """No callbacks called when object_group_plan has empty batches."""
        og_plan = ObjectGroupTransferPlan(
            object_group_id=0,
            object_keys=[],
            layout_desc=MemoryLayoutDesc(shapes=[], dtypes=[]),
            kernel_groups=[_make_kg_plan(0)],
            num_chunks=0,
            num_objects_to_skip=0,
            batches=[],
        )

        call_log: list[str] = []

        execute_transfer_plan_copy(
            og_plan,
            TransferDirection.STORE,
            before_object_batch=lambda *a: call_log.append("before"),
            copy_kernel_group_batch=lambda *a: call_log.append("copy"),
            after_object_batch=lambda *a: call_log.append("after"),
        )

        assert call_log == []

    def test_raises_key_error_for_unknown_kernel_group_id(self) -> None:
        """Raises KeyError if kg_batch references a non-existent kg plan."""
        kernel_groups = [_make_kg_plan(0)]
        # Batch references kernel group ID 99 which doesn't exist
        kg_batches = [_make_kg_batch(99, start=0, count=4)]
        batches = [
            ObjectBatchTransferPlan(
                start_object_idx=0,
                batch_len=1,
                kernel_groups=kg_batches,
            )
        ]
        og_plan = ObjectGroupTransferPlan(
            object_group_id=0,
            object_keys=[_make_object_key(0)],
            layout_desc=MemoryLayoutDesc(shapes=[], dtypes=[]),
            kernel_groups=kernel_groups,
            num_chunks=1,
            num_objects_to_skip=0,
            batches=batches,
        )

        with pytest.raises(KeyError):
            execute_transfer_plan_copy(
                og_plan,
                TransferDirection.STORE,
                copy_kernel_group_batch=lambda *a: None,
            )

    def test_direction_passed_through(self) -> None:
        """Direction argument is passed correctly to all callbacks."""
        og_plan = _make_object_group_plan(kg_ids=[0], num_batches=1)
        directions_seen: list[TransferDirection] = []

        def before(og, batch, direction):
            directions_seen.append(direction)

        def copy_kg(og, batch, kg_plan, kg_batch, direction):
            directions_seen.append(direction)

        def after(og, batch, direction):
            directions_seen.append(direction)

        execute_transfer_plan_copy(
            og_plan,
            TransferDirection.RETRIEVE,
            before_object_batch=before,
            copy_kernel_group_batch=copy_kg,
            after_object_batch=after,
        )

        assert all(d == TransferDirection.RETRIEVE for d in directions_seen)
        assert len(directions_seen) == 3  # before + copy + after
