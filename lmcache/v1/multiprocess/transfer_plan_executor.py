# SPDX-License-Identifier: Apache-2.0
"""Shared traversal executor for transfer plans.

This module provides a single function,
:func:`execute_transfer_plan_copy`, that owns the canonical nested
iteration order over an :class:`~.transfer_plan.ObjectGroupTransferPlan`:

.. code-block:: text

    for batch in object_group_plan.batches:
        before_object_batch(...)
        for kg_batch in batch.kernel_groups:
            kg_plan = kernel_group_plan_by_id[kg_batch.kernel_group_id]
            copy_kernel_group_batch(...)
        after_object_batch(...)

The function itself is transport-free and CUDA-free.  It does not know
about:

- ``cache_context``
- ``MemoryObj``
- pickle / SHM
- ``reserve_write`` / ``read_prefetched_results``
- ``finish_write`` / ``finish_read_prefetched``
- CUDA events / streams
- object key resolution

It only iterates the plan and dispatches callback invocations.
"""

# Standard
from typing import Callable

# First Party
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupBatchTransferPlan,
    KernelGroupTransferPlan,
    ObjectBatchTransferPlan,
    ObjectGroupTransferPlan,
    TransferDirection,
)


def execute_transfer_plan_copy(
    object_group_plan: ObjectGroupTransferPlan,
    direction: TransferDirection,
    *,
    before_object_batch: Callable[
        [ObjectGroupTransferPlan, ObjectBatchTransferPlan, TransferDirection],
        None,
    ]
    | None = None,
    copy_kernel_group_batch: Callable[
        [
            ObjectGroupTransferPlan,
            ObjectBatchTransferPlan,
            KernelGroupTransferPlan,
            KernelGroupBatchTransferPlan,
            TransferDirection,
        ],
        None,
    ],
    after_object_batch: Callable[
        [ObjectGroupTransferPlan, ObjectBatchTransferPlan, TransferDirection],
        None,
    ]
    | None = None,
) -> None:
    """Execute the copy traversal for one object group's transfer plan.

    Iterates the pre-computed batches in ``object_group_plan`` and
    dispatches caller-supplied callbacks at each stage.  The traversal
    order is:

    1. ``before_object_batch`` — once per batch, before kernel-group
       copies.
    2. ``copy_kernel_group_batch`` — once per kernel group within the
       batch.
    3. ``after_object_batch`` — once per batch, after all kernel-group
       copies.

    This function owns only the traversal logic.  All transport-specific
    or CUDA-specific work is delegated to the callbacks.

    Args:
        object_group_plan: The pre-computed plan for one object group,
            including batch-level geometry and kernel-group metadata.
        direction: The transfer direction (``STORE`` or ``RETRIEVE``).
        before_object_batch: Optional callback invoked before processing
            each batch.  Receives the object-group plan, the current
            batch plan, and the direction.
        copy_kernel_group_batch: Required callback invoked for each
            kernel-group batch within each object batch.  Receives the
            object-group plan, the current batch plan, the kernel-group
            plan (resolved by ID), the kernel-group batch plan, and the
            direction.
        after_object_batch: Optional callback invoked after processing
            each batch.  Receives the object-group plan, the current
            batch plan, and the direction.

    Raises:
        KeyError: If a ``KernelGroupBatchTransferPlan.kernel_group_id``
            in a batch does not match any entry in
            ``object_group_plan.kernel_groups``.
    """
    # Build lookup for kernel-group plans by ID.
    kg_plan_by_id: dict[int, KernelGroupTransferPlan] = {
        kgp.kernel_group_id: kgp for kgp in object_group_plan.kernel_groups
    }

    for batch in object_group_plan.batches:
        if before_object_batch is not None:
            before_object_batch(object_group_plan, batch, direction)

        for kg_batch in batch.kernel_groups:
            kg_plan = kg_plan_by_id[kg_batch.kernel_group_id]
            copy_kernel_group_batch(
                object_group_plan,
                batch,
                kg_plan,
                kg_batch,
                direction,
            )

        if after_object_batch is not None:
            after_object_batch(object_group_plan, batch, direction)
