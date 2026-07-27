# SPDX-License-Identifier: Apache-2.0
"""Path-neutral launch planning and execution for KV-cache transfers.

:mod:`.common_copy` decides *which* blocks and objects move. This module
decides *how* that movement is issued -- how objects are batched, which
launch covers which blocks, how a token-level APC skip becomes a per-launch
block skip, and in which order staging and launches happen -- and then runs
that plan through a path-supplied :class:`CopyEndpoint`.

Both multiprocess transfer paths use it::

    plans   = build_group_transfer_plans(...)        # common_copy
    batches = plan_copy_batches(plans, ...)          # here
    execute_copy_batches(batches, endpoint, ...)     # here

The endpoint is the only path-specific part. It owns the transport-level
concerns this module deliberately excludes: which buffers back the objects,
how they are staged to and from the device, which native primitive performs
the copy, and when the caller synchronizes or commits::

    LMCache-driven server   endpoint over IPC-opened paged KV + ``MemoryObj``
                            staging, issuing ``multi_layer_block_kv_transfer``
                            or accumulating one ``execute_object_group_transfer``
    Engine-driven worker    endpoint over worker-local KV tensors + SHM/pickle
                            chunk tensors, issuing
                            ``multi_layer_block_kv_transfer``

Batching contract
-----------------
Objects are indexed **relative to the first transferred object**, i.e. after
Sliding Window object-tail selection has already removed the leading objects
(``GroupTransferPlan.first_object``). ``GroupLaunch`` block offsets are
likewise relative to ``GroupTransferPlan.selected_block_ids``. An endpoint
that keeps untrimmed per-object state (the server's ``MemoryObj`` list) adds
``first_object`` itself.

A batch is dropped entirely when the APC skip already covers all of its
tokens, or when any of its objects is unavailable; the remaining partial
batch carries the leftover skip, converted per group through
:meth:`~.common_copy.RegisteredGroup.blocks_to_skip`.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.ops_types import TransferDirection

if TYPE_CHECKING:
    # Local
    from .common_copy import GroupTransferPlan


@dataclass(frozen=True)
class GroupLaunch:
    """One native copy launch for one group inside one batch.

    Attributes:
        group_index: Index into the plan sequence the batch was built from.
        block_offset: Offset into ``GroupTransferPlan.selected_block_ids``.
        num_blocks: Blocks covered by this launch.
        num_objects: Objects covered by this launch; equals the batch size.
        skip_blocks: Leading blocks of this launch left untouched by the APC
            prefix guard.
    """

    group_index: int
    block_offset: int
    num_blocks: int
    num_objects: int
    skip_blocks: int


@dataclass(frozen=True)
class CopyBatch:
    """One staging step and its group launches.

    Attributes:
        object_start: Index of the first object, relative to the first
            transferred object.
        num_objects: Objects staged and copied by this batch.
        launches: One launch per planned group, in plan order.
    """

    object_start: int
    num_objects: int
    launches: tuple[GroupLaunch, ...]


class CopyEndpoint(ABC):
    """Path-specific staging and launch target of :func:`execute_copy_batches`.

    Implementations own buffers, pointers, and native invocation; they must
    not re-derive block selection, batching, or skip geometry.
    """

    @abstractmethod
    def stage_objects_to_device(self, batch: CopyBatch) -> None:
        """Stage the batch's objects into device-side buffers (H2D only).

        Args:
            batch: Batch about to be launched.
        """

    @abstractmethod
    def launch_group_copy(self, batch: CopyBatch, launch: GroupLaunch) -> None:
        """Issue (or record) one group's copy for one batch.

        Args:
            batch: Batch being copied.
            launch: Planned block range, object count, and block skip.
        """

    @abstractmethod
    def stage_objects_from_device(self, batch: CopyBatch) -> None:
        """Stage the batch's objects out of device-side buffers (D2H only).

        Args:
            batch: Batch that was just launched.
        """

    def end_batch(self, batch: CopyBatch) -> None:  # noqa: B027
        """Finish one batch. Default: nothing to do.

        Endpoints that accumulate a deferred plan use this to close the
        current step.

        Args:
            batch: Batch that was just staged and launched.
        """

    def flush(self) -> None:  # noqa: B027
        """Issue any work the endpoint deferred. Default: nothing to do.

        Called exactly once, after the last batch and also when there was no
        batch at all. Endpoints that record instead of issuing -- such as the
        server's single ``execute_object_group_transfer`` call -- override this
        as their one submission point; endpoints that issue every batch
        immediately keep the default.
        """


def plan_copy_batches(
    plans: Sequence[GroupTransferPlan],
    *,
    max_objects_per_batch: int,
    available_objects: Sequence[bool] | None = None,
) -> list[CopyBatch]:
    """Plan the launches of one storage object group.

    This is the single implementation of object batching, APC skip placement,
    and per-launch block-range selection. Both transfer paths call it.

    Args:
        plans: Plans of the groups sharing one storage object, in the order the
            endpoint indexes them. They must agree on object geometry, which
            they do by construction: groups of one object group share the
            objects they are stored in.
        max_objects_per_batch: Largest number of objects one batch may stage
            and copy at once. The path chooses it from its own limits (native
            kernel object limit, staging buffer count).
        available_objects: One flag per transferred object; ``False`` marks an
            object the path cannot copy (e.g. a storage reservation the manager
            skipped), and every batch containing it is dropped. ``None`` means
            all objects are available.

    Returns:
        Batches in copy order; empty when nothing is left to copy.

    Raises:
        ValueError: If ``max_objects_per_batch`` is not positive, the plans
            disagree on object geometry, or ``available_objects`` has the wrong
            length.

    Example:
        Eight objects with ``max_objects_per_batch=4`` and an APC skip of five
        objects' tokens yields one batch of objects 4..7 whose first group
        launch skips the blocks of object 4.
    """
    if max_objects_per_batch < 1:
        raise ValueError("max_objects_per_batch must be at least one")
    if not plans:
        return []

    reference = plans[0]
    transfer_objects = reference.transfer_objects
    chunk_tokens = reference.group.chunk_tokens
    logical_skip = reference.logical_skip
    for plan in plans[1:]:
        if (
            plan.transfer_objects != transfer_objects
            or plan.group.chunk_tokens != chunk_tokens
            or plan.logical_skip != logical_skip
        ):
            raise ValueError(
                "groups sharing one object group must agree on object "
                f"geometry: group {reference.group.kernel_group_id} has "
                f"(objects={transfer_objects}, chunk_tokens={chunk_tokens}, "
                f"skip={logical_skip}) but group {plan.group.kernel_group_id} "
                f"has (objects={plan.transfer_objects}, "
                f"chunk_tokens={plan.group.chunk_tokens}, "
                f"skip={plan.logical_skip})"
            )
    if available_objects is not None and len(available_objects) != transfer_objects:
        raise ValueError(
            f"available_objects has {len(available_objects)} entries but the "
            f"plans transfer {transfer_objects} objects"
        )

    batches: list[CopyBatch] = []
    for object_start in range(0, transfer_objects, max_objects_per_batch):
        num_objects = min(max_objects_per_batch, transfer_objects - object_start)
        if available_objects is not None and not all(
            available_objects[object_start : object_start + num_objects]
        ):
            continue

        batch_start_token = object_start * chunk_tokens
        if logical_skip >= batch_start_token + num_objects * chunk_tokens:
            continue
        skip_tokens_in_batch = max(0, logical_skip - batch_start_token)

        launches = tuple(
            GroupLaunch(
                group_index=group_index,
                block_offset=object_start * plan.group.copy_blocks_per_chunk,
                num_blocks=num_objects * plan.group.copy_blocks_per_chunk,
                num_objects=num_objects,
                skip_blocks=plan.group.blocks_to_skip(skip_tokens_in_batch),
            )
            for group_index, plan in enumerate(plans)
        )
        batches.append(
            CopyBatch(
                object_start=object_start,
                num_objects=num_objects,
                launches=launches,
            )
        )
    return batches


def execute_copy_batches(
    batches: Sequence[CopyBatch],
    endpoint: CopyEndpoint,
    *,
    direction: TransferDirection,
) -> None:
    """Run planned batches through an endpoint in the canonical order.

    Per batch: stage objects to the device (H2D), issue every group launch in
    plan order, stage objects back from the device (D2H), then close the
    batch. ``flush()`` runs once at the end, even when there is nothing to
    copy, so endpoints that defer work always get their single issue point.

    Args:
        batches: Batches from :func:`plan_copy_batches`.
        endpoint: Path-specific staging and launch target.
        direction: ``H2D`` (retrieve) or ``D2H`` (store). Compared by value so
            both the native and the fallback enum are accepted.

    Raises:
        ValueError: If ``direction`` is neither ``H2D`` nor ``D2H``.
    """
    direction_value = int(direction)
    if direction_value not in (int(TransferDirection.H2D), int(TransferDirection.D2H)):
        raise ValueError(f"Unsupported transfer direction: {direction!r}")
    is_h2d = direction_value == int(TransferDirection.H2D)

    for batch in batches:
        if is_h2d:
            endpoint.stage_objects_to_device(batch)
        for launch in batch.launches:
            endpoint.launch_group_copy(batch, launch)
        if not is_h2d:
            endpoint.stage_objects_from_device(batch)
        endpoint.end_batch(batch)
    endpoint.flush()
