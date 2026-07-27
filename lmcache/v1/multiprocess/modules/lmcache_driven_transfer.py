# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from typing import Sequence, cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    build_staging_copies,
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.transfer_context.common_copy import (
    GroupTransferPlan,
    RegisteredGroup,
    build_group_transfer_plans,
    registered_groups_from_kv_layer_groups,
)
from lmcache.v1.multiprocess.transfer_context.common_exec import (
    CopyBatch,
    CopyEndpoint,
    GroupLaunch,
    execute_copy_batches,
    plan_copy_batches,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = hasattr(
    lmc_ops, "execute_object_group_transfer"
)


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


class _ObjectGroupEndpoint(CopyEndpoint):
    """Shared state of the two LMCache-driven object-group endpoints.

    Both endpoints copy one storage object group of one request: the same
    ``MemoryObj`` list, the same staged block IDs, and the same per-kernel-group
    device buffers. They differ only in how the copy is issued -- immediately on
    the stream, or as one recorded native plan.

    Args:
        cache_context: Cache context owning the KV pointers and staging buffers.
        plans: Common plans of the kernel groups in this object group.
        block_ids_gpu: Staged block IDs, indexed by kernel group ID. Each entry
            holds exactly the blocks of its plan's ``selected_block_ids``.
        memory_objs: Storage objects of this object group, indexed by *logical*
            object (Sliding Window skipped objects included).
        object_group_id: Index of the object group being copied.
    """

    def __init__(
        self,
        cache_context: BaseCacheContext,
        plans: Sequence[GroupTransferPlan],
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        object_group_id: int,
    ) -> None:
        self._cache_context = cache_context
        self._plans = plans
        self._block_ids_gpu = block_ids_gpu
        self._memory_objs = memory_objs
        self._object_group_id = object_group_id
        self._first_object = plans[0].first_object

    def batch_memory_objs(self, batch: CopyBatch) -> list[MemoryObj]:
        """Return the storage objects of one batch.

        Args:
            batch: Batch whose objects to resolve.

        Returns:
            One ``MemoryObj`` per object of the batch, in copy order.

        Raises:
            ValueError: If an object is missing. The planner drops batches with
                unavailable objects, so this indicates a planning error.
        """
        start = self._first_object + batch.object_start
        objs = self._memory_objs[start : start + batch.num_objects]
        if any(memory_obj is None for memory_obj in objs):
            raise ValueError(
                f"object group {self._object_group_id} batch at object {start} "
                "contains an unreserved object; it should not have been planned"
            )
        return cast(list[MemoryObj], list(objs))


class _StreamObjectGroupEndpoint(_ObjectGroupEndpoint):
    """Issue every staging copy and kernel launch immediately on the stream.

    Args:
        cache_context: Cache context owning the KV pointers and staging buffers.
        plans: Common plans of the kernel groups in this object group.
        block_ids_gpu: Staged block IDs, indexed by kernel group ID.
        memory_objs: Storage objects of this object group.
        object_group_id: Index of the object group being copied.
        direction: H2D (retrieve) or D2H (store).
    """

    def __init__(
        self,
        cache_context: BaseCacheContext,
        plans: Sequence[GroupTransferPlan],
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        object_group_id: int,
        direction: "lmc_ops.TransferDirection",
    ) -> None:
        super().__init__(
            cache_context, plans, block_ids_gpu, memory_objs, object_group_id
        )
        self._direction = direction

    def stage_objects_to_device(self, batch: CopyBatch) -> None:
        """Copy the batch's objects into the object-group staging buffers."""
        for slot, memory_obj in enumerate(self.batch_memory_objs(batch)):
            lmcache_memcpy_async_h2d(
                memory_obj,
                self._cache_context.get_temp_object_group_buffer(
                    slot, self._object_group_id
                ),
            )

    def stage_objects_from_device(self, batch: CopyBatch) -> None:
        """Copy the object-group staging buffers into the batch's objects."""
        for slot, memory_obj in enumerate(self.batch_memory_objs(batch)):
            lmcache_memcpy_async_d2h(
                self._cache_context.get_temp_object_group_buffer(
                    slot, self._object_group_id
                ),
                memory_obj,
            )

    def launch_group_copy(self, batch: CopyBatch, launch: GroupLaunch) -> None:
        """Launch one kernel group's paged copy for one batch."""
        kernel_group_id = self._plans[launch.group_index].group.kernel_group_id
        lmc_ops.multi_layer_block_kv_transfer(
            self._cache_context.get_kernel_group_kv_pointers(kernel_group_id),
            [
                self._cache_context.get_temp_kernel_group_buffer(
                    slot, kernel_group_id
                ).data_ptr()
                for slot in range(launch.num_objects)
            ],
            self._block_ids_gpu[kernel_group_id][
                launch.block_offset : launch.block_offset + launch.num_blocks
            ],
            self._cache_context.device,
            self._direction,
            self._cache_context.get_shape_desc(kernel_group_id),
            self._cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
            self._cache_context.get_engine_kv_format(kernel_group_id),
            launch.skip_blocks,
        )


class _NativeObjectGroupEndpoint(_ObjectGroupEndpoint):
    """Record the object group's transfer and issue it in one native call.

    Instead of issuing each staging copy and kernel launch immediately (each a
    GIL release/re-acquire), every argument is resolved to plain
    pointers/scalars (GIL held throughout) and the whole plan is handed to
    ``execute_object_group_transfer``, which issues all of it on the stream
    within a single GIL release.

    Requires every object to be non-GDS (staged through the lazy-allocator
    path); the caller skips groups that contain any GDS-backed object.

    Args:
        cache_context: Cache context owning the KV pointers and staging buffers.
        plans: Common plans of the kernel groups in this object group.
        block_ids_gpu: Staged block IDs, indexed by kernel group ID.
        memory_objs: Storage objects of this object group.
        object_group_id: Index of the object group being copied.
        direction: H2D (retrieve) or D2H (store).
    """

    def __init__(
        self,
        cache_context: BaseCacheContext,
        plans: Sequence[GroupTransferPlan],
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        object_group_id: int,
        direction: "lmc_ops.TransferDirection",
    ) -> None:
        super().__init__(
            cache_context, plans, block_ids_gpu, memory_objs, object_group_id
        )
        self._direction = direction
        self._is_h2d = int(direction) == int(lmc_ops.TransferDirection.H2D)
        max_batch_size = cache_context.max_batch_size

        # Per-kernel-group invariants, resolved once for the whole request.
        self._kernel_group_specs: list["lmc_ops.KernelGroupSpec"] = []
        for plan in plans:
            kernel_group_id = plan.group.kernel_group_id
            block_ids_tensor = block_ids_gpu[kernel_group_id]
            self._kernel_group_specs.append(
                lmc_ops.KernelGroupSpec(
                    cache_context.get_kernel_group_kv_pointers(
                        kernel_group_id
                    ).data_ptr(),
                    [
                        cache_context.get_temp_kernel_group_buffer(
                            slot, kernel_group_id
                        ).data_ptr()
                        for slot in range(max_batch_size)
                    ],
                    cache_context.get_shape_desc(kernel_group_id),
                    cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                    cache_context.get_engine_kv_format(kernel_group_id),
                    block_ids_tensor.data_ptr(),
                    block_ids_tensor.numel(),
                )
            )

        # Temp object-group staging buffers, reused per batch slot.
        self._object_group_buffers = [
            cache_context.get_temp_object_group_buffer(slot, object_group_id)
            for slot in range(max_batch_size)
        ]
        self._staging: list = []
        self._launches: list["lmc_ops.LaunchVar"] = []
        self._batch_steps: list["lmc_ops.BatchStep"] = []

    def stage_objects_to_device(self, batch: CopyBatch) -> None:
        """Record the batch's H2D staging copies."""
        self._record_staging(batch)

    def stage_objects_from_device(self, batch: CopyBatch) -> None:
        """Record the batch's D2H staging copies."""
        self._record_staging(batch)

    def launch_group_copy(self, batch: CopyBatch, launch: GroupLaunch) -> None:
        """Record one kernel group's launch variables for one batch."""
        self._launches.append(
            lmc_ops.LaunchVar(
                launch.group_index,
                launch.block_offset,
                launch.num_blocks,
                launch.num_objects,
                launch.skip_blocks,
            )
        )

    def end_batch(self, batch: CopyBatch) -> None:
        """Close the recorded batch step."""
        self._batch_steps.append(lmc_ops.BatchStep(self._staging, self._launches))
        self._staging = []
        self._launches = []

    def flush(self) -> None:
        """Issue the whole recorded object-group transfer in one native call."""
        if not self._batch_steps:
            return
        lmc_ops.execute_object_group_transfer(
            self._direction,
            self._cache_context.device,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
            self._kernel_group_specs,
            self._batch_steps,
        )

    def _record_staging(self, batch: CopyBatch) -> None:
        """Resolve the batch's staging copies to pointer/size descriptors."""
        self._staging = build_staging_copies(
            self.batch_memory_objs(batch),
            self._object_group_buffers[: batch.num_objects],
            self._is_h2d,
        )


def build_registered_groups(cache_context: BaseCacheContext) -> list[RegisteredGroup]:
    """Adapt the server's KV layer groups to the common registered-group model.

    Args:
        cache_context: The cache context owning the KV layer groups manager and
            the authoritative LMCache chunk size.

    Returns:
        One :class:`RegisteredGroup` per kernel group, in kernel-group order.
    """
    return registered_groups_from_kv_layer_groups(
        cache_context.kv_layer_groups_manager,
        cache_context.lmcache_tokens_per_chunk,
    )


def plan_and_stage_block_ids(
    cache_context: BaseCacheContext,
    block_ids: list[list[int]],
    *,
    for_retrieve: bool = False,
    skip_first_n_tokens: int = 0,
) -> tuple[list[GroupTransferPlan], list[torch.Tensor]]:
    """Plan the request with the common planner and stage its block IDs.

    The plan drops the blocks a group does not need -- the per-chunk blocks
    outside a Sliding Window sub-chunk, and, for retrieve, the leading objects
    that fell out of the window entirely -- so only the blocks that are actually
    copied are staged to the device.

    Args:
        cache_context: The cache context containing the KV cache information.
        block_ids: The original block id lists, indexed by LMCache KV group
            index. Mutated in place to the selected block IDs.
        for_retrieve: Apply Sliding Window object-tail selection.
        skip_first_n_tokens: Logical tokens at the head of the range that must
            not be overwritten (APC-shared blocks).

    Returns:
        The per-kernel-group plans, and the staged (device-side) block ID
        tensors indexed by LMCache KV group index. Entry ``i`` of the tensors
        holds exactly the blocks of plan ``i``.

    Raises:
        ValueError: If the block-ID lists do not exactly cover the same logical
            chunk count for every kernel group.

    Example:
        If a model have 2 kernel groups, one is full attention with block size
        32, one is swa attention with block size 32 and sliding window size 64,
        and LMCache has a chunk size of 128. And there are 2 chunks in total
        (256 tokens).

        The input will be:
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # block ids for the full attention group
          [11, 12, 13, 14, 15, 16, 17, 18], # block ids for the swa attention group
        ]

        The staged output will be
        [
          [1, 2, 3, 4, 5, 6, 7, 8],  # full attention group still needs all block ids
          [13, 14, 17, 18], # swa attention group only needs the last 2 block per chunk
        ]
    """
    plans = build_group_transfer_plans(
        build_registered_groups(cache_context),
        block_ids,
        for_retrieve=for_retrieve,
        skip_first_n_tokens=skip_first_n_tokens,
    )
    for plan in plans:
        block_ids[plan.group.kernel_group_id] = list(plan.selected_block_ids)

    # Stage the cut block ids into GPU tensors
    return plans, cache_context.stage_block_ids(block_ids)


def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    plans: Sequence[GroupTransferPlan],
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    direction: "lmc_ops.TransferDirection",
) -> None:
    """Transfer one object group's memory objects to/from the device.

    Batching, APC skip placement, and per-launch block ranges come from the
    common launch planner, so this path issues exactly the copies the
    Engine-driven worker path would issue for the same request.

    Args:
        cache_context: The GPU cache context containing the KV cache information.
        plans: The request's per-kernel-group plans, as returned by
            :func:`plan_and_stage_block_ids`.
        block_ids_gpu: Staged block IDs, indexed by LMCache KV group index.
        memory_objs: The storage objects of this object group, one per logical
            object of the request. It could be None when allocation or
            retrieval fails. For store (D2H), those objects are skipped. For
            retrieve (H2D), a None entry is an error.
        object_group_id: Index of the object group being copied.
        batch_size: The number of memory objects to perform batched copy.
        direction: The transfer direction, H2D (retrieve) or D2H (store).

    Raises:
        ValueError: If no plan covers ``object_group_id``, if ``memory_objs``
            does not match the object count the request block IDs cover, or if
            a None entry is found in ``memory_objs`` when direction is H2D.
    """
    group_plans = [
        plan for plan in plans if plan.group.object_group_id == object_group_id
    ]
    if not group_plans:
        raise ValueError(
            f"No transfer group is stored in object group {object_group_id}"
        )

    reference = group_plans[0]
    if len(memory_objs) != reference.total_objects:
        raise ValueError(
            f"object group {object_group_id} has {len(memory_objs)} objects but "
            f"the request block IDs cover {reference.total_objects}"
        )
    if reference.first_object:
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            reference.first_object,
        )

    transferred = memory_objs[reference.first_object :]
    is_h2d = int(direction) == int(lmc_ops.TransferDirection.H2D)
    if is_h2d and any(memory_obj is None for memory_obj in transferred):
        raise ValueError(
            "MemoryObj is None for some objects in object group "
            f"{object_group_id}, cannot perform H2D copy"
        )

    batches = plan_copy_batches(
        group_plans,
        max_objects_per_batch=batch_size,
        available_objects=[memory_obj is not None for memory_obj in transferred],
    )
    if not batches:
        return

    endpoint: _ObjectGroupEndpoint
    if _HAS_NATIVE_OBJECT_GROUP_TRANSFER and not any(
        isinstance(memory_obj, GDSMemoryObject) for memory_obj in transferred
    ):
        endpoint = _NativeObjectGroupEndpoint(
            cache_context,
            group_plans,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            direction,
        )
    else:
        endpoint = _StreamObjectGroupEndpoint(
            cache_context,
            group_plans,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            direction,
        )
    execute_copy_batches(batches, endpoint, direction=direction)


@dataclass
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._cache_contexts.pop(iid)))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            reaped.clear()
            self._release_entries(entries)
        return reaped_ids

    def _release_entries(self, entries: list[ContextEntry]) -> None:
        """Release a batch of entries and reclaim their device memory.

        Args:
            entries: The only remaining references to the released entries.
                The list is cleared before memory is reclaimed.
        """
        if not entries:
            return
        for entry in entries:
            entry.cache_context.close()
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
        del entry
        entries.clear()
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE,
                self.register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.STORE,
                self.store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.RETRIEVE,
                self.retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        with self._lock:
            entries = list(self._cache_contexts.values())
            self._cache_contexts.clear()
        self._release_entries(entries)

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
        )
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        kv_groups_manager = cache_context.kv_layer_groups_manager
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name, world_size, layout_desc, attn_desc
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # _release_entries' reclaim actually unmaps the IPC segments.
        self._release_entries(popped)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes).

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), or a copy fails, the whole store is skipped and nothing is
            committed (logged at WARNING); a subsequent retrieve simply misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            plans, block_ids_per_group_gpu = plan_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        obj_keys, layout_desc, "new"
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (skipped by the storage manager)
                    # become None entries; the helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    transfer_kv_per_object_group(
                        cache_context,
                        plans,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        direction=lmc_ops.TransferDirection.D2H,
                    )

                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        list(all_dict.keys()),
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
        )

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()

        entry = self.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            plans, block_ids_per_group_gpu = plan_and_stage_block_ids(
                cache_context,
                gpu_block_ids,
                for_retrieve=True,
                skip_first_n_tokens=skip_first_n_tokens,
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            event_backend.wait_event(producer_event, cache_context.stream)

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    with self._ctx.storage_manager.read_prefetched_results(
                        obj_keys
                    ) as memory_objs:
                        if not memory_objs or len(memory_objs) != len(obj_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in memory_objs)

                        transfer_kv_per_object_group(
                            cache_context,
                            plans,
                            block_ids_per_group_gpu,
                            memory_objs,
                            object_group_id=obj_group_id,
                            batch_size=cache_context.max_batch_size,
                            direction=lmc_ops.TransferDirection.H2D,
                        )
                        # Extend only after the copy is enqueued: on exception,
                        # read_prefetched_results releases this group's locks
                        # itself, and a key must not be released twice.
                        prefetched_keys.extend(obj_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == num_chunks * num_object_groups
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )
