# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_plan import compute_num_objects_to_skip
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


ResolveObjKeysFn = Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]]


class _PostPrefetchValidationError(Exception):
    """Signal invalid prefetched data so context cleanup releases held read locks.

    Raised when post-prefetch validation fails (e.g. count mismatch or missing
    tensor component). The exception path triggers ``read_prefetched_results``
    context cleanup, ensuring held read locks for the current group are released
    exactly once.

    This private control-flow exception is raised only inside
    ``_PickleLifecycleExecutor.retrieve`` and caught there immediately.
    It carries an optional descriptive message for logging/debugging.
    """


@dataclass(frozen=True)
class _ResolvedObjectGroup:
    """Object-group transfer inputs resolved for one request."""

    object_group_id: int
    obj_keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    sw_size_chunks: int


def _ordered_object_group_ids(context: EngineDrivenContextMetadata) -> list[int]:
    """Return deterministic object-group IDs for transfer execution."""
    transfer_metadata = context.transfer_metadata
    if transfer_metadata is None:
        return [0]
    return [group.object_group_id for group in transfer_metadata.object_groups]


def _is_legacy_single_group_context(context: EngineDrivenContextMetadata) -> bool:
    """Return whether transfer metadata still uses legacy single-group payload shape."""
    return context.transfer_metadata is None and not context.object_group_layout_descs


def _ordered_layout_descs(
    context: EngineDrivenContextMetadata,
    object_group_ids: list[int],
) -> list[MemoryLayoutDesc]:
    """Return per-object-group layouts in object-group ID order."""
    if context.object_group_layout_descs:
        return [
            context.object_group_layout_descs[group_id] for group_id in object_group_ids
        ]
    if object_group_ids == [0]:
        return [context.layout_desc]
    raise ValueError("missing object-group layout descriptors for multi-group transfer")


def _resolve_object_groups(
    key: IPCCacheServerKey,
    context: EngineDrivenContextMetadata,
    resolve_obj_keys: ResolveObjKeysFn,
) -> list[_ResolvedObjectGroup]:
    """Resolve object-group ordering, keys, layouts, and attention windows."""
    object_group_ids = _ordered_object_group_ids(context)
    obj_keys_by_group = resolve_obj_keys(key, object_group_ids)
    layout_descs = _ordered_layout_descs(context, object_group_ids)
    if len(obj_keys_by_group) != len(object_group_ids):
        raise ValueError("resolved object-group keys do not match object-group count")
    if len(layout_descs) != len(object_group_ids):
        raise ValueError("object-group layouts do not match object-group count")

    transfer_metadata = context.transfer_metadata
    sw_sizes_by_group_id: dict[int, int] = {}
    if transfer_metadata is None:
        sw_sizes_by_group_id[0] = -1
    else:
        sw_sizes_by_group_id = {
            object_group.object_group_id: object_group.sw_size_chunks
            for object_group in transfer_metadata.object_groups
        }

    return [
        _ResolvedObjectGroup(
            object_group_id=group_id,
            obj_keys=obj_keys,
            layout_desc=layout_desc,
            sw_size_chunks=sw_sizes_by_group_id[group_id],
        )
        for group_id, obj_keys, layout_desc in zip(
            object_group_ids, obj_keys_by_group, layout_descs, strict=True
        )
    ]


def _select_windowed_obj_keys_for_retrieve(
    group: _ResolvedObjectGroup,
) -> list[ObjectKey]:
    """Select retrieve keys for one object group using its sliding-window size."""
    num_objects_to_skip = compute_num_objects_to_skip(
        group.sw_size_chunks,
        len(group.obj_keys),
        True,
    )
    return group.obj_keys[num_objects_to_skip:]


def _normalize_store_payload(payload: list[Any]) -> list[list[torch.Tensor]] | None:
    """Normalize store payload into object-major list-of-list tensor form."""
    if all(isinstance(payload_object, torch.Tensor) for payload_object in payload):
        return [[payload_object] for payload_object in payload]
    if all(isinstance(payload_object, list) for payload_object in payload):
        normalized_payload: list[list[torch.Tensor]] = []
        for payload_object in payload:
            if not all(
                isinstance(component, torch.Tensor) for component in payload_object
            ):
                return None
            normalized_payload.append(payload_object)
        return normalized_payload
    return None


def _validate_store_payload(
    payload_objects: list[list[torch.Tensor]],
    resolved_groups: list[_ResolvedObjectGroup],
) -> bool:
    """Validate object count and per-component shape/dtype before reserving writes."""
    expected_num_objects = sum(len(group.obj_keys) for group in resolved_groups)
    if len(payload_objects) != expected_num_objects:
        return False

    payload_idx = 0
    for group in resolved_groups:
        for _ in group.obj_keys:
            payload_object = payload_objects[payload_idx]
            payload_idx += 1
            if len(payload_object) != len(group.layout_desc.shapes):
                return False
            for component, expected_shape, expected_dtype in zip(
                payload_object,
                group.layout_desc.shapes,
                group.layout_desc.dtypes,
                strict=True,
            ):
                if component.shape != expected_shape:
                    return False
                if component.dtype != expected_dtype:
                    return False
    return payload_idx == len(payload_objects)


def _memory_tensor(memory_obj: Any, tensor_idx: int) -> torch.Tensor | None:
    """Return tensor component from a reserved memory object."""
    get_tensor = getattr(memory_obj, "get_tensor", None)
    if callable(get_tensor):
        tensor = get_tensor(tensor_idx)
        if isinstance(tensor, torch.Tensor):
            return tensor
    if tensor_idx == 0:
        tensor = getattr(memory_obj, "tensor", None)
        if isinstance(tensor, torch.Tensor):
            return tensor
    return None


class _PickleLifecycleExecutor:
    """Encapsulate pickle-mode storage reserve/read lifecycle."""

    def __init__(self, storage_manager: "StorageManager") -> None:
        self._storage_manager = storage_manager

    def store(
        self,
        payload_objects: list[list[torch.Tensor]],
        resolved_groups: list[_ResolvedObjectGroup],
    ) -> bool:
        """Reserve, copy, and either finish or rollback write reservations."""
        reserved_by_group: list[tuple[_ResolvedObjectGroup, dict[ObjectKey, Any]]] = []
        reserved_keys: list[ObjectKey] = []
        try:
            for group in resolved_groups:
                reserved = self._storage_manager.reserve_write(
                    group.obj_keys,
                    group.layout_desc,
                    "new",
                )
                reserved_by_group.append((group, reserved))
                reserved_keys.extend(reserved.keys())

            payload_idx = 0
            for group, reserved in reserved_by_group:
                for obj_key in group.obj_keys:
                    payload_object = payload_objects[payload_idx]
                    payload_idx += 1
                    memory_obj = reserved.get(obj_key)
                    if memory_obj is None:
                        continue
                    for tensor_idx, component in enumerate(payload_object):
                        memory_tensor = _memory_tensor(memory_obj, tensor_idx)
                        if memory_tensor is None:
                            raise ValueError(
                                "reserved memory object missing tensor component"
                            )
                        memory_tensor.copy_(component)

            if payload_idx != len(payload_objects):
                return False
            if reserved_keys:
                self._storage_manager.finish_write(reserved_keys)
            return True
        except Exception:
            if reserved_keys:
                self._storage_manager.delete_l1_keys(reserved_keys, force=True)
            return False

    def retrieve(
        self,
        resolved_groups: list[_ResolvedObjectGroup],
        flatten_single_component_payload: bool,
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and serialize object-major pickle payload.

        Notes:
            Each group's read locks are managed by a ``read_prefetched_results``
            context manager.  When post-prefetch validation fails for a group,
            ``_PostPrefetchValidationError`` is raised inside the context so its
            exception path releases that group's locks.  Keys from groups that
            were already successfully read are accumulated in ``prefetched_keys``
            and released via ``finish_read_prefetched`` in the outer ``finally``.
            Only groups that exit their context normally without exception have
            their keys entered into ``prefetched_keys``; the outer ``finally``
            does not overlap with the context's own cleanup.
        """
        payload_objects: list[list[torch.Tensor]] = []
        prefetched_keys: list[ObjectKey] = []
        try:
            for group in resolved_groups:
                selected_obj_keys = _select_windowed_obj_keys_for_retrieve(group)
                if not selected_obj_keys:
                    continue
                read_ctx = self._storage_manager.read_prefetched_results(
                    selected_obj_keys
                )
                group_payload: list[list[torch.Tensor]] = []
                with read_ctx as maybe_memory_objs:
                    if not maybe_memory_objs:
                        # Context yielded None (some keys missing); its own cleanup
                        # releases any partial read locks for this group.
                        raise _PostPrefetchValidationError(
                            f"prefetch returned no objects for group "
                            f"{group.object_group_id} "
                            f"(expected {len(selected_obj_keys)} keys)"
                        )
                    if len(maybe_memory_objs) != len(selected_obj_keys):
                        # Count mismatch: raise so context releases this group's locks.
                        raise _PostPrefetchValidationError(
                            f"prefetch count mismatch for group "
                            f"{group.object_group_id}: "
                            f"expected {len(selected_obj_keys)}, "
                            f"got {len(maybe_memory_objs)}"
                        )
                    for memory_obj in maybe_memory_objs:
                        payload_object: list[torch.Tensor] = []
                        for tensor_idx in range(len(group.layout_desc.shapes)):
                            memory_tensor = _memory_tensor(memory_obj, tensor_idx)
                            if memory_tensor is None:
                                # Missing tensor component: raise so context releases.
                                raise _PostPrefetchValidationError(
                                    f"missing tensor component {tensor_idx} in "
                                    f"prefetched object for group "
                                    f"{group.object_group_id}"
                                )
                            payload_object.append(memory_tensor.cpu().clone())
                        group_payload.append(payload_object)
                # Context exited normally; caller now owns these read locks.
                prefetched_keys.extend(selected_obj_keys)
                payload_objects.extend(group_payload)
            serialized_payload: list[torch.Tensor] | list[list[torch.Tensor]]
            if flatten_single_component_payload:
                serialized_payload = [po[0] for po in payload_objects]
            else:
                serialized_payload = payload_objects
            return PrepareRetrieveResponse(
                success=True,
                data=pickle.dumps(serialized_payload),
                context={},
            )
        except _PostPrefetchValidationError:
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)


def create_transfer_strategy(
    storage_manager: "StorageManager",
    *,
    shm_name: str,
    pool_size: int,
    pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
    pending_lock: LockType,
    transfer_key_factory: Callable[
        [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
    ],
) -> "TransferStrategy":
    """Create the non-GPU transfer strategy for a registered context.

    Args:
        storage_manager: Storage manager used by the selected strategy.
        shm_name: Shared-memory pool name advertised to workers.
        pool_size: Shared-memory pool size in bytes.
        pending_writes: Map of pending SHM write reservations keyed by transfer key.
        pending_reads: Map of pending SHM read reservations keyed by transfer key.
        pending_lock: Lock guarding shared pending SHM reservation state.
        transfer_key_factory: Factory that builds the `(instance_id, key)` lookup key
            used in the pending SHM reservation maps.

    Returns:
        ``ShmTransferStrategy`` when SHM is configured with a non-empty pool name and
        positive pool size, otherwise ``PickleTransferStrategy``.
    """
    if shm_name and pool_size > 0:
        logger.info("Using shm non-GPU transfer strategy")
        return ShmTransferStrategy(
            storage_manager=storage_manager,
            pending_writes=pending_writes,
            pending_reads=pending_reads,
            pending_lock=pending_lock,
            transfer_key_factory=transfer_key_factory,
            fallback_strategy=PickleTransferStrategy(storage_manager),
        )

    logger.info("Using pickle non-GPU transfer strategy")
    return PickleTransferStrategy(storage_manager)


class TransferStrategy(abc.ABC):
    """Contract for non-GPU transport backends used by the server.

    Implementations encapsulate the transport-specific prepare/commit lifecycle for
    store and retrieve operations, allowing the server to use either pickle-based or
    shared-memory-based transfers behind a common interface.
    """

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific store preparation response.
        """

    @abc.abstractmethod
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            resolve_obj_keys: Callable that resolves object keys from ``key``.

        Returns:
            Transport-specific retrieve preparation response.
        """

    @abc.abstractmethod
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.

        Returns:
            ``True`` when retrieve finalization succeeds.
        """


class PickleTransferStrategy(TransferStrategy):
    """Pickle-based transport for non-GPU transfer requests.

    This is the default transport when SHM is unavailable, and it is also used as a
    fallback by the SHM strategy when the worker sends an inline serialized payload.
    ``prepare_store`` returns an empty context, while ``commit_store`` deserializes
    the pickle payload and writes the resulting tensors into reserved objects.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
    ) -> None:
        """Initialize pickle transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
        """
        self._storage_manager = storage_manager
        self._executor = _PickleLifecycleExecutor(storage_manager)

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareStoreResponse:
        """Return empty store context for pickle mode.

        Pickle transport does not pre-allocate SHM slots during prepare.
        """
        return PrepareStoreResponse(context={})

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
            Returns ``False`` when payload deserialization fails before any
            storage reservation.
        """
        try:
            payload = pickle.loads(cpu_data)
        except (
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
        ):
            # pickle.PickleError / EOFError: malformed or truncated payload bytes.
            # AttributeError / ImportError: unresolved classes or modules.
            # IndexError: malformed opcode/data stream edge cases.
            # All decode failures must happen before any reserve_write side effect.
            logger.exception("Failed to deserialize engine-driven store payload")
            return False
        if not isinstance(payload, list):
            return False
        payload_objects = _normalize_store_payload(payload)
        if payload_objects is None:
            return False
        try:
            resolved_groups = _resolve_object_groups(key, context, resolve_obj_keys)
        except (IndexError, KeyError, ValueError):
            return False
        if not _validate_store_payload(payload_objects, resolved_groups):
            return False
        return self._executor.store(payload_objects, resolved_groups)

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        try:
            resolved_groups = _resolve_object_groups(key, context, resolve_obj_keys)
        except (IndexError, KeyError, ValueError):
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        return self._executor.retrieve(
            resolved_groups,
            flatten_single_component_payload=_is_legacy_single_group_context(context),
        )

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """No-op for pickle mode; data was already copied during prepare."""
        return True


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory transport for non-GPU transfer requests.

    This strategy exposes SHM slot descriptors during ``prepare_store`` and
    ``prepare_retrieve`` so workers can access storage buffers directly. It tracks
    pending SHM reservations until the matching commit step releases them, and it
    falls back to pickle-based commit handling when ``cpu_data`` is non-empty.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        pending_writes: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_reads: dict[tuple[int, IPCCacheServerKey], list[ObjectKey]],
        pending_lock: LockType,
        transfer_key_factory: Callable[
            [IPCCacheServerKey, int], tuple[int, IPCCacheServerKey]
        ],
        fallback_strategy: PickleTransferStrategy,
    ) -> None:
        """Initialize SHM transfer strategy.

        Args:
            storage_manager: Storage manager used for reserve/read/finish calls.
            pending_writes: Shared pending SHM write reservations map.
            pending_reads: Shared pending SHM read reservations map.
            pending_lock: Lock guarding shared pending SHM maps.
            transfer_key_factory: Factory to build `(instance_id, key)` transfer keys.
            fallback_strategy: Pickle fallback for non-empty ``cpu_data`` payloads.
        """
        self._storage_manager = storage_manager
        self._pending_writes = pending_writes
        self._pending_reads = pending_reads
        self._pending_lock = pending_lock
        self._transfer_key_factory = transfer_key_factory
        self._fallback_strategy = fallback_strategy

    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Context with ``slots`` and ``chunk_indices``.
        """
        obj_keys = resolve_obj_keys(key, [0])[0]
        reserved = self._storage_manager.reserve_write(
            obj_keys, context.layout_desc, "new"
        )
        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        reserved_keys: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                memory_obj = reserved.get(obj_key)
                if memory_obj is None or memory_obj.tensor is None:
                    continue
                slots.append(
                    ShmSlotDescriptor(
                        offset=memory_obj.shm_offset,
                        length=memory_obj.shm_byte_length,
                        shape=list(memory_obj.tensor.shape),
                        dtype=_dtype_to_name(memory_obj.tensor.dtype),
                    ).to_dict()
                )
                chunk_indices.append(idx)
                reserved_keys.append(obj_key)
        finally:
            reserved_keys_set = set(reserved_keys)
            unused_keys = [
                obj_key for obj_key in reserved if obj_key not in reserved_keys_set
            ]
            if unused_keys:
                self._storage_manager.finish_write(unused_keys)
        if not reserved_keys:
            return PrepareStoreResponse(context={"slots": [], "chunk_indices": []})
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            self._pending_writes[transfer_key] = reserved_keys
        return PrepareStoreResponse(
            context={"slots": slots, "chunk_indices": chunk_indices}
        )

    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> bool:
        """Finalize SHM store write locks or fallback to pickle commit.

        Returns:
            ``True`` when pending SHM reservation is committed successfully.
        """
        if cpu_data != b"":
            return self._fallback_strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=context,
                resolve_obj_keys=resolve_obj_keys,
            )
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            reserved_keys = self._pending_writes.pop(transfer_key, None)
        if reserved_keys is None:
            return False
        if reserved_keys:
            self._storage_manager.finish_write(reserved_keys)
        return True

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        obj_keys = resolve_obj_keys(key, [0])[0]
        shm_prefetched_keys, shm_memory_objs = self._storage_manager.unsafe_read(
            obj_keys
        )
        if (
            not shm_memory_objs
            or len(shm_prefetched_keys) != len(obj_keys)
            or len(shm_memory_objs) != len(obj_keys)
        ):
            if shm_prefetched_keys:
                self._storage_manager.finish_read_prefetched(shm_prefetched_keys)
            return PrepareRetrieveResponse(success=False, data=b"", context={})
        slots: list[dict[str, Any]] = []
        for memory_obj in shm_memory_objs:
            if memory_obj.tensor is None:
                self._storage_manager.finish_read_prefetched(shm_prefetched_keys)
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            slots.append(
                ShmSlotDescriptor(
                    offset=memory_obj.shm_offset,
                    length=memory_obj.shm_byte_length,
                    shape=list(memory_obj.tensor.shape),
                    dtype=_dtype_to_name(memory_obj.tensor.dtype),
                ).to_dict()
            )
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            self._pending_reads[transfer_key] = shm_prefetched_keys
        return PrepareRetrieveResponse(success=True, data=b"", context={"slots": slots})

    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Release pending SHM read locks for the completed retrieve request."""
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            prefetched_keys = self._pending_reads.pop(transfer_key, [])
        if prefetched_keys:
            self._storage_manager.finish_read_prefetched(prefetched_keys)
        return True
