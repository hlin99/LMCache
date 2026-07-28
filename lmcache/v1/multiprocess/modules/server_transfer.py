# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
from _thread import LockType
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import abc
import pickle

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
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
        """
        transfer_metadata = context.transfer_metadata
        is_multi_group = transfer_metadata is not None and (
            len(transfer_metadata.object_groups) > 1
            or len(transfer_metadata.kernel_groups) > 1
        )

        payload = pickle.loads(cpu_data)
        if not isinstance(payload, list):
            return False

        if not is_multi_group:
            obj_keys = resolve_obj_keys(key, [0])[0]
            if not all(isinstance(chunk, torch.Tensor) for chunk in payload):
                return False
            chunks: list[torch.Tensor] = payload
            reserved_dict = self._storage_manager.reserve_write(
                obj_keys, context.layout_desc, "new"
            )
            written_keys: list[ObjectKey] = []
            try:
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key not in reserved_dict:
                        continue
                    if idx >= len(chunks):
                        continue
                    memory_obj = reserved_dict[obj_key]
                    if memory_obj.tensor is None:
                        continue
                    chunk_cpu = chunks[idx]
                    if chunk_cpu.shape != memory_obj.tensor.shape:
                        continue
                    memory_obj.tensor.copy_(chunk_cpu)
                    written_keys.append(obj_key)
            finally:
                if written_keys:
                    self._storage_manager.finish_write(written_keys)
            return len(written_keys) == len(reserved_dict)

        assert transfer_metadata is not None
        object_group_ids = list(range(len(transfer_metadata.object_groups)))
        obj_keys_by_group = resolve_obj_keys(key, object_group_ids)
        layout_descs = context.object_group_layout_descs
        if len(layout_descs) != len(obj_keys_by_group):
            return False

        expected_num_objects = sum(len(obj_keys) for obj_keys in obj_keys_by_group)
        if len(payload) != expected_num_objects:
            return False

        payload_idx = 0
        reserved_keys: list[ObjectKey] = []
        pending_copies: list[tuple[Any, int, torch.Tensor]] = []
        success = False
        try:
            for obj_keys, layout_desc in zip(
                obj_keys_by_group,
                layout_descs,
                strict=True,
            ):
                reserved = self._storage_manager.reserve_write(
                    obj_keys,
                    layout_desc,
                    "new",
                )
                reserved_keys.extend(reserved.keys())
                for obj_key in obj_keys:
                    payload_object = payload[payload_idx]
                    payload_idx += 1
                    if obj_key not in reserved:
                        continue
                    if not isinstance(payload_object, list):
                        return False
                    if len(payload_object) != len(layout_desc.shapes):
                        return False
                    memory_obj = reserved[obj_key]
                    for tensor_idx, component in enumerate(payload_object):
                        if not isinstance(component, torch.Tensor):
                            return False
                        memory_tensor = memory_obj.get_tensor(tensor_idx)
                        if memory_tensor is None:
                            return False
                        if component.shape != memory_tensor.shape:
                            return False
                        if component.dtype != memory_tensor.dtype:
                            return False
                        pending_copies.append((memory_obj, tensor_idx, component))
            if payload_idx != len(payload):
                return False
            for memory_obj, tensor_idx, component in pending_copies:
                memory_tensor = memory_obj.get_tensor(tensor_idx)
                if memory_tensor is None:
                    return False
                memory_tensor.copy_(component)
            success = True
        except Exception:
            success = False
        finally:
            if reserved_keys:
                self._storage_manager.finish_write(reserved_keys)
        return success

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: ResolveObjKeysFn,
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        transfer_metadata = context.transfer_metadata
        is_multi_group = transfer_metadata is not None and (
            len(transfer_metadata.object_groups) > 1
            or len(transfer_metadata.kernel_groups) > 1
        )
        if not is_multi_group:
            obj_keys = resolve_obj_keys(key, [0])[0]
            prefetched_keys: list[ObjectKey] = []
            try:
                read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
                with read_ctx as maybe_memory_objs:
                    if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                        return PrepareRetrieveResponse(
                            success=False,
                            data=b"",
                            context={},
                        )
                    prefetched_keys = obj_keys[: len(maybe_memory_objs)]
                    chunks = []
                    for memory_obj in maybe_memory_objs:
                        if memory_obj.tensor is None:
                            return PrepareRetrieveResponse(
                                success=False, data=b"", context={}
                            )
                        chunks.append(memory_obj.tensor.cpu().clone())
                    return PrepareRetrieveResponse(
                        success=True, data=pickle.dumps(chunks), context={}
                    )
            finally:
                if prefetched_keys:
                    self._storage_manager.finish_read_prefetched(prefetched_keys)

        assert transfer_metadata is not None
        object_group_ids = list(range(len(transfer_metadata.object_groups)))
        obj_keys_by_group = resolve_obj_keys(key, object_group_ids)
        layout_descs = context.object_group_layout_descs
        if len(layout_descs) != len(obj_keys_by_group):
            return PrepareRetrieveResponse(success=False, data=b"", context={})

        payload_objects: list[list[torch.Tensor]] = []
        prefetched_keys: list[ObjectKey] = []
        try:
            for obj_keys, layout_desc in zip(
                obj_keys_by_group,
                layout_descs,
                strict=True,
            ):
                read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
                with read_ctx as maybe_memory_objs:
                    if (
                        not maybe_memory_objs
                        or len(maybe_memory_objs) != len(obj_keys)
                    ):
                        return PrepareRetrieveResponse(
                            success=False,
                            data=b"",
                            context={},
                        )
                    prefetched_keys.extend(obj_keys)
                    for memory_obj in maybe_memory_objs:
                        object_parts: list[torch.Tensor] = []
                        for tensor_idx in range(len(layout_desc.shapes)):
                            memory_tensor = memory_obj.get_tensor(tensor_idx)
                            if memory_tensor is None:
                                return PrepareRetrieveResponse(
                                    success=False,
                                    data=b"",
                                    context={},
                                )
                            object_parts.append(memory_tensor.cpu().clone())
                        payload_objects.append(object_parts)
            return PrepareRetrieveResponse(
                success=True,
                data=pickle.dumps(payload_objects),
                context={},
            )
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

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
