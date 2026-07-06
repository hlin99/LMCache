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
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        payload: Any = pickle.loads(cpu_data)
        if isinstance(payload, list):
            payload = {
                "object_groups": {
                    0: {
                        "chunk_indices": list(range(len(payload))),
                        "chunks": payload,
                    }
                }
            }
        if not isinstance(payload, dict):
            return False

        object_groups = payload.get("object_groups")
        if not isinstance(object_groups, dict) or not object_groups:
            return False

        reserved_by_group: dict[int, dict[ObjectKey, Any]] = {}
        layout_desc_by_group = self._get_layout_desc_by_group(context)
        written_keys: list[ObjectKey] = []
        success = True
        try:
            for object_group_id_raw in sorted(object_groups):
                object_group_id = int(object_group_id_raw)
                object_group_payload = object_groups[object_group_id_raw]
                if not isinstance(object_group_payload, dict):
                    success = False
                    break
                chunks = object_group_payload.get("chunks")
                if not isinstance(chunks, list):
                    success = False
                    break
                obj_keys = resolve_obj_keys(key, [object_group_id])[0]
                layout_desc = layout_desc_by_group.get(object_group_id, context.layout_desc)
                reserved = self._storage_manager.reserve_write(obj_keys, layout_desc, "new")
                reserved_by_group[object_group_id] = reserved
                for chunk_idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved.get(obj_key)
                    if memory_obj is None or memory_obj.tensor is None:
                        success = False
                        continue
                    if chunk_idx >= len(chunks):
                        success = False
                        continue
                    if not self._copy_chunk_to_memory(chunks[chunk_idx], memory_obj.tensor):
                        success = False
                        continue
                    written_keys.append(obj_key)
            if not success:
                return False
            expected_writes = sum(len(reserved) for reserved in reserved_by_group.values())
            return len(written_keys) == expected_writes
        finally:
            if written_keys:
                self._storage_manager.finish_write(written_keys)
            if not success:
                written_set = set(written_keys)
                for reserved in reserved_by_group.values():
                    rollback_keys = [
                        obj_key for obj_key in reserved if obj_key not in written_set
                    ]
                    if rollback_keys:
                        self._storage_manager.finish_write(rollback_keys)

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        layout_desc_by_group = self._get_layout_desc_by_group(context)
        object_groups = sorted(layout_desc_by_group) if layout_desc_by_group else [0]
        prefetched_keys: list[ObjectKey] = []
        payload: dict[str, Any] = {"object_groups": {}}
        try:
            for object_group_id in object_groups:
                obj_keys = resolve_obj_keys(key, [object_group_id])[0]
                read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
                with read_ctx as maybe_memory_objs:
                    if (
                        not maybe_memory_objs
                        or len(maybe_memory_objs) != len(obj_keys)
                    ):
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    prefetched_keys.extend(obj_keys[: len(maybe_memory_objs)])
                    chunks: list[Any] = []
                    for memory_obj in maybe_memory_objs:
                        if memory_obj.tensor is None:
                            return PrepareRetrieveResponse(
                                success=False, data=b"", context={}
                            )
                        chunks.append(self._clone_chunk_from_memory(memory_obj.tensor))
                    payload["object_groups"][object_group_id] = {
                        "chunk_indices": list(range(len(chunks))),
                        "chunks": chunks,
                    }
                if object_group_id in layout_desc_by_group and not obj_keys:
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
            if len(payload["object_groups"]) == 1 and 0 in payload["object_groups"]:
                chunks = payload["object_groups"][0]["chunks"]
                return PrepareRetrieveResponse(success=True, data=pickle.dumps(chunks), context={})
            return PrepareRetrieveResponse(
                success=True, data=pickle.dumps(payload), context={}
            )
        finally:
            if prefetched_keys:
                self._storage_manager.finish_read_prefetched(prefetched_keys)

    @staticmethod
    def _copy_chunk_to_memory(chunk_cpu: Any, memory_tensor: Any) -> bool:
        if isinstance(chunk_cpu, torch.Tensor) and isinstance(memory_tensor, torch.Tensor):
            if chunk_cpu.shape != memory_tensor.shape:
                return False
            memory_tensor.copy_(chunk_cpu)
            return True
        if isinstance(chunk_cpu, (list, tuple)) and isinstance(memory_tensor, (list, tuple)):
            if len(chunk_cpu) != len(memory_tensor):
                return False
            for src, dst in zip(chunk_cpu, memory_tensor, strict=True):
                if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
                    return False
                if src.shape != dst.shape:
                    return False
                dst.copy_(src)
            return True
        return False

    @staticmethod
    def _clone_chunk_from_memory(memory_tensor: Any) -> Any:
        if isinstance(memory_tensor, torch.Tensor):
            return memory_tensor.cpu().clone()
        if isinstance(memory_tensor, (list, tuple)):
            cloned = []
            for tensor in memory_tensor:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("unexpected non-tensor memory object chunk")
                cloned.append(tensor.cpu().clone())
            return cloned
        raise TypeError("unsupported memory object tensor type")

    @staticmethod
    def _get_layout_desc_by_group(
        context: EngineDrivenContextMetadata,
    ) -> dict[int, MemoryLayoutDesc]:
        if not context.engine_group_infos:
            return {0: context.layout_desc}
        object_group_id_by_sw: dict[int, int] = {}
        grouped_shapes: dict[int, list[torch.Size]] = {}
        grouped_dtypes: dict[int, list[torch.dtype]] = {}
        next_object_group_id = 0
        for info in context.engine_group_infos:
            sw_size_tokens = info.sw_size_tokens
            if sw_size_tokens not in object_group_id_by_sw:
                object_group_id_by_sw[sw_size_tokens] = next_object_group_id
                next_object_group_id += 1
            object_group_id = object_group_id_by_sw[sw_size_tokens]
            grouped_shapes.setdefault(object_group_id, []).append(context.layout_desc.shapes[0])
            grouped_dtypes.setdefault(object_group_id, []).append(context.layout_desc.dtypes[0])
        return {
            object_group_id: MemoryLayoutDesc(
                shapes=grouped_shapes[object_group_id],
                dtypes=grouped_dtypes[object_group_id],
            )
            for object_group_id in grouped_shapes
        }

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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, list[int]], list[list[ObjectKey]]],
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
