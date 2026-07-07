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
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


def _flat_uint8_tensor(tensor: torch.Tensor, num_bytes: int) -> torch.Tensor:
    """Return a flat uint8 view over the first bytes of a tensor.

    Args:
        tensor: Source tensor with any dtype or shape.
        num_bytes: Number of bytes to expose from the start of ``tensor``.

    Returns:
        A one-dimensional ``torch.uint8`` tensor of length ``num_bytes`` that
        shares storage with ``tensor``; no data copy is performed.

    Raises:
        ValueError: If ``num_bytes`` is negative or exceeds ``tensor.nbytes``.
    """
    if num_bytes < 0 or num_bytes > tensor.nbytes:
        raise ValueError(
            f"num_bytes must be in inclusive range [0, {tensor.nbytes}], "
            f"got {num_bytes}"
        )
    return tensor.view(torch.uint8)[:num_bytes]


def _memory_obj_size(memory_obj: "MemoryObj", tensor: torch.Tensor) -> int:
    """Return a memory object's logical byte size, falling back to tensor bytes."""
    try:
        size = memory_obj.get_size()
    except (AttributeError, NotImplementedError):
        return tensor.nbytes
    return size if isinstance(size, int) else tensor.nbytes


def _memory_obj_tensor_view(memory_obj: "MemoryObj") -> torch.Tensor | None:
    """Return a tensor view suitable for engine-driven CPU transport.

    Multi-shape object groups cannot be represented by ``MemoryObj.tensor``
    because that legacy property reshapes to the first shape only. For those
    objects, return a flat uint8 view over the full logical object bytes.

    Args:
        memory_obj: Storage memory object reserved or prefetched for one chunk.

    Returns:
        ``None`` when the memory object has no backing tensor; otherwise the
        legacy tensor view for single-shape objects or a flat uint8 view for
        multi-shape objects.
    """
    try:
        shapes = memory_obj.get_shapes()
    except (AttributeError, NotImplementedError):
        shapes = []
    try:
        raw_tensor = memory_obj.raw_tensor
    except (AttributeError, NotImplementedError):
        raw_tensor = None
    if not isinstance(raw_tensor, torch.Tensor):
        try:
            tensor = memory_obj.tensor
        except (AttributeError, NotImplementedError):
            return None
        return tensor if isinstance(tensor, torch.Tensor) else None
    if len(shapes) <= 1:
        try:
            tensor = memory_obj.tensor
        except (AttributeError, NotImplementedError):
            return None
        return tensor if isinstance(tensor, torch.Tensor) else None
    return _flat_uint8_tensor(raw_tensor, _memory_obj_size(memory_obj, raw_tensor))


def _copy_tensor_to_memory_obj(src: torch.Tensor, memory_obj: "MemoryObj") -> bool:
    """Copy ``src`` bytes into ``memory_obj`` when sizes are compatible.

    Args:
        src: CPU tensor containing the serialized object-group bytes.
        memory_obj: Destination storage memory object.

    Returns:
        True when ``src.nbytes`` equals ``memory_obj.get_size()`` and the copy
        is performed; otherwise False.
    """
    dst = _memory_obj_tensor_view(memory_obj)
    if dst is None or src.nbytes != _memory_obj_size(memory_obj, dst):
        return False
    dst_view = _flat_uint8_tensor(dst, dst.nbytes)
    src_view = _flat_uint8_tensor(src, src.nbytes)
    dst_view.copy_(src_view)
    return True


def _slot_descriptor_from_memory_obj(
    memory_obj: "MemoryObj",
) -> dict[str, Any] | None:
    """Build an SHM slot descriptor for a reserved memory object.

    Args:
        memory_obj: Storage memory object backed by the SHM pool.

    Returns:
        ``None`` when the object has no tensor view; otherwise a dict with the
        slot byte offset, byte length, shape, and dtype for worker-side mapping.
    """
    tensor = _memory_obj_tensor_view(memory_obj)
    if tensor is None:
        return None
    return ShmSlotDescriptor(
        offset=memory_obj.shm_offset,
        length=memory_obj.shm_byte_length,
        shape=list(tensor.shape),
        dtype=_dtype_to_name(tensor.dtype),
    ).to_dict()


def _flatten_object_keys(
    key: IPCCacheServerKey,
    context: EngineDrivenContextMetadata,
    resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
) -> list[ObjectKey]:
    """Resolve object keys in object-group-major order.

    Args:
        key: IPC cache key for the transfer.
        context: Registered engine-driven metadata that determines the object
            group count.
        resolve_obj_keys: Callable that resolves one object group's keys.

    Returns:
        All chunks for object group 0, then all chunks for object group 1, and
        so on.
    """
    obj_keys: list[ObjectKey] = []
    for object_group_id in range(context.num_object_groups):
        obj_keys.extend(resolve_obj_keys(key, object_group_id))
    return obj_keys


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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``
                and an object-group index.

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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> bool:
        """Finalize a store request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            cpu_data: Serialized payload from the worker.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``
                and an object-group index.

        Returns:
            ``True`` when the strategy successfully commits the store request.
        """

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request.

        Args:
            key: Cache key identifying the requested token range.
            instance_id: Worker instance identifier.
            context: Non-GPU transfer metadata for the instance.
            resolve_obj_keys: Callable that resolves object keys from ``key``
                and an object-group index.

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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> bool:
        """Deserialize and write pickled chunks into reserved objects.

        Returns:
            ``True`` when every reserved object is written successfully.
        """
        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
        chunk_offset = 0
        written_keys: list[ObjectKey] = []
        reserved_count = 0
        for object_group_id in range(context.num_object_groups):
            obj_keys = resolve_obj_keys(key, object_group_id)
            layout_desc = context.layout_desc_for_object_group(object_group_id)
            reserved_dict = self._storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )
            reserved_count += len(reserved_dict)
            try:
                group_written: list[ObjectKey] = []
                for chunk_idx, obj_key in enumerate(obj_keys):
                    if obj_key not in reserved_dict:
                        continue
                    flat_idx = chunk_offset + chunk_idx
                    if flat_idx >= len(chunks):
                        continue
                    memory_obj = reserved_dict[obj_key]
                    if _copy_tensor_to_memory_obj(chunks[flat_idx], memory_obj):
                        written_keys.append(obj_key)
                        group_written.append(obj_key)
            finally:
                if group_written:
                    self._storage_manager.finish_write(group_written)
            chunk_offset += len(obj_keys)

        return len(written_keys) == reserved_count

    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        context: EngineDrivenContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Read prefetched objects and return serialized pickle payload."""
        obj_keys = _flatten_object_keys(key, context, resolve_obj_keys)
        prefetched_keys: list[ObjectKey] = []
        try:
            read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
            with read_ctx as maybe_memory_objs:
                if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
                prefetched_keys = obj_keys[: len(maybe_memory_objs)]
                chunks = []
                for memory_obj in maybe_memory_objs:
                    tensor = _memory_obj_tensor_view(memory_obj)
                    if tensor is None:
                        return PrepareRetrieveResponse(
                            success=False, data=b"", context={}
                        )
                    chunks.append(tensor.cpu().clone())
                return PrepareRetrieveResponse(
                    success=True, data=pickle.dumps(chunks), context={}
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Reserve SHM-backed objects and return slot descriptors.

        Returns:
            Context with ``slots`` and ``chunk_indices``.
        """
        slots: list[dict[str, Any]] = []
        chunk_indices: list[int] = []
        reserved_keys: list[ObjectKey] = []
        chunk_offset = 0
        for object_group_id in range(context.num_object_groups):
            obj_keys = resolve_obj_keys(key, object_group_id)
            layout_desc = context.layout_desc_for_object_group(object_group_id)
            reserved = self._storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )
            try:
                for idx, obj_key in enumerate(obj_keys):
                    memory_obj = reserved.get(obj_key)
                    if memory_obj is None:
                        continue
                    slot = _slot_descriptor_from_memory_obj(memory_obj)
                    if slot is None:
                        continue
                    slots.append(slot)
                    chunk_indices.append(chunk_offset + idx)
                    reserved_keys.append(obj_key)
            finally:
                reserved_keys_set = set(reserved_keys)
                unused_keys = [
                    obj_key for obj_key in reserved if obj_key not in reserved_keys_set
                ]
                if unused_keys:
                    self._storage_manager.finish_write(unused_keys)
            chunk_offset += len(obj_keys)
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
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
        resolve_obj_keys: Callable[[IPCCacheServerKey, int], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Read SHM objects and return slot descriptors for worker access."""
        obj_keys = _flatten_object_keys(key, context, resolve_obj_keys)
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
            slot = _slot_descriptor_from_memory_obj(memory_obj)
            if slot is None:
                self._storage_manager.finish_read_prefetched(shm_prefetched_keys)
                return PrepareRetrieveResponse(success=False, data=b"", context={})
            slots.append(slot)
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
