# SPDX-License-Identifier: Apache-2.0
"""Transfer strategy implementations for non-GPU transport paths."""

# Standard
import abc
import pickle
from collections.abc import Callable
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transport.base import NonGpuContextMetadata
from lmcache.v1.multiprocess.transport.shm_types import ShmSlotDescriptor


def _dtype_to_name(dtype: torch.dtype) -> str:
    """Return a stable torch dtype name without module prefix."""
    return str(dtype).split(".")[-1]


class TransferStrategy(abc.ABC):
    """Abstract strategy for non-GPU data transfer operations."""

    @abc.abstractmethod
    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        """Prepare destination resources for a store request."""

    @abc.abstractmethod
    def commit_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        cpu_data: bytes,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> bool:
        """Finalize a store request."""

    @abc.abstractmethod
    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        """Prepare source resources for a retrieve request."""

    @abc.abstractmethod
    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve request."""


class PickleTransferStrategy(TransferStrategy):
    """Pickle/ZMQ non-GPU transfer implementation."""

    def __init__(self, storage_manager: StorageManager) -> None:
        self._storage_manager = storage_manager

    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        return PrepareStoreResponse(context={})

    def commit_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        cpu_data: bytes,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> bool:
        obj_keys = resolve_obj_keys(key)
        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
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

    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        obj_keys = resolve_obj_keys(key)
        prefetched_keys: list[ObjectKey] = []
        try:
            read_ctx = self._storage_manager.read_prefetched_results(obj_keys)
            with read_ctx as maybe_memory_objs:
                if not maybe_memory_objs or len(maybe_memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(success=False, data=b"", context={})
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

    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        return True


class ShmTransferStrategy(TransferStrategy):
    """Shared-memory non-GPU transfer implementation."""

    def __init__(
        self,
        storage_manager: StorageManager,
        pending_writes: dict[tuple[int, IPCCacheEngineKey], list[ObjectKey]],
        pending_reads: dict[tuple[int, IPCCacheEngineKey], list[ObjectKey]],
        pending_lock: Any,
        transfer_key_factory: Callable[
            [IPCCacheEngineKey, int], tuple[int, IPCCacheEngineKey]
        ],
        fallback_strategy: PickleTransferStrategy,
    ) -> None:
        self._storage_manager = storage_manager
        self._pending_writes = pending_writes
        self._pending_reads = pending_reads
        self._pending_lock = pending_lock
        self._transfer_key_factory = transfer_key_factory
        self._fallback_strategy = fallback_strategy

    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareStoreResponse:
        obj_keys = resolve_obj_keys(key)
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
        key: IPCCacheEngineKey,
        instance_id: int,
        cpu_data: bytes,
        context: NonGpuContextMetadata,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> bool:
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
        key: IPCCacheEngineKey,
        instance_id: int,
        resolve_obj_keys: Callable[[IPCCacheEngineKey], list[ObjectKey]],
    ) -> PrepareRetrieveResponse:
        obj_keys = resolve_obj_keys(key)
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
        return PrepareRetrieveResponse(
            success=True, data=b"", context={"slots": slots}
        )

    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        transfer_key = self._transfer_key_factory(key, instance_id)
        with self._pending_lock:
            prefetched_keys = self._pending_reads.pop(transfer_key, [])
        if prefetched_keys:
            self._storage_manager.finish_read_prefetched(prefetched_keys)
        return True
