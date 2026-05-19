# SPDX-License-Identifier: Apache-2.0
"""Shared-memory NonGpuContext implementation for multiprocess mode."""

# Standard
from multiprocessing import shared_memory
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.non_gpu_context import (
    NonGpuContext,
    NonGpuContextMetadata,
)
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)

logger = init_logger(__name__)


class NonGpuContextShm(NonGpuContext):
    """Shared-memory implementation of :class:`NonGpuContext`."""

    def __init__(
        self,
        metadata: NonGpuContextMetadata,
        mq_client: Any,
        mq_timeout: float,
        shm_name: str,
        pool_size: int,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)
        self._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        if self._shm.size < pool_size:
            self._shm.close()
            raise ValueError(
                f"SHM pool size mismatch: expected {pool_size}, got {self._shm.size}"
            )
        self._buffer = self._shm.buf

    def allocate_store_buffers(
        self, num_chunks: int, chunk_shape: list[int], dtype: torch.dtype
    ) -> list[torch.Tensor] | None:
        """Reserve SHM slots and return shm-backed tensor views for store."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [None, None, num_chunks],
            get_response_class(RequestType.PREPARE_STORE),
        )
        response: PrepareStoreResponse = future.result(timeout=self.mq_timeout)
        if not response.slots:
            return None
        buffers: list[torch.Tensor] = []
        for slot in response.slots:
            buffers.append(
                self._make_tensor_view(slot.offset, slot.length, slot.shape, slot.dtype)
            )
        # Stash the response for prepare_store to use.
        self._pending_store_response = response
        return buffers

    def allocate_retrieve_buffers(
        self, num_chunks: int, chunk_shape: list[int], dtype: torch.dtype
    ) -> list[torch.Tensor] | None:
        """Reserve SHM slots and return shm-backed tensor views for retrieve."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [None, None, num_chunks],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        response: PrepareRetrieveResponse = future.result(timeout=self.mq_timeout)
        if not response.success or not response.slots:
            return None
        buffers: list[torch.Tensor] = []
        for slot in response.slots:
            buffers.append(
                self._make_tensor_view(slot.offset, slot.length, slot.shape, slot.dtype)
            )
        self._pending_retrieve_response = response
        return buffers

    def prepare_store(
        self, key: Any, instance_id: int, chunks: list[torch.Tensor]
    ) -> Any:
        """Prepare a SHM store.

        If allocate_store_buffers was called, data is already in SHM slots
        and we just need to finalize with key/instance_id. Otherwise fall back
        to the original copy path.
        """
        pending = getattr(self, "_pending_store_response", None)
        if pending is not None:
            # Data already in SHM via allocate_store_buffers; just record key.
            self._pending_store_response = None
            return (key, instance_id, True)

        # Fallback: RPC for slot reservation and copy chunks in.
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        response: PrepareStoreResponse = future.result(timeout=self.mq_timeout)
        success = True
        try:
            for slot in response.slots:
                if slot.chunk_index >= len(chunks):
                    success = False
                    continue
                self._make_tensor_view(
                    slot.offset, slot.length, slot.shape, slot.dtype
                ).copy_(chunks[slot.chunk_index])
        except (RuntimeError, ValueError, IndexError):
            logger.exception("Failed to copy prepared store chunks into SHM")
            success = False
        return (key, instance_id, success)

    def commit_store(self, handle: Any) -> bool:
        """Commit a prepared SHM store."""
        key, instance_id, success = handle
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return False
        return bool(success)

    def prepare_retrieve(
        self, key: Any, instance_id: int
    ) -> tuple[Any, list[torch.Tensor] | None]:
        """Prepare a SHM retrieve and return zero-copy tensor views."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        response: PrepareRetrieveResponse = future.result(timeout=self.mq_timeout)
        if not response.success or not response.slots:
            return (None, None)
        try:
            chunks = [
                self._make_tensor_view(slot.offset, slot.length, slot.shape, slot.dtype)
                for slot in response.slots
            ]
        except (RuntimeError, ValueError):
            logger.exception("Failed to construct SHM tensor views for retrieve")
            return ((key, instance_id), None)
        return ((key, instance_id), chunks)

    def commit_retrieve(self, handle: Any) -> None:
        """Finish a prepared SHM retrieve and release read locks."""
        if handle is None:
            return
        key, instance_id = handle
        future = self.mq_client.submit_request(
            RequestType.FINISH_READ,
            [key, instance_id],
            get_response_class(RequestType.FINISH_READ),
        )
        future.result(timeout=self.mq_timeout)

    def close(self) -> None:
        """Detach from the shared-memory pool."""
        self._buffer.release()
        self._shm.close()

    def _make_tensor_view(
        self, offset: int, length: int, shape: list[int], dtype_name: str
    ) -> torch.Tensor:
        """Construct a zero-copy tensor view over the SHM pool."""
        dtype = getattr(torch, dtype_name, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid SHM tensor dtype {dtype_name!r}")
        return torch.frombuffer(
            self._buffer[offset : offset + length],
            dtype=dtype,
        ).view(*shape)
