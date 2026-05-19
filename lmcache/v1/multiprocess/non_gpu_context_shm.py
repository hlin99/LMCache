# SPDX-License-Identifier: Apache-2.0
"""Shared-memory NonGpuContext implementation for multiprocess mode."""

# Standard
import mmap
import os
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.non_gpu_context import (
    NonGpuContext,
    NonGpuContextMetadata,
)
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


class NonGpuContextShm(NonGpuContext):
    """Shared-memory implementation of :class:`NonGpuContext`.

    Attaches to a named shared-memory segment on init and uses zero-copy
    tensor views for data transfer. The server reserves slots and returns
    offset/shape metadata; the worker constructs torch views directly into
    shared memory.
    """

    def __init__(
        self,
        metadata: NonGpuContextMetadata,
        mq_client: Any,
        mq_timeout: float,
        shm_name: str,
        pool_size: int,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)
        # Attach to the server's shm segment via mmap (not SharedMemory)
        # to avoid Python's resource_tracker unlinking on worker exit.
        shm_path = f"/dev/shm/{shm_name}"
        fd = os.open(shm_path, os.O_RDWR)
        try:
            self._mmap = mmap.mmap(fd, pool_size)
        finally:
            os.close(fd)
        self._pool_size = pool_size
        self._buffer = self._mmap

    def _make_tensor_view(
        self, offset: int, length: int, shape: list[int], dtype_str: str
    ) -> torch.Tensor:
        """Create a tensor view into the shared-memory buffer."""
        dtype = getattr(torch, dtype_str)
        t = torch.frombuffer(self._buffer, dtype=dtype, count=length // dtype.itemsize, offset=offset)
        return t.view(shape)

    def prepare_store(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Send PREPARE_STORE RPC and return pre-allocated SHM tensor views."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None

        slots = response.context.get("slots")
        if not slots:
            return None

        views = []
        for slot in slots:
            view = self._make_tensor_view(
                slot["offset"], slot["length"], slot["shape"], slot["dtype"]
            )
            views.append(view)
        return views

    def commit_store(
        self, key: Any, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Notify server that data has been written to SHM (send empty bytes)."""
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, b""],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE and return SHM tensor views on hit."""
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None

        if not response.success:
            return None

        slots = response.context.get("slots")
        if not slots:
            return None

        views = []
        for slot in slots:
            view = self._make_tensor_view(
                slot["offset"], slot["length"], slot["shape"], slot["dtype"]
            )
            views.append(view)
        return views

    def commit_retrieve(self, key: Any, instance_id: int) -> bool:
        """Release read locks on the server side."""
        future = self.mq_client.submit_request(
            RequestType.COMMIT_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.COMMIT_RETRIEVE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return True

    def close(self) -> None:
        """Release buffer and close shared memory attachment."""
        self._buffer = None
        try:
            self._mmap.close()
        except Exception:
            pass
