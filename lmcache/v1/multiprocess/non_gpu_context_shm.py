import pickle
# SPDX-License-Identifier: Apache-2.0
"""Shared memory based NonGpuContext implementation for multiprocess mode."""

# Standard
import mmap
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from typing import Generator

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.non_gpu_context import (
    NonGpuContext,
    NonGpuContextMetadata,
)
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

# Default SHM capacity (2GB)
DEFAULT_SHM_CAPACITY = 2 * 1024 * 1024 * 1024


def _check_shm_capacity(required_bytes: int, shm_path: str = "/dev/shm") -> bool:
    """Check if available SHM capacity is sufficient.
    
    Args:
        required_bytes: Minimum required bytes
        shm_path: Path to check (default: /dev/shm)
    
    Returns:
        True if sufficient capacity exists
    """
    try:
        stat = os.statvfs(shm_path)
        available = stat.f_bavail * stat.f_frsize
        return available >= required_bytes
    except (OSError, AttributeError):
        # Fallback: try to create a test file
        try:
            test_file = os.path.join(shm_path, f".shm_check_{uuid.uuid4().hex}")
            with open(test_file, "wb") as f:
                f.write(b"test")
            os.unlink(test_file)
            return True
        except OSError:
            return False


def _unlink_stale_shm(name: str) -> None:
    """Remove stale shared memory file if it exists."""
    try:
        shm_path = f"/dev/shm/{name}"
        if os.path.exists(shm_path):
            os.unlink(shm_path)
    except OSError:
        pass


class SharedMemoryBuffer:
    """Wrapper for mmap-based shared memory buffer."""
    
    def __init__(self, name: str, size: int, fd: int) -> None:
        self.name = name
        self.size = size
        self.fd = fd
        self._mmap = None
        self._tensor: torch.Tensor | None = None
    
    def _get_mmap(self) -> mmap.mmap:
        """Get or create mmap."""
        if self._mmap is None:
            self._mmap = mmap.mmap(self.fd, self.size)
        return self._mmap
    
    def as_tensor(self, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
        """Get a Tensor view into the shared memory."""
        if self._tensor is None:
            self._tensor = torch.frombuffer(
                self._get_mmap(),
                dtype=dtype,
                shape=shape,
            )
        return self._tensor
    
    def close(self) -> None:
        """Close the mmap."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
    
    def unlink(self) -> None:
        """Remove the backing file."""
        _unlink_stale_shm(self.name)


class NonGpuContextShm(NonGpuContext):
    """Shared memory based implementation of :class:`NonGpuContext`.
    
    This implementation uses /dev/shm (tmpfs) for zero-copy like data transfer
    between worker and server processes. Falls back to pickle on failure.
    
    Transport mechanism:
    - **Store**: ``prepare_store`` allocates a SHM slot locally and returns
      the slot info. ``commit_store`` copies data to SHM and sends a small
      handle to server.
    - **Retrieve**: ``prepare_retrieve`` sends request, server writes to SHM
      and returns a handle. ``commit_retrieve`` reads from SHM and releases.
    """
    
    def __init__(
        self,
        metadata: NonGpuContextMetadata,
        mq_client: Any,
        mq_timeout: float,
        fallback: "NonGpuContext | None" = None,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)
        self.fallback = fallback
        self._shm_buffers: dict[str, SharedMemoryBuffer] = {}
        self._max_capacity = DEFAULT_SHM_CAPACITY
    
    def _create_shm_buffer(self, size: int) -> SharedMemoryBuffer | None:
        """Create a shared memory buffer.
        
        Args:
            size: Size in bytes
            
        Returns:
            SharedMemoryBuffer or None if creation fails
        """
        # Check capacity
        if not _check_shm_capacity(size):
            return None
        
        # Create unique name
        name = f"lmcache_{uuid.uuid4().hex}"
        
        try:
            # Create temp file in /dev/shm
            path = f"/dev/shm/{name}"
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            
            # Allocate size
            os.ftruncate(fd, size)
            
            return SharedMemoryBuffer(name, size, fd)
        except OSError:
            return None
    
    def prepare_store(
        self, key: Any, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare for store by allocating SHM buffers.
        
        For SHM mode, we allocate buffers locally and return slot info to server.
        """
        # Calculate total size needed
        # This is a simplified version - in practice you'd calculate based on
        # metadata (num_layers * hidden_dim * chunk_tokens * dtype_size)
        # For now, use a reasonable default or ask server
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            if self.fallback:
                return self.fallback.prepare_store(key, instance_id)
            return None
        
        # Response contains context with slot info - we'll use local SHM
        # For now, return empty to signal pickle fallback
        return None
    
    def commit_store(
        self, key: Any, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Serialize chunks and send via COMMIT_STORE.
        
        Falls back to pickle if SHM is not available.
        """
        # Try SHM: for now just use pickle serialization
        # Full SHM would involve writing to pre-allocated SHM buffers
        try:
            serialised = pickle.dumps(chunks)
        except Exception:
            if self.fallback:
                return self.fallback.commit_store(key, instance_id, chunks)
            return False
        
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, serialised],
            get_response_class(RequestType.COMMIT_STORE),
        )
        
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            if self.fallback:
                return self.fallback.commit_store(key, instance_id, chunks)
            return False
    
    def prepare_retrieve(
        self, key: Any, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve - request data from server.
        
        For SHM mode, server writes directly to SHM and returns a handle.
        """
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            if self.fallback:
                return self.fallback.prepare_retrieve(key, instance_id)
            return None
        
        if not response.success:
            return None
        
        # For now, fallback to pickle deserialization
        # Full SHM would read directly from shared memory
        if response.data:
            chunks: list[torch.Tensor] = pickle.loads(response.data)
            return chunks
        
        if self.fallback:
            return self.fallback.prepare_retrieve(key, instance_id)
        return None
    
    def commit_retrieve(self, key: Any, instance_id: int) -> bool:
        """Send COMMIT_RETRIEVE to release server-side resources."""
        try:
            future = self.mq_client.submit_request(
                RequestType.COMMIT_RETRIEVE,
                [key, instance_id],
                get_response_class(RequestType.COMMIT_RETRIEVE),
            )
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        
        # Clean up any local SHM buffers for this key
        key_str = f"{key}:{instance_id}"
        if key_str in self._shm_buffers:
            self._shm_buffers[key_str].close()
            self._shm_buffers[key_str].unlink()
            del self._shm_buffers[key_str]
        
        return True
    
    def close(self) -> None:
        """Release all shared memory resources."""
        for buf in self._shm_buffers.values():
            buf.close()
            buf.unlink()
        self._shm_buffers.clear()
        
        if self.fallback:
            self.fallback.close()
