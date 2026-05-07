# SPDX-License-Identifier: Apache-2.0
"""Shared-memory ring-buffer helpers for multiprocess CPU bounce buffers."""

# Standard
from dataclasses import dataclass
from multiprocessing import shared_memory
import os
import struct
import time

# Third Party
import torch

_HEADER_BYTES = 128
_WRITE_PTR_OFFSET = 0
_READ_PTR_OFFSET = 8
_U64 = struct.Struct("<Q")
_DEFAULT_RING_SIZE_GB = 128


def get_default_shm_ring_size_bytes() -> int:
    """Return the configured shared-memory ring-buffer size in bytes.

    The ``LMCACHE_SHM_RING_SIZE_GB`` environment variable overrides the
    default size of 128 GiB per ring buffer.
    """

    raw_value = os.getenv(
        "LMCACHE_SHM_RING_SIZE_GB",
        str(_DEFAULT_RING_SIZE_GB),
    )
    ring_size_gb = int(raw_value)
    if ring_size_gb <= 0:
        raise ValueError("LMCACHE_SHM_RING_SIZE_GB must be a positive integer")
    return ring_size_gb * 1024 * 1024 * 1024


@dataclass
class ShmTransferMetadata:
    """Metadata describing tensors written into a shared-memory ring buffer."""

    offsets: list[int]
    lengths: list[int]
    shape: list[int]
    dtype: str


class ShmRingBuffer:
    """Single-writer single-reader shared-memory ring buffer.

    Notes:
        This implementation targets the current LMCache multiprocess bounce
        path: one writer, one reader, and aligned 64-bit pointer updates on
        64-bit platforms. The writer publishes completed payloads by updating
        the absolute write pointer after copying data into shared memory, while
        the reader advances the absolute read pointer after consuming the
        corresponding metadata.
    """

    def __init__(self, name: str, size: int, create: bool = True) -> None:
        """Create or attach to a named shared-memory ring buffer."""

        shm_name = name.lstrip("/")
        if size <= _HEADER_BYTES:
            raise ValueError("Shared-memory ring buffer size must exceed header size")

        if create:
            try:
                stale = shared_memory.SharedMemory(name=shm_name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass

        self._owns_shm = create
        self._shm = shared_memory.SharedMemory(name=shm_name, create=create, size=size)
        self.name = self._shm.name
        self.size = size
        self.capacity = size - _HEADER_BYTES
        self._header = self._shm.buf[:_HEADER_BYTES]
        self._data = self._shm.buf[_HEADER_BYTES:]

        if create:
            self._set_u64(_WRITE_PTR_OFFSET, 0)
            self._set_u64(_READ_PTR_OFFSET, 0)

    def write(self, data: bytes | memoryview) -> tuple[int, int]:
        """Write a contiguous byte payload into the ring buffer."""

        payload = data if isinstance(data, memoryview) else memoryview(data)
        length = len(payload)
        if length == 0:
            return 0, 0
        if length > self.capacity:
            raise ValueError(
                f"Payload size {length} exceeds ring-buffer capacity {self.capacity}"
            )

        while True:
            write_ptr = self._get_u64(_WRITE_PTR_OFFSET)
            read_ptr = self._get_u64(_READ_PTR_OFFSET)
            used = write_ptr - read_ptr
            available = self.capacity - used
            offset = write_ptr % self.capacity
            padding = 0
            tail = self.capacity - offset
            if length > tail:
                padding = tail
            if length + padding <= available:
                break
            time.sleep(0.001)

        if padding:
            offset = 0
        self._data[offset : offset + length] = payload
        self._set_u64(_WRITE_PTR_OFFSET, write_ptr + padding + length)
        return offset, length

    def read(self, offset: int, length: int) -> memoryview:
        """Return a zero-copy view for a previously written contiguous payload.

        Callers must only read ranges whose metadata has already been published
        by the writer and not yet released by advancing the read pointer.
        """

        if length < 0:
            raise ValueError("length must be non-negative")
        if offset < 0 or offset + length > self.capacity:
            raise ValueError("Requested range exceeds ring-buffer bounds")
        return self._data[offset : offset + length]

    def write_tensor(self, tensor: torch.Tensor) -> tuple[int, int]:
        """Write a CPU tensor into the shared-memory ring buffer."""

        tensor_cpu = tensor.detach().cpu().contiguous()
        payload = memoryview(tensor_cpu.numpy()).cast("B")
        return self.write(payload)

    def read_tensor(
        self,
        offset: int,
        length: int,
        shape: list[int],
        dtype: str | torch.dtype,
    ) -> torch.Tensor:
        """Read a CPU tensor view from the shared-memory ring buffer."""

        torch_dtype = _normalize_torch_dtype(dtype)
        itemsize = torch.empty((), dtype=torch_dtype).element_size()
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        if expected_numel * itemsize != length:
            raise ValueError(
                "Shared-memory tensor metadata does not match payload size: "
                f"shape={shape}, dtype={torch_dtype}, length={length}"
            )
        return torch.frombuffer(
            self.read(offset, length),
            dtype=torch_dtype,
        ).view(*shape)

    def get_read_ptr(self) -> int:
        """Return the absolute reader pointer."""

        return self._get_u64(_READ_PTR_OFFSET)

    def advance_read_ptr(self, length: int, offset: int | None = None) -> int:
        """Advance the absolute reader pointer after consuming one payload."""

        if length < 0:
            raise ValueError("length must be non-negative")
        read_ptr = self._get_u64(_READ_PTR_OFFSET)
        if offset is not None:
            current_offset = read_ptr % self.capacity
            if current_offset != offset:
                if offset != 0:
                    raise ValueError(
                        "Shared-memory ring-buffer reader desynchronized: "
                        f"expected offset {current_offset}, got {offset}"
                    )
                read_ptr += self.capacity - current_offset
        read_ptr += length
        self._set_u64(_READ_PTR_OFFSET, read_ptr)
        return read_ptr

    def close(self) -> None:
        """Close the shared-memory handle."""

        self._data.release()
        self._header.release()
        self._shm.close()

    def unlink(self) -> None:
        """Unlink the shared-memory segment if this process created it.

        The worker process is the single owner of bounce-buffer ring names and
        is responsible for unlinking them during shutdown.
        """

        if not self._owns_shm:
            return
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def _get_u64(self, offset: int) -> int:
        return _U64.unpack_from(self._header, offset)[0]

    def _set_u64(self, offset: int, value: int) -> None:
        _U64.pack_into(self._header, offset, value)


def _normalize_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a serialized dtype string into a torch dtype."""

    if isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.removeprefix("torch.")
    torch_dtype = getattr(torch, normalized, None)
    if torch_dtype is None or not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype '{dtype}'")
    return torch_dtype
