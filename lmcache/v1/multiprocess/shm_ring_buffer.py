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

    Returns:
        The configured ring-buffer size in bytes.

    Raises:
        ValueError: If the environment variable is not a positive integer.
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
        """Create or attach to a named shared-memory ring buffer.

        Args:
            name: Shared-memory segment name.
            size: Total segment size in bytes, including the ring header.
            create: Whether to create a new segment or attach to an existing one.

        Raises:
            ValueError: If ``size`` does not exceed the ring header size.
            FileExistsError: If ``create`` is True and the segment already exists.
        """

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
        """Write a contiguous byte payload into the ring buffer.

        Args:
            data: Byte payload to write into the ring.

        Returns:
            Tuple ``(offset, length)`` describing the contiguous region that was
            written.

        Raises:
            ValueError: If the payload is larger than the ring-buffer capacity.

        Notes:
            This method blocks until enough free space is available.
        """

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
        """Write a CPU tensor into the shared-memory ring buffer.

        Args:
            tensor: Tensor to serialize into shared memory. The tensor is first
                converted to a contiguous CPU tensor view.

        Returns:
            Tuple ``(offset, length)`` describing the written payload.
        """

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
        """Read a CPU tensor view from the shared-memory ring buffer.

        Args:
            offset: Byte offset within the ring-buffer data region.
            length: Byte length of the serialized tensor payload.
            shape: Tensor shape used to reconstruct the view.
            dtype: Serialized torch dtype name or ``torch.dtype``.

        Returns:
            A CPU tensor view backed directly by shared memory.

        Raises:
            ValueError: If the metadata does not match the payload size.
        """

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
        """Return the absolute reader pointer.

        Returns:
            The total number of bytes the reader has consumed from the ring.
        """

        return self._get_u64(_READ_PTR_OFFSET)

    def advance_read_ptr(self, length: int, offset: int | None = None) -> int:
        """Advance the absolute reader pointer after consuming one payload.

        Args:
            length: Number of payload bytes consumed.
            offset: Expected payload offset within the current ring cycle.

        Returns:
            The new absolute read pointer.

        Raises:
            ValueError: If ``length`` is negative or the reader is
                desynchronized from the provided offset.
        """

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
        """Close the shared-memory handle without unlinking the segment."""

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
    """Convert a serialized dtype string into a torch dtype.

    Args:
        dtype: Serialized torch dtype name or an existing ``torch.dtype``.

    Returns:
        The normalized torch dtype.

    Raises:
        ValueError: If ``dtype`` does not resolve to a valid torch dtype.
    """

    if isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.removeprefix("torch.")
    torch_dtype = getattr(torch, normalized, None)
    if torch_dtype is None or not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype '{dtype}'")
    return torch_dtype
