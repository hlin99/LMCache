# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
import os
import threading
import time
from typing import Any

# Third Party
import msgspec
import torch


@contextmanager
def _make_ring_buffer(size: int = 4096) -> Any:
    """Create and clean up a shared-memory ring buffer for tests."""

    # First Party
    from lmcache.v1.multiprocess.shm_ring_buffer import ShmRingBuffer

    ring_buffer = ShmRingBuffer(
        f"test_shm_ring_buffer_{os.getpid()}_{time.time_ns()}",
        size,
        create=True,
    )
    try:
        yield ring_buffer
    finally:
        ring_buffer.close()
        ring_buffer.unlink()


def test_shm_ring_buffer_write_read_roundtrip() -> None:
    """Validate basic shared-memory byte round-trip."""

    with _make_ring_buffer() as ring_buffer:
        payload = b"hello shared memory"
        offset, length, padding = ring_buffer.write(payload)

        assert bytes(ring_buffer.read(offset, length)) == payload
        ring_buffer.advance_read_ptr(length, padding=padding)


def test_shm_ring_buffer_wraparound_write() -> None:
    """Ensure writes wrap to the start when the tail region is too small."""

    with _make_ring_buffer(size=1024) as ring_buffer:
        first_offset, first_length, first_padding = ring_buffer.write(b"a" * 400)
        ring_buffer.advance_read_ptr(first_length, padding=first_padding)

        second_offset, second_length, second_padding = ring_buffer.write(b"b" * 300)
        ring_buffer.advance_read_ptr(second_length, padding=second_padding)

        wrapped_offset, wrapped_length, wrapped_padding = ring_buffer.write(b"c" * 250)

        assert wrapped_offset == 0
        assert wrapped_padding > 0
        assert bytes(ring_buffer.read(wrapped_offset, wrapped_length)) == b"c" * 250
        ring_buffer.advance_read_ptr(wrapped_length, padding=wrapped_padding)


def test_shm_ring_buffer_sequential_writes() -> None:
    """Validate sequential writes and reads preserve payload ordering."""

    with _make_ring_buffer() as ring_buffer:
        payloads = [b"one", b"two" * 10, b"three" * 20]
        positions = [ring_buffer.write(payload) for payload in payloads]

        for payload, (offset, length, padding) in zip(
            payloads, positions, strict=True
        ):
            assert bytes(ring_buffer.read(offset, length)) == payload
            ring_buffer.advance_read_ptr(length, padding=padding)


def test_shm_ring_buffer_tensor_roundtrip() -> None:
    """Validate tensor write/read round-trip with shape reconstruction."""

    with _make_ring_buffer() as ring_buffer:
        tensor = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        offset, length, padding = ring_buffer.write_tensor(tensor)
        recovered = ring_buffer.read_tensor(
            offset,
            length,
            list(tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
        )

        assert torch.equal(recovered, tensor)
        del recovered
        ring_buffer.advance_read_ptr(length, padding=padding)


def test_shm_ring_buffer_blocks_until_space_available() -> None:
    """Ensure writes wait until the reader advances when the ring is full."""

    with _make_ring_buffer(size=512) as ring_buffer:
        first = ring_buffer.write(b"a" * 160)
        second = ring_buffer.write(b"b" * 160)
        third_result: dict[str, tuple[int, int, int]] = {}
        finished = threading.Event()

        def _writer() -> None:
            third_result["value"] = ring_buffer.write(b"c" * 160)
            finished.set()

        thread = threading.Thread(target=_writer)
        thread.start()
        time.sleep(0.05)
        assert not finished.is_set()

        ring_buffer.advance_read_ptr(first[1], padding=first[2])
        thread.join(timeout=1.0)

        assert finished.is_set()
        third_offset, third_length, third_padding = third_result["value"]
        assert bytes(ring_buffer.read(third_offset, third_length)) == b"c" * 160
        ring_buffer.advance_read_ptr(second[1], padding=second[2])
        ring_buffer.advance_read_ptr(third_length, padding=third_padding)


def test_shm_ring_buffer_preserves_padding_for_repeated_offset_zero() -> None:
    """Ensure explicit padding keeps repeated offset-zero writes unambiguous."""

    with _make_ring_buffer(size=1024) as ring_buffer:
        first = ring_buffer.write(b"a" * 700)
        ring_buffer.advance_read_ptr(first[1], padding=first[2])

        wrapped = ring_buffer.write(b"b" * 400)
        assert wrapped[0] == 0
        assert wrapped[2] > 0
        ring_buffer.advance_read_ptr(wrapped[1], padding=wrapped[2])

        fill_to_boundary = ring_buffer.write(b"d" * 496)
        assert fill_to_boundary[0] == 400
        assert fill_to_boundary[2] == 0
        ring_buffer.advance_read_ptr(fill_to_boundary[1], padding=fill_to_boundary[2])

        at_zero_again = ring_buffer.write(b"c" * 100)
        assert at_zero_again[0] == 0
        assert at_zero_again[2] == 0
        assert bytes(ring_buffer.read(at_zero_again[0], at_zero_again[1])) == b"c" * 100
        ring_buffer.advance_read_ptr(at_zero_again[1], padding=at_zero_again[2])


def test_shm_transfer_metadata_msgpack_roundtrip() -> None:
    """Ensure transfer metadata is msgpack-serializable for the MQ protocol."""

    # First Party
    from lmcache.v1.multiprocess.shm_ring_buffer import ShmTransferMetadata

    metadata = ShmTransferMetadata(
        offsets=[1, 2],
        lengths=[16, 16],
        paddings=[0, 4],
        shape=[2, 2, 4],
        dtype="float16",
    )

    encoded = msgspec.msgpack.encode(metadata)
    decoded = msgspec.msgpack.decode(encoded, type=ShmTransferMetadata)

    assert decoded == metadata
