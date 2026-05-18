# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the SHM-backed L1 pool used by non-CUDA MP mode."""

# Standard
from multiprocessing import shared_memory
import fcntl
import os

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import (
    L1MemoryManager,
    _check_shm_capacity,
    _unlink_stale_shm,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    ShmSlotMetadata,
)


def test_l1_memory_manager_config_defaults_shm_name() -> None:
    """The default SHM name should include the current process id."""
    config = L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False)
    assert config.shm_name.startswith("lmcache_l1_pool_")
    assert str(os.getpid()) in config.shm_name


def test_check_shm_capacity_raises_when_insufficient() -> None:
    """A clearly impossible SHM request should fail fast."""
    with pytest.raises(RuntimeError, match="Insufficient /dev/shm space"):
        _check_shm_capacity(1 << 60)


def test_unlink_stale_shm_removes_unlocked_segment() -> None:
    """Unlocked stale SHM files should be removed."""
    shm_name = "lmcache_l1_pool_test_stale_remove"
    _unlink_stale_shm(shm_name)

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=4096)
    shm.close()
    shm_path = f"/dev/shm/{shm_name}"
    assert os.path.exists(shm_path)

    _unlink_stale_shm(shm_name)
    assert not os.path.exists(shm_path)


def test_unlink_stale_shm_skips_locked_segment() -> None:
    """Locked SHM files should not be removed until their flock is released."""
    shm_name = "lmcache_l1_pool_test_locked_keep"
    _unlink_stale_shm(shm_name)

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=4096)
    shm.close()
    shm_path = f"/dev/shm/{shm_name}"
    fd = os.open(shm_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _unlink_stale_shm(shm_name)
        assert os.path.exists(shm_path)
    finally:
        os.close(fd)

    _unlink_stale_shm(shm_name)
    assert not os.path.exists(shm_path)


def test_l1_memory_manager_reports_and_cleans_shm_pool() -> None:
    """The manager should expose SHM info and unlink the pool on close."""
    shm_name = "lmcache_l1_pool_test_manager_info"
    _unlink_stale_shm(shm_name)
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            shm_name=shm_name,
        )
    )
    try:
        info = manager.get_shm_pool_info()
        assert info == {"shm_name": shm_name, "pool_size": 1 << 20}
        assert os.path.exists(f"/dev/shm/{shm_name}")
    finally:
        manager.close()

    assert not os.path.exists(f"/dev/shm/{shm_name}")


def test_memory_obj_exposes_shm_offset_and_byte_length() -> None:
    """Allocated L1 objects should expose SHM offset metadata."""
    shm_name = "lmcache_l1_pool_test_object_props"
    _unlink_stale_shm(shm_name)
    manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            shm_name=shm_name,
        )
    )
    try:
        error, objects = manager.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.float32]),
            1,
        )
        assert error == L1Error.SUCCESS
        assert len(objects) == 1
        obj = objects[0]
        element_size = torch.tensor([], dtype=torch.float32).element_size()
        assert obj.shm_offset >= 0
        assert obj.shm_byte_length == 2 * 4 * element_size
        manager.free(objects)
    finally:
        manager.close()


def test_shm_protocol_structs_roundtrip() -> None:
    """SHM protocol response structs should msgpack round-trip cleanly."""
    payload = PrepareStoreResponse(
        slots=[
            ShmSlotMetadata(
                key="k",
                shm_name="pool",
                offset=64,
                length=128,
                shape=[2, 4],
                dtype="float32",
                chunk_index=1,
            )
        ]
    )
    encoded = msgspec.msgpack.encode(payload)
    decoded = msgspec.msgpack.decode(encoded, type=PrepareStoreResponse)
    assert decoded == payload

    retrieve_payload = PrepareRetrieveResponse(success=True, slots=payload.slots)
    encoded_retrieve = msgspec.msgpack.encode(retrieve_payload)
    decoded_retrieve = msgspec.msgpack.decode(
        encoded_retrieve, type=PrepareRetrieveResponse
    )
    assert decoded_retrieve == retrieve_payload
