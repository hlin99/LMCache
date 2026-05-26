# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager import create_memory_allocator


def test_create_memory_allocator_rejects_shm_with_lazy() -> None:
    """create_memory_allocator rejects shm_name when lazy allocation is enabled."""
    config = L1MemoryManagerConfig(
        size_in_bytes=1024,
        use_lazy=False,
        init_size_in_bytes=1024,
        align_bytes=0x1000,
        shm_name="lmcache_l1_pool_test",
    )
    config.use_lazy = True

    with pytest.raises(
        ValueError,
        match="Shared memory mode \\(shm_name\\) is incompatible with lazy allocation",
    ):
        create_memory_allocator(config)
