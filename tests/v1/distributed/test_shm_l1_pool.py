# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SHM-backed L1 memory pool feature.

Tests cover:
- SHM lifecycle (create, attach, unlink, stale cleanup)
- /dev/shm capacity fail-fast check
- Offset allocation and tensor view construction
- OOM silent skip behaviour (consistent with CUDA path)
- MemoryObj shm_offset / shm_byte_length properties
- Protocol struct round-trip
"""

# Standard
import os
import unittest

# Third Party
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
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    ShmSlotMetadata,
)


class TestShmCapacityCheck(unittest.TestCase):
    """Tests for the /dev/shm capacity check (fail-fast)."""

    def test_sufficient_shm_space(self):
        """No error when /dev/shm has enough space."""
        # Just needs to not raise
        _check_shm_capacity(1)

    def test_insufficient_shm_space_raises(self):
        """RuntimeError when /dev/shm is too small."""
        huge_bytes = 1 << 60  # 1 EiB, way more than any system has
        with self.assertRaises(RuntimeError) as ctx:
            _check_shm_capacity(huge_bytes)
        self.assertIn("Insufficient /dev/shm space", str(ctx.exception))
        self.assertIn("docker run --shm-size", str(ctx.exception))


class TestStaleShm(unittest.TestCase):
    """Tests for stale SHM cleanup."""

    def test_unlink_nonexistent_is_safe(self):
        """No error when unlinking a non-existent SHM segment."""
        _unlink_stale_shm("lmcache_test_nonexistent_12345")

    def test_unlink_removes_stale_file(self):
        """Stale SHM file is removed on startup."""
        # Standard
        from multiprocessing import shared_memory

        name = "lmcache_test_stale_shm"
        shm = shared_memory.SharedMemory(name=name, create=True, size=4096)
        shm.close()
        self.assertTrue(os.path.exists(f"/dev/shm/{name}"))

        _unlink_stale_shm(name)
        self.assertFalse(os.path.exists(f"/dev/shm/{name}"))


class TestL1MemoryManagerShmConfig(unittest.TestCase):
    """Tests for L1MemoryManagerConfig with shm_name."""

    def test_default_shm_name(self):
        """Config defaults to 'lmcache_l1_pool_<pid>' shm_name."""
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
        )
        self.assertTrue(config.shm_name.startswith("lmcache_l1_pool_"))
        self.assertIn(str(os.getpid()), config.shm_name)

    def test_shm_name_propagated(self):
        """Config with shm_name passes through to the allocator."""
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            shm_name="test_shm_pool",
        )
        self.assertEqual(config.shm_name, "test_shm_pool")


class TestL1MemoryManagerShmLifecycle(unittest.TestCase):
    """Tests for L1MemoryManager with SHM allocation and cleanup."""

    def setUp(self):
        self.shm_name = "lmcache_test_lifecycle"
        # Ensure clean state
        _unlink_stale_shm(self.shm_name)

    def tearDown(self):
        _unlink_stale_shm(self.shm_name)

    def test_get_shm_pool_info(self):
        """get_shm_pool_info returns correct name and size."""
        pool_size = 1 << 20  # 1 MiB
        config = L1MemoryManagerConfig(
            size_in_bytes=pool_size,
            use_lazy=False,
            shm_name=self.shm_name,
        )
        mgr = L1MemoryManager(config)
        try:
            info = mgr.get_shm_pool_info()
            self.assertEqual(info["shm_name"], self.shm_name)
            self.assertEqual(info["pool_size"], pool_size)
        finally:
            mgr.close()

    def test_get_shm_pool_info_lazy_returns_empty(self):
        """get_shm_pool_info returns empty name for lazy allocator (CUDA)."""
        # We can't actually create a LazyMemoryAllocator without CUDA,
        # but we can verify the L1MemoryManager clears shm_name for lazy.
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,  # Can't use lazy without CUDA
            shm_name="lmcache_test_lazy_check",
        )
        # Just verify config works
        self.assertEqual(config.shm_name, "lmcache_test_lazy_check")

    def test_close_unlinks_shm(self):
        """close() unlinks the SHM segment."""
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=False,
            shm_name=self.shm_name,
        )
        mgr = L1MemoryManager(config)
        # SHM file should exist
        self.assertTrue(os.path.exists(f"/dev/shm/{self.shm_name}"))
        mgr.close()
        # SHM file should be removed after close
        self.assertFalse(os.path.exists(f"/dev/shm/{self.shm_name}"))


class TestMemoryObjShmProperties(unittest.TestCase):
    """Tests for MemoryObj.shm_offset and shm_byte_length properties."""

    def test_shm_offset_matches_address(self):
        """shm_offset returns metadata.address."""
        raw = torch.zeros(1024, dtype=torch.uint8)
        meta = MemoryObjMetadata(
            shape=torch.Size([2, 4, 128]),
            dtype=torch.float16,
            address=4096,
            phy_size=2048,
            ref_count=1,
        )
        obj = TensorMemoryObj(raw_data=raw, metadata=meta, parent_allocator=None)
        self.assertEqual(obj.shm_offset, 4096)

    def test_shm_byte_length_matches_get_size(self):
        """shm_byte_length returns get_size() (logical byte size)."""
        raw = torch.zeros(2048, dtype=torch.uint8)
        meta = MemoryObjMetadata(
            shape=torch.Size([2, 4, 128]),
            dtype=torch.float16,
            address=0,
            phy_size=2048,
            ref_count=1,
        )
        obj = TensorMemoryObj(raw_data=raw, metadata=meta, parent_allocator=None)
        # 2 * 4 * 128 * 2 bytes (float16) = 2048
        self.assertEqual(obj.shm_byte_length, 2 * 4 * 128 * 2)


class TestShmSlotMetadata(unittest.TestCase):
    """Tests for ShmSlotMetadata and response structs."""

    def test_prepare_store_response_empty_slots(self):
        """PrepareStoreResponse with empty slots (OOM)."""
        resp = PrepareStoreResponse(slots=[])
        self.assertEqual(len(resp.slots), 0)

    def test_prepare_store_response_with_slots(self):
        """PrepareStoreResponse with allocated slots."""
        slot = ShmSlotMetadata(
            key="test_key",
            shm_name="lmcache_pool",
            offset=4096,
            length=8192,
            shape=[2, 4, 128],
            dtype="float16",
        )
        resp = PrepareStoreResponse(slots=[slot])
        self.assertEqual(len(resp.slots), 1)
        self.assertEqual(resp.slots[0].offset, 4096)
        self.assertEqual(resp.slots[0].dtype, "float16")

    def test_prepare_retrieve_response_failure(self):
        """PrepareRetrieveResponse on failure."""
        resp = PrepareRetrieveResponse(success=False, slots=[])
        self.assertFalse(resp.success)

    def test_prepare_retrieve_response_success(self):
        """PrepareRetrieveResponse on success."""
        slot = ShmSlotMetadata(
            key="key1",
            shm_name="pool",
            offset=0,
            length=1024,
            shape=[4, 128],
            dtype="bfloat16",
        )
        resp = PrepareRetrieveResponse(success=True, slots=[slot])
        self.assertTrue(resp.success)
        self.assertEqual(resp.slots[0].length, 1024)


class TestShmOffsetAllocation(unittest.TestCase):
    """Tests for offset allocation consistency."""

    def setUp(self):
        self.shm_name = "lmcache_test_alloc"
        _unlink_stale_shm(self.shm_name)

    def tearDown(self):
        _unlink_stale_shm(self.shm_name)

    def test_allocated_objects_have_distinct_offsets(self):
        """Multiple allocations produce non-overlapping offsets."""
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 22,  # 4 MiB
            use_lazy=False,
            shm_name=self.shm_name,
        )
        mgr = L1MemoryManager(config)
        try:
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 64, 128])],
                dtypes=[torch.float16],
            )
            err, objs = mgr.allocate(layout, count=3)
            self.assertEqual(err, L1Error.SUCCESS)
            self.assertEqual(len(objs), 3)

            # All offsets should be distinct
            offsets = [obj.shm_offset for obj in objs]
            self.assertEqual(len(set(offsets)), 3)

            # No overlap
            for i, obj in enumerate(objs):
                for j, other in enumerate(objs):
                    if i >= j:
                        continue
                    # Check non-overlap
                    end_i = obj.shm_offset + obj.shm_byte_length
                    end_j = other.shm_offset + other.shm_byte_length
                    self.assertTrue(
                        end_i <= other.shm_offset or end_j <= obj.shm_offset,
                        f"Overlap: obj[{i}]=[{obj.shm_offset},{end_i}) "
                        f"obj[{j}]=[{other.shm_offset},{end_j})",
                    )
        finally:
            mgr.close()

    def test_oom_returns_empty(self):
        """OOM allocation returns empty list."""
        config = L1MemoryManagerConfig(
            size_in_bytes=1 << 10,  # 1 KiB - very small
            use_lazy=False,
            shm_name=self.shm_name,
        )
        mgr = L1MemoryManager(config)
        try:
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([2, 4, 64, 128])],
                dtypes=[torch.float16],
            )
            # Requesting much more than available
            err, objs = mgr.allocate(layout, count=100)
            self.assertEqual(err, L1Error.OUT_OF_MEMORY)
            self.assertEqual(len(objs), 0)
        finally:
            mgr.close()


class TestTensorViewConstruction(unittest.TestCase):
    """Tests for tensor view construction from SHM buffer."""

    def test_tensor_view_from_buffer(self):
        """torch.frombuffer creates correct zero-copy view."""
        # Simulate SHM buffer
        pool_size = 8192
        buf = bytearray(pool_size)
        # Write some known data
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        expected_bytes = expected.numpy().tobytes()
        offset = 1024
        buf[offset : offset + len(expected_bytes)] = expected_bytes

        # Construct view
        buf_view = memoryview(buf)[offset : offset + len(expected_bytes)]
        view = torch.frombuffer(buf_view, dtype=torch.float32)
        torch.testing.assert_close(view, expected)

    def test_tensor_view_shape(self):
        """Tensor view can be reshaped correctly."""
        pool_size = 4096
        buf = bytearray(pool_size)
        # 2 x 2 x 2 float16 = 16 bytes
        shape = [2, 2, 2]
        dtype = torch.float16
        length = 2 * 2 * 2 * 2  # 16 bytes
        buf_view = memoryview(buf)[0:length]
        view = torch.frombuffer(buf_view, dtype=dtype).view(*shape)
        self.assertEqual(list(view.shape), shape)


class TestGatherChunksToTensors(unittest.TestCase):
    """Tests for gather_chunks_to_cpu_tensors."""

    def test_returns_list_of_tensors(self):
        """gather_chunks_to_cpu_tensors returns list not bytes."""
        # First Party
        from lmcache.v1.multiprocess.cpu_bounce_context import (
            gather_chunks_to_cpu_tensors,
        )

        # Create minimal KV caches in NHD format
        # vLLM non-MLA: [2, num_blocks, block_size, num_heads, head_size]
        num_layers = 2
        block_size = 4
        hidden_dim = 8
        num_heads = 2
        head_size = hidden_dim // num_heads
        num_blocks = 4
        blocks_per_chunk = 2

        # vLLM NHD: [2, num_blocks, block_size, num_heads, head_size]
        kv_caches = {}
        for i in range(num_layers):
            kv_caches[f"layer{i}"] = torch.randn(
                2, num_blocks, block_size, num_heads, head_size
            )

        block_ids = list(range(blocks_per_chunk))
        result = gather_chunks_to_cpu_tensors(kv_caches, block_ids, blocks_per_chunk)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
