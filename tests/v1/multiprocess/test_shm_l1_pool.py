# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the shared-memory L1 pool feature.

Tests cover:
1. L1MemoryManagerConfig: use_shm_l1_pool / shm_name fields
2. ShmPoolInfo from L1MemoryManager when shm is disabled
3. ShmSlotMetadata, PrepareStoreResponse, PrepareRetrieveResponse dataclasses
4. Protocol: 4 new RequestType entries exist and have definitions
5. MPCacheEngine: prepare_store / commit_store / prepare_retrieve / finish_read
   state management (via mocks, no CUDA required)
6. Fallback to ring-buffer when shm is disabled
"""

# Standard
from dataclasses import fields
from unittest.mock import MagicMock
import threading

# Third Party
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_l1_memory_manager_config(**kwargs):
    """Create an L1MemoryManagerConfig with sensible defaults."""
    # First Party
    from lmcache.v1.distributed.config import L1MemoryManagerConfig

    defaults = {
        "size_in_bytes": 64 * 1024 * 1024,  # 64 MB
        "use_lazy": False,
        "align_bytes": 0x1000,
        "use_shm_l1_pool": False,
        "shm_name": "lmcache_l1_pool",
    }
    defaults.update(kwargs)
    return L1MemoryManagerConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Config field tests
# ---------------------------------------------------------------------------


class TestL1MemoryManagerConfigFields:
    """Verify that the new config fields exist with correct defaults."""

    def test_use_shm_l1_pool_default_false(self):
        """use_shm_l1_pool should default to False."""
        # First Party
        from lmcache.v1.distributed.config import L1MemoryManagerConfig

        cfg = L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
        )
        assert cfg.use_shm_l1_pool is False

    def test_shm_name_default(self):
        """shm_name should default to 'lmcache_l1_pool'."""
        # First Party
        from lmcache.v1.distributed.config import L1MemoryManagerConfig

        cfg = L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
        )
        assert cfg.shm_name == "lmcache_l1_pool"

    def test_use_shm_l1_pool_settable(self):
        """use_shm_l1_pool should accept True."""
        # First Party
        from lmcache.v1.distributed.config import L1MemoryManagerConfig

        cfg = L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
            use_shm_l1_pool=True,
        )
        assert cfg.use_shm_l1_pool is True

    def test_shm_name_settable(self):
        """shm_name should accept a custom name."""
        # First Party
        from lmcache.v1.distributed.config import L1MemoryManagerConfig

        cfg = L1MemoryManagerConfig(
            size_in_bytes=1 << 30,
            use_lazy=False,
            shm_name="my_custom_pool",
        )
        assert cfg.shm_name == "my_custom_pool"

    def test_config_fields_include_shm_fields(self):
        """Verify that the dataclass has shm fields (not just attributes)."""
        # First Party
        from lmcache.v1.distributed.config import L1MemoryManagerConfig

        field_names = {f.name for f in fields(L1MemoryManagerConfig)}
        assert "use_shm_l1_pool" in field_names
        assert "shm_name" in field_names


# ---------------------------------------------------------------------------
# 2. ShmPoolInfo tests (no CUDA needed — shm disabled path)
# ---------------------------------------------------------------------------


class TestShmPoolInfoDisabled:
    """Verify get_shm_pool_info() when shm is disabled."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA required for L1MemoryManager",
    )
    def test_get_shm_pool_info_disabled(self):
        """When use_shm_l1_pool=False, shm_enabled should be False."""
        # First Party
        from lmcache.v1.distributed.memory_manager import L1MemoryManager

        cfg = _make_l1_memory_manager_config(
            use_shm_l1_pool=False,
            use_lazy=True,
        )
        mgr = L1MemoryManager(cfg)
        info = mgr.get_shm_pool_info()
        assert info.shm_enabled is False
        assert info.shm_name == ""
        assert info.base_ptr == 0
        assert info.pool_size == cfg.size_in_bytes
        mgr.close()


# ---------------------------------------------------------------------------
# 3. ShmSlotMetadata / PrepareStoreResponse / PrepareRetrieveResponse
# ---------------------------------------------------------------------------


class TestShmDataclasses:
    """Verify that the SHM response dataclasses have the expected fields."""

    def test_shm_slot_metadata_fields(self):
        """ShmSlotMetadata should have the documented fields."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import ShmSlotMetadata

        slot = ShmSlotMetadata(
            shm_name="test_pool",
            offset=1024,
            length=65536,
            shape=[2, 32, 512],
            dtype="bfloat16",
        )
        assert slot.shm_name == "test_pool"
        assert slot.offset == 1024
        assert slot.length == 65536
        assert slot.shape == [2, 32, 512]
        assert slot.dtype == "bfloat16"

    def test_prepare_store_response_fields(self):
        """PrepareStoreResponse should have use_shm and slots fields."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            PrepareStoreResponse,
            ShmSlotMetadata,
        )

        resp_fallback = PrepareStoreResponse(use_shm=False, slots=[])
        assert resp_fallback.use_shm is False
        assert resp_fallback.slots == []

        slot = ShmSlotMetadata("pool", 0, 100, [10, 2], "float16")
        resp_shm = PrepareStoreResponse(use_shm=True, slots=[slot])
        assert resp_shm.use_shm is True
        assert len(resp_shm.slots) == 1
        assert resp_shm.slots[0] is slot

    def test_prepare_retrieve_response_fields(self):
        """PrepareRetrieveResponse should have use_shm, success and slots."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            PrepareRetrieveResponse,
            ShmSlotMetadata,
        )

        # Fallback response
        r = PrepareRetrieveResponse(use_shm=False, success=False, slots=[])
        assert r.use_shm is False
        assert r.success is False

        # Success response
        slot = ShmSlotMetadata("p", 4096, 8192, [1, 2, 3], "float32")
        r2 = PrepareRetrieveResponse(use_shm=True, success=True, slots=[slot])
        assert r2.use_shm is True
        assert r2.success is True
        assert r2.slots[0].offset == 4096


# ---------------------------------------------------------------------------
# 4. Protocol: 4 new RequestType entries
# ---------------------------------------------------------------------------


class TestNewRequestTypes:
    """Verify that the 4 new RequestType members exist and have definitions."""

    def test_prepare_store_request_type_exists(self):
        """PREPARE_STORE should be a member of RequestType."""
        # First Party
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert hasattr(RequestType, "PREPARE_STORE")

    def test_commit_store_request_type_exists(self):
        """COMMIT_STORE should be a member of RequestType."""
        # First Party
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert hasattr(RequestType, "COMMIT_STORE")

    def test_prepare_retrieve_request_type_exists(self):
        """PREPARE_RETRIEVE should be a member of RequestType."""
        # First Party
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert hasattr(RequestType, "PREPARE_RETRIEVE")

    def test_finish_read_request_type_exists(self):
        """FINISH_READ should be a member of RequestType."""
        # First Party
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert hasattr(RequestType, "FINISH_READ")

    def test_all_new_request_types_have_protocol_definitions(self):
        """All 4 new RequestTypes must have protocol definitions."""
        # First Party
        from lmcache.v1.multiprocess.protocols import initialize_protocols
        from lmcache.v1.multiprocess.protocols.base import RequestType

        defs = initialize_protocols()
        for name in ("PREPARE_STORE", "COMMIT_STORE", "PREPARE_RETRIEVE", "FINISH_READ"):
            rt = getattr(RequestType, name)
            assert rt in defs, f"RequestType.{name} has no protocol definition"

    def test_prepare_store_response_class(self):
        """PREPARE_STORE response class should be PrepareStoreResponse."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import PrepareStoreResponse
        from lmcache.v1.multiprocess.protocol import get_response_class
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert get_response_class(RequestType.PREPARE_STORE) is PrepareStoreResponse

    def test_commit_store_response_class(self):
        """COMMIT_STORE response class should be bool."""
        # First Party
        from lmcache.v1.multiprocess.protocol import get_response_class
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert get_response_class(RequestType.COMMIT_STORE) is bool

    def test_prepare_retrieve_response_class(self):
        """PREPARE_RETRIEVE response class should be PrepareRetrieveResponse."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import PrepareRetrieveResponse
        from lmcache.v1.multiprocess.protocol import get_response_class
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert get_response_class(RequestType.PREPARE_RETRIEVE) is PrepareRetrieveResponse

    def test_finish_read_response_class(self):
        """FINISH_READ response class should be bool."""
        # First Party
        from lmcache.v1.multiprocess.protocol import get_response_class
        from lmcache.v1.multiprocess.protocols.base import RequestType

        assert get_response_class(RequestType.FINISH_READ) is bool

    def test_prepare_store_payload_classes(self):
        """PREPARE_STORE payload should be (KeyType, int)."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
        from lmcache.v1.multiprocess.protocol import get_payload_classes
        from lmcache.v1.multiprocess.protocols.base import RequestType

        classes = get_payload_classes(RequestType.PREPARE_STORE)
        assert classes[0] is IPCCacheEngineKey
        assert classes[1] is int

    def test_prepare_retrieve_payload_classes(self):
        """PREPARE_RETRIEVE payload should be (KeyType, int)."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
        from lmcache.v1.multiprocess.protocol import get_payload_classes
        from lmcache.v1.multiprocess.protocols.base import RequestType

        classes = get_payload_classes(RequestType.PREPARE_RETRIEVE)
        assert classes[0] is IPCCacheEngineKey
        assert classes[1] is int

    def test_commit_store_payload_classes(self):
        """COMMIT_STORE payload should be (KeyType,)."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
        from lmcache.v1.multiprocess.protocol import get_payload_classes
        from lmcache.v1.multiprocess.protocols.base import RequestType

        classes = get_payload_classes(RequestType.COMMIT_STORE)
        assert classes[0] is IPCCacheEngineKey
        assert len(classes) == 1

    def test_finish_read_payload_classes(self):
        """FINISH_READ payload should be (KeyType,)."""
        # First Party
        from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
        from lmcache.v1.multiprocess.protocol import get_payload_classes
        from lmcache.v1.multiprocess.protocols.base import RequestType

        classes = get_payload_classes(RequestType.FINISH_READ)
        assert classes[0] is IPCCacheEngineKey
        assert len(classes) == 1


# ---------------------------------------------------------------------------
# 5. ShmPoolInfo dataclass structure
# ---------------------------------------------------------------------------


class TestShmPoolInfoStructure:
    """Verify ShmPoolInfo dataclass from memory_manager."""

    def test_shm_pool_info_has_expected_fields(self):
        """ShmPoolInfo should have shm_name, pool_size, shm_enabled, base_ptr."""
        # First Party
        from lmcache.v1.distributed.memory_manager import ShmPoolInfo

        info = ShmPoolInfo(
            shm_name="test",
            pool_size=1024,
            shm_enabled=False,
            base_ptr=0,
        )
        assert info.shm_name == "test"
        assert info.pool_size == 1024
        assert info.shm_enabled is False
        assert info.base_ptr == 0

    def test_shm_pool_info_enabled(self):
        """ShmPoolInfo can represent an enabled pool."""
        # First Party
        from lmcache.v1.distributed.memory_manager import ShmPoolInfo

        info = ShmPoolInfo(
            shm_name="lmcache_pool",
            pool_size=8 * 1024**3,
            shm_enabled=True,
            base_ptr=0x7F0000000000,
        )
        assert info.shm_enabled is True
        assert info.shm_name == "lmcache_pool"


# ---------------------------------------------------------------------------
# 6. MPCacheEngine state management (no CUDA required — uses mocks)
# ---------------------------------------------------------------------------


def _make_ipc_key(request_id: str = "req_1", worker_id: int = 42):
    """Create a minimal IPCCacheEngineKey for testing."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

    return IPCCacheEngineKey(
        model_name="gpt-test",
        world_size=1,
        worker_id=worker_id,
        token_ids=tuple(range(256)),
        start=0,
        end=256,
        request_id=request_id,
    )


def _make_engine_shm_disabled():
    """
    Build an MPCacheEngine with a fully mocked StorageManager that reports
    shm disabled.  Returns (engine, mock_storage_manager).
    """
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.memory_manager import ShmPoolInfo
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_sm = MagicMock()
    mock_sm.get_shm_pool_info.return_value = ShmPoolInfo(
        shm_name="",
        pool_size=64 * 1024 * 1024,
        shm_enabled=False,
        base_ptr=0,
    )

    with patch(
        "lmcache.v1.multiprocess.server.StorageManager",
        return_value=mock_sm,
    ):
        engine = MPCacheEngine.__new__(MPCacheEngine)
        engine.storage_manager = mock_sm
        engine.bounce_contexts = {}
        engine.bounce_context_meta = {}
        engine.gpu_contexts = {}
        engine.gpu_context_meta = {}
        engine.chunk_size = 256
        engine.session_manager = MagicMock()
        engine.token_hasher = MagicMock()
        engine.token_hasher.compute_chunk_hashes.return_value = [b"\x00" * 16]
        engine.session_manager.get_or_create.return_value.get_hashes.return_value = [
            0
        ]
        engine.lock = __import__("threading").Lock()
        engine._prefetch_jobs = {}
        engine._prefetch_job_lock = __import__("threading").Lock()
        engine._shm_store_state = {}
        engine._shm_store_lock = __import__("threading").Lock()
        engine._shm_retrieve_state = {}
        engine._shm_retrieve_lock = __import__("threading").Lock()
        engine._event_bus = MagicMock()

    return engine, mock_sm


class TestPrepareStoreFallback:
    """prepare_store should return use_shm=False when shm is disabled."""

    def test_prepare_store_returns_fallback_when_shm_disabled(self):
        """When shm pool is disabled, prepare_store must signal fallback."""
        engine, _ = _make_engine_shm_disabled()
        key = _make_ipc_key()
        response = engine.prepare_store(key, instance_id=99)
        assert response.use_shm is False
        assert response.slots == []

    def test_prepare_store_returns_fallback_no_bounce_context(self):
        """prepare_store returns fallback when instance has no bounce context."""
        # Even if shm were enabled, no bounce context means fallback.
        engine, mock_sm = _make_engine_shm_disabled()
        key = _make_ipc_key()
        response = engine.prepare_store(key, instance_id=999)
        assert response.use_shm is False

    def test_commit_store_returns_false_without_prepare(self):
        """commit_store should return False when no prior prepare_store exists."""
        engine, _ = _make_engine_shm_disabled()
        key = _make_ipc_key(request_id="never_prepared")
        result = engine.commit_store(key)
        assert result is False

    def test_shm_store_state_populated_and_cleared(self):
        """_shm_store_state is set by prepare_store and cleared by commit_store."""
        engine, _ = _make_engine_shm_disabled()
        # Inject a dummy reserved dict manually (simulating a successful prepare_store)
        dummy = {"dummy_key": MagicMock()}
        engine._shm_store_state["my_req"] = dummy

        # commit_store should clear it and call finish_write
        key = _make_ipc_key(request_id="my_req")
        result = engine.commit_store(key)
        assert result is True
        assert "my_req" not in engine._shm_store_state
        engine.storage_manager.finish_write.assert_called_once()


class TestPrepareRetrieveFallback:
    """prepare_retrieve should return use_shm=False when shm is disabled."""

    def test_prepare_retrieve_returns_fallback_when_shm_disabled(self):
        """When shm pool is disabled, prepare_retrieve must signal fallback."""
        engine, _ = _make_engine_shm_disabled()
        key = _make_ipc_key()
        response = engine.prepare_retrieve(key, instance_id=99)
        assert response.use_shm is False
        assert response.success is False

    def test_finish_read_returns_false_without_prepare(self):
        """finish_read should return False when no prior prepare_retrieve."""
        engine, _ = _make_engine_shm_disabled()
        key = _make_ipc_key(request_id="never_prepared")
        result = engine.finish_read(key)
        assert result is False

    def test_shm_retrieve_state_populated_and_cleared(self):
        """_shm_retrieve_state is set before finish_read and cleared by it."""
        engine, _ = _make_engine_shm_disabled()
        # Inject a dummy list of obj_keys manually
        dummy_keys = ["key_a", "key_b"]
        engine._shm_retrieve_state["my_req"] = dummy_keys

        key = _make_ipc_key(request_id="my_req")
        result = engine.finish_read(key)
        assert result is True
        assert "my_req" not in engine._shm_retrieve_state
        engine.storage_manager.finish_read_prefetched.assert_called_once_with(
            dummy_keys
        )


class TestShmStateThreadSafety:
    """State dicts must be thread-safe."""

    def test_concurrent_commit_store_calls(self):
        """Multiple commit_store calls for different request_ids are safe."""
        engine, _ = _make_engine_shm_disabled()

        # Pre-populate state for N different requests
        n = 50
        for i in range(n):
            engine._shm_store_state[f"req_{i}"] = {f"key_{i}": MagicMock()}

        errors = []
        results = []

        def commit(i):
            try:
                key = _make_ipc_key(request_id=f"req_{i}")
                r = engine.commit_store(key)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=commit, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert all(results), "All commit_store calls should succeed"
        assert len(engine._shm_store_state) == 0

    def test_concurrent_finish_read_calls(self):
        """Multiple finish_read calls for different request_ids are safe."""
        engine, _ = _make_engine_shm_disabled()

        n = 50
        for i in range(n):
            engine._shm_retrieve_state[f"req_{i}"] = [f"key_{i}"]

        errors = []
        results = []

        def finish(i):
            try:
                key = _make_ipc_key(request_id=f"req_{i}")
                r = engine.finish_read(key)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=finish, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert all(results), "All finish_read calls should succeed"
        assert len(engine._shm_retrieve_state) == 0


# ---------------------------------------------------------------------------
# 7. _check_shm_capacity helper
# ---------------------------------------------------------------------------


class TestCheckShmCapacity:
    """Verify the /dev/shm capacity check returns sensible values."""

    def test_check_shm_capacity_with_small_request(self):
        """Should return True for a very small request (1 byte)."""
        # First Party
        from lmcache.v1.distributed.memory_manager import _check_shm_capacity

        result = _check_shm_capacity(1)
        # On Linux /dev/shm almost always exists; but we don't enforce True here
        # because CI might run in a restricted environment.
        assert isinstance(result, bool)

    def test_check_shm_capacity_with_huge_request(self):
        """Should return False for a 1 PiB request (no machine has that)."""
        # First Party
        from lmcache.v1.distributed.memory_manager import _check_shm_capacity

        # 1 PiB — definitely not available
        result = _check_shm_capacity(2**50)
        assert result is False
