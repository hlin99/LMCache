# SPDX-License-Identifier: Apache-2.0
"""
Unit tests reproducing PDBackendAsync race conditions with shared prefix keys.

These tests directly exercise the PDBackendAsync.put(), get_blocking(), and
remove() methods to demonstrate two critical bugs:

Bug 1 (put overwrite releases in-flight buffer):
    When two requests share the same CacheEngineKey, the second put(K, obj_B)
    releases obj_A via ref_count_down(). If Sender A is still writing to
    obj_A.meta.address via RDMA, this is a use-after-free.

Bug 2 (remove_after_retrieve deletes another request's data):
    After Req A retrieves and removes key K, Req B's subsequent
    get_blocking(K) fails with AssertionError because the data dict
    no longer contains K.

No GPU, RDMA, or ZMQ required — we mock the allocator and directly
call the data-path methods.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(chunk_hash: int, worker_id: int = 0) -> CacheEngineKey:
    """Create a CacheEngineKey with a specific chunk_hash."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


def _make_memory_obj(address: int) -> MemoryObj:
    """Create a lightweight mock MemoryObj with a distinguishable address."""
    meta = MagicMock(spec=MemoryObjMetadata)
    meta.address = address
    meta.fmt = MemoryFormat.KV_2LTD
    meta.shape = torch.Size([2, 1, 256, 128])
    meta.dtype = torch.float16

    obj = MagicMock(spec=MemoryObj)
    obj.meta = meta
    obj.get_ref_count.return_value = 1
    obj.get_size.return_value = 131072
    # Track ref_count_down calls
    obj._freed = False

    def _ref_down():
        obj._freed = True

    obj.ref_count_down.side_effect = _ref_down
    return obj


def _make_pd_backend_data_dict():
    """
    Create a minimal stand-in for PDBackendAsync's data dict and lock,
    along with the put/get_blocking/remove/contains methods copied from
    the real implementation. This avoids needing to instantiate the full
    PDBackendAsync (which requires ZMQ, transfer channels, etc).
    """

    class FakePDBackendDataPath:
        """Mimics PDBackendAsync data-path methods exactly."""

        def __init__(self):
            self.data: dict[CacheEngineKey, MemoryObj] = {}
            self.data_lock = threading.Lock()

        def put(self, key: CacheEngineKey, mem_obj: MemoryObj) -> None:
            # Exact copy from pd_backend_async.py:1417-1440
            with self.data_lock:
                old = self.data.pop(key, None)
                if old is not None:
                    old.ref_count_down()
                self.data[key] = mem_obj

        def get_blocking(self, key: CacheEngineKey):
            # Exact copy from pd_backend_async.py:1442-1458
            with self.data_lock:
                mem_obj = self.data.get(key, None)
                assert mem_obj is not None, f"Key {key} not found in local data."
                return mem_obj

        def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
            # Exact copy from pd_backend_async.py:632-648
            with self.data_lock:
                if mem_obj := self.data.get(key, None):
                    if pin:
                        mem_obj.ref_count_up()
                    return True
                return False

        def remove(self, key: CacheEngineKey) -> bool:
            # Exact copy from pd_backend_async.py:1460-1502 (simplified)
            with self.data_lock:
                mem_obj = self.data.pop(key, None)
                if mem_obj is not None:
                    mem_obj.ref_count_down()
                    return True
                return False

    return FakePDBackendDataPath()


# ---------------------------------------------------------------------------
# Bug 1: put() overwrites in-flight buffer (use-after-free)
# ---------------------------------------------------------------------------


class TestPutOverwriteReleasesInflightBuffer:
    """
    Demonstrates that when two AllocRequests produce the same key K,
    the second put(K, obj_B) frees obj_A's buffer while the first sender
    may still be writing to it via RDMA.
    """

    def test_put_overwrites_and_frees_old_obj(self):
        """
        Scenario:
          1. Receiver allocates obj_A for Req A → put(K, obj_A)
          2. Receiver allocates obj_B for Req B → put(K, obj_B)
          3. obj_A.ref_count_down() is called → buffer freed
          4. Sender A is still doing RDMA to obj_A.meta.address → UAF!

        This test verifies that put(K, obj_B) does indeed free obj_A.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=12345)

        obj_a = _make_memory_obj(address=0x1000)
        obj_b = _make_memory_obj(address=0x2000)

        # Step 1: Receiver handles AllocRequest for Req A
        backend.put(key, obj_a)
        assert backend.contains(key)
        assert not obj_a._freed, "obj_A should NOT be freed yet"

        # Step 2: Receiver handles AllocRequest for Req B (same key!)
        backend.put(key, obj_b)

        # Step 3: Verify obj_A was freed (use-after-free!)
        assert obj_a._freed, (
            "BUG: obj_A was freed by put() even though Sender A's RDMA "
            "may still be writing to obj_A.meta.address (0x1000). "
            "This is a use-after-free."
        )

        # The data dict now points to obj_B
        result = backend.get_blocking(key)
        assert result is obj_b

    def test_concurrent_put_from_two_threads(self):
        """
        Simulate two AllocRequest handlers running concurrently for the
        same key (as would happen with shared prefix in xP1D topology).
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=99999)

        obj_a = _make_memory_obj(address=0xA000)
        obj_b = _make_memory_obj(address=0xB000)

        barrier = threading.Barrier(2)

        def put_a():
            barrier.wait()
            backend.put(key, obj_a)

        def put_b():
            barrier.wait()
            # Small delay to increase chance of interleaving
            time.sleep(0.001)
            backend.put(key, obj_b)

        t1 = threading.Thread(target=put_a)
        t2 = threading.Thread(target=put_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # One of them was freed; this demonstrates the race
        freed_count = sum([obj_a._freed, obj_b._freed])
        # At least one object was freed (the loser of the race)
        # In practice obj_a is freed since obj_b's put() comes second
        assert freed_count >= 1, (
            "Expected at least one MemoryObj to be freed due to overwrite"
        )


# ---------------------------------------------------------------------------
# Bug 2: remove_after_retrieve causes get_blocking AssertionError
# ---------------------------------------------------------------------------


class TestRemoveAfterRetrieveCausesGetBlockingFailure:
    """
    Demonstrates the crash: Req A's remove(K) after retrieve deletes the
    entry that Req B needs for its get_blocking(K).

    This is the exact scenario from the bug report:
      AssertionError: Key CacheEngineKey(...) not found in local data.
    """

    def test_remove_then_get_blocking_asserts(self):
        """
        Scenario (single-threaded, deterministic):
          1. put(K, obj_A) — Req A's data arrives
          2. contains(K) → True — Req B's lookup sees the key
          3. get_blocking(K) → obj_A — Req A's retrieve succeeds
          4. remove(K) — Req A's remove_after_retrieve
          5. get_blocking(K) → AssertionError! — Req B's retrieve fails

        Steps 2-5 happen on the decoder side. The key point is that
        Req B's lookup (step 2) happened before Req A's remove (step 4),
        but Req B's get_blocking (step 5) happens after.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=12345)

        obj_a = _make_memory_obj(address=0x1000)

        # Step 1: Data arrives for Req A
        backend.put(key, obj_a)

        # Step 2: Req B's lookup — sees the key exists
        assert backend.contains(key), "Req B's lookup should see key K"

        # Step 3: Req A's retrieve
        result = backend.get_blocking(key)
        assert result is obj_a

        # Step 4: Req A's remove_after_retrieve
        backend.remove(key)

        # Step 5: Req B's retrieve — CRASH
        with pytest.raises(AssertionError, match="not found in local data"):
            backend.get_blocking(key)

    def test_concurrent_retrieve_and_remove(self):
        """
        More realistic: two threads racing — one removes while the other
        tries get_blocking. The assertion failure is non-deterministic but
        this test maximizes the chance of hitting it.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=55555)

        errors: list[Exception] = []
        N_ITERATIONS = 200

        def retrieve_and_remove():
            """Simulates Req A: get_blocking then remove."""
            for _ in range(N_ITERATIONS):
                obj = _make_memory_obj(address=0xAAAA)
                backend.put(key, obj)
                time.sleep(0.0001)
                try:
                    backend.get_blocking(key)
                except AssertionError:
                    pass  # Expected in race
                backend.remove(key)

        def lookup_and_retrieve():
            """Simulates Req B: contains check then get_blocking."""
            for _ in range(N_ITERATIONS):
                if backend.contains(key):
                    try:
                        backend.get_blocking(key)
                    except AssertionError as e:
                        errors.append(e)

        t1 = threading.Thread(target=retrieve_and_remove)
        t2 = threading.Thread(target=lookup_and_retrieve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # We expect at least some assertion errors from the race
        # If the code were correct, this list would be empty
        assert len(errors) > 0, (
            "Expected AssertionError from get_blocking() due to race between "
            "remove_after_retrieve and concurrent get_blocking on shared key. "
            "The race was not triggered in this run — try increasing N_ITERATIONS."
        )


# ---------------------------------------------------------------------------
# Bug 1+2 combined: full scenario
# ---------------------------------------------------------------------------


class TestFullSharedPrefixScenario:
    """
    End-to-end simulation of the shared prefix race condition:
    Two requests with the same prefix → same CacheEngineKey.
    """

    def test_full_race_scenario(self):
        """
        Timeline:
          1. Sender A → AllocRequest → receiver put(K, obj_A)
          2. Sender A RDMA completes → ProxyNotif
          3. Decoder: Req A lookup(K) → hit
          4. Sender B → AllocRequest → receiver put(K, obj_B)
             → obj_A freed! (but Req A already got_blocking it, so OK this time)
          5. Decoder: Req A retrieve → get_blocking(K) → gets obj_B (WRONG DATA!)
          6. Decoder: Req A remove(K)
          7. Decoder: Req B lookup(K) was True earlier
          8. Decoder: Req B retrieve → get_blocking(K) → ASSERT FAIL

        This demonstrates both bugs in sequence.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=67890)

        obj_a = _make_memory_obj(address=0xA000)
        obj_b = _make_memory_obj(address=0xB000)

        # Step 1: Receiver allocates for Req A
        backend.put(key, obj_a)

        # Step 3: Req A's lookup succeeds
        assert backend.contains(key)

        # Step 4: Receiver allocates for Req B (same key)
        # This frees obj_A — Bug 1!
        backend.put(key, obj_b)
        assert obj_a._freed, "Bug 1: obj_A freed while potentially still in RDMA"

        # Step 5: Req A's retrieve — gets obj_B instead of obj_A!
        # This is silent data corruption: Req A reads Req B's (possibly
        # incomplete) buffer instead of its own.
        result = backend.get_blocking(key)
        assert result is obj_b, (
            "Req A gets obj_B (wrong data!) because put() overwrote obj_A"
        )
        assert result is not obj_a, "Req A should have gotten obj_A but got obj_B"

        # Step 6: Req A's remove_after_retrieve
        backend.remove(key)

        # Step 7-8: Req B tries to retrieve — CRASH (Bug 2)
        with pytest.raises(AssertionError, match="not found in local data"):
            backend.get_blocking(key)
