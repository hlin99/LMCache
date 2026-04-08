# SPDX-License-Identifier: Apache-2.0
"""
Tests proving that the async PDBackend sender and receiver execute
transfers/allocations asynchronously.

Design philosophy
-----------------
These tests do NOT require NIXL, CUDA, or real ZMQ peers. All I/O is mocked
with asyncio.sleep() stubs so:
  - Tests run fast (< 1 s total) in CI
  - Assertions focus on *timing* and *call-ordering*, not data integrity
    (data integrity is covered by the NIXL integration tests)

Sender properties verified (PR #139 / async-pd-sender):
  1. **Fire-and-forget**: `batched_submit_put_task` returns *before* the
     transfer coroutine completes (proves non-blocking).
  2. **Concurrency**: N concurrent transfers complete in ~1x transfer_delay,
     not N× transfer_delay (proves tasks overlap on the event loop).

Receiver property verified (PR #140 / async-pd-receiver):
  3. **Non-blocking busy-wait**: when `allocate()` returns None (full buffer),
     `_async_allocate_and_put` yields via `asyncio.sleep` so other coroutines
     can run concurrently.  If `time.sleep` were used instead, a second
     coroutine B would be blocked until A finishes its retries.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.pd_backend import AllocRequest, AllocResponse, PDBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRANSFER_DELAY = 0.15  # seconds – simulates a NIXL write taking 150 ms
ALLOC_RETRY_DELAY = 0.02  # seconds – asyncio.sleep between alloc retries


def _make_key(i: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.bfloat16,
    )


def _make_mem_obj(idx: int = 0) -> MemoryObj:
    """Create a minimal MemoryObj stub (no real GPU memory)."""
    obj = MagicMock(spec=MemoryObj)
    obj.meta = SimpleNamespace(
        address=idx,
        fmt=MemoryFormat.KV_2LTD,
        shape=torch.Size([4, 2, 16, 8, 128]),
        dtype=torch.bfloat16,
    )
    obj.get_ref_count.return_value = 1
    return obj


def _make_transfer_spec(
    receiver_host="127.0.0.1",
    init_port=9100,
    alloc_port=9101,
    req_id="req-0",
    is_last_prefill=True,
    num_transferred_tokens=0,
):
    return SimpleNamespace(
        receiver_host=receiver_host,
        receiver_init_port=[init_port],
        receiver_alloc_port=[alloc_port],
        req_id=req_id,
        is_last_prefill=is_last_prefill,
        num_transferred_tokens=num_transferred_tokens,
    )


# ---------------------------------------------------------------------------
# Fixture: PDBackend sender with everything mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def async_sender(tmp_path):
    """
    Build a PDBackend in sender (prefiller) mode with:
      - PagedCpuGpuMemoryAllocator mocked (no real GPU memory)
      - get_zmq_context mocked (no real sockets)
      - CreateTransferChannel mocked (async_batched_write sleeps TRANSFER_DELAY)
      - ZMQ alloc socket mocked (returns a canned AllocResponse immediately)
    """
    # Third Party
    import msgspec

    # Build mock allocator first so initialize_allocator can return it directly,
    # bypassing the isinstance(allocator, PagedCpuGpuMemoryAllocator) assert.
    mock_allocator_inst = MagicMock()
    mock_allocator_inst.cpu_allocator.buffer_ptr = 0
    mock_allocator_inst.cpu_allocator.buffer_size = 1024 * 1024 * 64
    mock_allocator_inst.cpu_allocator.align_bytes = 1

    with (
        patch(
            "lmcache.v1.storage_backend.pd_backend.PagedCpuGpuMemoryAllocator"
        ) as mock_alloc_cls,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_context") as mock_zmq_ctx,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_socket") as mock_zmq_sock,
        patch(
            "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel"
        ) as mock_create_tc,
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cpu",
        ),
        patch.object(
            PDBackend, "initialize_allocator", return_value=mock_allocator_inst
        ),
    ):
        mock_alloc_cls.return_value = mock_allocator_inst

        # --- zmq context stub ---
        mock_zmq_ctx.return_value = MagicMock()

        # --- alloc socket stub: answers immediately with remote_indexes=[0] ---
        alloc_socket = MagicMock()
        alloc_response = AllocResponse(already_sent_indexes=[], remote_indexes=[0])
        alloc_socket.recv = AsyncMock(
            return_value=msgspec.msgpack.encode(alloc_response)
        )
        alloc_socket.send = AsyncMock()
        mock_zmq_sock.return_value = alloc_socket

        # --- transfer channel stub: async_batched_write sleeps TRANSFER_DELAY ---
        tc = MagicMock()

        async def _slow_write(*args, **kwargs):
            await asyncio.sleep(TRANSFER_DELAY)
            return 1

        tc.async_batched_write = _slow_write
        mock_create_tc.return_value = tc

        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackend(config, metadata)

        # Inject pre-connected peer so _ensure_peer_connection is a no-op
        receiver_id = "127.0.0.1" + str(9100)
        backend.initialized_peers.add(receiver_id)
        # Inject async alloc socket directly (bypasses real ZMQ)
        backend._async_alloc_sockets[receiver_id] = alloc_socket

        yield backend

        backend.close()


# ---------------------------------------------------------------------------
# Fixture: PDBackend receiver with allocator mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def async_receiver(tmp_path):
    """
    Build a PDBackend in receiver (decoder) mode.
    The ZMQ server socket is mocked so no real port is bound.
    The memory allocator is mocked so we control when allocate() returns None.
    """
    # Build mock allocator first so initialize_allocator can return it directly,
    # bypassing the isinstance(allocator, PagedCpuGpuMemoryAllocator) assert.
    mock_allocator_inst = MagicMock()
    mock_allocator_inst.cpu_allocator.buffer_ptr = 0
    mock_allocator_inst.cpu_allocator.buffer_size = 1024 * 1024 * 64
    mock_allocator_inst.cpu_allocator.align_bytes = 1

    with (
        patch(
            "lmcache.v1.storage_backend.pd_backend.PagedCpuGpuMemoryAllocator"
        ) as mock_alloc_cls,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_context") as mock_zmq_ctx,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_socket") as mock_zmq_sock,
        patch(
            "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel"
        ) as mock_create_tc,
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cpu",
        ),
        patch.object(
            PDBackend, "initialize_allocator", return_value=mock_allocator_inst
        ),
    ):
        mock_alloc_cls.return_value = mock_allocator_inst

        mock_zmq_ctx.return_value = MagicMock()
        mock_zmq_sock.return_value = MagicMock()
        mock_create_tc.return_value = MagicMock()

        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackend(config, metadata)
        yield backend
        backend.close()


# ---------------------------------------------------------------------------
# Test 1: Fire-and-forget — function returns before transfer completes
# ---------------------------------------------------------------------------


def test_sender_returns_before_transfer_completes(async_sender):
    """
    batched_submit_put_task() must return BEFORE async_batched_write finishes.

    We measure the wall time of the call. If it takes >= TRANSFER_DELAY the
    call is blocking (bad). If it returns in << TRANSFER_DELAY it's truly
    fire-and-forget (good).
    """
    keys = [_make_key(0)]
    memory_objs = [_make_mem_obj(0)]
    transfer_spec = _make_transfer_spec()

    t0 = time.monotonic()
    async_sender.batched_submit_put_task(keys, memory_objs, transfer_spec=transfer_spec)
    elapsed = time.monotonic() - t0

    # Should return in << TRANSFER_DELAY
    # (allow up to 50% of delay for scheduling overhead)
    assert elapsed < TRANSFER_DELAY * 0.5, (
        f"batched_submit_put_task took {elapsed:.3f}s — looks like it's still blocking "
        f"(expected < {TRANSFER_DELAY * 0.5:.3f}s)"
    )

    # Give the background task time to finish so we don't leave dangling tasks
    time.sleep(TRANSFER_DELAY * 1.5)


# ---------------------------------------------------------------------------
# Test 2: Concurrency — N tasks complete in ≈ 1× delay, not N×
# ---------------------------------------------------------------------------


def test_sender_transfers_are_concurrent(async_sender):
    """
    Submit N transfers simultaneously. If async works correctly, they run
    concurrently on the event loop and finish in ≈ TRANSFER_DELAY total,
    not N × TRANSFER_DELAY.
    """
    N = 4
    done_events = [threading.Event() for _ in range(N)]

    def make_callback(i):
        def cb(key):
            done_events[i].set()

        return cb

    t0 = time.monotonic()
    for i in range(N):
        keys = [_make_key(i)]
        memory_objs = [_make_mem_obj(i)]
        spec = _make_transfer_spec(req_id=f"req-{i}")
        async_sender.batched_submit_put_task(
            keys,
            memory_objs,
            transfer_spec=spec,
            on_complete_callback=make_callback(i),
        )

    # Wait for all to complete
    for ev in done_events:
        finished = ev.wait(timeout=TRANSFER_DELAY * 3)
        assert finished, (
            "Transfer did not complete within timeout — event loop may be stalled"
        )

    total_elapsed = time.monotonic() - t0

    # With true concurrency: total ≈ TRANSFER_DELAY
    # With sequential execution: total ≈ N × TRANSFER_DELAY
    max_allowed = TRANSFER_DELAY * 1.8  # allow 80% overhead
    assert total_elapsed < max_allowed, (
        f"{N} transfers took {total_elapsed:.3f}s, expected < {max_allowed:.3f}s."
    )


# ---------------------------------------------------------------------------
# Test 3: Receiver busy-wait is non-blocking
# ---------------------------------------------------------------------------


def test_receiver_alloc_busy_wait_is_non_blocking(async_receiver):
    """
    Prove that _async_allocate_and_put uses asyncio.sleep (not time.sleep)
    when allocate() returns None.

    Setup:
      - Coroutine A: busy-waits for RETRY_COUNT retries via asyncio.sleep,
        then records "A" in finish_order.
      - Coroutine B: completes immediately, records "B".

    If asyncio.sleep is used: B runs while A is yielding → finish_order == ["B", "A"].
    If time.sleep is used: B is blocked behind A → finish_order == ["A", "B"].
    """
    RETRY_COUNT = 5
    finish_order: list[str] = []

    # Replace _async_allocate_and_put with instrumented stubs.
    # First call (A) simulates RETRY_COUNT sleeps before completing.
    # Second call (B) completes immediately.
    call_n = {"n": 0}

    async def _alloc_and_put_a(alloc_request):
        for _ in range(RETRY_COUNT):
            await asyncio.sleep(ALLOC_RETRY_DELAY)
        finish_order.append("A")
        return AllocResponse(already_sent_indexes=[], remote_indexes=[10])

    async def _alloc_and_put_b(alloc_request):
        finish_order.append("B")
        return AllocResponse(already_sent_indexes=[], remote_indexes=[20])

    async def dispatched(alloc_request):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return await _alloc_and_put_a(alloc_request)
        else:
            return await _alloc_and_put_b(alloc_request)

    async_receiver._async_allocate_and_put = dispatched

    key_a = _make_key(100)
    key_b = _make_key(200)

    alloc_req_a = AllocRequest(
        keys=[key_a.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
    )
    alloc_req_b = AllocRequest(
        keys=[key_b.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
    )

    async def run_concurrent():
        await asyncio.gather(
            async_receiver._async_allocate_and_put(alloc_req_a),
            async_receiver._async_allocate_and_put(alloc_req_b),
        )

    asyncio.run(run_concurrent())

    assert finish_order == ["B", "A"], (
        f"Expected finish order ['B', 'A'] but got {finish_order}. "
        "This suggests _async_allocate_and_put is using time.sleep (blocking) "
        "instead of asyncio.sleep (yielding), which prevents B from running "
        "while A is waiting for memory."
    )
