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
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.storage_backend.pd_backend import AllocRequest, AllocResponse, PDBackend, ProxyNotif, PDMsg

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

    # Use spec= so isinstance(mock, PagedCpuGpuMemoryAllocator) returns True.
    mock_allocator_inst = MagicMock()
    mock_allocator_inst.cpu_allocator.buffer_ptr = 0
    mock_allocator_inst.cpu_allocator.buffer_size = 1024 * 1024 * 64
    mock_allocator_inst.cpu_allocator.align_bytes = 1
    # Make isinstance(mock, PagedCpuGpuMemoryAllocator) return True
    mock_allocator_inst.__class__ = PagedCpuGpuMemoryAllocator

    with (
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_context") as mock_zmq_ctx,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_socket") as mock_zmq_sock,
        patch(
            "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel"
        ) as mock_create_tc,
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cpu",
        ),
    ):
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
    # Use spec= so isinstance(mock, PagedCpuGpuMemoryAllocator) returns True.
    mock_allocator_inst = MagicMock()
    mock_allocator_inst.cpu_allocator.buffer_ptr = 0
    mock_allocator_inst.cpu_allocator.buffer_size = 1024 * 1024 * 64
    mock_allocator_inst.cpu_allocator.align_bytes = 1
    # Make isinstance(mock, PagedCpuGpuMemoryAllocator) return True
    mock_allocator_inst.__class__ = PagedCpuGpuMemoryAllocator

    with (
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_context") as mock_zmq_ctx,
        patch("lmcache.v1.storage_backend.pd_backend.get_zmq_socket") as mock_zmq_sock,
        patch(
            "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel"
        ) as mock_create_tc,
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cpu",
        ),
    ):
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
    Prove that the *real* _async_allocate_and_put uses asyncio.sleep (not
    time.sleep) when allocate() returns None.

    Two AllocRequests run concurrently via asyncio.gather:
      - req_a: allocate() returns None for RETRY_COUNT calls (per task),
               then succeeds. The real busy-wait loop fires RETRY_COUNT times.
      - req_b: allocate() succeeds immediately.

    We use asyncio.current_task() inside patched allocate() to distinguish
    which coroutine is calling, so retry counts are tracked per-task.

    If asyncio.sleep is used (correct):
        req_b runs while req_a is yielding → finish_order == ["b", "a"].
    If time.sleep is used (blocking):
        req_b cannot run until req_a finishes → finish_order == ["a", "b"].
    """
    RETRY_COUNT = 5

    key_a = _make_key(100)
    key_b = _make_key(200)
    mem_obj_a = _make_mem_obj(idx=10)
    mem_obj_b = _make_mem_obj(idx=20)

    finish_order: list[str] = []

    # Wrap put() to record store order
    original_put = async_receiver.put

    def tracked_put(key, mem_obj):
        if key == key_a:
            finish_order.append("a")
        elif key == key_b:
            finish_order.append("b")
        return original_put(key, mem_obj)

    async_receiver.put = tracked_put

    # Patch allocate() to use asyncio.current_task() as per-coroutine context.
    # Each task tracks its own call count independently.
    task_alloc_calls: dict[int, int] = {}

    def patched_allocate(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kwargs):
        task = asyncio.current_task()
        task_id = id(task)
        task_alloc_calls[task_id] = task_alloc_calls.get(task_id, 0) + 1
        n = task_alloc_calls[task_id]
        # The task handling key_a was submitted first (call n=1 initially);
        # distinguish by whether this task has already been seen before the
        # *other* task's first call. Use a simple heuristic: first task to
        # call allocate is A (gets retries), second is B (immediate success).
        if task_id not in _task_roles:
            _task_roles[task_id] = "a" if len(_task_roles) == 0 else "b"
        role = _task_roles[task_id]
        if role == "a" and n <= RETRY_COUNT:
            return None
        return mem_obj_a if role == "a" else mem_obj_b

    _task_roles: dict[int, str] = {}
    async_receiver.allocate = patched_allocate

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

    assert finish_order == ["b", "a"], (
        f"Expected finish order ['b', 'a'] but got {finish_order}. "
        "This suggests _async_allocate_and_put uses time.sleep (blocking) "
        "instead of asyncio.sleep (non-blocking): req_b could not run while "
        "req_a was busy-waiting for memory."
    )


# ---------------------------------------------------------------------------
# Test 4: Proxy notification sent on last prefill
# ---------------------------------------------------------------------------


def test_sender_proxy_notification_on_last_prefill(async_sender):
    """
    When transfer_spec.is_last_prefill is True, the sender must send a
    ProxyNotif message to proxy_side_channel after transfer completes.
    """
    keys = [_make_key(0)]
    memory_objs = [_make_mem_obj(0)]
    transfer_spec = _make_transfer_spec(is_last_prefill=True, req_id="req-notify")

    done = threading.Event()

    def cb(key):
        done.set()

    async_sender.batched_submit_put_task(
        keys, memory_objs, transfer_spec=transfer_spec, on_complete_callback=cb
    )

    assert done.wait(timeout=TRANSFER_DELAY * 3), "Transfer did not complete"

    # Verify proxy notification was sent
    async_sender.proxy_side_channel.send.assert_called_once()
    sent_bytes = async_sender.proxy_side_channel.send.call_args[0][0]
    notif = msgspec.msgpack.decode(sent_bytes, type=PDMsg)
    assert isinstance(notif, ProxyNotif)
    assert notif.req_id == "req-notify"


# ---------------------------------------------------------------------------
# Test 5: No proxy notification when not last prefill
# ---------------------------------------------------------------------------


def test_sender_no_proxy_notification_when_not_last_prefill(async_sender):
    """
    When transfer_spec.is_last_prefill is False, no ProxyNotif should be sent.
    """
    keys = [_make_key(0)]
    memory_objs = [_make_mem_obj(0)]
    transfer_spec = _make_transfer_spec(is_last_prefill=False)

    done = threading.Event()
    async_sender.batched_submit_put_task(
        keys, memory_objs, transfer_spec=transfer_spec,
        on_complete_callback=lambda k: done.set()
    )

    assert done.wait(timeout=TRANSFER_DELAY * 3)
    async_sender.proxy_side_channel.send.assert_not_called()
