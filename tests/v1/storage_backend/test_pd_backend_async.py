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

Test matrix (behaviour × role)
------------------------------
  1. Sender  — Non-blocking FIFO: batched_submit_put_task returns immediately;
               N requests processed serially in ≈ N× delay
  2. Receiver — Non-blocking: asyncio.sleep yields, not time.sleep
  3. Sender  — Flow control: allocate() blocks when staging buffer full
  4. Receiver — Flow control: _async_allocate_and_put blocks when inflight full
  5. Sender  — Close: close() stops _sender_loop and joins _sender_thread
  6. Receiver — Close: close() stops _recv_loop and joins _recv_thread
  7. Receiver — Data correctness: already_sent dedup + last_chunk_toks shape
  8. Sender  — Chunk ordering: last prefill waits for prior slow chunk
  9. Receiver — Fail-fast: C_req > max_inflight returns all -1
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
)
from lmcache.v1.storage_backend.pd_backend import (
    AllocRequest,
    AllocResponse,
    PDBackend,
    PDMsg,
    ProxyNotif,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRANSFER_DELAY = 0.15  # seconds – simulates a NIXL write taking 150 ms


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
# Shared patch context for PDBackend construction
# ---------------------------------------------------------------------------


def _pd_backend_patches():
    """Return a combined patch context that mocks out all external deps."""
    return (
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_zmq_context",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_zmq_socket",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cpu",
        ),
    )


# ---------------------------------------------------------------------------
# Fixture: PDBackend sender with everything mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def async_sender():
    """
    Build a PDBackend in sender (prefiller) mode with:
      - get_zmq_context / get_zmq_socket mocked (no real sockets)
      - CreateTransferChannel mocked (async_batched_write sleeps TRANSFER_DELAY)
      - ZMQ alloc socket mocked (returns a canned AllocResponse immediately)
    """
    p1, p2, p3, p4 = _pd_backend_patches()

    with p1, p2 as mock_zmq_sock, p3 as mock_create_tc, p4:
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

        # Replace proxy_side_channel with a dedicated mock so it doesn't
        # share call history with the alloc socket mock.
        backend.proxy_side_channel = MagicMock()

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
def async_receiver():
    """
    Build a PDBackend in receiver (decoder) mode.
    The ZMQ server socket is mocked so no real port is bound.
    The memory allocator is mocked so we control when allocate() returns None.
    """
    p1, p2, p3, p4 = _pd_backend_patches()

    with p1, p2, p3, p4:
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
        # Release all allocated MemoryObjs to avoid ref_count warnings
        for mem_obj in backend.data.values():
            try:
                mem_obj.ref_count_down()
            except Exception:
                pass
        backend.close()


# ---------------------------------------------------------------------------
# Test 1: Sender — Non-blocking concurrency
# ---------------------------------------------------------------------------


def test_sender_nonblocking_fifo_transfers(async_sender):
    """
    Submit N transfers with distinct req_ids.

    Verifies two properties of the new global FIFO worker design:
    1. ``batched_submit_put_task`` is fire-and-forget — all N calls return
       well before any transfer completes.
    2. All transfers eventually complete (the single worker processes them
       serially in FIFO order, so the total time is ≈ N × TRANSFER_DELAY).
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

    enqueue_elapsed = time.monotonic() - t0
    # All enqueue calls should complete well before a single TRANSFER_DELAY
    assert enqueue_elapsed < TRANSFER_DELAY / 4, (
        f"batched_submit_put_task calls took {enqueue_elapsed:.3f}s — "
        f"should be non-blocking (< {TRANSFER_DELAY / 4:.3f}s)"
    )

    # With serial FIFO execution, all N requests complete in ≈ N × TRANSFER_DELAY.
    serial_timeout = TRANSFER_DELAY * N * 3  # generous margin for CI
    for i, ev in enumerate(done_events):
        finished = ev.wait(timeout=serial_timeout)
        assert finished, (
            f"Transfer for req-{i} did not complete within "
            f"{serial_timeout:.1f}s (serial FIFO timeout)"
        )


# ---------------------------------------------------------------------------
# Test 2: Receiver — Non-blocking busy-wait uses asyncio.sleep
# ---------------------------------------------------------------------------


def test_receiver_nonblocking_async_sleep(async_receiver):
    """
    Prove that _async_allocate_and_put uses asyncio.sleep (not time.sleep)
    when allocate() returns None.

    Two AllocRequests run concurrently via asyncio.gather:
      - req_a: allocate() returns None for RETRY_COUNT calls, then succeeds.
      - req_b: allocate() succeeds immediately.

    We distinguish them by shape: req_a uses shape [4,2,16,8,128] (token=16),
    req_b uses shape [4,2,8,8,128] (token=8). The patched allocate inspects
    shape to determine which request is calling, avoiding any dependence on
    asyncio scheduling order.

    If asyncio.sleep is used (correct):
        req_b runs while req_a is yielding → finish_order == ["b", "a"].
    If time.sleep is used (blocking):
        req_b cannot run until req_a finishes → finish_order == ["a", "b"].
    """
    RETRY_COUNT = 5
    SHAPE_A_TOKS = 16
    SHAPE_B_TOKS = 8

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

    # Per-shape call counters; shape determines role, not call order
    shape_alloc_calls: dict[int, int] = {}

    def patched_allocate(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kwargs):
        token_dim = MemoryFormat.KV_2LTD.token_dim()
        toks = (
            shapes[token_dim] if isinstance(shapes, torch.Size) else shapes[token_dim]
        )
        shape_alloc_calls[toks] = shape_alloc_calls.get(toks, 0) + 1
        n = shape_alloc_calls[toks]

        if toks == SHAPE_A_TOKS and n <= RETRY_COUNT:
            return None  # req_a retries
        return mem_obj_a if toks == SHAPE_A_TOKS else mem_obj_b

    async_receiver.allocate = patched_allocate

    alloc_req_a = AllocRequest(
        keys=[key_a.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, SHAPE_A_TOKS, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=SHAPE_A_TOKS,
    )
    alloc_req_b = AllocRequest(
        keys=[key_b.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, SHAPE_B_TOKS, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=SHAPE_B_TOKS,
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
# Test 3: Sender — Flow control backpressure
# ---------------------------------------------------------------------------


def test_sender_flow_control_backpressure(async_sender):
    """
    When the sender staging buffer is full (_sender_inflight_chunks >=
    _sender_max_inflight_chunks), allocate() must block until
    _release_sender_staging_chunks() makes room.
    """
    sentinel = _make_mem_obj(idx=77)

    # Patch memory_allocator.allocate to always return sentinel
    async_sender.memory_allocator.allocate = MagicMock(return_value=sentinel)

    # Saturate the inflight counter
    max_chunks = async_sender._sender_max_inflight_chunks
    with async_sender._sender_staging_condition:
        async_sender._sender_inflight_chunks = max_chunks

    result_holder: list = []
    blocked_event = threading.Event()
    unblocked_event = threading.Event()

    def allocating_thread():
        blocked_event.set()  # Signal that we are about to call allocate()
        mem_obj = async_sender.allocate(torch.Size([4, 2, 16, 8, 128]), torch.bfloat16)
        result_holder.append(mem_obj)
        unblocked_event.set()

    t = threading.Thread(target=allocating_thread, daemon=True)
    t.start()

    # Wait until the thread is definitely inside allocate()
    assert blocked_event.wait(timeout=2.0), "Thread did not start in time"
    # Give it a moment to enter the wait() call
    time.sleep(0.1)

    # Thread should still be blocked
    assert not unblocked_event.is_set(), (
        "allocate() returned before the staging slot was freed — "
        "backpressure is not working"
    )

    # Release one slot — the blocked thread should wake up
    async_sender._release_sender_staging_chunks(1)

    assert unblocked_event.wait(timeout=2.0), (
        "allocate() did not unblock within 2 s after slot was freed"
    )
    assert result_holder and result_holder[0] is sentinel, (
        "allocate() did not return the expected MemoryObj after unblocking"
    )
    t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Test 4: Receiver — Flow control inflight backpressure
# ---------------------------------------------------------------------------


def test_receiver_flow_control_inflight(async_receiver):
    """
    When _inflight_chunks >= _max_inflight_chunks, _async_allocate_and_put
    must wait on _inflight_condition. After the condition is notified
    (simulating remove() freeing a slot), the blocked allocation proceeds.
    """
    mem_obj = _make_mem_obj(idx=60)

    # Patch allocate to always succeed
    async_receiver.allocate = (
        lambda shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw: mem_obj
    )

    key = _make_key(600)
    alloc_req = AllocRequest(
        keys=[key.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
    )

    async def saturate_and_test():
        # Saturate inflight counter to the max
        async with async_receiver._inflight_condition:
            async_receiver._inflight_chunks = async_receiver._max_inflight_chunks

        alloc_completed = asyncio.Event()
        resp_holder: list = []

        async def do_alloc():
            resp = await async_receiver._async_allocate_and_put(alloc_req)
            resp_holder.append(resp)
            alloc_completed.set()

        async def free_slot_later():
            # Wait a bit to ensure do_alloc is blocked on the condition
            await asyncio.sleep(0.05)
            # Should NOT have completed yet
            assert not alloc_completed.is_set(), (
                "_async_allocate_and_put returned before inflight slot was freed"
            )
            # Free a slot — simulates what _notify_inflight_freed does
            async with async_receiver._inflight_condition:
                async_receiver._inflight_chunks -= 1
                async_receiver._inflight_condition.notify_all()

        # Run both concurrently
        await asyncio.gather(do_alloc(), free_slot_later())

        # Allocation should have succeeded
        assert len(resp_holder) == 1
        assert resp_holder[0].remote_indexes == [mem_obj.meta.address]

    asyncio.run(saturate_and_test())


# ---------------------------------------------------------------------------
# Test 5: Sender — close() shuts down event loop and thread
# ---------------------------------------------------------------------------


def test_sender_close_stops_event_loop(async_sender):
    """
    After close(), the sender's _sender_loop should be stopped and
    _sender_thread should have joined (not alive).
    """
    assert async_sender._sender_thread.is_alive(), (
        "Sender thread should be alive before close()"
    )
    async_sender.close()

    assert not async_sender._sender_thread.is_alive(), (
        "Sender thread should not be alive after close()"
    )
    # Prevent fixture's close() from double-closing
    async_sender.running = False


# ---------------------------------------------------------------------------
# Test 6: Receiver — close() shuts down event loop and thread
# ---------------------------------------------------------------------------


def test_receiver_close_stops_event_loop(async_receiver):
    """
    After close(), the receiver's _recv_loop should be stopped and
    _recv_thread should have joined (not alive).
    """
    assert async_receiver._recv_thread.is_alive(), (
        "Receiver thread should be alive before close()"
    )
    async_receiver.close()

    assert not async_receiver._recv_thread.is_alive(), (
        "Receiver thread should not be alive after close()"
    )
    # Prevent fixture's close() from double-closing
    async_receiver.running = False


# ---------------------------------------------------------------------------
# Test 7: Receiver — Data correctness: dedup + last_chunk_toks shape
# ---------------------------------------------------------------------------


def test_receiver_data_correctness_dedup_and_shape(async_receiver):
    """
    Combined test for _async_allocate_and_put correctness:
      - key_existing: already in backend → in already_sent_indexes, no allocate()
      - key_full:     new key → allocate with original token dim
      - key_last:     new key (last) → allocate with overridden last_chunk_toks

    Validates both deduplication and shape override in a single pass.
    """
    key_existing = _make_key(300)
    key_full = _make_key(301)
    key_last = _make_key(302)
    mem_obj = _make_mem_obj(idx=30)

    FULL_TOKENS = 16
    LAST_TOKENS = 7

    # Pre-populate backend with key_existing
    async_receiver.put(key_existing, _make_mem_obj(idx=99))

    alloc_shapes: list[torch.Size] = []

    def tracking_allocate(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kwargs):
        alloc_shapes.append(shapes)
        return mem_obj

    async_receiver.allocate = tracking_allocate

    alloc_req = AllocRequest(
        keys=[
            key_existing.to_string(),
            key_full.to_string(),
            key_last.to_string(),
        ],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, FULL_TOKENS, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=LAST_TOKENS,
    )

    resp = asyncio.run(async_receiver._async_allocate_and_put(alloc_req))

    # --- Deduplication assertions ---
    # Index 0 (key_existing) should be in already_sent
    assert 0 in resp.already_sent_indexes, (
        f"Expected index 0 in already_sent_indexes, got {resp.already_sent_indexes}"
    )
    # Only 2 allocations (key_full + key_last), NOT 3
    assert len(alloc_shapes) == 2, (
        f"Expected 2 allocate() calls but got {len(alloc_shapes)}"
    )
    # remote_indexes should contain 2 entries for the allocated keys
    assert len(resp.remote_indexes) == 2

    # --- Shape override assertions ---
    token_dim = MemoryFormat.KV_2LTD.token_dim()
    # First alloc (key_full): token dim should remain FULL_TOKENS
    assert alloc_shapes[0][token_dim] == FULL_TOKENS, (
        f"Full chunk token dim should be {FULL_TOKENS}, "
        f"got {alloc_shapes[0][token_dim]}"
    )
    # Second alloc (key_last): token dim should be overridden to LAST_TOKENS
    assert alloc_shapes[1][token_dim] == LAST_TOKENS, (
        f"Last chunk token dim should be {LAST_TOKENS}, "
        f"got {alloc_shapes[1][token_dim]}"
    )


# ---------------------------------------------------------------------------
# Test 8: Sender — Chunk ordering: last prefill waits for prior tasks
# ---------------------------------------------------------------------------


def test_sender_chunk_ordering_waits_for_prior_tasks(async_sender):
    """
    When a long prompt is chunked into multiple prefills, the final chunk
    (is_last_prefill=True) must NOT send ProxyNotif until all prior chunks'
    RDMA transfers have completed.

    Also implicitly validates:
      - ProxyNotif IS sent when is_last_prefill=True
      - ProxyNotif is NOT sent for the non-last chunk (is_last_prefill=False)

    Strategy:
      - Submit chunk #1 (is_last_prefill=False) with a SLOW transfer.
      - Submit chunk #2 (is_last_prefill=True) with a FAST transfer.
      - ProxyNotif must arrive no earlier than the slow chunk's delay.
    """
    TRANSFER_DELAY_SLOW = 0.30
    TRANSFER_DELAY_FAST = 0.05
    REQ_ID = "req-chunked"

    write_call_count = 0
    write_call_lock = threading.Lock()

    async def _controlled_write(*args, **kwargs):
        nonlocal write_call_count
        with write_call_lock:
            write_call_count += 1
            call_index = write_call_count
        if call_index == 1:
            await asyncio.sleep(TRANSFER_DELAY_SLOW)
        else:
            await asyncio.sleep(TRANSFER_DELAY_FAST)
        return 1

    async_sender.transfer_channel.async_batched_write = _controlled_write

    notify_time: list[float] = []
    sent_data: list[bytes] = []  # <-- capture sent bytes ourselves

    def recording_send(data):
        notify_time.append(time.monotonic())
        sent_data.append(data)  # <-- save it here

    async_sender.proxy_side_channel.send = recording_send

    # Submit chunk #1: slow RDMA, is_last_prefill=False
    spec1 = _make_transfer_spec(req_id=REQ_ID, is_last_prefill=False)
    async_sender.batched_submit_put_task(
        keys=[_make_key(0)],
        memory_objs=[_make_mem_obj(0)],
        transfer_spec=spec1,
    )

    # Small gap so chunk #1 is enqueued first
    time.sleep(0.01)

    # Submit chunk #2: fast RDMA, is_last_prefill=True
    done = threading.Event()
    spec2 = _make_transfer_spec(req_id=REQ_ID, is_last_prefill=True)
    async_sender.batched_submit_put_task(
        keys=[_make_key(1)],
        memory_objs=[_make_mem_obj(1)],
        transfer_spec=spec2,
        on_complete_callback=lambda k: done.set(),
    )

    t_submit = time.monotonic()

    # Wait long enough for both transfers to complete
    finished = done.wait(timeout=TRANSFER_DELAY_SLOW * 3)
    assert finished, "Transfer did not complete within timeout"

    # --- ProxyNotif assertions ---
    # Exactly one ProxyNotif should have been sent (for chunk #2 only)
    assert len(notify_time) == 1, (
        f"Expected exactly one ProxyNotif, got {len(notify_time)}"
    )

    # Verify the ProxyNotif content from our captured data
    notif = msgspec.msgpack.decode(sent_data[0], type=PDMsg)  # <-- use sent_data
    assert isinstance(notif, ProxyNotif)
    assert notif.req_id == REQ_ID

    # --- Timing assertion ---
    # The notify must have arrived at least TRANSFER_DELAY_SLOW after submit,
    # meaning the fast chunk waited for the slow chunk.
    TIMING_TOLERANCE = 0.8  # allow 20% scheduling overhead
    elapsed = notify_time[0] - t_submit
    assert elapsed >= TRANSFER_DELAY_SLOW * TIMING_TOLERANCE, (
        f"ProxyNotif sent after only {elapsed:.3f}s — "
        f"expected >= {TRANSFER_DELAY_SLOW * TIMING_TOLERANCE:.3f}s "
        f"(slow chunk delay). "
        "The final chunk did not wait for the prior slow chunk to finish."
    )


# ---------------------------------------------------------------------------
# Test 9: Receiver — Fail-fast when C_req > max_inflight_chunks
# ---------------------------------------------------------------------------


def test_receiver_fail_fast_request_too_large(async_receiver):
    """
    When a request's total chunk count (accumulated across batches) exceeds
    ``max_inflight_chunks``, ``_async_allocate_and_put`` must return
    ``remote_indexes`` filled with ``-1`` for the offending batch and log an
    error rather than waiting indefinitely.

    This prevents the deadlock scenario where the decoder cannot start
    consuming (needs all chunks) but the buffer is full and no request
    will ever complete.

    Uses a small override for ``_max_inflight_chunks`` so the test runs fast
    without allocating hundreds of real chunks.
    """
    # Override max_inflight so we don't need to allocate hundreds of real chunks.
    MAX_T = 5
    async_receiver._max_inflight_chunks = MAX_T
    req_id = "req-too-large"

    mem_obj = _make_mem_obj(idx=42)
    async_receiver.allocate = lambda *a, **kw: mem_obj

    def _make_req(n_keys: int, req_id_val: str = req_id) -> AllocRequest:
        """Build an AllocRequest with *n_keys* distinct keys."""
        keys = [_make_key(i).to_string() for i in range(n_keys)]
        return AllocRequest(
            keys=keys,
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[4, 2, 16, 8, 128],
            dtype="bfloat16",
            last_chunk_toks=16,
            req_id=req_id_val,
        )

    async def run():
        # Batch 1: exactly MAX_T keys — should succeed (cumulative == MAX_T).
        resp1 = await async_receiver._async_allocate_and_put(_make_req(MAX_T))
        assert -1 not in resp1.remote_indexes, (
            f"Batch 1 should succeed (cumulative == MAX_T={MAX_T}), "
            f"but got remote_indexes={resp1.remote_indexes}"
        )
        assert async_receiver._req_chunk_counts.get(req_id) == MAX_T, (
            "Cumulative chunk count should equal MAX_T after batch 1"
        )

        # Batch 2: 1 more key — cumulative = MAX_T + 1 > MAX_T → fail fast.
        # The fail-fast check fires BEFORE inflight wait, so this returns
        # immediately without blocking.
        resp2 = await async_receiver._async_allocate_and_put(_make_req(1))
        assert resp2.remote_indexes == [-1], (
            f"Batch 2 should fail fast with [-1] when cumulative > MAX_T, "
            f"but got remote_indexes={resp2.remote_indexes}"
        )
        assert resp2.already_sent_indexes == [], (
            "Fail-fast response should have no already_sent_indexes"
        )

    asyncio.run(run())

    # A different req_id should not be affected by the too-large req's tracking.
    # Reset inflight counter (from batch 1's MAX_T allocations) so the
    # backpressure wait for the new req doesn't block indefinitely.
    async_receiver._inflight_chunks = 0

    other_req_id = "req-small"

    async def run_other():
        resp = await async_receiver._async_allocate_and_put(
            _make_req(1, req_id_val=other_req_id)
        )
        assert -1 not in resp.remote_indexes, (
            "A different small request should not be affected by the fail-fast "
            "tracking of an unrelated req_id"
        )

    asyncio.run(run_other())
