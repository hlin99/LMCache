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
               N requests to same receiver processed serially (per-receiver FIFO)
  2. Receiver — Non-blocking: asyncio.sleep yields, not time.sleep
  3. Sender  — Flow control: allocate() blocks when staging buffer full
  4. Receiver — Flow control: _async_allocate_and_put blocks when inflight full
  5. Sender  — Close: close() stops _sender_loop and joins _sender_thread
  6. Receiver — Close: close() stops _recv_loop and joins _recv_thread
  7. Receiver — Data correctness: already_sent dedup + last_chunk_toks shape
  8. Sender  — Chunk ordering: last prefill waits for prior slow chunk
  9. Receiver — Fail-fast: C_req > max_inflight raises RuntimeError
 10. Receiver — Alloc timeout raises RuntimeError (no rollback)
 11. Receiver — is_last_batch cleans up _req_allocated_keys tracking
 12. Receiver — Fail-fast raises RuntimeError (no rollback of prior batches)
 13. Receiver — Admission control prevents req interleaving
 14. Receiver/Sender — pd_max_prefill_len capacity check raises ValueError on init
 15. Receiver — _handle_alloc_request sends error response on exception
 16. Sender  — Per-receiver workers: different-receiver requests run concurrently;
               same-receiver requests are serialized
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
from lmcache.v1.storage_backend.pd_backend_async import (
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

# Fraction of TRANSFER_DELAY used as the threshold for verifying that
# batched_submit_put_task() is non-blocking (fire-and-forget).  All N
# submit calls must complete in < TRANSFER_DELAY * NONBLOCKING_THRESHOLD_RATIO
# of a single TRANSFER_DELAY.
NONBLOCKING_THRESHOLD_RATIO = 0.25

# Multiplier applied to N × TRANSFER_DELAY for the serial FIFO completion
# timeout in CI environments (generous to accommodate scheduling jitter).
CI_SERIAL_TIMEOUT_MARGIN = 3


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
            "lmcache.v1.storage_backend.pd_backend_async.get_zmq_context",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_zmq_socket",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.CreateTransferChannel",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_correct_device",
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
        # DEALER socket uses send_multipart/recv_multipart (DEALER-ROUTER protocol).
        # recv_multipart returns [empty_delimiter, payload].
        alloc_socket = MagicMock()
        alloc_response = AllocResponse(remote_indexes=[0])
        alloc_socket.recv_multipart = AsyncMock(
            return_value=[b"", msgspec.msgpack.encode(alloc_response)]
        )
        alloc_socket.send_multipart = AsyncMock()
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
    Submit N transfers with distinct req_ids, all targeting the same receiver.

    Verifies two properties of the per-receiver FIFO worker design:
    1. ``batched_submit_put_task`` is fire-and-forget — all N calls return
       well before any transfer completes.
    2. All transfers eventually complete.  Since all requests go to the same
       receiver they are serialized by that receiver's worker (FIFO order).
    3. Transfers complete in FIFO order (req-0 before req-1 before ...).
    """
    N = 4
    done_events = [threading.Event() for _ in range(N)]
    completion_order: list[int] = []
    completion_lock = threading.Lock()

    def make_callback(i):
        def cb(key):
            with completion_lock:
                completion_order.append(i)
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
    # All enqueue calls should complete well before a single TRANSFER_DELAY.
    nonblocking_threshold = TRANSFER_DELAY * NONBLOCKING_THRESHOLD_RATIO
    assert enqueue_elapsed < nonblocking_threshold, (
        f"batched_submit_put_task calls took {enqueue_elapsed:.3f}s — "
        f"should be non-blocking (< {nonblocking_threshold:.3f}s)"
    )

    # With serial FIFO execution, all N requests complete in ≈ N × TRANSFER_DELAY.
    serial_timeout = TRANSFER_DELAY * N * CI_SERIAL_TIMEOUT_MARGIN
    for i, ev in enumerate(done_events):
        finished = ev.wait(timeout=serial_timeout)
        assert finished, (
            f"Transfer for req-{i} did not complete within "
            f"{serial_timeout:.1f}s (serial FIFO timeout)"
        )

    # Verify FIFO completion order: req-0 must complete before req-1, etc.
    assert completion_order == list(range(N)), (
        f"Transfers did not complete in FIFO order: {completion_order} "
        f"(expected {list(range(N))})"
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
    Verify _async_allocate_and_put allocates all keys and applies the
    last_chunk_toks shape override on the final slot.

    All three keys are allocated (dedup logic was removed); the last key
    receives a token-dim override of LAST_TOKENS.
    """
    key_existing = _make_key(300)
    key_full = _make_key(301)
    key_last = _make_key(302)
    mem_obj = _make_mem_obj(idx=30)

    FULL_TOKENS = 16
    LAST_TOKENS = 7

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

    asyncio.run(async_receiver._async_allocate_and_put(alloc_req))

    # All 3 keys should be allocated (no dedup)
    assert len(alloc_shapes) == 3, (
        f"Expected 3 allocate() calls but got {len(alloc_shapes)}"
    )

    # --- Shape override assertion ---
    token_dim = MemoryFormat.KV_2LTD.token_dim()
    # Last alloc (key_last): token dim should be overridden to LAST_TOKENS
    assert alloc_shapes[-1][token_dim] == LAST_TOKENS, (
        f"Last chunk token dim should be {LAST_TOKENS}, "
        f"got {alloc_shapes[-1][token_dim]}"
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
    ``max_inflight_chunks``, ``_async_allocate_and_put`` must raise a
    ``RuntimeError`` for the offending batch rather than waiting indefinitely.

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

    def _make_req(
        n_keys: int, req_id_val: str = req_id, key_offset: int = 0
    ) -> AllocRequest:
        """Build an AllocRequest with *n_keys* distinct keys."""
        keys = [_make_key(key_offset + i).to_string() for i in range(n_keys)]
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
        assert len(resp1.remote_indexes) == MAX_T, (
            f"Expected {MAX_T} remote_indexes in batch 1, "
            f"got {len(resp1.remote_indexes)}"
        )

        # Batch 2: 1 more key — cumulative = MAX_T + 1 > MAX_T → RuntimeError.
        # The fail-fast check fires BEFORE inflight wait.
        # Use key_offset=MAX_T to avoid overlap with Batch 1 keys (0..MAX_T-1).
        with pytest.raises(RuntimeError, match="max_inflight_chunks"):
            await async_receiver._async_allocate_and_put(
                _make_req(1, key_offset=MAX_T)
            )

    asyncio.run(run())

    # A different req_id should not be affected by the too-large req's tracking.
    # Directly reset the inflight counter so the backpressure wait in run_other()
    # doesn't block indefinitely.  This is consistent with the approach used in
    # Test 4, which also directly writes _inflight_chunks for test setup; the
    # alternative (calling remove() per key) would introduce a race since
    # _notify_inflight_freed() is scheduled asynchronously on _recv_loop.
    async_receiver._inflight_chunks = 0

    other_req_id = "req-small"

    async def run_other():
        # Use key_offset=2000 to avoid overlap with any keys used above.
        resp = await async_receiver._async_allocate_and_put(
            _make_req(1, req_id_val=other_req_id, key_offset=2000)
        )
        assert -1 not in resp.remote_indexes, (
            "A different small request should not be affected by the fail-fast "
            "tracking of an unrelated req_id"
        )

    asyncio.run(run_other())


# ---------------------------------------------------------------------------
# Test 10: Receiver — Allocation timeout raises RuntimeError (no rollback)
# ---------------------------------------------------------------------------


def test_receiver_alloc_timeout_raises_runtime_error(async_receiver):
    """
    When allocate() returns None (timeout) for the Nth key in a batch,
    _async_allocate_and_put must raise RuntimeError. No rollback is performed;
    previously allocated keys are left in place.

    Strategy:
      - Batch 1 (3 keys): succeeds, keys stored in backend and tracked.
      - Batch 2 (3 keys): allocate() returns None on the 2nd key (idx=1).
        RuntimeError must be raised; batch 1 keys remain in the backend.
    """
    MAX_T = 10
    async_receiver._max_inflight_chunks = MAX_T
    req_id = "req-timeout"

    mem_idx = [0]

    def make_unique_mem_obj():
        obj = _make_mem_obj(idx=mem_idx[0])
        mem_idx[0] += 1
        return obj

    # Batch 1: 3 keys, all succeed.
    batch1_keys = [_make_key(1000 + i) for i in range(3)]

    def always_succeed(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        return make_unique_mem_obj()

    async_receiver.allocate = always_succeed

    alloc_req1 = AllocRequest(
        keys=[k.to_string() for k in batch1_keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
    )

    resp1 = asyncio.run(async_receiver._async_allocate_and_put(alloc_req1))
    assert -1 not in resp1.remote_indexes, "Batch 1 should succeed"
    assert len(resp1.remote_indexes) == 3

    # Verify batch 1 keys are in the backend
    for k in batch1_keys:
        assert async_receiver.contains(k, pin=False), (
            f"Key {k} should be in backend after batch 1"
        )

    # Batch 2: 3 keys. allocate() succeeds on idx=0, fails (None) on idx=1.
    batch2_keys = [_make_key(2000 + i) for i in range(3)]
    alloc_call_count = [0]

    def fail_on_second(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        alloc_call_count[0] += 1
        if alloc_call_count[0] == 1:
            return make_unique_mem_obj()
        return None  # timeout simulation

    async_receiver.allocate = fail_on_second
    # Use a very short timeout so the test doesn't hang
    async_receiver._allocation_timeout = 0.05

    alloc_req2 = AllocRequest(
        keys=[k.to_string() for k in batch2_keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
    )

    async def run_batch2():
        with pytest.raises(RuntimeError, match="timeout"):
            await async_receiver._async_allocate_and_put(alloc_req2)

    asyncio.run(run_batch2())

    # Batch 1 keys should NOT have been rolled back (no rollback on hard error).
    for k in batch1_keys:
        assert async_receiver.contains(k, pin=False), (
            f"Key {k} from batch 1 should remain in backend (no rollback on error)"
        )


# ---------------------------------------------------------------------------
# Test 11: Receiver — is_last_batch cleans up _req_allocated_keys tracking
# ---------------------------------------------------------------------------


def test_receiver_is_last_batch_cleans_up_tracking(async_receiver):
    """
    When is_last_batch=True and all allocations succeed, the receiver
    must pop the req_id from _req_allocated_keys (lifecycle cleanup).

    Strategy:
      - Batch 1 (is_last_batch=False): keys tracked in _req_allocated_keys.
      - Batch 2 (is_last_batch=True): after success, req_id removed from tracking.
    """
    MAX_T = 10
    async_receiver._max_inflight_chunks = MAX_T
    req_id = "req-lifecycle"

    mem_idx = [0]

    def always_succeed(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        obj = _make_mem_obj(idx=mem_idx[0])
        mem_idx[0] += 1
        return obj

    async_receiver.allocate = always_succeed

    # Batch 1: is_last_batch=False → tracking should persist
    alloc_req1 = AllocRequest(
        keys=[_make_key(3000 + i).to_string() for i in range(3)],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
        is_last_batch=False,
    )

    resp1 = asyncio.run(async_receiver._async_allocate_and_put(alloc_req1))
    assert -1 not in resp1.remote_indexes
    assert req_id in async_receiver._req_allocated_keys, (
        "After batch 1 (not last), req_id should still be tracked"
    )
    assert len(async_receiver._req_allocated_keys[req_id]) == 3, (
        "Should track 3 keys from batch 1"
    )

    # Batch 2: is_last_batch=True → tracking should be cleaned up
    alloc_req2 = AllocRequest(
        keys=[_make_key(4000 + i).to_string() for i in range(2)],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
        is_last_batch=True,
    )

    resp2 = asyncio.run(async_receiver._async_allocate_and_put(alloc_req2))
    assert -1 not in resp2.remote_indexes
    assert req_id not in async_receiver._req_allocated_keys, (
        "After is_last_batch=True, req_id should be removed from tracking"
    )


# ---------------------------------------------------------------------------
# Test 12: Receiver — Fail-fast raises RuntimeError (no rollback of prior batches)
# ---------------------------------------------------------------------------


def test_receiver_fail_fast_raises_runtime_error(async_receiver):
    """
    When the fail-fast check triggers (cumulative > max_inflight_chunks),
    _async_allocate_and_put must raise RuntimeError. Prior-batch keys must NOT
    be rolled back (hard-error, no rollback).
    """
    MAX_T = 4
    async_receiver._max_inflight_chunks = MAX_T
    req_id = "req-failfast-error"

    mem_idx = [0]

    def always_succeed(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        obj = _make_mem_obj(idx=mem_idx[0])
        mem_idx[0] += 1
        return obj

    async_receiver.allocate = always_succeed

    # Batch 1: 3 keys, succeeds (3 <= 4).
    batch1_keys = [_make_key(5000 + i) for i in range(3)]
    alloc_req1 = AllocRequest(
        keys=[k.to_string() for k in batch1_keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
    )

    resp1 = asyncio.run(async_receiver._async_allocate_and_put(alloc_req1))
    assert -1 not in resp1.remote_indexes
    for k in batch1_keys:
        assert async_receiver.contains(k, pin=False)

    # Batch 2: 2 keys → cumulative = 3 + 2 = 5 > 4 → fail-fast RuntimeError
    batch2_keys = [_make_key(6000 + i) for i in range(2)]
    alloc_req2 = AllocRequest(
        keys=[k.to_string() for k in batch2_keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id=req_id,
    )

    async def run_batch2():
        with pytest.raises(RuntimeError, match="max_inflight_chunks"):
            await async_receiver._async_allocate_and_put(alloc_req2)

    asyncio.run(run_batch2())

    # Batch 1 keys should NOT have been rolled back (no rollback on hard error).
    for k in batch1_keys:
        assert async_receiver.contains(k, pin=False), (
            f"Key {k} from batch 1 should remain in backend (no rollback on error)"
        )


# ---------------------------------------------------------------------------
# Test 13: Receiver — Admission control prevents req interleaving
# ---------------------------------------------------------------------------


def test_receiver_admission_control_prevents_interleaving(async_receiver):
    """
    Two different req_ids submitted concurrently: the second must wait until
    the first finishes (is_last_batch=True) before its allocation proceeds.

    Strategy:
      - req-A batch 1 (is_last_batch=False): starts, grabs admission.
      - req-B batch 1 (is_last_batch=True): must wait (admission owned by A).
      - req-A batch 2 (is_last_batch=True): proceeds (same req_id), releases.
      - req-B then proceeds.

    Timing: req-A batches use a slow allocator to create a window where req-B
    must be blocked. We verify ordering via an event log.
    """
    MAX_T = 20
    async_receiver._max_inflight_chunks = MAX_T

    event_log: list[str] = []
    mem_idx = [0]

    def make_obj():
        obj = _make_mem_obj(idx=mem_idx[0])
        mem_idx[0] += 1
        return obj

    async_receiver.allocate = (
        lambda shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw: make_obj()
    )

    def _req(
        req_id: str,
        n_keys: int,
        key_offset: int,
        is_last: bool = False,
    ) -> AllocRequest:
        return AllocRequest(
            keys=[_make_key(key_offset + i).to_string() for i in range(n_keys)],
            fmt=MemoryFormat.KV_2LTD.value,
            shape=[4, 2, 16, 8, 128],
            dtype="bfloat16",
            last_chunk_toks=16,
            req_id=req_id,
            is_last_batch=is_last,
        )

    async def run():
        # req-A batch 1: grabs admission
        event_log.append("A1-start")
        resp = await async_receiver._async_allocate_and_put(
            _req("req-A", 2, 7000, is_last=False)
        )
        assert -1 not in resp.remote_indexes
        event_log.append("A1-done")

        # Now launch req-B and req-A batch 2 concurrently.
        # req-A batch 2 should proceed immediately; req-B should wait.

        b_started = asyncio.Event()
        b_acquired = asyncio.Event()

        async def do_b():
            b_started.set()
            event_log.append("B-start")
            resp = await async_receiver._async_allocate_and_put(
                _req("req-B", 1, 8000, is_last=True)
            )
            event_log.append("B-done")
            b_acquired.set()
            return resp

        async def do_a2():
            # Small delay to ensure B starts waiting first
            await asyncio.sleep(0.02)
            event_log.append("A2-start")
            resp = await async_receiver._async_allocate_and_put(
                _req("req-A", 1, 9000, is_last=True)
            )
            event_log.append("A2-done")
            return resp

        results = await asyncio.gather(do_b(), do_a2())
        resp_b, resp_a2 = results

        assert -1 not in resp_a2.remote_indexes
        assert -1 not in resp_b.remote_indexes

    asyncio.run(run())

    # Verify ordering: A2 must complete before B, because admission is held
    # by req-A until A2 (is_last_batch=True) releases it.
    a2_done_idx = event_log.index("A2-done")
    b_done_idx = event_log.index("B-done")
    assert a2_done_idx < b_done_idx, (
        f"req-A batch 2 should complete before req-B, "
        f"but event_log={event_log}"
    )


# ---------------------------------------------------------------------------
# Test 14: Receiver/Sender — pd_max_prefill_len capacity check on init
# ---------------------------------------------------------------------------


def test_pd_max_prefill_len_capacity_check():
    """
    When pd_max_prefill_len > capacity_tokens, PDBackend.__init__ must raise
    ValueError for both receiver and sender roles.

    The capacity formula is:
        capacity_tokens = (aligned_buffer_size // chunk_size_bytes) * chunk_token_size

    With the test fixtures:
        kv_shape = (4, 2, 16, 8, 128)
        kv_dtype = bfloat16 (2 bytes)
        chunk_size_bytes = 4 * 2 * 16 * 8 * 128 * 2 = 262144
        pd_buffer_size = 64 MiB = 67108864
        aligned_buffer_size = (67108864 // 262144) * 262144 = 67108864
        chunk_token_size = kv_shape[2] = 16
        capacity_tokens = (67108864 // 262144) * 16 = 256 * 16 = 4096

    So pd_max_prefill_len=5000 must trigger ValueError (5000 > 4096),
    while pd_max_prefill_len=4096 must succeed (== capacity_tokens),
    and pd_max_prefill_len=0 must skip the check.
    """
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

    p1, p2, p3, p4 = _pd_backend_patches()

    def _make_receiver_config(pd_max_prefill_len: int) -> LMCacheEngineConfig:
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=pd_max_prefill_len,
        )

    def _make_sender_config(pd_max_prefill_len: int) -> LMCacheEngineConfig:
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=pd_max_prefill_len,
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

    # pd_max_prefill_len > capacity_tokens → ValueError for receiver
    with p1, p2, p3, p4:
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackend(_make_receiver_config(5000), metadata)

    # pd_max_prefill_len > capacity_tokens → ValueError for sender
    with p1, p2, p3, p4:
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackend(_make_sender_config(5000), metadata)

    # pd_max_prefill_len == capacity_tokens → success (boundary: exactly fits)
    with p1, p2, p3, p4:
        backend = PDBackend(_make_receiver_config(4096), metadata)
        backend.close()

    # pd_max_prefill_len = 0 → check skipped (default, legacy behaviour)
    with p1, p2, p3, p4:
        backend = PDBackend(_make_receiver_config(0), metadata)
        backend.close()


# ---------------------------------------------------------------------------
# Test 15: Receiver — _handle_alloc_request sends error response on exception
# ---------------------------------------------------------------------------


def test_receiver_handle_alloc_request_sends_error_on_exception(async_receiver):
    """
    When ``_async_allocate_and_put`` raises an exception inside
    ``_handle_alloc_request``, the method must send an error AllocResponse
    (all -1 remote_indexes) back to the sender rather than silently dropping
    the reply.  Without this, the sender would hang forever on
    ``recv_multipart``.

    Strategy: patch ``_async_allocate_and_put`` to raise RuntimeError, capture
    calls to the ROUTER send, and verify an error response is sent.
    """
    key = _make_key(0)
    alloc_req = AllocRequest(
        keys=[key.to_string()],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=16,
        req_id="req-err",
    )
    payload = msgspec.msgpack.encode(alloc_req)
    identity = b"fake-sender-identity"

    sent_frames: list[list[bytes]] = []

    # Mock ROUTER socket
    mock_socket = MagicMock()

    async def capture_send(frames):
        sent_frames.append(frames)

    mock_socket.send_multipart = AsyncMock(side_effect=capture_send)

    # Patch _async_allocate_and_put to raise
    original_allocate = async_receiver._async_allocate_and_put

    async def _failing_allocate(req):
        raise RuntimeError("injected failure")

    async_receiver._async_allocate_and_put = _failing_allocate

    async def run():
        await async_receiver._handle_alloc_request(mock_socket, identity, payload)

    asyncio.run(run())

    async_receiver._async_allocate_and_put = original_allocate

    # Exactly one send_multipart call (the error response)
    assert len(sent_frames) == 1, (
        f"Expected exactly one send_multipart call (error response), got {len(sent_frames)}"
    )
    # Verify frame layout: [identity, empty_delimiter, payload]
    frames = sent_frames[0]
    assert frames[0] == identity, "First frame should be the sender identity"
    assert frames[1] == b"", "Second frame should be the empty delimiter"
    # Decode the error response
    resp = msgspec.msgpack.decode(frames[2], type=PDMsg)
    assert isinstance(resp, AllocResponse), (
        f"Expected AllocResponse, got {type(resp)}"
    )
    assert resp.remote_indexes == [-1], (
        f"Expected all -1 remote_indexes, got {resp.remote_indexes}"
    )


# ---------------------------------------------------------------------------
# Test 16: Sender — per-receiver workers allow concurrent transfers to
#          different receivers while serializing within the same receiver
# ---------------------------------------------------------------------------


def test_sender_per_receiver_worker_concurrency(async_sender):
    """
    Verify that requests to *different* receivers proceed concurrently, while
    requests to the *same* receiver are still serialized (FIFO).

    Setup:
      - req-A and req-C target receiver-1 (SLOW transfer).
      - req-B targets receiver-2 (FAST transfer).

    With per-receiver workers:
      - req-B (receiver-2) should complete BEFORE req-A (receiver-1) because
        it does not wait for receiver-1's slow worker.
      - req-C (receiver-1) should complete AFTER req-A, not before.

    With the old global single worker, req-B would have been blocked behind
    req-A, and its completion would be delayed by the slow transfer.
    """
    SLOW = 0.25
    FAST = 0.05

    # Add a second receiver to the sender
    receiver_1_id = "127.0.0.1" + str(9100)  # already injected by fixture
    receiver_2_id = "127.0.0.1" + str(9200)

    alloc_response = AllocResponse(remote_indexes=[0])

    alloc_socket_2 = MagicMock()
    alloc_socket_2.recv_multipart = AsyncMock(
        return_value=[b"", msgspec.msgpack.encode(alloc_response)]
    )
    alloc_socket_2.send_multipart = AsyncMock()
    async_sender.initialized_peers.add(receiver_2_id)
    async_sender._async_alloc_sockets[receiver_2_id] = alloc_socket_2

    # Patch transfer_channel.async_batched_write per-receiver
    write_delays: dict[str, float] = {
        receiver_1_id: SLOW,
        receiver_2_id: FAST,
    }

    # We need the delay to vary by receiver; patch _async_transfer_task instead
    original_async_transfer = async_sender._async_transfer_task

    async def _slow_transfer_task(**kwargs):
        recv_id = kwargs.get("receiver_id", "")
        delay = write_delays.get(recv_id, FAST)
        num_chunks = len(kwargs.get("memory_objs", []))
        await asyncio.sleep(delay)
        # Minimal post-processing: trigger callback
        on_cb = kwargs.get("on_complete_callback")
        for key in kwargs.get("keys", []):
            if on_cb:
                try:
                    on_cb(key)
                except Exception:
                    pass
        # Release staging chunks like the real _async_transfer_task does.
        async_sender._release_sender_staging_chunks(num_chunks)

    async_sender._async_transfer_task = _slow_transfer_task

    done_events: dict[str, threading.Event] = {
        "req-A": threading.Event(),
        "req-B": threading.Event(),
        "req-C": threading.Event(),
    }
    completion_times: dict[str, float] = {}
    lock = threading.Lock()

    def make_cb(req_name: str):
        def cb(key):
            with lock:
                completion_times[req_name] = time.monotonic()
            done_events[req_name].set()

        return cb

    # Submit req-A → receiver-1 (slow), req-B → receiver-2 (fast),
    # req-C → receiver-1 (slow, after A)
    spec_a = _make_transfer_spec(
        receiver_host="127.0.0.1", init_port=9100, alloc_port=9101,
        req_id="req-A", is_last_prefill=True,
    )
    spec_b = _make_transfer_spec(
        receiver_host="127.0.0.1", init_port=9200, alloc_port=9201,
        req_id="req-B", is_last_prefill=True,
    )
    spec_c = _make_transfer_spec(
        receiver_host="127.0.0.1", init_port=9100, alloc_port=9101,
        req_id="req-C", is_last_prefill=True,
    )

    async_sender.batched_submit_put_task(
        [_make_key(10)], [_make_mem_obj(10)], transfer_spec=spec_a,
        on_complete_callback=make_cb("req-A"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(20)], [_make_mem_obj(20)], transfer_spec=spec_b,
        on_complete_callback=make_cb("req-B"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(30)], [_make_mem_obj(30)], transfer_spec=spec_c,
        on_complete_callback=make_cb("req-C"),
    )

    # Wait for all to complete
    for req_name, ev in done_events.items():
        finished = ev.wait(timeout=SLOW * 6)
        assert finished, f"{req_name} did not complete within timeout"

    # req-B (fast, different receiver) should finish before req-A (slow, receiver-1)
    assert completion_times["req-B"] < completion_times["req-A"], (
        f"Expected req-B (fast, receiver-2) to finish before req-A (slow, "
        f"receiver-1), but req-B={completion_times['req-B']:.3f} >= "
        f"req-A={completion_times['req-A']:.3f}. "
        "Per-receiver workers should not block each other."
    )
    # req-C should finish after req-A (both on receiver-1, serialized)
    assert completion_times["req-C"] >= completion_times["req-A"], (
        f"Expected req-C to finish after req-A (same receiver-1, FIFO), "
        f"but req-C={completion_times['req-C']:.3f} < "
        f"req-A={completion_times['req-A']:.3f}"
    )

    async_sender._async_transfer_task = original_async_transfer

