# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PdL2Adapter (async PD sender/receiver L2 adapter).

Covers the adapter's L2AdapterInterface implementation: event fds,
store/lookup/load task submission and completion, reservation-based
admission control, and graceful shutdown.

All I/O is mocked — no NIXL, CUDA, or real ZMQ peers are required.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import asyncio

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.pd_l2_adapter import (
    AllocRequest,
    AllocResponse,
    CancelNotif,
    PDMsg,
    PdL2Adapter,
    PdL2AdapterConfig,
    ProxyNotif,
    ReservationManager,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SHAPE = [4, 2, 16, 8, 128]


def _make_object_key(i: int) -> ObjectKey:
    """Create a minimal ObjectKey for testing."""
    return ObjectKey(
        chunk_hash=i.to_bytes(8, "big"),
        model_name="test-model",
        kv_rank=0,
    )


def _make_mem_obj(idx: int = 0) -> MemoryObj:
    """Create a mock MemoryObj."""
    obj = MagicMock(spec=MemoryObj)
    obj.meta = SimpleNamespace(
        address=idx,
        fmt=MemoryFormat.KV_2LTD,
        shape=torch.Size(_DEFAULT_SHAPE),
        dtype=torch.bfloat16,
    )
    obj.get_ref_count.return_value = 1
    obj.get_size.return_value = 1024
    obj.tensor = torch.zeros(1)
    return obj


def _make_alloc_req(
    keys: list[ObjectKey],
    last_chunk_toks: int = 16,
    req_id: str = "",
    is_last_batch: bool = False,
    total_chunks: int = 0,
    shape: list[int] | None = None,
) -> AllocRequest:
    """Create an AllocRequest for testing."""
    return AllocRequest(
        keys=[str(k) for k in keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=list(shape or _DEFAULT_SHAPE),
        dtype="bfloat16",
        last_chunk_toks=last_chunk_toks,
        req_id=req_id,
        is_last_batch=is_last_batch,
        total_chunks=total_chunks,
    )


def _sender_config() -> PdL2AdapterConfig:
    """Create a minimal sender config."""
    return PdL2AdapterConfig.from_dict({
        "role": "sender",
        "peer_host": "127.0.0.1",
        "peer_init_port": [9051],
        "peer_alloc_port": [9052],
    })


def _receiver_config() -> PdL2AdapterConfig:
    """Create a minimal receiver config."""
    return PdL2AdapterConfig.from_dict({
        "role": "receiver",
        "peer_host": "127.0.0.1",
        "peer_init_port": [9051],
        "peer_alloc_port": [9052],
    })


# ---------------------------------------------------------------------------
# Wire-protocol message tests
# ---------------------------------------------------------------------------


class TestWireProtocol:
    """Tests for msgspec-based message serialization."""

    def test_alloc_request_roundtrip(self) -> None:
        """AllocRequest serializes and deserializes correctly."""
        req = AllocRequest(
            keys=["k0", "k1"],
            fmt=0,
            shape=_DEFAULT_SHAPE,
            dtype="bfloat16",
            last_chunk_toks=7,
            req_id="test-req",
            is_last_batch=True,
            total_chunks=5,
        )
        data = msgspec.msgpack.encode(req)
        decoded = msgspec.msgpack.decode(data, type=PDMsg)
        assert isinstance(decoded, AllocRequest)
        assert decoded.keys == ["k0", "k1"]
        assert decoded.req_id == "test-req"
        assert decoded.is_last_batch is True
        assert decoded.total_chunks == 5

    def test_alloc_response_roundtrip(self) -> None:
        """AllocResponse serializes and deserializes correctly."""
        resp = AllocResponse(remote_indexes=[10, 20, -1])
        data = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(data, type=PDMsg)
        assert isinstance(decoded, AllocResponse)
        assert decoded.remote_indexes == [10, 20, -1]
        assert decoded.already_sent_indexes == []

    def test_proxy_notif_roundtrip(self) -> None:
        """ProxyNotif serializes and deserializes correctly."""
        notif = ProxyNotif(req_id="req-123")
        data = msgspec.msgpack.encode(notif)
        decoded = msgspec.msgpack.decode(data, type=PDMsg)
        assert isinstance(decoded, ProxyNotif)
        assert decoded.req_id == "req-123"

    def test_cancel_notif_roundtrip(self) -> None:
        """CancelNotif serializes and deserializes correctly."""
        cancel = CancelNotif(req_id="req-abort", keys=["k0", "k1"])
        data = msgspec.msgpack.encode(cancel)
        decoded = msgspec.msgpack.decode(data, type=PDMsg)
        assert isinstance(decoded, CancelNotif)
        assert decoded.req_id == "req-abort"
        assert decoded.keys == ["k0", "k1"]

    def test_alloc_request_defaults(self) -> None:
        """AllocRequest optional fields have correct defaults."""
        req = AllocRequest(
            keys=["k0"],
            fmt=0,
            shape=_DEFAULT_SHAPE,
            dtype="bfloat16",
            last_chunk_toks=16,
        )
        assert req.req_id == ""
        assert req.is_last_batch is False
        assert req.total_chunks == 0


# ---------------------------------------------------------------------------
# ReservationManager tests
# ---------------------------------------------------------------------------


class TestReservationManager:
    """Tests for async reservation-based admission control."""

    def test_admit_within_capacity(self) -> None:
        """Admission succeeds when capacity is available."""
        mgr = ReservationManager(
            total_chunks=10,
            allocation_timeout=1.0,
            condition_poll_interval=0.1,
        )

        async def run() -> None:
            mgr.init_async_admit_condition()
            assert await mgr.async_try_admit("req-a", 5)
            assert await mgr.async_try_admit("req-b", 5)

        asyncio.run(run())
        assert mgr._total_reserved == 10

    def test_admit_times_out_when_full(self) -> None:
        """Admission times out when buffer is fully reserved."""
        mgr = ReservationManager(
            total_chunks=5,
            allocation_timeout=0.05,
            condition_poll_interval=0.01,
        )

        async def run() -> None:
            mgr.init_async_admit_condition()
            assert await mgr.async_try_admit("req-a", 5)
            assert not await mgr.async_try_admit("req-b", 1)

        asyncio.run(run())

    def test_release_unblocks_waiting(self) -> None:
        """Releasing a reservation unblocks a waiting admission."""
        mgr = ReservationManager(
            total_chunks=5,
            allocation_timeout=2.0,
            condition_poll_interval=0.01,
        )

        async def run() -> None:
            mgr.init_async_admit_condition()
            await mgr.async_try_admit("req-a", 5)

            result: list[bool] = []

            async def waiter() -> None:
                result.append(await mgr.async_try_admit("req-b", 3))

            async def releaser() -> None:
                await asyncio.sleep(0.05)
                await mgr.async_release_reservation("req-a")

            await asyncio.gather(waiter(), releaser())
            assert result == [True]
            assert mgr._total_reserved == 3

        asyncio.run(run())

    def test_get_total_chunks(self) -> None:
        """get_total_chunks returns the initialised value."""
        mgr = ReservationManager(10, 1.0, 0.1)
        assert mgr.get_total_chunks() == 10


# ---------------------------------------------------------------------------
# PdL2Adapter receiver allocation tests
# ---------------------------------------------------------------------------


class TestReceiverAllocateAndPut:
    """Tests for _async_allocate_and_put on the receiver."""

    @pytest.fixture()
    def receiver(self) -> PdL2Adapter:
        """Create a receiver adapter with mocked internals."""
        with patch(
            "lmcache.v1.distributed.l2_adapters.pd_l2_adapter.zmq"
        ):
            adapter = PdL2Adapter(_receiver_config())
        yield adapter
        adapter.close()

    def test_allocate_and_put_single_batch(self, receiver: PdL2Adapter) -> None:
        """Single-batch allocation succeeds and stores keys."""
        counter = [0]

        def alloc(shape, dtype, fmt):
            counter[0] += 1
            return _make_mem_obj(idx=counter[0])

        receiver._receiver_allocate = alloc

        keys = [_make_object_key(i) for i in range(3)]
        req = _make_alloc_req(
            keys, req_id="req-1", total_chunks=3, is_last_batch=True
        )

        resp = asyncio.run(receiver._async_allocate_and_put(req))
        assert len(resp.remote_indexes) == 3
        assert -1 not in resp.remote_indexes

    def test_reject_legacy_zero_total_chunks(
        self, receiver: PdL2Adapter
    ) -> None:
        """req_id + total_chunks=0 raises RuntimeError."""
        receiver._receiver_allocate = lambda *a, **kw: _make_mem_obj()
        req = _make_alloc_req(
            [_make_object_key(0)],
            req_id="req-legacy",
            total_chunks=0,
        )

        async def run() -> None:
            with pytest.raises(RuntimeError, match="total_chunks"):
                await receiver._async_allocate_and_put(req)

        asyncio.run(run())

    def test_fail_fast_overflow(self, receiver: PdL2Adapter) -> None:
        """Cumulative chunks > declared total_chunks raises RuntimeError."""
        counter = [0]

        def alloc(shape, dtype, fmt):
            counter[0] += 1
            return _make_mem_obj(idx=counter[0])

        receiver._receiver_allocate = alloc
        req_id = "req-overflow"

        async def run() -> None:
            # First batch: 3 keys with total_chunks=3.
            r1 = await receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_object_key(i) for i in range(3)],
                    req_id=req_id,
                    total_chunks=3,
                )
            )
            assert len(r1.remote_indexes) == 3

            # Second batch overflows.
            with pytest.raises(RuntimeError, match="total_chunks"):
                await receiver._async_allocate_and_put(
                    _make_alloc_req(
                        [_make_object_key(100)],
                        req_id=req_id,
                        total_chunks=3,
                    )
                )

        asyncio.run(run())

    def test_is_last_batch_cleanup(self, receiver: PdL2Adapter) -> None:
        """is_last_batch=True removes req_id from _req_allocated_keys."""
        counter = [0]

        def alloc(shape, dtype, fmt):
            counter[0] += 1
            return _make_mem_obj(idx=counter[0])

        receiver._receiver_allocate = alloc
        req_id = "req-lifecycle"

        async def run() -> None:
            await receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_object_key(i) for i in range(2)],
                    req_id=req_id,
                    total_chunks=3,
                    is_last_batch=False,
                )
            )
            assert req_id in receiver._req_allocated_keys

            await receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_object_key(10)],
                    req_id=req_id,
                    total_chunks=3,
                    is_last_batch=True,
                )
            )
            assert req_id not in receiver._req_allocated_keys

        asyncio.run(run())

    def test_alloc_timeout_rollback(self, receiver: PdL2Adapter) -> None:
        """Allocation timeout rolls back current and prior batches."""
        counter = [0]

        def alloc_first_only(shape, dtype, fmt):
            counter[0] += 1
            return _make_mem_obj(idx=counter[0]) if counter[0] <= 3 else None

        receiver._receiver_allocate = alloc_first_only
        receiver._allocation_timeout = 0.05
        req_id = "req-timeout"

        async def run() -> None:
            r1 = await receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_object_key(i) for i in range(3)],
                    req_id=req_id,
                    total_chunks=6,
                    is_last_batch=False,
                )
            )
            assert len(r1.remote_indexes) == 3

            with pytest.raises(RuntimeError, match="timeout"):
                await receiver._async_allocate_and_put(
                    _make_alloc_req(
                        [_make_object_key(100 + i) for i in range(3)],
                        req_id=req_id,
                        total_chunks=6,
                        is_last_batch=True,
                    )
                )

            # Prior batch keys should be cleaned up.
            assert req_id not in receiver._req_allocated_keys

        asyncio.run(run())


# ---------------------------------------------------------------------------
# PdL2Adapter L2 interface tests (receiver)
# ---------------------------------------------------------------------------


class TestReceiverL2Interface:
    """Tests for the L2AdapterInterface methods on the receiver."""

    @pytest.fixture()
    def receiver(self) -> PdL2Adapter:
        """Create a receiver adapter with mocked ZMQ."""
        with patch(
            "lmcache.v1.distributed.l2_adapters.pd_l2_adapter.zmq"
        ):
            adapter = PdL2Adapter(_receiver_config())
        yield adapter
        adapter.close()

    def test_event_fds_are_distinct(self, receiver: PdL2Adapter) -> None:
        """All three event fds are unique file descriptors."""
        fds = {
            receiver.get_store_event_fd(),
            receiver.get_lookup_and_lock_event_fd(),
            receiver.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_submit_store_receiver_immediate(
        self, receiver: PdL2Adapter
    ) -> None:
        """On receiver, submit_store_task completes immediately."""
        task_id = receiver.submit_store_task([], [])
        completed = receiver.pop_completed_store_tasks()
        assert task_id in completed
        assert completed[task_id] is True

    def test_lookup_finds_stored_keys(
        self, receiver: PdL2Adapter
    ) -> None:
        """Lookup returns bits for keys present in local data."""
        k1 = _make_object_key(1)
        k2 = _make_object_key(2)
        k3 = _make_object_key(3)

        # Manually put some keys into the receiver's data store.
        receiver._data[k1] = _make_mem_obj(1)
        receiver._data[k3] = _make_mem_obj(3)

        task_id = receiver.submit_lookup_and_lock_task([k1, k2, k3])
        bitmap = receiver.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True  # k1 found
        assert bitmap.test(1) is False  # k2 not found
        assert bitmap.test(2) is True  # k3 found

    def test_lookup_result_consumed_once(
        self, receiver: PdL2Adapter
    ) -> None:
        """query_lookup_and_lock_result returns None after first call."""
        task_id = receiver.submit_lookup_and_lock_task([])
        assert receiver.query_lookup_and_lock_result(task_id) is not None
        assert receiver.query_lookup_and_lock_result(task_id) is None

    def test_submit_unlock(self, receiver: PdL2Adapter) -> None:
        """submit_unlock decrements lock count."""
        k = _make_object_key(42)
        receiver._data[k] = _make_mem_obj(42)
        receiver.submit_lookup_and_lock_task([k])
        assert receiver._locked_keys.get(k, 0) == 1
        receiver.submit_unlock([k])
        assert k not in receiver._locked_keys

    def test_load_copies_data(self, receiver: PdL2Adapter) -> None:
        """submit_load_task copies tensor data from stored objects."""
        k = _make_object_key(99)
        src = MagicMock(spec=MemoryObj)
        src.tensor = torch.ones(4)
        receiver._data[k] = src

        dst = MagicMock(spec=MemoryObj)
        dst.tensor = torch.zeros(4)
        dst.meta = SimpleNamespace(
            address=0,
            fmt=MemoryFormat.KV_2LTD,
            shape=torch.Size(_DEFAULT_SHAPE),
            dtype=torch.bfloat16,
        )

        task_id = receiver.submit_load_task([k], [dst])
        bitmap = receiver.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is True
        # dst.tensor.copy_ should have been called with src.tensor.
        dst.tensor.copy_.assert_called_once_with(src.tensor)

    def test_load_missing_key(self, receiver: PdL2Adapter) -> None:
        """submit_load_task returns 0-bit for missing keys."""
        k = _make_object_key(999)
        dst = _make_mem_obj(0)
        task_id = receiver.submit_load_task([k], [dst])
        bitmap = receiver.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False


# ---------------------------------------------------------------------------
# PdL2Adapter sender tests
# ---------------------------------------------------------------------------


class TestSenderL2Interface:
    """Tests for the L2AdapterInterface methods on the sender."""

    @pytest.fixture()
    def sender(self) -> PdL2Adapter:
        """Create a sender adapter with mocked ZMQ."""
        with patch(
            "lmcache.v1.distributed.l2_adapters.pd_l2_adapter.zmq"
        ):
            adapter = PdL2Adapter(_sender_config())
        yield adapter
        adapter.close()

    def test_event_fds_are_distinct(self, sender: PdL2Adapter) -> None:
        """All three event fds are unique file descriptors."""
        fds = {
            sender.get_store_event_fd(),
            sender.get_lookup_and_lock_event_fd(),
            sender.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_lookup_always_empty_on_sender(
        self, sender: PdL2Adapter
    ) -> None:
        """On sender, lookup returns all zeros."""
        keys = [_make_object_key(i) for i in range(3)]
        task_id = sender.submit_lookup_and_lock_task(keys)
        bitmap = sender.query_lookup_and_lock_result(task_id)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i) is False

    def test_load_always_empty_on_sender(
        self, sender: PdL2Adapter
    ) -> None:
        """On sender, load returns all zeros."""
        keys = [_make_object_key(0)]
        objs = [_make_mem_obj(0)]
        task_id = sender.submit_load_task(keys, objs)
        bitmap = sender.query_load_result(task_id)
        assert bitmap is not None
        assert bitmap.test(0) is False

    def test_report_status(self, sender: PdL2Adapter) -> None:
        """report_status returns expected fields."""
        status = sender.report_status()
        assert status["is_healthy"] is True
        assert status["role"] == "sender"


# ---------------------------------------------------------------------------
# Close tests
# ---------------------------------------------------------------------------


class TestClose:
    """Tests for adapter shutdown."""

    def test_close_sender(self) -> None:
        """Sender close() stops the background thread."""
        with patch(
            "lmcache.v1.distributed.l2_adapters.pd_l2_adapter.zmq"
        ):
            adapter = PdL2Adapter(_sender_config())
        assert adapter._sender_thread.is_alive()
        adapter.close()
        assert not adapter._sender_thread.is_alive()

    def test_close_receiver(self) -> None:
        """Receiver close() stops the background thread."""
        with patch(
            "lmcache.v1.distributed.l2_adapters.pd_l2_adapter.zmq"
        ):
            adapter = PdL2Adapter(_receiver_config())
        assert adapter._recv_thread.is_alive()
        adapter.close()
        assert not adapter._recv_thread.is_alive()
