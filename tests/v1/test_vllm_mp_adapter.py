# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter`` and
``LMCacheMPSchedulerAdapter``.

Behavioural coverage of the heartbeat-driven recovery path
(``HeartbeatThread.register_recover_callback`` →
worker re-registration) lives in the buildkite end-to-end test
``.buildkite/k3_tests/multiprocess/scripts/run-restart-recovery.sh``.
That path requires driving the periodic-thread tick loop, which is
deliberately not reachable through any public interface.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.protocol import RequestType


def _make_parallel_strategy() -> ParallelStrategy:
    return ParallelStrategy(
        use_mla=False,
        kv_world_size=1,
        kv_worker_id=0,
        actual_world_size=1,
        actual_worker_id=0,
        tp_size=1,
        pp_size=1,
    )


def _chunk_size_timeout(*args, **kwargs) -> int:
    """Stub for ``get_lmcache_chunk_size`` that always raises TimeoutError."""
    raise TimeoutError("server down")


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter through its real ``__init__`` with the network
    boundary stubbed out. Returns ``(adapter, send_mock, future)`` where
    ``send_mock`` is the patched ``send_lmcache_request`` and ``future``
    is its return value (a ``MagicMock`` whose ``result()`` defaults to
    succeed; tests can attach ``side_effect`` to simulate failures).
    """
    # Stub the MQ boundary so chunk-size query and any later
    # send_lmcache_request call don't touch a real socket.
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))
    # ``vllm_layout_hints`` returns a ``LayoutHints`` (TypedDict / dict at
    # runtime); the production path performs item assignment on it
    # (``layout_hints["inference_engine_logical_block_size"] = ...``), so
    # the stub must also be a real dict — a string would raise
    # ``TypeError: 'str' object does not support item assignment``.
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
    )

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=5.0,
    )
    # chunk_size is now fetched lazily (not in __init__), so send_mock starts
    # clean with zero calls.
    return adapter, send_mock, future


# ---------------------------------------------------------------------------
# Lazy chunk-size initialisation tests
# ---------------------------------------------------------------------------


def test_worker_init_does_not_fetch_chunk_size(monkeypatch):
    """__init__ must NOT block on the server: blocks_in_chunk is None after init."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)

    chunk_size_calls = []

    def _spy_chunk_size(*a, **kw):
        chunk_size_calls.append(1)
        return 256

    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _spy_chunk_size)

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=5.0,
    )

    assert chunk_size_calls == [], (
        "get_lmcache_chunk_size must not be called in __init__"
    )
    assert adapter.blocks_in_chunk is None


def test_scheduler_init_does_not_fetch_chunk_size(monkeypatch):
    """LMCacheMPSchedulerAdapter.__init__ must not block on the server."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)

    chunk_size_calls = []

    def _spy_chunk_size(*a, **kw):
        chunk_size_calls.append(1)
        return 256

    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _spy_chunk_size)

    adapter = LMCacheMPSchedulerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=5.0,
    )

    assert chunk_size_calls == [], (
        "get_lmcache_chunk_size must not be called in __init__"
    )
    assert adapter.chunk_size is None
    assert adapter.blocks_in_chunk is None


def test_register_kv_caches_skipped_when_server_unavailable(monkeypatch):
    """register_kv_caches is a no-op when the server never responds."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _chunk_size_timeout)
    send_mock = MagicMock(name="send_lmcache_request")
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=1.0,
    )

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    # Should not raise and should be a no-op
    adapter.register_kv_caches({"layer.0": fake_tensor})

    assert not adapter.is_healthy
    assert send_mock.call_count == 0
    # kv_caches should remain empty because registration was skipped
    assert adapter.kv_caches == {}


def test_submit_store_skipped_when_chunk_size_unavailable(monkeypatch):
    """submit_store_request is a no-op when the chunk-size fetch times out."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _chunk_size_timeout)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", MagicMock())

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=1.0,
    )

    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[0], start=0, end=4)
    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert "req-1" not in adapter.store_futures


def test_submit_retrieve_marks_error_blocks_when_chunk_size_unavailable(monkeypatch):
    """submit_retrieve_request adds block IDs to error set when server is down."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _chunk_size_timeout)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", MagicMock())

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=1.0,
    )

    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[7, 8], start=0, end=4)
    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    assert {7, 8}.issubset(adapter.error_block_ids)


def test_chunk_size_fetched_only_once(monkeypatch):
    """get_lmcache_chunk_size is called exactly once across multiple operations."""
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)

    call_count = []

    def _counting_chunk_size(*a, **kw):
        call_count.append(1)
        return 256

    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", _counting_chunk_size)
    future = MagicMock()
    future.result.return_value = None
    monkeypatch.setattr(
        adapter_mod, "send_lmcache_request", MagicMock(return_value=future)
    )
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))
    monkeypatch.setattr("lmcache.integration.vllm.utils.vllm_layout_hints", lambda: {})

    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=_make_parallel_strategy(),
        mq_timeout=5.0,
    )
    assert len(call_count) == 0, "should not call chunk_size in __init__"

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})
    assert len(call_count) == 1, "should call chunk_size exactly once on first use"

    adapter.register_kv_caches({"layer.0": fake_tensor})
    assert len(call_count) == 1, "should not call chunk_size again on second use"


# ---------------------------------------------------------------------------
# Existing register_kv_caches tests
# ---------------------------------------------------------------------------


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, send_mock, _ = fake_adapter
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    new_caches = {"layer.0": fake_tensor, "layer.1": fake_tensor}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_register_kv_caches_raises_connection_error_on_timeout(fake_adapter):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    adapter, _send_mock, future = fake_adapter
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        fake_tensor = MagicMock()
        fake_tensor.device.type = "cuda"
        adapter.register_kv_caches({"layer.0": fake_tensor})


def test_register_kv_caches_cpu_submits_non_gpu_context_registration(
    fake_adapter, monkeypatch
):
    """CPU KV cache registration routes to REGISTER_KV_CACHE_NON_GPU_CONTEXT."""
    adapter, send_mock, _ = fake_adapter
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
        raising=False,
    )
    cpu_kv = {"layer.0": torch.randn(2, 8, 4, 2, 8)}

    adapter.register_kv_caches(cpu_kv)

    assert adapter.kv_caches is cpu_kv
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT
    assert len(args[2]) == 1


def test_submit_store_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_store_request stores the returned future in store_futures."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_store.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[0], start=0, end=4)

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.called
    assert transfer_ctx.submit_store.call_args.kwargs == {}
    assert adapter.store_futures["req-1"] is fake_future


def test_submit_retrieve_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_retrieve_request stores returned future and block IDs."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_retrieve.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[0],
        start=0,
        end=4,
        skip_first_n_tokens=1,
    )

    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_retrieve.called
    assert transfer_ctx.submit_retrieve.call_args.kwargs == {"skip_first_n_tokens": 1}
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])
