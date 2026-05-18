# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter.register_kv_caches``.

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

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.protocol import RequestType


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter through its real ``__init__`` with the network
    boundary stubbed out. Returns ``(adapter, send_mock, future)`` where
    ``send_mock`` is the patched ``send_lmcache_request`` and ``future``
    is its return value (a ``MagicMock`` whose ``result()`` defaults to
    succeed; tests can attach ``side_effect`` to simulate failures).
    """
    # Stub the MQ boundary so __init__'s chunk-size query and any later
    # send_lmcache_request call don't touch a real socket.
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda mq: 256)

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    monkeypatch.setattr(adapter_mod, "wrap_kv_caches", lambda kv: list(kv.values()))
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: "fake-layout",
        raising=False,
    )

    parallel_strategy = ParallelStrategy(
        use_mla=False,
        kv_world_size=1,
        kv_worker_id=0,
        actual_world_size=1,
        actual_worker_id=0,
        tp_size=1,
        pp_size=1,
    )
    adapter = LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
    )
    # __init__ issues exactly one MQ call (the chunk-size query). Reset
    # so individual tests start with a clean call count.
    send_mock.reset_mock()
    return adapter, send_mock, future


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, send_mock, _ = fake_adapter
    new_caches = {"layer.0": object(), "layer.1": object()}

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
        adapter.register_kv_caches({"layer.0": object()})


def test_get_finished_logs_store_failure_context(fake_adapter, monkeypatch):
    """Store failure should log request context for narrowing down root cause."""
    adapter, send_mock, future = fake_adapter
    adapter._ensure_heartbeat_started = lambda: None
    adapter._health_event.set()
    logger_error = MagicMock(name="logger_error")
    monkeypatch.setattr(adapter_mod.logger, "error", logger_error)
    request_id = "req-store-fail"
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[10, 11],
        start=0,
        end=4,
    )
    event = MagicMock(name="cuda_event")
    event.ipc_handle.return_value = b"fake-ipc"
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = True
    cuda_future.result.return_value = False
    future.to_cuda_future.return_value = cuda_future

    adapter.submit_store_request(request_id, op, event, cache_salt="salt")

    adapter.get_finished(set())
    logger_error.assert_called_once()
    log_args = logger_error.call_args[0]
    assert log_args[0].startswith("mp_store_path store_result_false")
    assert log_args[1:] == (
        "req-store-fail",
        "test-model",
        0,
        1,
        4,
        2,
        0,
        4,
        10,
        11,
    )
    send_mock.assert_called_once()


def test_get_finished_logs_store_result_exception(fake_adapter, monkeypatch):
    """Store result exceptions should include exception type in logs."""
    adapter, _send_mock, future = fake_adapter
    adapter._ensure_heartbeat_started = lambda: None
    adapter._health_event.set()
    logger_exception = MagicMock(name="logger_exception")
    monkeypatch.setattr(adapter_mod.logger, "exception", logger_exception)
    request_id = "req-store-exception"
    op = LoadStoreOp(
        token_ids=[9, 8, 7],
        block_ids=[21],
        start=0,
        end=3,
    )
    event = MagicMock(name="cuda_event")
    event.ipc_handle.return_value = b"fake-ipc"
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = True
    cuda_future.result.side_effect = RuntimeError("store boom")
    future.to_cuda_future.return_value = cuda_future

    adapter.submit_store_request(request_id, op, event)

    with pytest.raises(RuntimeError, match="store boom"):
        adapter.get_finished(set())
    logger_exception.assert_called_once()
    log_args = logger_exception.call_args[0]
    assert log_args[0].startswith("mp_store_path store_result_exception")
    (
        request_id_arg,
        model_name_arg,
        worker_id_arg,
        world_size_arg,
        token_count_arg,
        block_count_arg,
        start_arg,
        end_arg,
        exception_type_arg,
        exception_arg,
    ) = log_args[1:]
    assert request_id_arg == "req-store-exception"
    assert model_name_arg == "test-model"
    assert worker_id_arg == 0
    assert world_size_arg == 1
    assert token_count_arg == 3
    assert block_count_arg == 1
    assert start_arg == 0
    assert end_arg == 3
    assert exception_type_arg == "RuntimeError"
    assert isinstance(exception_arg, RuntimeError)
    assert str(exception_arg) == "store boom"
