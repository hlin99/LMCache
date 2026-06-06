# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.worker_transfer import DataTransferContext


@dataclass
class _FakeStoreContext:
    """Minimal non-GPU context for async store tests."""

    commit_impl: Callable[[list[torch.Tensor]], bool]
    prepare_result: tuple[list[torch.Tensor], list[int]] | None = None

    def __post_init__(self) -> None:
        self.layout_desc = SimpleNamespace(
            shapes=[torch.Size([2, 1, 1, 1])], dtypes=[torch.float32]
        )

    def prepare_store(
        self, _key: object, _instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        return self.prepare_result

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def close(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, gate: threading.Event):
        self._gate = gate

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        self._gate.wait(timeout=2)

    def query(self) -> bool:
        return self._gate.is_set()


class _FakeTorchDev:
    def __init__(self, gather_gate: threading.Event):
        self._stream = object()
        self._gather_gate = gather_gate

    def Stream(self) -> object:
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def current_stream(self) -> object:
        return self._stream

    def Event(self, interprocess: bool = False) -> _FakeEvent:
        return _FakeEvent(self._gather_gate)


def _install_fake_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_in_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        out = kwargs.get("out")
        if out is None:
            # Sync fallback path passes out=None: gather allocates its own
            # buffers and returns them, so mirror that contract here.
            return [torch.ones(1)]
        assert isinstance(out, list)
        for tensor in out:
            tensor.fill_(1.0)
        return out

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)


def _new_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gather_gate: threading.Event,
    commit_impl: Callable[[list[torch.Tensor]], bool],
    max_inflight: int = 8,
) -> DataTransferContext:
    monkeypatch.setattr(worker_transfer, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    ctx = DataTransferContext(max_inflight_stores=max_inflight)
    ctx._non_gpu_context = _FakeStoreContext(commit_impl=commit_impl)
    # Async tests exercise the async store path directly; enable it explicitly
    # so the capability probe (which needs real pinned memory) is bypassed.
    ctx._async_capable = True
    ctx._create_async_resources()
    return ctx


def test_submit_store_returns_pending_future_until_gather_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not future.query()
    gather_gate.set()
    assert future.result(timeout=1) is True
    ctx.close()


def test_submit_store_commit_waits_for_gather_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    commit_called = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_called.set()
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not commit_called.wait(timeout=0.05)
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert commit_called.is_set()
    ctx.close()


def test_submit_store_backpressure_blocks_when_inflight_cap_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    commit_gate = threading.Event()
    gather_gate.set()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_gate.wait(timeout=2)
        return True

    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=_commit, max_inflight=1
    )
    first = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    done = threading.Event()

    def _submit_second() -> None:
        try:
            ctx.submit_store(
                "r2",
                object(),
                1,
                {"k": torch.zeros(1)},
                [[0]],
                _FakeEvent(gather_gate),
                1,
            )
        finally:
            done.set()

    t = threading.Thread(target=_submit_second, daemon=True)
    t.start()
    assert not done.wait(timeout=0.1)
    commit_gate.set()
    assert first.result(timeout=1) is True
    t.join(timeout=1)
    assert done.is_set()
    ctx.close()


def test_close_drains_inflight_async_store(monkeypatch: pytest.MonkeyPatch) -> None:
    gather_gate = threading.Event()
    commit_gate = threading.Event()
    gather_gate.set()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_gate.wait(timeout=2)
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    closed = threading.Event()

    def _close() -> None:
        ctx.close()
        closed.set()

    t = threading.Thread(target=_close, daemon=True)
    t.start()
    assert not closed.wait(timeout=0.05)
    commit_gate.set()
    t.join(timeout=1)
    assert closed.is_set()
    assert future.result(timeout=1) is True


def test_commit_failure_sets_false_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        raise RuntimeError("commit failed")

    log_exception = MagicMock()
    monkeypatch.setattr(worker_transfer.logger, "exception", log_exception)
    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    gather_gate.set()
    assert future.result(timeout=1) is False
    log_exception.assert_called_once()
    ctx.close()


class _RecordingTorchDev:
    """torch_dev stub that records whether async primitives are touched."""

    def __init__(self) -> None:
        self.synchronize_calls = 0
        self.stream_calls = 0
        self.event_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def Stream(self) -> object:
        self.stream_calls += 1
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def Event(self, interprocess: bool = False) -> _FakeEvent:
        self.event_calls += 1
        return _FakeEvent(threading.Event())


def test_submit_store_sync_path_returns_resolved_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _RecordingTorchDev()
    monkeypatch.setattr(worker_transfer, "torch_dev", fake)
    _install_fake_gather(monkeypatch)
    ctx = DataTransferContext()
    ctx._non_gpu_context = _FakeStoreContext(commit_impl=lambda _c: True)
    assert ctx._async_capable is False

    future = ctx.submit_store(
        "r1",
        object(),
        1,
        {"k": torch.zeros(1)},
        [[0]],
        _FakeEvent(threading.Event()),
        1,
    )

    # Sync path resolves inline and never touches copy-stream / event primitives.
    assert future.query()
    assert future.result(timeout=1) is True
    assert fake.stream_calls == 0
    assert fake.event_calls == 0
    assert fake.synchronize_calls >= 1
    ctx.close()


def test_submit_store_dispatches_on_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = DataTransferContext()
    ctx._non_gpu_context = MagicMock()
    async_mock = MagicMock(return_value="async")
    sync_mock = MagicMock(return_value="sync")
    monkeypatch.setattr(ctx, "_submit_store_async", async_mock)
    monkeypatch.setattr(ctx, "_submit_store_sync", sync_mock)

    ctx._async_capable = True
    assert ctx.submit_store("r", None, 1, {}, [[0]], None, 1) == "async"
    async_mock.assert_called_once()
    sync_mock.assert_not_called()

    async_mock.reset_mock()
    ctx._async_capable = False
    assert ctx.submit_store("r", None, 1, {}, [[0]], None, 1) == "sync"
    sync_mock.assert_called_once()
    async_mock.assert_not_called()


def test_init_async_capability_non_capable_skips_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch_dev without Stream/Event is not async-capable.
    monkeypatch.setattr(worker_transfer, "torch_dev", object())
    ctx = DataTransferContext()
    assert ctx._copy_stream is None
    assert ctx._commit_executor is None

    ctx._init_async_capability()

    assert ctx._async_capable is False
    assert ctx._copy_stream is None
    assert ctx._commit_executor is None
    assert ctx._inflight_semaphore is None
    # close() must not raise even though async resources were never created.
    ctx.close()


def test_init_async_capability_capable_creates_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_transfer, "torch_dev", _FakeTorchDev(threading.Event()))
    # Make the pinned-memory probe succeed regardless of host CUDA support.
    real_empty = torch.empty

    def _fake_empty(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(worker_transfer.torch, "empty", _fake_empty)

    ctx = DataTransferContext()
    ctx._init_async_capability()

    assert ctx._async_capable is True
    assert ctx._copy_stream is not None
    assert ctx._commit_executor is not None
    assert ctx._inflight_semaphore is not None
    ctx.close()
