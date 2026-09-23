# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the XPU host-memory registration backend."""

# Standard
from types import SimpleNamespace
import importlib

# Third Party
import pytest

# First Party
from lmcache.v1.platform.devices.xpu import XpuDeviceSpec
from lmcache.v1.platform.devices.xpu.pin_memory import XpuPinMemoryBackend


class _FakeXpuOps:
    """Native-extension double that records host-registration calls."""

    def __init__(
        self,
        register_result: bool | Exception = True,
        unregister_result: bool | Exception = True,
    ) -> None:
        """Configure native call outcomes."""
        self.register_result = register_result
        self.unregister_result = unregister_result
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []

    def xpu_host_register(self, ptr: int, n_bytes: int) -> bool:
        """Record and emulate native host registration."""
        self.register_calls.append((ptr, n_bytes))
        if isinstance(self.register_result, Exception):
            raise self.register_result
        return self.register_result

    def xpu_host_unregister(self, ptr: int) -> bool:
        """Record and emulate native host unregistration."""
        self.unregister_calls.append(ptr)
        if isinstance(self.unregister_result, Exception):
            raise self.unregister_result
        return self.unregister_result


def _install_xpu_ops(
    monkeypatch: pytest.MonkeyPatch,
    ops: _FakeXpuOps,
) -> None:
    """Make a native XPU operation double importable for one test."""
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(
            xpu_host_register=ops.xpu_host_register,
            xpu_host_unregister=ops.xpu_host_unregister,
        ),
    )


def test_pin_memory_registers_the_requested_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public backend forwards a host range to the native extension."""
    ops = _FakeXpuOps()
    _install_xpu_ops(monkeypatch, ops)
    backend = XpuPinMemoryBackend()

    assert backend.pin_memory(0x1000, 64 << 20, 0x02) is True
    assert ops.register_calls == [(0x1000, 64 << 20)]


def test_unpin_memory_releases_the_original_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public backend releases the exact originally registered pointer."""
    ops = _FakeXpuOps()
    _install_xpu_ops(monkeypatch, ops)
    backend = XpuPinMemoryBackend()

    assert backend.unpin_memory(0x2000) is True
    assert ops.unregister_calls == [0x2000]


def test_backend_reports_native_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native ``False`` result remains a failed registration."""
    _install_xpu_ops(monkeypatch, _FakeXpuOps(register_result=False))

    assert XpuPinMemoryBackend().pin_memory(0x1000, 4096) is False


def test_backend_rejects_invalid_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null pointers and empty ranges never reach native registration."""
    ops = _FakeXpuOps()
    _install_xpu_ops(monkeypatch, ops)
    backend = XpuPinMemoryBackend()

    assert backend.pin_memory(0, 4096) is False
    assert backend.pin_memory(0x1000, 0) is False
    assert ops.register_calls == []


def test_backend_handles_native_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native runtime failure degrades to the synchronous-copy fallback."""
    _install_xpu_ops(
        monkeypatch,
        _FakeXpuOps(register_result=RuntimeError("registration failed")),
    )

    assert XpuPinMemoryBackend().pin_memory(0x1000, 4096) is False


def test_backend_handles_native_unregistration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native release failure is visible as a failed unregistration."""
    _install_xpu_ops(
        monkeypatch,
        _FakeXpuOps(unregister_result=RuntimeError("release failed")),
    )

    assert XpuPinMemoryBackend().unpin_memory(0x1000) is False


def test_backend_is_unsupported_without_native_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing native extension cannot claim pinning support."""

    def _raise_import_error(name: str) -> SimpleNamespace:
        raise ImportError(f"{name} is unavailable")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    backend = XpuPinMemoryBackend()

    assert backend.is_pin_supported is False
    assert backend.pin_memory(0x1000, 4096) is False
    assert backend.unpin_memory(0x1000) is False


def test_xpu_device_spec_uses_xpu_pin_memory_backend() -> None:
    """The XPU device specification exposes the XPU pinning backend."""
    assert XpuDeviceSpec().pin_memory_backend is XpuPinMemoryBackend
