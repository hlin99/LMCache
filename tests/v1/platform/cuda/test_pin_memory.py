# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
import importlib
import sys

# Third Party
import pytest


class _FakeCudartFunction:
    def __init__(self, result: int) -> None:
        self._result = result
        self.argtypes = None
        self.calls: list[tuple[object, ...]] = []
        self.restype = None

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self._result


class _FakeLibcudart:
    def __init__(self, result: int = 0) -> None:
        self.cudaHostRegister = _FakeCudartFunction(result)
        self.cudaHostUnregister = _FakeCudartFunction(result)


class _FakeTorchCudart:
    def __init__(self, register_result: int = 0, unregister_result: int = 0) -> None:
        self._register_result = register_result
        self._unregister_result = unregister_result
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.register_calls.append((ptr, size, flags))
        return self._register_result

    def cudaHostUnregister(self, ptr: int) -> int:
        self.unregister_calls.append(ptr)
        return self._unregister_result


def _clear_lmcache_modules() -> None:
    """Remove cached ``lmcache`` modules before each fresh test import.

    This keeps module-level initialization from one test's torch stub from
    leaking into the next test.
    """
    for name in list(sys.modules):
        if name == "lmcache" or name.startswith("lmcache."):
            sys.modules.pop(name, None)


def _install_torch_stub(
    monkeypatch: pytest.MonkeyPatch,
    cudart_factory: Callable[[], object] | None = None,
) -> None:
    """Install a minimal torch stub for importing the CUDA pinning module.

    Args:
        monkeypatch: Pytest monkeypatch helper used to inject the stub.
        cudart_factory: Optional callable used as ``torch.cuda.cudart``.
    """
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.xpu = SimpleNamespace(is_available=lambda: False)
    torch.hpu = SimpleNamespace(is_available=lambda: False)

    if cudart_factory is not None:
        torch.cuda.cudart = cudart_factory

    monkeypatch.setitem(sys.modules, "torch", torch)


def _import_pin_memory_module() -> ModuleType:
    """Import the CUDA pinning module after clearing cached ``lmcache`` state."""
    _clear_lmcache_modules()
    return importlib.import_module("lmcache.v1.platform.cuda.pin_memory")


def _fail_cudart_lookup() -> object:
    """Raise a cudart lookup failure for fallback-path tests."""
    raise RuntimeError("no cudart")


def test_load_libcudart_binds_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(monkeypatch)
    module = _import_pin_memory_module()
    fake_lib = _FakeLibcudart()

    monkeypatch.setattr(module.ctypes.util, "find_library", lambda name: "libcudart.so")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: fake_lib)

    loaded = module._load_libcudart()

    assert loaded is fake_lib
    assert fake_lib.cudaHostRegister.restype is module.ctypes.c_int
    assert fake_lib.cudaHostRegister.argtypes == [
        module.ctypes.c_void_p,
        module.ctypes.c_size_t,
        module.ctypes.c_uint,
    ]
    assert fake_lib.cudaHostUnregister.restype is module.ctypes.c_int
    assert fake_lib.cudaHostUnregister.argtypes == [module.ctypes.c_void_p]


def test_backend_uses_torch_cudart_first(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cudart = _FakeTorchCudart()
    _install_torch_stub(monkeypatch, cudart_factory=lambda: fake_cudart)
    module = _import_pin_memory_module()
    monkeypatch.setattr(
        module,
        "_load_libcudart",
        lambda: pytest.fail("libcudart fallback should not run when torch works"),
    )

    backend = module.CudaPinMemoryBackend()

    assert backend.is_pin_supported() is True
    assert backend.pin_memory(1234, 64) is True
    assert backend.unpin_memory(1234) is True
    assert fake_cudart.register_calls == [
        (1234, 64, module.CudaPinMemoryBackend.PIN_FLAGS)
    ]
    assert fake_cudart.unregister_calls == [1234]


def test_backend_falls_back_to_libcudart(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_torch_stub(
        monkeypatch,
        cudart_factory=_fail_cudart_lookup,
    )
    module = _import_pin_memory_module()
    fake_lib = _FakeLibcudart()
    monkeypatch.setattr(module, "_load_libcudart", lambda: fake_lib)

    backend = module.CudaPinMemoryBackend()

    assert backend.is_pin_supported() is True
    assert backend.pin_memory(4321, 128) is True
    assert backend.unpin_memory(4321) is True

    register_ptr, register_size, register_flags = fake_lib.cudaHostRegister.calls[0]
    (unregister_ptr,) = fake_lib.cudaHostUnregister.calls[0]
    assert int(register_ptr.value) == 4321
    assert int(register_size.value) == 128
    assert int(register_flags.value) == module.CudaPinMemoryBackend.PIN_FLAGS
    assert int(unregister_ptr.value) == 4321


def test_backend_reports_unsupported_when_no_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_stub(monkeypatch)
    module = _import_pin_memory_module()
    monkeypatch.setattr(module, "_load_libcudart", lambda: None)

    backend = module.CudaPinMemoryBackend()

    assert backend.is_pin_supported() is False
    assert backend.pin_memory(1, 2) is False
    assert backend.unpin_memory(1) is False
