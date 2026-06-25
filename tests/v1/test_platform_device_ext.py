# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
import lmcache.v1.cache_engine as cache_engine_module
import lmcache.v1.platform.device_ext as device_ext_module
import lmcache.v1.transfer_channel.transfer_utils as transfer_utils
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.device_ext import DeviceExt, register_pin_memory_backend


class _FakeBuffer:
    def __init__(self, device: str, ptr: int = 1234) -> None:
        self.device = torch.device(device)
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


@pytest.fixture(autouse=True)
def restore_pin_memory_backends() -> Generator[None, None, None]:
    saved = dict(device_ext_module._PIN_MEMORY_BACKENDS)
    try:
        yield
    finally:
        device_ext_module._PIN_MEMORY_BACKENDS.clear()
        device_ext_module._PIN_MEMORY_BACKENDS.update(saved)


def _make_config(buffer_device: str) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    config.extra_config = {"enable_nixl_storage": True}
    config.nixl_buffer_device = buffer_device
    config.nixl_buffer_size = 4096
    return config


def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="device-ext-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 1, 16, 1, 1),
    )


def test_device_ext_falls_back_to_base_backend_for_unknown_device() -> None:
    ext = DeviceExt("custom-device")

    assert ext.pin_memory(1, 2, 3) is False
    assert ext.unpin_memory(1) is False
    assert ext.is_pin_supported is False


def test_register_pin_memory_backend_dispatches_through_registry() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class _RecordingBackend(PinMemoryBackend):
        def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
            calls.append(("pin", (ptr, size, flags)))
            return True

        def unpin_memory(self, ptr: int) -> bool:
            calls.append(("unpin", (ptr,)))
            return True

        def is_pin_supported(self) -> bool:
            return True

    register_pin_memory_backend("test-device", _RecordingBackend)

    ext = DeviceExt("test-device")

    assert ext.pin_memory(11, 22, 33) is True
    assert ext.unpin_memory(11) is True
    assert ext.is_pin_supported is True
    assert calls == [("pin", (11, 22, 33)), ("unpin", (11,))]


def test_cuda_backend_registers_pin_memory_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    def _fake_init(self: CudaPinMemoryBackend) -> None:
        return None

    def _fake_pin_memory(
        self: CudaPinMemoryBackend, ptr: int, size: int, flags: int = 0
    ) -> bool:
        calls.append((ptr, size, flags))
        return True

    monkeypatch.setattr(CudaPinMemoryBackend, "__init__", _fake_init)
    monkeypatch.setattr(CudaPinMemoryBackend, "pin_memory", _fake_pin_memory)

    ext = DeviceExt("cuda")

    assert ext.pin_memory(7, 8, 9) is True
    assert calls == [(7, 8, 9)]


@pytest.mark.no_shared_allocator
def test_create_memory_allocator_pins_cpu_buffer_by_tensor_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config("cpu")
    metadata = _make_metadata()
    buffer = _FakeBuffer("cpu")
    pin_calls: list[tuple[int, int, int]] = []
    set_device_calls: list[Any] = []
    sentinel = object()

    def _fake_pin_memory(ptr: int, size: int, flags: int = 0) -> bool:
        pin_calls.append((ptr, size, flags))
        return True

    def _fake_empty_cpu(*args: Any, **kwargs: Any) -> _FakeBuffer:
        return buffer

    monkeypatch.setattr(
        transfer_utils, "get_correct_device", lambda device, worker_id: "cpu:0"
    )
    monkeypatch.setattr(cache_engine_module.torch, "empty", _fake_empty_cpu)
    monkeypatch.setattr(
        cache_engine_module,
        "torch_dev",
        SimpleNamespace(
            ext=SimpleNamespace(pin_memory=_fake_pin_memory),
            set_device=lambda device: set_device_calls.append(device),
        ),
    )
    monkeypatch.setattr(
        cache_engine_module,
        "PagedTensorMemoryAllocator",
        lambda *args, **kwargs: sentinel,
    )

    allocator = LMCacheEngineBuilder._Create_memory_allocator(config, metadata)

    assert allocator is sentinel
    assert pin_calls == [(buffer.data_ptr(), config.nixl_buffer_size, 0)]
    assert set_device_calls == []


@pytest.mark.no_shared_allocator
def test_create_memory_allocator_sets_device_for_non_cpu_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config("cuda")
    metadata = _make_metadata()
    buffer = _FakeBuffer("cuda:0")
    pin_calls: list[tuple[int, int, int]] = []
    set_device_calls: list[Any] = []
    sentinel = object()

    def _fake_pin_memory(ptr: int, size: int, flags: int = 0) -> bool:
        pin_calls.append((ptr, size, flags))
        return True

    def _fake_empty_cuda(*args: Any, **kwargs: Any) -> _FakeBuffer:
        return buffer

    monkeypatch.setattr(
        transfer_utils, "get_correct_device", lambda device, worker_id: "cuda:3"
    )
    monkeypatch.setattr(cache_engine_module.torch, "empty", _fake_empty_cuda)
    monkeypatch.setattr(
        cache_engine_module,
        "torch_dev",
        SimpleNamespace(
            ext=SimpleNamespace(pin_memory=_fake_pin_memory),
            set_device=lambda device: set_device_calls.append(device),
        ),
    )
    monkeypatch.setattr(
        cache_engine_module,
        "PagedTensorMemoryAllocator",
        lambda *args, **kwargs: sentinel,
    )

    allocator = LMCacheEngineBuilder._Create_memory_allocator(config, metadata)

    assert allocator is sentinel
    assert pin_calls == []
    assert set_device_calls == ["cuda:3"]


def test_stub_cpu_device_set_device_raises() -> None:
    with pytest.raises(
        RuntimeError,
        match="StubCPUDevice does not support set_device",
    ):
        StubCPUDevice().set_device(0)
