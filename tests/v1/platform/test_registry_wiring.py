# SPDX-License-Identifier: Apache-2.0
"""Production wiring tests for registry-based platform backend resolution."""

# Standard

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform._registry import (
    _discover_base_classes,
    get_impl,
    resolve_impl,
)
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.base.tensor_from_ptr import TensorFromPtrBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.cuda.tensor_from_ptr import CudaTensorFromPtrBackend
from lmcache.v1.platform.cpu.tensor_from_ptr import CpuTensorFromPtrBackend
from lmcache.v1.platform.device_ext import DeviceExt
from lmcache.python_ops_fallback import _tensor_from_ptr


def test_pin_memory_backend_is_discovered_from_base_package() -> None:
    """PinMemoryBackend is discoverable from ``lmcache.v1.platform.base``."""
    assert PinMemoryBackend in _discover_base_classes()


def test_tensor_from_ptr_backend_is_discovered_from_base_package() -> None:
    """TensorFromPtrBackend is discoverable from ``lmcache.v1.platform.base``."""
    assert TensorFromPtrBackend in _discover_base_classes()


def test_get_impl_returns_cuda_pin_memory_backend() -> None:
    """Strict lookup resolves the CUDA concrete implementation."""
    assert get_impl(PinMemoryBackend, "cuda", "default") is CudaPinMemoryBackend


def test_get_impl_returns_tensor_from_ptr_backends() -> None:
    """Strict lookup resolves CPU and CUDA tensor-from-pointer backends."""
    assert get_impl(TensorFromPtrBackend, "cpu", "default") is CpuTensorFromPtrBackend
    assert get_impl(TensorFromPtrBackend, "cuda", "default") is (
        CudaTensorFromPtrBackend
    )


def test_get_impl_cpu_is_strict_and_raises() -> None:
    """Strict lookup raises when no CPU pin-memory backend exists."""
    with pytest.raises(ValueError):
        get_impl(PinMemoryBackend, "cpu", "default")


def test_resolve_impl_cpu_returns_noop_base_fallback() -> None:
    """resolve_impl falls back to PinMemoryBackend for CPU."""
    assert resolve_impl(PinMemoryBackend, "cpu", "default") is PinMemoryBackend


def test_resolve_impl_unsupported_tensor_from_ptr_device_raises() -> None:
    """TensorFromPtrBackend stays fail-fast for unsupported device types."""
    with pytest.raises(ValueError, match="No TensorFromPtrBackend implementation"):
        resolve_impl(TensorFromPtrBackend, "xpu", "default")


def test_device_ext_uses_override_backend_when_provided() -> None:
    """DeviceExt resolves CUDA pin memory through the registry."""

    class _FakeCudaDeviceInfo:
        @property
        def device_type(self) -> str:
            return "cuda"

    ext = DeviceExt(_FakeCudaDeviceInfo())  # type: ignore[arg-type]
    assert isinstance(ext._pin, CudaPinMemoryBackend)


def test_device_ext_uses_noop_backend_for_cpu_without_override() -> None:
    """DeviceExt resolves to no-op PinMemoryBackend for CPU/no override."""

    class _FakeCpuDeviceInfo:
        @property
        def device_type(self) -> str:
            return "cpu"

        @property
        def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
            return None

    ext = DeviceExt(_FakeCpuDeviceInfo())  # type: ignore[arg-type]
    assert type(ext._pin) is PinMemoryBackend
    assert ext.is_pin_supported is False


def test_tensor_from_ptr_cpu_zero_copy_round_trip() -> None:
    """CPU tensor-from-pointer reconstruction shares the original storage."""
    original = torch.arange(6, dtype=torch.int32).view(2, 3)

    rebuilt = _tensor_from_ptr(
        original.data_ptr(),
        tuple(original.shape),
        original.dtype,
        original.device,
    )

    rebuilt[1, 2] = -7
    assert original[1, 2].item() == -7
    assert rebuilt.data_ptr() == original.data_ptr()


def test_tensor_from_ptr_wrapper_rejects_unsupported_device() -> None:
    """The compatibility wrapper surfaces registry lookup failures."""
    with pytest.raises(ValueError, match="No TensorFromPtrBackend implementation"):
        _tensor_from_ptr(1, (1,), torch.uint8, "xpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tensor_from_ptr_cuda_round_trip() -> None:
    """CUDA tensor-from-pointer reconstruction materializes on the right device."""
    original = torch.arange(8, dtype=torch.bfloat16, device="cuda").view(2, 4)

    rebuilt = _tensor_from_ptr(
        original.data_ptr(),
        tuple(original.shape),
        original.dtype,
        original.device,
    )

    assert rebuilt.device == original.device
    assert rebuilt.dtype == original.dtype
    assert torch.equal(rebuilt.cpu(), original.cpu())
