# SPDX-License-Identifier: Apache-2.0
"""Production wiring tests for registry-based platform backend resolution."""

# Standard
from typing import Generator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform._registry import (
    _discover_base_classes,
    _register_impl,
    get_impl,
    reset_registry_for_tests,
    resolve_impl,
    restore_registry,
    snapshot_registry,
)
from lmcache.v1.platform.base.device_info import DeviceInfo
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
from lmcache.v1.platform.base.tensor_from_ptr import TensorFromPtrBackend
from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
from lmcache.v1.platform.cuda.tensor_from_ptr import CudaTensorFromPtrBackend
from lmcache.v1.platform.cpu.tensor_from_ptr import CpuTensorFromPtrBackend
from lmcache.v1.platform.device_ext import DeviceExt
from lmcache.python_ops_fallback import _tensor_from_ptr


@pytest.fixture()
def isolated_registry() -> Generator[None, None, None]:
    """Isolate universal registry state between tests."""
    state = snapshot_registry()
    reset_registry_for_tests()
    try:
        yield
    finally:
        restore_registry(state)


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


def test_device_info_is_discovered_from_base_package() -> None:
    """DeviceInfo is discoverable from ``lmcache.v1.platform.base``."""
    assert DeviceInfo in _discover_base_classes()


def test_cuda_device_info_has_device_type_classvar() -> None:
    """CudaDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceInfo

    # The registry reads device_type via getattr on the class (not an instance).
    assert getattr(CudaDeviceInfo, "device_type", None) == "cuda"
    # Instance access also works — ClassVar satisfies the abstract property.
    instance = CudaDeviceInfo()
    assert instance.device_type == "cuda"
    # Verify the ABC contract is fully satisfied (no unimplemented abstract methods).
    assert isinstance(instance, DeviceInfo)


def test_musa_device_info_has_device_type_classvar() -> None:
    """MusaDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.musa import MusaDeviceInfo

    assert getattr(MusaDeviceInfo, "device_type", None) == "musa"
    assert isinstance(MusaDeviceInfo(), DeviceInfo)


def test_xpu_device_info_has_device_type_classvar() -> None:
    """XpuDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.xpu import XpuDeviceInfo

    assert getattr(XpuDeviceInfo, "device_type", None) == "xpu"
    assert isinstance(XpuDeviceInfo(), DeviceInfo)


def test_hpu_device_info_has_device_type_classvar() -> None:
    """HpuDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.hpu import HpuDeviceInfo

    assert getattr(HpuDeviceInfo, "device_type", None) == "hpu"
    assert isinstance(HpuDeviceInfo(), DeviceInfo)


def test_all_device_info_subclasses_can_be_registered(
    isolated_registry: None,
) -> None:
    """All built-in DeviceInfo subclasses can be registered in the universal registry.

    Verifies that the ClassVar pattern works consistently across all
    implementations, not just CUDA, by registering each subclass and
    retrieving it by its declared device_type.
    """
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceInfo
    from lmcache.v1.platform.hpu import HpuDeviceInfo
    from lmcache.v1.platform.musa import MusaDeviceInfo
    from lmcache.v1.platform.xpu import XpuDeviceInfo

    impls: list[tuple[type, str]] = [
        (CudaDeviceInfo, "cuda"),
        (MusaDeviceInfo, "musa"),
        (XpuDeviceInfo, "xpu"),
        (HpuDeviceInfo, "hpu"),
    ]
    for cls, expected_dt in impls:
        _register_impl(DeviceInfo, cls)
        result = get_impl(DeviceInfo, expected_dt, "default")
        assert result is cls, "Expected %s for device_type=%r, got %s" % (
            cls.__name__,
            expected_dt,
            result.__name__,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_device_info_is_registered_in_universal_registry() -> None:
    """CudaDeviceInfo is registered in the universal 3-D registry."""
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceInfo

    result = get_impl(DeviceInfo, "cuda", "default")
    assert result is CudaDeviceInfo
