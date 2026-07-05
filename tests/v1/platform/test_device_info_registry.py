# SPDX-License-Identifier: Apache-2.0
"""Tests for DeviceInfo registry integration.

Verifies that:
- ``DeviceInfo`` is discoverable from ``lmcache.v1.platform.base`` by
  ``_discover_base_classes()``.
- The backward-compatibility shim in ``base_device_info.py`` still works.
- Concrete subclasses (at least ``CudaDeviceInfo``) are registered in the
  universal registry when CUDA is available.
"""

# Standard
from typing import Generator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform._registry import (
    _discover_base_classes,
    get_impl,
    reset_registry_for_tests,
    restore_registry,
    snapshot_registry,
)
from lmcache.v1.platform.base.device_info import DeviceInfo


@pytest.fixture(autouse=True)
def isolated_registry() -> Generator[None, None, None]:
    """Isolate universal registry state between tests."""
    state = snapshot_registry()
    reset_registry_for_tests()
    try:
        yield
    finally:
        restore_registry(state)


def test_device_info_is_discovered_from_base_package() -> None:
    """DeviceInfo is discoverable from ``lmcache.v1.platform.base``."""
    assert DeviceInfo in _discover_base_classes()


def test_backward_compat_shim_imports_device_info() -> None:
    """The re-export shim in base_device_info.py resolves to the same class."""
    # First Party
    from lmcache.v1.platform.base_device_info import DeviceInfo as ShimDeviceInfo

    assert ShimDeviceInfo is DeviceInfo


def test_cuda_device_info_has_device_type_classvar() -> None:
    """CudaDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceInfo

    # The registry reads device_type via getattr on the class (not an instance).
    assert getattr(CudaDeviceInfo, "device_type", None) == "cuda"
    # Instance access also works.
    assert CudaDeviceInfo().device_type == "cuda"


def test_musa_device_info_has_device_type_classvar() -> None:
    """MusaDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.musa import MusaDeviceInfo

    assert getattr(MusaDeviceInfo, "device_type", None) == "musa"
    assert MusaDeviceInfo().device_type == "musa"


def test_xpu_device_info_has_device_type_classvar() -> None:
    """XpuDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.xpu import XpuDeviceInfo

    assert getattr(XpuDeviceInfo, "device_type", None) == "xpu"
    assert XpuDeviceInfo().device_type == "xpu"


def test_hpu_device_info_has_device_type_classvar() -> None:
    """HpuDeviceInfo exposes device_type as a ClassVar for registry indexing."""
    # First Party
    from lmcache.v1.platform.hpu import HpuDeviceInfo

    assert getattr(HpuDeviceInfo, "device_type", None) == "hpu"
    assert HpuDeviceInfo().device_type == "hpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_device_info_is_registered_in_universal_registry() -> None:
    """CudaDeviceInfo is registered in the universal 3-D registry."""
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceInfo

    result = get_impl(DeviceInfo, "cuda", "default")
    assert result is CudaDeviceInfo
