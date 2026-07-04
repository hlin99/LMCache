# SPDX-License-Identifier: Apache-2.0

# Standard
import abc

# First Party
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper, RawCudaIPCWrapper


def test_base_ipc_wrapper_path_is_backward_compatible() -> None:
    """Legacy and new DeviceIPCWrapper import paths resolve to the same class."""
    # First Party
    from lmcache.v1.platform.base_ipc_wrapper import (
        DeviceIPCWrapper as LegacyDeviceIPCWrapper,
    )

    assert LegacyDeviceIPCWrapper is DeviceIPCWrapper
    assert issubclass(DeviceIPCWrapper, abc.ABC)


def test_registry_discovers_default_ipc_wrapper_impls() -> None:
    """Discovery indexes default IPC wrappers by ``(device_type, impl_key)``."""
    snapshot = platform_registry.snapshot()
    try:
        platform_registry.reset_for_tests()
        all_impls = platform_registry.get_all_impls(DeviceIPCWrapper)

        assert all_impls["cpu"]["default"].__name__ == "CpuShmTensorWrapper"
        assert all_impls["cuda"]["default"] is CudaIPCWrapper
        assert RawCudaIPCWrapper not in all_impls["cuda"].values()
    finally:
        platform_registry.restore(snapshot)


def test_registry_supports_three_dimensional_lookup() -> None:
    """Manual registration supports non-default ``impl_key`` variants."""

    class _AltCpuWrapper(DeviceIPCWrapper):
        device_type = "cpu"
        impl_key = "alt"

    snapshot = platform_registry.snapshot()
    try:
        platform_registry.register_impl(DeviceIPCWrapper, "cpu", "alt", _AltCpuWrapper)
        assert (
            platform_registry.get_impl(DeviceIPCWrapper, "cpu", "alt")
            is _AltCpuWrapper
        )
    finally:
        platform_registry.restore(snapshot)
