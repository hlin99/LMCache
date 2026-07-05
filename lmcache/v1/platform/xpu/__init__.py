# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel SYCL) platform helpers."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_info import DeviceInfo

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class XpuDeviceInfo(DeviceInfo):
    """XPU device information for the detection registry."""

    #: ``torch.device.type`` this backend handles (used by auto-discovery).
    device_type: ClassVar[str] = "xpu"
    #: Implementation key (used by the universal registry).
    impl_key: ClassVar[str] = "default"

    @property
    def torch_module_name(self) -> str:
        return "xpu"

    @property
    def ops_module(self) -> str | None:
        return "lmcache.xpu_ops"

    def is_available(self) -> bool:
        """Check XPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False
