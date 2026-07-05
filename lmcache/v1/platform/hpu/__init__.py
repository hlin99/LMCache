# SPDX-License-Identifier: Apache-2.0
"""HPU (Habana Gaudi) platform helpers."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_info import DeviceInfo

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class HpuDeviceInfo(DeviceInfo):
    """HPU device information for the detection registry."""

    #: ``torch.device.type`` this backend handles (used by auto-discovery).
    device_type: ClassVar[str] = "hpu"
    #: Implementation key (used by the universal registry).
    impl_key: ClassVar[str] = "default"

    @property
    def torch_module_name(self) -> str:
        return "hpu"

    @property
    def ops_module(self) -> str | None:
        return None

    def is_available(self) -> bool:
        """Check HPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "hpu") and torch.hpu.is_available()
        except Exception:
            return False
