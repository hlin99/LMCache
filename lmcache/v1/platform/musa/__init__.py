# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific platform primitives."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_info import DeviceInfo

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class MusaDeviceInfo(DeviceInfo):
    """MUSA device information for the detection registry."""

    #: ``torch.device.type`` this backend handles (used by auto-discovery).
    device_type: ClassVar[str] = "musa"
    #: Implementation key (used by the universal registry).
    impl_key: ClassVar[str] = "default"

    @property
    def torch_module_name(self) -> str:
        return "musa"

    @property
    def ops_module(self) -> str | None:
        return "lmcache.v1.platform.musa.ops"

    def is_available(self) -> bool:
        """Check MUSA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "musa") and torch.musa.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    def is_handle_transfer_available(self) -> bool:
        # First Party
        from lmcache.v1.platform.musa.ipc import is_musa_handle_transfer_available

        return is_musa_handle_transfer_available()
