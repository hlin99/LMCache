# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific platform primitives."""

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_info import DeviceInfo
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class CudaDeviceInfo(DeviceInfo):
    """CUDA device information for the detection registry."""

    #: ``torch.device.type`` this backend handles (used by auto-discovery).
    device_type: ClassVar[str] = "cuda"
    #: Implementation key (used by the universal registry).
    impl_key: ClassVar[str] = "default"

    @property
    def torch_module_name(self) -> str:
        return "cuda"

    @property
    def ops_module(self) -> str | None:
        return "lmcache.c_ops"

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        # First Party
        from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

        return CudaPinMemoryBackend

    def is_available(self) -> bool:
        """Check CUDA availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
