# SPDX-License-Identifier: Apache-2.0
"""XPU-specific memory pinning.

XPU (Intel GPU) allocates host memory via SYCL Unified Shared Memory (USM),
which is already accessible from the device at allocation time.  There is no
explicit ``cudaHostRegister``-equivalent, so pin/unpin are no-ops that report
success so callers do not need to handle a platform-specific code path.
"""

# First Party
from lmcache.v1.platform.base import PinMemoryBackend


class XPUPinMemoryBackend(PinMemoryBackend):
    """XPU memory pinning backend.

    Pinning is a no-op on XPU because SYCL USM host allocations are already
    device-accessible.  Both methods return ``True`` to signal success.
    """

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """No-op pin for XPU — memory is pinned at allocation time.

        Args:
            ptr: Raw pointer to the memory region (unused).
            size: Size in bytes (unused).
            flags: Registration flags (unused).

        Returns:
            Always True.
        """
        return True

    def unpin_memory(self, ptr: int) -> bool:
        """No-op unpin for XPU.

        Args:
            ptr: Raw pointer to the memory region (unused).

        Returns:
            Always True.
        """
        return True

    def is_pin_supported(self) -> bool:
        """Whether XPU memory pinning is supported.

        Returns:
            Always True for this backend.
        """
        return True
