# SPDX-License-Identifier: Apache-2.0
"""XPU-specific memory pinning.

On XPU, memory pinning is handled at allocation time via SYCL USM host
allocations (``torch.xpu.pin_memory=True``).  There is no separate
register/unregister step for pre-allocated buffers, so both operations
are effectively no-ops that return ``True`` to signal that pinning is
considered supported.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base import PinMemoryBackend


class XPUPinMemoryBackend(PinMemoryBackend):
    """XPU implementation of :class:`PinMemoryBackend`.

    Pinning on XPU is done implicitly at allocation time (SYCL USM host
    allocation).  For buffers that are already allocated, pin/unpin are
    no-ops that return ``True`` so callers can treat XPU the same as CUDA
    without any platform-specific branches.
    """

    def pin_memory(self, ptr: int, size: int) -> bool:
        """No-op pin — memory is already pinned at allocation time on XPU.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Size in bytes of the region (unused).

        Returns:
            Always ``True``.
        """
        return True

    def unpin_memory(self, ptr: int, size: int = 0) -> bool:
        """No-op unpin — memory lifecycle is managed by SYCL USM on XPU.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Size in bytes of the region (unused).

        Returns:
            Always ``True``.
        """
        return True

    def is_pin_supported(self) -> bool:
        """Whether XPU memory pinning is supported.

        Returns:
            Always ``True`` for the XPU backend.
        """
        return True
