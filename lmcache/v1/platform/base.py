# SPDX-License-Identifier: Apache-2.0
"""Platform-abstraction base classes for LMCache.

This module is designed to hold multiple platform-abstraction base classes.
:class:`PinMemoryBackend` is the first one; additional abstractions can be
added here as needed.

A module-level singleton :data:`pin_memory_backend` is auto-selected at
import time based on the current ``torch_device_type``:

- ``"cuda"``  → :class:`~lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend`
- ``"xpu"``   → :class:`~lmcache.v1.platform.xpu.pin_memory.XPUPinMemoryBackend`
- otherwise  → :class:`PinMemoryBackend` (no-op, returns ``False``)
"""

# Future
from __future__ import annotations


class PinMemoryBackend:
    """Abstract base for host-memory pinning per platform.

    The default implementation is a no-op that always returns ``False``.
    Platform-specific sub-classes override the methods below to provide
    real pinning support.
    """

    def pin_memory(self, ptr: int, size: int) -> bool:
        """Pin a host memory region.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Size in bytes of the region to pin.

        Returns:
            ``True`` on success, ``False`` otherwise (including when pinning
            is not supported on this platform).
        """
        return False

    def unpin_memory(self, ptr: int, size: int = 0) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Size in bytes of the region (some platforms need this).

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        return False

    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning.

        Returns:
            ``True`` if pinning is supported, ``False`` otherwise.
        """
        return False


# ---------------------------------------------------------------------------
# Singleton selection
# ---------------------------------------------------------------------------


def _create_pin_memory_backend() -> PinMemoryBackend:
    """Instantiate the appropriate :class:`PinMemoryBackend` for the
    running device type.

    Returns:
        A :class:`PinMemoryBackend` instance for the current platform.
    """
    # First Party
    from lmcache import torch_device_type

    if torch_device_type == "cuda":
        # First Party
        from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

        return CudaPinMemoryBackend()
    elif torch_device_type == "xpu":
        # First Party
        from lmcache.v1.platform.xpu.pin_memory import XPUPinMemoryBackend

        return XPUPinMemoryBackend()
    else:
        return PinMemoryBackend()


pin_memory_backend: PinMemoryBackend = _create_pin_memory_backend()
"""Module-level singleton :class:`PinMemoryBackend`.

Upper-level code should import and use this object directly::

    from lmcache.v1.platform.base import pin_memory_backend

    if not pin_memory_backend.pin_memory(ptr, size):
        raise RuntimeError("Failed to pin memory")
"""
