# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific memory pinning via ``cudaHostRegister`` / ``cudaHostUnregister``."""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base import PinMemoryBackend


class CudaPinMemoryBackend(PinMemoryBackend):
    """CUDA implementation of :class:`PinMemoryBackend`.

    Uses ``torch.cuda.cudart()`` to call ``cudaHostRegister`` and
    ``cudaHostUnregister`` for explicit host-memory pinning.
    """

    # cudaHostRegisterMapped — makes the registered memory accessible from
    # the device via a device pointer returned by cudaHostGetDevicePointer.
    PIN_FLAGS = 0x02

    def __init__(self) -> None:
        """Initialise the CUDA runtime handle.

        Raises:
            RuntimeError: If ``torch.cuda.cudart`` is not available.
        """
        # Third Party
        import torch

        if not hasattr(torch.cuda, "cudart"):
            raise RuntimeError(
                "torch.cuda.cudart is not available; "
                "CudaPinMemoryBackend cannot be used on this build."
            )
        self._cudart = torch.cuda.cudart()

    def pin_memory(self, ptr: int, size: int) -> bool:
        """Pin a host memory region using ``cudaHostRegister``.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Size in bytes of the region to pin.

        Returns:
            ``True`` if ``cudaHostRegister`` returned 0 (success),
            ``False`` otherwise.
        """
        err = self._cudart.cudaHostRegister(ptr, size, self.PIN_FLAGS)
        return err == 0

    def unpin_memory(self, ptr: int, size: int = 0) -> bool:
        """Unpin a host memory region using ``cudaHostUnregister``.

        Args:
            ptr: Raw pointer (``tensor.data_ptr()``) to the memory region.
            size: Unused; present for interface compatibility.

        Returns:
            ``True`` if ``cudaHostUnregister`` returned 0 (success),
            ``False`` otherwise.
        """
        err = self._cudart.cudaHostUnregister(ptr)
        return err == 0

    def is_pin_supported(self) -> bool:
        """Whether CUDA memory pinning is supported.

        Returns:
            Always ``True`` for the CUDA backend.
        """
        return True
