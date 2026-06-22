# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific memory pinning via cudaHostRegister/cudaHostUnregister."""

# First Party
from lmcache.v1.platform.base import PinMemoryBackend


class CudaPinMemoryBackend(PinMemoryBackend):
    """CUDA memory pinning using the CUDA runtime (cudart).

    Uses ``cudaHostRegisterMapped`` (flag ``0x02``) so that CUDA can
    perform DMA directly from the pinned host buffer.
    """

    PIN_FLAGS = 0x02  # cudaHostRegisterMapped

    def __init__(self) -> None:
        # Third Party
        import torch

        self._cudart = torch.cuda.cudart()

    def pin_memory(self, ptr: int, size: int) -> bool:
        """Pin a host memory region using cudaHostRegister.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.

        Returns:
            True if cudaHostRegister returned 0 (success), False otherwise.
        """
        err = self._cudart.cudaHostRegister(ptr, size, self.PIN_FLAGS)
        return int(err) == 0

    def unpin_memory(self, ptr: int, size: int = 0) -> bool:
        """Unpin a previously pinned host memory region using cudaHostUnregister.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Unused; present for interface compatibility.

        Returns:
            True if cudaHostUnregister returned 0 (success), False otherwise.
        """
        err = self._cudart.cudaHostUnregister(ptr)
        return int(err) == 0

    def is_pin_supported(self) -> bool:
        """Whether CUDA memory pinning is supported.

        Returns:
            Always True for this backend.
        """
        return True
