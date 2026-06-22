# SPDX-License-Identifier: Apache-2.0
"""CUDA memory pinning: try torch cudart first, then libcudart via ctypes."""

# Standard
import ctypes
import ctypes.util

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base import PinMemoryBackend

logger = init_logger(__name__)


def _load_libcudart() -> ctypes.CDLL | None:
    """Try to load ``libcudart`` and bind the pinning symbols.

    Returns:
        The loaded ``ctypes.CDLL`` library with bound symbols on success, or
        ``None`` if the library cannot be found or loaded.

    Notes:
        Missing symbols or load failures are treated as an unavailable
        fallback path and cause this helper to return ``None``.
    """
    path = ctypes.util.find_library("cudart")
    if path is None:
        return None

    try:
        lib = ctypes.CDLL(path)
        lib.cudaHostRegister.restype = ctypes.c_int
        lib.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.cudaHostUnregister.restype = ctypes.c_int
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        return lib
    except (AttributeError, OSError):
        return None


class CudaPinMemoryBackend(PinMemoryBackend):
    """CUDA memory pinning backend.

    Pinning prefers ``torch.cuda.cudart()``. When the torch binding is
    unavailable, the backend falls back to loading ``libcudart`` directly.
    """

    PIN_FLAGS = 0x02  # cudaHostRegisterMapped

    def __init__(self) -> None:
        """Initialize the backend with torch-first, libcudart-second fallback.

        The backend first tries ``torch.cuda.cudart()`` because it is the
        lightest path when torch already exposes the CUDA runtime binding. If
        that import or lookup fails, it falls back to loading ``libcudart``
        directly via :mod:`ctypes`.
        """
        self._cudart = None
        self._libcudart = None

        try:
            # Third Party
            import torch
        except ImportError as exc:
            logger.debug(
                "CudaPinMemoryBackend: torch import failed, "
                "attempting libcudart fallback: %s",
                exc,
            )
        else:
            try:
                self._cudart = torch.cuda.cudart()
                logger.info("CudaPinMemoryBackend: using torch cudart")
                return
            except (AttributeError, RuntimeError) as exc:
                logger.debug(
                    "CudaPinMemoryBackend: torch cudart unavailable, "
                    "attempting libcudart fallback: %s",
                    exc,
                )

        self._libcudart = _load_libcudart()
        if self._libcudart is not None:
            logger.info("CudaPinMemoryBackend: using libcudart via ctypes")
        else:
            logger.warning(
                "CudaPinMemoryBackend: neither torch cudart nor libcudart is available"
            )

    def pin_memory(self, ptr: int, size: int) -> bool:
        """Pin a host memory region using ``cudaHostRegister``.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.

        Returns:
            True if ``cudaHostRegister`` succeeded, False otherwise.
        """
        if self._cudart is not None:
            err = self._cudart.cudaHostRegister(ptr, size, self.PIN_FLAGS)
            return int(err) == 0

        if self._libcudart is not None:
            err = self._libcudart.cudaHostRegister(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(self.PIN_FLAGS),
            )
            return err == 0

        return False

    def unpin_memory(self, ptr: int, size: int = 0) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Unused; present for interface compatibility.

        Returns:
            True if ``cudaHostUnregister`` succeeded, False otherwise.
        """
        if self._cudart is not None:
            err = self._cudart.cudaHostUnregister(ptr)
            return int(err) == 0

        if self._libcudart is not None:
            err = self._libcudart.cudaHostUnregister(ctypes.c_void_p(ptr))
            return err == 0

        return False

    def is_pin_supported(self) -> bool:
        """Whether CUDA memory pinning is supported.

        Returns:
            True if either the torch cudart binding or ``libcudart`` is
            available, False otherwise.
        """
        return self._cudart is not None or self._libcudart is not None
