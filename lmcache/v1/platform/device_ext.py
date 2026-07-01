# SPDX-License-Identifier: Apache-2.0
"""Platform device extension namespace.

This module defines :class:`DeviceExt`, which is attached to the torch
device module as ``torch_dev.ext``.  It exposes platform-specific
capabilities (e.g. memory pinning) that do not exist on the original
torch device module.
"""

# First Party
from lmcache.v1.platform._registry import get_impl
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend

_PIN_MEMORY_BACKENDS: dict[str, type[PinMemoryBackend]] = {}


def register_pin_memory_backend(device_type: str, cls: type[PinMemoryBackend]) -> None:
    """Register a pin-memory backend implementation for a device type.

    This is the manual registration path kept for backward compatibility.
    New backends should instead set ``device_type`` on their
    :class:`~lmcache.v1.platform.base.pin_memory.PinMemoryBackend`
    subclass and let the universal registry discover them automatically.

    Args:
        device_type: The device type string (for example, ``"cuda"``).
        cls: A :class:`PinMemoryBackend` subclass (or the base class
            itself) to instantiate for ``device_type``.

    Notes:
        Re-registering the same ``device_type`` overwrites the previous
        backend class. Registration is expected to happen during module
        import, so this helper does not add extra synchronization for
        concurrent writes. Existing :class:`DeviceExt` instances keep the
        backend object they already created; later registrations affect
        only newly constructed instances.

    Raises:
        TypeError: If ``cls`` is not a :class:`PinMemoryBackend`
            subclass.
    """
    if not isinstance(cls, type) or not issubclass(cls, PinMemoryBackend):
        raise TypeError(
            "register_pin_memory_backend expects a PinMemoryBackend subclass"
        )
    _PIN_MEMORY_BACKENDS[device_type] = cls


def _get_pin_memory_backend(device_type: str) -> type[PinMemoryBackend]:
    """Resolve the pin-memory backend class for *device_type*.

    Checks the manual ``_PIN_MEMORY_BACKENDS`` table first (backward
    compat), then falls back to the universal registry, then falls back
    to the no-op base class.

    Args:
        device_type: The ``torch.device.type`` string.

    Returns:
        A :class:`PinMemoryBackend` subclass (or the base class itself as
        a no-op fallback).
    """
    # Manual registration takes priority for backward compatibility.
    cls = _PIN_MEMORY_BACKENDS.get(device_type)
    if cls is not None:
        return cls

    # Fall back to the universal registry.
    try:
        return get_impl(PinMemoryBackend, device_type)  # type: ignore[return-value]
    except ValueError:
        return PinMemoryBackend


class DeviceExt:
    """Extension namespace attached as ``torch_dev.ext``.

    Holds platform-specific capabilities that do not exist on the original
    torch device module.  New capabilities can be added as methods or
    properties here without changing call-sites.

    Intended usage::

        torch_dev.ext.pin_memory(ptr, size)
        torch_dev.ext.pin_memory(ptr, size, flags)
        torch_dev.ext.unpin_memory(ptr)
        if not torch_dev.ext.is_pin_supported:
            raise RuntimeError(...)
    """

    def __init__(self, device_type: str) -> None:
        backend_cls = _get_pin_memory_backend(device_type)
        self._pin: PinMemoryBackend = backend_cls()

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region for DMA access.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Platform-specific registration flags (e.g.
                ``cudaHostRegisterDefault = 0``).

        Returns:
            True if pinning succeeded, False otherwise.
        """
        return self._pin.pin_memory(ptr, size, flags)

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return self._pin.unpin_memory(ptr)

    @property
    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning."""
        return self._pin.is_pin_supported()
