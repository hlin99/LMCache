# SPDX-License-Identifier: Apache-2.0
"""Global accelerator abstraction for LMCache.

This module provides a unified interface for accessing the available
hardware accelerator (CUDA, XPU, HPU, etc.), replacing direct use of
``torch.cuda`` throughout the codebase.

At initialization time the module probes which accelerator is available
and exposes:

* :func:`get_accelerator` – the torch device module
  (e.g. ``torch.cuda``, ``torch.xpu``).
* :func:`get_device_name` – a short name string
  (``"cuda"``, ``"xpu"``, ``"hpu"``, or ``"cpu"``).
* :func:`is_accelerator_available` – whether *any* accelerator is
  present.

Initialization is lazy (happens on first call) and thread-safe.
"""

# Standard
from types import ModuleType
from typing import Optional, Tuple
import logging
import threading

# Third Party
import torch

logger = logging.getLogger(__name__)

_accelerator: Optional[ModuleType] = None
_device_name: str = "cpu"
_initialized: bool = False
_lock = threading.Lock()


def _detect_accelerator() -> Tuple[Optional[ModuleType], str]:
    """Detect the first available accelerator.

    Returns:
        A tuple of (torch device module or ``None``, device name string).
    """
    if torch.cuda.is_available():
        return torch.cuda, "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu, "xpu"
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu, "hpu"
    return None, "cpu"


def _ensure_initialized() -> None:
    """Lazily initialize the global accelerator state (thread-safe)."""
    global _accelerator, _device_name, _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        _accelerator, _device_name = _detect_accelerator()
        _initialized = True
        logger.info("LMCache detected accelerator: %s", _device_name)


def get_accelerator() -> Optional[ModuleType]:
    """Return the global accelerator module.

    Returns:
        The torch device module for the detected accelerator
        (e.g. ``torch.cuda``, ``torch.xpu``, ``torch.hpu``),
        or ``None`` when only CPU is available.
    """
    _ensure_initialized()
    return _accelerator


def get_device_name() -> str:
    """Return the name of the detected accelerator.

    Returns:
        One of ``"cuda"``, ``"xpu"``, ``"hpu"``, or ``"cpu"``.
    """
    _ensure_initialized()
    return _device_name


def is_accelerator_available() -> bool:
    """Check whether a hardware accelerator is available.

    Returns:
        ``True`` if CUDA, XPU, or HPU is available; ``False`` otherwise.
    """
    _ensure_initialized()
    return _accelerator is not None
