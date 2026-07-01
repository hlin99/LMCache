# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

The canonical location of :class:`PinMemoryBackend` is now
:mod:`lmcache.v1.platform.base.pin_memory`.  This module re-exports
it so existing imports continue to work without change.
"""

# First Party
from lmcache.v1.platform.base.pin_memory import (  # noqa: F401
    PinMemoryBackend as PinMemoryBackend,
)

__all__ = ["PinMemoryBackend"]
