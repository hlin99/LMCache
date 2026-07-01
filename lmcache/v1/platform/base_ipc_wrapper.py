# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

The canonical location of :class:`DeviceIPCWrapper` is now
:mod:`lmcache.v1.platform.base.ipc_wrapper`.  This module re-exports
it so existing imports continue to work without change.
"""

# First Party
from lmcache.v1.platform.base.ipc_wrapper import (  # noqa: F401
    DeviceIPCWrapper as DeviceIPCWrapper,
)

__all__ = ["DeviceIPCWrapper"]
