# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim.

The canonical location of :class:`BaseCacheContext` is now
:mod:`lmcache.v1.platform.base.cache_context`.  This module re-exports
it so existing imports continue to work without change.
"""

# First Party
from lmcache.v1.platform.base.cache_context import (  # noqa: F401
    BaseCacheContext as BaseCacheContext,
)

__all__ = ["BaseCacheContext"]
