# SPDX-License-Identifier: Apache-2.0
"""Pickle-based NonGpuContext implementation for multiprocess mode.

This module re-exports :class:`NonGpuContextPickle` from
:mod:`lmcache.v1.multiprocess.worker_transfer.pickle` so that both import
paths work interchangeably.
"""

# Local
from lmcache.v1.multiprocess.worker_transfer.pickle import NonGpuContextPickle  # noqa: F401

__all__ = ["NonGpuContextPickle"]
