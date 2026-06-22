# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim — re-exports PerfStepTimer as StoreTimer."""

# First Party
from lmcache.perf_step_timer import PerfStepTimer as StoreTimer

__all__ = ["StoreTimer"]
