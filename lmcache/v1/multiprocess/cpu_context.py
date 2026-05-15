# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for legacy CPU context module path."""

# First Party
from lmcache.v1.multiprocess.none_gpu_context import *  # noqa: F401,F403
from lmcache.v1.multiprocess.none_gpu_context import (  # noqa: F401
    NoneGpuContext as CPUContext,
    NoneGpuContextMetadata as CPUContextMetadata,
    create_none_gpu_context as create_cpu_context,
)
