# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for legacy CPU context module path."""

# First Party
# Backward compatibility — canonical module is non_gpu_context
from lmcache.v1.multiprocess.non_gpu_context import *  # noqa: F401,F403
from lmcache.v1.multiprocess.non_gpu_context import (  # noqa: F401
    NonGpuContext as CPUContext,
    NonGpuContextMetadata as CPUContextMetadata,
    create_non_gpu_context as create_cpu_context,
)
