# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports for legacy CPU context module path."""

# First Party
from lmcache.v1.multiprocess.none_gpu_context import (  # noqa: F401
    NoneGpuContext as CPUContext,
    NoneGpuContextMetadata as CPUContextMetadata,
    compute_kv_layout,
    create_none_gpu_context as create_cpu_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
