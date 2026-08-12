# SPDX-License-Identifier: Apache-2.0
"""KVFormatSpec geometry layer.

``base.py`` is the interface + shape rendering, ``<engine_kv_format>.py`` are
the per-format implementations, and ``registry.py`` provides lookup and
engine-scoped detection helpers.
"""

# First Party
from lmcache.v1.kv_format.specs.base import (
    KVFormatSpec,
    concrete_shape,
    describe_shape,
)
from lmcache.v1.kv_format.specs.registry import (
    detect_format_for_engine,
    get_detectable_specs,
    get_spec,
    get_spec_class,
)

__all__ = [
    "KVFormatSpec",
    "concrete_shape",
    "describe_shape",
    "detect_format_for_engine",
    "get_detectable_specs",
    "get_spec",
    "get_spec_class",
]
