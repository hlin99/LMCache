# SPDX-License-Identifier: Apache-2.0
"""Per-engine KV cache discovery interface.

One :class:`EngineDetector` per serving engine prepares engine-wide hints and
delegates format-specific recognition and normalization to registered probes.
The engine -> detector table is in ``registry.py``.
"""

# Standard
from abc import ABC
from typing import ClassVar

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.probes import probe_format
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class EngineDetector(ABC):
    """Prepare engine context and dispatch registered format probes."""

    engine_type: ClassVar[EngineType]

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[lmcache_native.EngineKVFormat | None, DiscoverableKVCache]":
        """Return the uniquely probed format and canonical KV caches.

        Args:
            kv_caches: Raw KV-cache structure supplied by the serving engine.
            layout_hints: Engine-supplied layout and normalization hints.

        Returns:
            ``(format, canonical_kv_caches)`` when one registered probe
            matches, otherwise ``(None, kv_caches)``.

        Raises:
            ValueError: Probe matching is ambiguous or normalization fails.
        """
        return probe_format(kv_caches, self.engine_type, layout_hints)
