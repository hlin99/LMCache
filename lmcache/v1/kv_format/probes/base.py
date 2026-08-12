# SPDX-License-Identifier: Apache-2.0
"""Probe interface for engine-specific KV-cache format discovery."""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.kv_format.specs.base import KVFormatSpec


class KVFormatProbe(ABC):
    """Recognize and normalize one format produced by one serving engine."""

    engine_type: ClassVar[EngineType]
    format_spec: ClassVar[type["KVFormatSpec"]]

    @classmethod
    @abstractmethod
    def probe(
        cls,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache | None:
        """Return normalized KV caches when this probe matches.

        Args:
            kv_caches: Raw KV-cache structure supplied by the serving engine.
            layout_hints: Engine-supplied hints needed to resolve layouts that
                cannot be distinguished from tensor shapes alone.

        Returns:
            The canonical KV-cache structure for ``format_spec`` when matched,
            otherwise ``None``.

        Raises:
            ValueError: The structure targets this format but violates a
                required normalization invariant.
        """
