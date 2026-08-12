# SPDX-License-Identifier: Apache-2.0
"""Automatic registry and unique-match dispatch for format probes."""

# Standard
from collections import defaultdict
from types import ModuleType
import importlib

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.probes.base import KVFormatProbe
from lmcache.v1.kv_format.specs.registry import SPECS
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


def _probe_classes(module: ModuleType) -> list[type[KVFormatProbe]]:
    """Return probe classes declared directly in ``module``."""
    return [
        value
        for value in vars(module).values()
        if (
            isinstance(value, type)
            and issubclass(value, KVFormatProbe)
            and value is not KVFormatProbe
            and value.__module__ == module.__name__
        )
    ]


def _discover_probes() -> dict[EngineType, tuple[type[KVFormatProbe], ...]]:
    """Discover probes colocated with registered format specs."""
    probes: defaultdict[EngineType, list[type[KVFormatProbe]]] = defaultdict(list)
    seen: set[type[KVFormatProbe]] = set()
    for spec in SPECS.values():
        module = importlib.import_module(spec.__module__)
        for probe in _probe_classes(module):
            if probe in seen:
                continue
            if probe.format_spec is not spec:
                raise ValueError(
                    f"{probe.__name__} must be declared beside "
                    f"{probe.format_spec.__name__}"
                )
            probes[probe.engine_type].append(probe)
            seen.add(probe)
    return {
        engine_type: tuple(
            sorted(
                engine_probes,
                key=lambda probe: int(probe.format_spec.engine_kv_format),
            )
        )
        for engine_type, engine_probes in probes.items()
    }


_PROBES_BY_ENGINE = _discover_probes()


def get_probes(engine_type: EngineType) -> tuple[type[KVFormatProbe], ...]:
    """Return all registered format probes for a serving engine.

    Args:
        engine_type: Serving engine whose probes should be returned.

    Returns:
        An immutable tuple of registered probe classes.
    """
    return _PROBES_BY_ENGINE.get(engine_type, ())


def probe_format(
    kv_caches: DiscoverableKVCache,
    engine_type: EngineType,
    layout_hints: LayoutHints,
) -> "tuple[lmcache_native.EngineKVFormat | None, DiscoverableKVCache]":
    """Probe all formats for an engine and require a unique match.

    Args:
        kv_caches: Raw KV-cache structure supplied by the serving engine.
        engine_type: Serving engine that produced ``kv_caches``.
        layout_hints: Engine-supplied layout and normalization hints.

    Returns:
        The uniquely matched format and its normalized KV caches. Returns
        ``(None, kv_caches)`` when no probe matches.

    Raises:
        ValueError: More than one probe matches the same input.
    """
    matches: list[tuple[lmcache_native.EngineKVFormat, DiscoverableKVCache]] = []
    for probe in get_probes(engine_type):
        normalized = probe.probe(kv_caches, layout_hints)
        if normalized is not None:
            matches.append((probe.format_spec.engine_kv_format, normalized))
    if len(matches) > 1:
        formats = ", ".join(fmt.name for fmt, _normalized in matches)
        raise ValueError(
            f"ambiguous KV cache format for {engine_type}: matched {formats}"
        )
    if matches:
        return matches[0]
    return None, kv_caches
