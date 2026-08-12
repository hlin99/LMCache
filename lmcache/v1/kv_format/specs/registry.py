# SPDX-License-Identifier: Apache-2.0
"""The ``EngineKVFormat`` -> spec table, discovered from this folder.

Every ``specs/<engine_kv_format>.py`` defines one :class:`KVFormatSpec`
subclass. This imports them all and indexes each by the ``engine_kv_format`` it
declares, so adding a format is just dropping a new file here -- nothing in this
file changes.
"""

# Standard
from pathlib import Path
import importlib
import pkgutil

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.specs.base import KVFormatSpec
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


def _discover_specs() -> dict["lmcache_native.EngineKVFormat", type[KVFormatSpec]]:
    """Import every spec module in this folder and index it by its format."""
    specs: dict["lmcache_native.EngineKVFormat", type[KVFormatSpec]] = {}
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module.name in ("base", "registry"):
            continue
        imported = importlib.import_module(f"{__package__}.{module.name}")
        for value in vars(imported).values():
            if (
                isinstance(value, type)
                and issubclass(value, KVFormatSpec)
                and value is not KVFormatSpec
            ):
                specs[value.engine_kv_format] = value
    return specs


SPECS = _discover_specs()

# Indexed by enum *value*: the native pybind ``EngineKVFormat`` and the
# pure-Python fallback one are distinct types with the same members, and both
# reach this table (e.g. from ``lmcache.v1.platform.torch_ops``).
_SPECS_BY_VALUE = {int(fmt): spec for fmt, spec in SPECS.items()}


def get_spec_class(fmt: "lmcache_native.EngineKVFormat") -> type[KVFormatSpec]:
    """Return the spec class for *fmt* -- the owner of its static facts.

    Args:
        fmt: The Engine KV format to look up.

    Returns:
        The :class:`KVFormatSpec` subclass declaring *fmt*, which carries the
        format's static layout facts (``is_mla``, ``is_hnd``, the structural
        shape, ``attention_backends``, ...) as class attributes.

    Raises:
        ValueError: If *fmt* has no spec.
    """
    spec = _SPECS_BY_VALUE.get(int(fmt))
    if spec is None:
        raise ValueError(f"Unknown Engine KV Format: {fmt}")
    return spec


def get_spec(
    kv_caches: DiscoverableKVCache, fmt: "lmcache_native.EngineKVFormat"
) -> KVFormatSpec:
    """Return a spec instance wrapping *kv_caches* of *fmt*."""
    return get_spec_class(fmt)(kv_caches)


def get_detectable_specs(engine_type: EngineType) -> tuple[type[KVFormatSpec], ...]:
    """Return all specs detectable for a serving engine.

    Args:
        engine_type: Serving engine to query.

    Returns:
        An immutable tuple of spec classes whose ``engine_type`` matches,
        sorted by their ``engine_kv_format`` integer value.
    """
    return tuple(
        sorted(
            (spec for spec in SPECS.values() if spec.engine_type == engine_type),
            key=lambda spec: int(spec.engine_kv_format),
        )
    )


def detect_format_for_engine(
    kv_caches: "DiscoverableKVCache",
    engine_type: EngineType,
    layout_hints: "LayoutHints",
) -> "tuple[lmcache_native.EngineKVFormat | None, DiscoverableKVCache]":
    """Probe all detectable specs for an engine and require a unique match.

    Args:
        kv_caches: Raw KV-cache structure supplied by the serving engine.
        engine_type: Serving engine that produced ``kv_caches``.
        layout_hints: Engine-supplied layout and normalization hints (already
            resolved by the engine detector before this is called).

    Returns:
        The uniquely matched format and its normalized KV caches. Returns
        ``(None, kv_caches)`` when no spec matches.

    Raises:
        ValueError: More than one spec matches the same input, or
            normalization fails.
    """
    matches: list[tuple[lmcache_native.EngineKVFormat, DiscoverableKVCache]] = []
    for spec in get_detectable_specs(engine_type):
        normalized = spec.try_normalize(kv_caches, layout_hints)
        if normalized is not None:
            matches.append((spec.engine_kv_format, normalized))
    if len(matches) > 1:
        formats = ", ".join(fmt.name for fmt, _normalized in matches)
        raise ValueError(
            f"ambiguous KV cache format for {engine_type}: matched {formats}"
        )
    if matches:
        return matches[0]
    return None, kv_caches
