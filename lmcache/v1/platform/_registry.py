# SPDX-License-Identifier: Apache-2.0
"""Universal platform registry.

Scans ``platform/base/`` for ABC-based registry roots, then scans
``platform/<device>/`` implementations and indexes them by
``(base_class, device_type, impl_key)``.

Only IPC wrappers are wired through convenience helpers in this port.
Other base-class families will migrate in follow-up changes.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Dict
import abc
import importlib
import inspect
import pkgutil
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

DEFAULT_BACKEND: str = "cpu"
DEFAULT_IMPL_KEY: str = "default"

# {base_class: {device_type: {impl_key: concrete_class}}}
_REGISTRY: Dict[type, Dict[str, Dict[str, type]]] = {}

# Backward-compatible manual override table used by tests/callers that
# still use register_kv_wrapper().
_KV_WRAPPER_FACTORIES: Dict[str, Callable[..., Any]] = {}

_AVAILABILITY: Dict[str, Callable[[], bool]] = {}
_DISCOVERED: bool = False
_DISCOVERY_LOCK = threading.Lock()


def _collect_base_classes() -> list[type]:
    """Collect base classes directly defined under ``platform/base``.

    A class qualifies iff:
    - it is defined in a direct ``platform/base/*.py`` module;
    - it subclasses :class:`abc.ABC`;
    - it is not :class:`abc.ABC` itself.
    """
    # First Party
    import lmcache.v1.platform.base as base_pkg

    base_classes: list[type] = []
    pkg_path = getattr(base_pkg, "__path__", None)
    if pkg_path is None:
        return base_classes

    for _, module_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if is_pkg:
            continue
        full_name = f"{base_pkg.__name__}.{module_name}"
        try:
            mod = importlib.import_module(full_name)
        except Exception:
            logger.warning("Failed to import base module %s", full_name, exc_info=True)
            continue

        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            if cls is abc.ABC or not issubclass(cls, abc.ABC):
                continue
            base_classes.append(cls)
            _REGISTRY.setdefault(cls, {})

    return base_classes


def _discover_all_once() -> None:
    """Populate :data:`_REGISTRY` on first use."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    with _DISCOVERY_LOCK:
        if _DISCOVERED:
            return

        # First Party
        from lmcache.v1.utils.subclass_discovery import discover_subclasses
        import lmcache.v1.platform as platform_pkg

        for base_cls in _collect_base_classes():
            for sub_cls in discover_subclasses(
                platform_pkg,
                base_cls,  # type: ignore[type-abstract]
                levels=[2, 2],
                include_abstract=False,
            ):
                device_type = getattr(sub_cls, "device_type", "")
                if not device_type:
                    logger.warning(
                        "Skipping %s: empty device_type ClassVar; subclasses "
                        "of %s must override device_type.",
                        sub_cls.__name__,
                        base_cls.__name__,
                    )
                    continue

                impl_key_is_explicit = "impl_key" in sub_cls.__dict__
                if (
                    not impl_key_is_explicit
                    and getattr(sub_cls, "_is_default_wrapper", None) is False
                ):
                    continue

                impl_key = getattr(sub_cls, "impl_key", DEFAULT_IMPL_KEY)
                if not impl_key:
                    logger.warning(
                        "Skipping %s: empty impl_key ClassVar; subclasses "
                        "of %s must set a non-empty impl_key.",
                        sub_cls.__name__,
                        base_cls.__name__,
                    )
                    continue

                device_table = _REGISTRY[base_cls].setdefault(device_type, {})
                existing = device_table.get(impl_key)
                if existing is not None and existing is not sub_cls:
                    logger.warning(
                        "Multiple %s subclasses claim device_type=%r, impl_key=%r "
                        "(%s vs %s); keeping the first.",
                        base_cls.__name__,
                        device_type,
                        impl_key,
                        existing.__name__,
                        sub_cls.__name__,
                    )
                    continue
                device_table[impl_key] = sub_cls

        _DISCOVERED = True


def get_impl(
    base_class: type,
    device_type: str,
    impl_key: str = DEFAULT_IMPL_KEY,
) -> type:
    """Get implementation for ``(base_class, device_type, impl_key)``."""
    _discover_all_once()
    by_device = _REGISTRY.get(base_class)
    if by_device is None:
        raise ValueError("Base class %r is not registered." % base_class)

    by_impl = by_device.get(device_type)
    if by_impl is None:
        raise ValueError(
            "No %s implementation registered for device_type=%r"
            % (base_class.__name__, device_type)
        )

    impl = by_impl.get(impl_key)
    if impl is None:
        raise ValueError(
            "No %s implementation registered for device_type=%r, impl_key=%r"
            % (base_class.__name__, device_type, impl_key)
        )
    return impl


def get_all_impls(base_class: type) -> Dict[str, Dict[str, type]]:
    """Return all implementations for ``base_class``."""
    _discover_all_once()
    return {
        device: dict(impls) for device, impls in _REGISTRY.get(base_class, {}).items()
    }


def register_impl(
    base_class: type,
    device_type: str,
    impl_key: str,
    impl_class: type,
) -> None:
    """Register a concrete implementation in the universal registry."""
    _REGISTRY.setdefault(base_class, {}).setdefault(device_type, {})[impl_key] = (
        impl_class
    )


def register_availability(device_type: str, predicate: Callable[[], bool]) -> None:
    """Register an availability predicate for a device type."""
    _AVAILABILITY[device_type] = predicate


def is_available(device_type: str) -> bool:
    """Check whether a device type is available."""
    pred = _AVAILABILITY.get(device_type)
    if pred is None:
        return True
    try:
        return bool(pred())
    except Exception:
        return False


def register_kv_wrapper(device_type: str, factory: Callable[..., Any]) -> None:
    """Backward-compatible manual registration for KV wrapper factories."""
    _KV_WRAPPER_FACTORIES[device_type] = factory


def get_kv_wrapper_factory(device_type: str) -> Callable[..., Any]:
    """Pick the default KV-cache wrapper factory for ``device_type``."""
    manual = _KV_WRAPPER_FACTORIES.get(device_type)
    if manual is not None:
        return manual

    # First Party
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

    cls = get_impl(DeviceIPCWrapper, device_type, DEFAULT_IMPL_KEY)
    return getattr(cls, "wrap", cls)


def snapshot() -> Dict[str, Any]:
    """Return a copy of registry state for tests."""
    return {
        "registry": {
            base: {
                device_type: dict(impls)
                for device_type, impls in by_device.items()
            }
            for base, by_device in _REGISTRY.items()
        },
        "kv_wrapper": dict(_KV_WRAPPER_FACTORIES),
        "availability": dict(_AVAILABILITY),
        "discovered": _DISCOVERED,
    }


def restore(state: Dict[str, Any]) -> None:
    """Restore previously snapshotted registry state."""
    global _DISCOVERED
    _REGISTRY.clear()
    for base, by_device in state.get("registry", {}).items():
        _REGISTRY[base] = {
            device_type: dict(impls) for device_type, impls in by_device.items()
        }

    _KV_WRAPPER_FACTORIES.clear()
    _KV_WRAPPER_FACTORIES.update(state.get("kv_wrapper", {}))

    _AVAILABILITY.clear()
    _AVAILABILITY.update(state.get("availability", {}))
    _DISCOVERED = bool(state.get("discovered", False))


def reset_for_tests() -> None:
    """Wipe registry tables and force re-discovery on next access."""
    global _DISCOVERED
    _REGISTRY.clear()
    _KV_WRAPPER_FACTORIES.clear()
    _AVAILABILITY.clear()
    _DISCOVERED = False
