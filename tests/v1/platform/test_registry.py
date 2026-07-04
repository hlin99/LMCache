# SPDX-License-Identifier: Apache-2.0
"""Tests for the universal 3-D platform backend registry.

Two test groups:

**Mechanism tests**
    Synthetic ABC base classes and stub implementations exercise the
    registry machinery -- base discovery, 3-D lookup, multiple
    ``impl_key`` variants, strict ``get_impl`` semantics, and
    ``resolve_impl`` fallback -- without touching any production backend.

**Real-wiring tests**
    Verify the production pin-memory wiring:

    * :class:`~lmcache.v1.platform.base.pin_memory.PinMemoryBackend` is
      discovered as a registry base class.
    * ``get_impl(PinMemoryBackend, "cuda", "default")`` returns
      :class:`~lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend`.
    * ``resolve_impl(PinMemoryBackend, "cpu", "default")`` returns
      :class:`~lmcache.v1.platform.base.pin_memory.PinMemoryBackend`
      (no-op fallback) when no CPU implementation is registered.
    * :class:`~lmcache.v1.platform.device_ext.DeviceExt` selects the
      correct backend for CUDA and the no-op fallback for devices without
      a concrete pin-memory implementation.
"""

# Standard
from typing import ClassVar
import abc

# Third Party
import pytest

# First Party
from lmcache.v1.platform._registry import (
    _discover_base_classes,
    _register_impl,
    get_impl,
    reset_registry_for_tests,
    resolve_impl,
    restore_registry,
    snapshot_registry,
)
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
import lmcache.v1.platform._registry as _reg_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot the universal registry before each test and restore it
    after, so individual tests can mutate state without polluting others."""
    state = snapshot_registry()
    reset_registry_for_tests()
    try:
        yield
    finally:
        restore_registry(state)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


class _SyntheticBase(abc.ABC):  # noqa: B024
    """A synthetic registry base class used by mechanism tests."""

    @classmethod
    def registry_fallback(
        cls, device_type: str, impl_key: str = "default"
    ) -> "type[_SyntheticBase]":
        """Return the no-op base as fallback."""
        return cls


class _SyntheticNoFallbackBase(abc.ABC):  # noqa: B024
    """A synthetic base class with *no* fallback -- strict capability."""


class _ImplA(_SyntheticBase):
    device_type: ClassVar[str] = "alpha"
    impl_key: ClassVar[str] = "default"


class _ImplB(_SyntheticBase):
    device_type: ClassVar[str] = "alpha"
    impl_key: ClassVar[str] = "variant"


class _ImplC(_SyntheticBase):
    device_type: ClassVar[str] = "beta"
    impl_key: ClassVar[str] = "default"


class _ImplD(_SyntheticNoFallbackBase):
    device_type: ClassVar[str] = "alpha"
    impl_key: ClassVar[str] = "default"


# ---------------------------------------------------------------------------
# Mechanism tests
# ---------------------------------------------------------------------------


class TestRegisterImplAndGetImpl:
    """_register_impl / get_impl basic wiring."""

    def test_register_and_retrieve(self) -> None:
        """A registered implementation is returned by get_impl."""
        _register_impl(_SyntheticBase, _ImplA)
        result = get_impl(_SyntheticBase, "alpha", "default")
        assert result is _ImplA

    def test_multiple_impl_keys_same_device(self) -> None:
        """Two implementations sharing a device but different impl_keys
        coexist without conflict."""
        _register_impl(_SyntheticBase, _ImplA)
        _register_impl(_SyntheticBase, _ImplB)
        assert get_impl(_SyntheticBase, "alpha", "default") is _ImplA
        assert get_impl(_SyntheticBase, "alpha", "variant") is _ImplB

    def test_multiple_device_types(self) -> None:
        """Implementations for different device types are independent."""
        _register_impl(_SyntheticBase, _ImplA)
        _register_impl(_SyntheticBase, _ImplC)
        assert get_impl(_SyntheticBase, "alpha") is _ImplA
        assert get_impl(_SyntheticBase, "beta") is _ImplC

    def test_get_impl_default_impl_key(self) -> None:
        """impl_key defaults to 'default'."""
        _register_impl(_SyntheticBase, _ImplA)
        assert get_impl(_SyntheticBase, "alpha") is _ImplA

    def test_get_impl_raises_unregistered_base(self) -> None:
        """ValueError for a base class not in the registry."""
        with pytest.raises(ValueError, match="not registered"):
            get_impl(_SyntheticBase, "alpha")

    def test_get_impl_raises_unregistered_device(self) -> None:
        """ValueError for a missing device_type within a registered base."""
        _register_impl(_SyntheticBase, _ImplA)
        with pytest.raises(ValueError, match="device_type"):
            get_impl(_SyntheticBase, "gamma")

    def test_get_impl_raises_unregistered_impl_key(self) -> None:
        """ValueError for a missing impl_key within a registered device."""
        _register_impl(_SyntheticBase, _ImplA)
        with pytest.raises(ValueError, match="impl_key"):
            get_impl(_SyntheticBase, "alpha", "nonexistent")

    def test_first_registration_wins_on_collision(self, caplog) -> None:
        """Duplicate (base, device_type, impl_key) keeps the first entry
        and emits a warning."""

        class _ImplADuplicate(_SyntheticBase):
            device_type: ClassVar[str] = "alpha"
            impl_key: ClassVar[str] = "default"

        _register_impl(_SyntheticBase, _ImplA)
        _register_impl(_SyntheticBase, _ImplADuplicate)
        assert get_impl(_SyntheticBase, "alpha") is _ImplA

    def test_no_device_type_skipped(self, caplog) -> None:
        """An implementation without device_type is not registered."""

        class _NoDeviceType(_SyntheticBase):
            pass

        _register_impl(_SyntheticBase, _NoDeviceType)
        # The base class entry may not even exist if nothing was registered.
        with pytest.raises(ValueError):
            get_impl(_SyntheticBase, "")


class TestResolveImpl:
    """resolve_impl fallback semantics."""

    def test_returns_concrete_when_registered(self) -> None:
        """resolve_impl returns the concrete class when registered."""
        _register_impl(_SyntheticBase, _ImplA)
        assert resolve_impl(_SyntheticBase, "alpha") is _ImplA

    def test_fallback_when_device_not_registered(self) -> None:
        """resolve_impl returns the base fallback for an unknown device."""
        # No implementations registered at all for _SyntheticBase.
        _reg_module._REGISTRY.setdefault(_SyntheticBase, {})
        result = resolve_impl(_SyntheticBase, "unknown_device")
        assert result is _SyntheticBase

    def test_no_fallback_raises(self) -> None:
        """A base class without registry_fallback propagates ValueError."""
        _reg_module._REGISTRY.setdefault(_SyntheticNoFallbackBase, {})
        with pytest.raises(ValueError):
            resolve_impl(_SyntheticNoFallbackBase, "alpha")

    def test_fallback_with_impl_key(self) -> None:
        """Fallback is also triggered for an unknown impl_key."""
        _reg_module._REGISTRY.setdefault(_SyntheticBase, {})
        result = resolve_impl(_SyntheticBase, "anything", "missing_key")
        assert result is _SyntheticBase


# ---------------------------------------------------------------------------
# Real-wiring tests
# ---------------------------------------------------------------------------


class TestPinMemoryBaseDiscovery:
    """PinMemoryBackend is discoverable via _discover_base_classes."""

    def test_pin_memory_backend_is_discovered(self) -> None:
        """`PinMemoryBackend` appears in the list returned by
        _discover_base_classes because it lives in
        ``lmcache.v1.platform.base.pin_memory`` and subclasses abc.ABC."""
        base_classes = _discover_base_classes()
        assert PinMemoryBackend in base_classes


class TestPinMemoryRegistryWiring:
    """Production (base_class, device_type, impl_key) lookup for pin memory."""

    def _prime_registry(self) -> None:
        """Force discovery so the real backends are in _REGISTRY."""
        _reg_module._discover_all_once()

    def test_get_impl_cuda(self) -> None:
        """get_impl(PinMemoryBackend, 'cuda') returns CudaPinMemoryBackend."""
        # First Party
        from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend

        self._prime_registry()
        result = get_impl(PinMemoryBackend, "cuda", "default")
        assert result is CudaPinMemoryBackend

    def test_resolve_impl_cpu_fallback(self) -> None:
        """resolve_impl(PinMemoryBackend, 'cpu') returns the no-op base when
        no CPU pin-memory implementation is registered."""
        self._prime_registry()
        result = resolve_impl(PinMemoryBackend, "cpu", "default")
        # No CPU pin-memory backend exists; fallback is the base class.
        assert result is PinMemoryBackend

    def test_get_impl_cpu_raises(self) -> None:
        """get_impl (strict) raises ValueError for 'cpu' since no CPU
        pin-memory implementation is registered."""
        self._prime_registry()
        with pytest.raises(ValueError):
            get_impl(PinMemoryBackend, "cpu", "default")


class TestDeviceExtPinMemoryWiring:
    """DeviceExt selects pin-memory backend correctly."""

    def test_device_ext_uses_cuda_backend_for_cuda_device_info(self) -> None:
        """DeviceExt.__init__ honours DeviceInfo.pin_memory_backend when
        it is non-None (backward-compatibility path)."""
        # First Party
        from lmcache.v1.platform.cuda.pin_memory import CudaPinMemoryBackend
        from lmcache.v1.platform.device_ext import DeviceExt

        class _FakeCudaDeviceInfo:
            @property
            def device_type(self) -> str:
                return "cuda"

            @property
            def pin_memory_backend(self):
                return CudaPinMemoryBackend

        ext = DeviceExt(_FakeCudaDeviceInfo())  # type: ignore[arg-type]
        assert isinstance(ext._pin, CudaPinMemoryBackend)

    def test_device_ext_uses_noop_for_cpu_via_registry(self) -> None:
        """DeviceExt falls back to the no-op PinMemoryBackend for a device
        with no registered concrete backend and no DeviceInfo override."""
        # First Party
        from lmcache.v1.platform.device_ext import DeviceExt

        class _FakeCpuDeviceInfo:
            @property
            def device_type(self) -> str:
                return "cpu"

            @property
            def pin_memory_backend(self):
                return None

        _reg_module._discover_all_once()
        ext = DeviceExt(_FakeCpuDeviceInfo())  # type: ignore[arg-type]
        # CPU has no concrete backend; resolve_impl falls back to PinMemoryBackend.
        assert type(ext._pin) is PinMemoryBackend
        assert not ext.is_pin_supported

    def test_device_ext_without_device_info_uses_noop(self) -> None:
        """DeviceExt(None) resolves 'cpu' through the registry and gets
        the no-op fallback."""
        # First Party
        from lmcache.v1.platform.device_ext import DeviceExt

        _reg_module._discover_all_once()
        ext = DeviceExt(None)
        assert type(ext._pin) is PinMemoryBackend
        assert not ext.is_pin_supported
