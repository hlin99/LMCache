# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Iterator
from types import ModuleType
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform.base import PlatformBase


class _EnumNamespace:
    """Minimal attribute namespace for import-only enum lookups."""

    def __getattr__(self, name: str) -> str:
        return name


@pytest.fixture
def stub_c_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``lmcache.c_ops`` stub for import-only tests."""
    stub = types.ModuleType("lmcache.c_ops")
    stub.EngineKVFormat = _EnumNamespace()
    stub.PageBufferShapeDesc = type("PageBufferShapeDesc", (), {})
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", stub)


def test_collect_base_classes_uses_platformbase_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only local ``PlatformBase`` subclasses qualify as platform bases."""
    saved = platform_registry.snapshot()
    fake_mod = ModuleType("lmcache.v1.platform.base.fake")
    exec(
        "\n".join(
            [
                "from lmcache.v1.platform.base import PlatformBase",
                "class _MarkedBase(PlatformBase):",
                "    pass",
                "class MarkedBase(PlatformBase):",
                "    pass",
                "class Helper:",
                "    pass",
            ]
        ),
        fake_mod.__dict__,
    )
    imported_marked = type("ImportedMarked", (PlatformBase,), {})
    fake_mod.ImportedMarked = imported_marked

    def fake_iter_modules(_: object) -> Iterator[tuple[None, str, bool]]:
        return iter([(None, "fake", False)])

    def fake_import_module(name: str) -> ModuleType:
        if name != "lmcache.v1.platform.base.fake":
            raise AssertionError("unexpected module import %s" % name)
        return fake_mod

    monkeypatch.setattr(platform_registry.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(
        platform_registry.importlib, "import_module", fake_import_module
    )

    try:
        platform_registry.reset_for_tests()
        base_classes = platform_registry._collect_base_classes()
        assert set(base_classes) == {fake_mod._MarkedBase, fake_mod.MarkedBase}
        assert fake_mod.Helper not in base_classes
        assert imported_marked not in base_classes
    finally:
        platform_registry.restore(saved)


def test_registry_discovers_real_context_and_wrapper_impls(
    stub_c_ops: None,
) -> None:
    """The real marker bases discover CPU/CUDA context and wrapper classes."""
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
    from lmcache.v1.platform.base.pin_memory import PinMemoryBackend
    from lmcache.v1.platform.cpu.cache_context import CPUCacheContext
    from lmcache.v1.platform.cpu.shm import CpuShmTensorWrapper
    from lmcache.v1.platform.cuda.cache_context import GPUCacheContext
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

    saved = platform_registry.snapshot()
    try:
        platform_registry.reset_for_tests()

        base_classes = platform_registry._collect_base_classes()
        assert set(base_classes) == {
            BaseCacheContext,
            DeviceIPCWrapper,
            PinMemoryBackend,
        }
        assert PlatformBase not in base_classes

        cache_context_impls = platform_registry.get_all_impls(BaseCacheContext)
        assert cache_context_impls == {
            "cpu": CPUCacheContext,
            "cuda": GPUCacheContext,
        }
        assert (
            platform_registry.get_impl(DeviceIPCWrapper, "cpu")
            is CpuShmTensorWrapper
        )
        assert platform_registry.get_impl(DeviceIPCWrapper, "cuda") is CudaIPCWrapper
        assert PlatformBase not in platform_registry.snapshot()["registry"]
    finally:
        platform_registry.restore(saved)
