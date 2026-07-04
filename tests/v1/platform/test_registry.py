# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Iterator
from types import ModuleType
from typing import Any
import pathlib
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform.base._base import PlatformBase
from lmcache.v1.utils.subclass_discovery import discover_subclasses


class _FakeBaseModule(ModuleType):
    """Typed stub module for the synthetic ``platform/base/fake`` module."""

    ImportedMarked: type[PlatformBase]


def test_collect_base_classes_uses_platformbase_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_collect_base_classes`` selects only locally-defined ``PlatformBase``
    subclasses from each ``base/`` module — fully synthetic, no real
    production-class assumptions.

    Property validated: the filter is based solely on the ``PlatformBase``
    marker and the ``cls.__module__ == mod.__name__`` locality check.
    Naming conventions (leading underscore, capitalisation, etc.) play no
    role.  Specifically:

    * Classes with both conventional and underscore-prefixed names are
      included as long as they subclass ``PlatformBase`` and are defined
      in the scanned module.
    * A class that does not subclass ``PlatformBase`` is excluded.
    * A class that subclasses ``PlatformBase`` but whose ``__module__``
      belongs to a different module (i.e. re-exported) is excluded.
    * ``PlatformBase`` itself is excluded.
    """
    saved = platform_registry.snapshot()
    fake_mod = _FakeBaseModule("lmcache.v1.platform.base.fake")
    # exec into fake_mod.__dict__ so the defined classes' __module__
    # matches the module name, exactly as _collect_base_classes expects.
    exec(
        "\n".join(
            [
                "from lmcache.v1.platform.base._base import PlatformBase",
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


def test_device_subclass_discovery_by_device_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """``discover_subclasses`` finds concrete device subclasses of a given
    synthetic base class and never conflates subclasses of different bases —
    fully synthetic, no real production-class assumptions.

    Property validated: given a package tree with depth-2 leaf modules
    (``synth_platform/cpu/impl.py``, ``synth_platform/cuda/impl.py``),
    calling ``discover_subclasses(synth_pkg, FakeBaseA, levels=[2, 2])``
    yields exactly the concrete subclasses of ``FakeBaseA`` defined there,
    and not subclasses of the unrelated ``FakeBaseB``.  Building the
    ``{device_type: cls}`` index (mirroring ``_discover_all_once`` keying)
    produces the expected mapping for each base.

    The synthetic package is built by:
    1. Creating real empty directories/files in ``tmp_path`` so that
       ``pkgutil.iter_modules`` can discover them.
    2. Pre-registering ``ModuleType`` objects (carrying the fake classes,
       with ``__module__`` set via ``exec``) in ``sys.modules`` so that
       ``importlib.import_module`` returns them without touching disk.
    """
    # Build the directory skeleton so pkgutil.iter_modules can walk it.
    # levels=[min_depth, max_depth]=[2, 2] means only depth-2 leaf modules are
    # scanned (the package root is depth 0, its direct children are depth 1):
    #   depth 1: cpu/, cuda/  (sub-packages, recursed into but not scanned)
    #   depth 2: cpu/impl.py, cuda/impl.py  (leaf modules that are scanned)
    for sub in ("cpu", "cuda"):
        subdir = tmp_path / sub
        subdir.mkdir()
        (subdir / "__init__.py").write_text("")
        (subdir / "impl.py").write_text("")

    # Synthetic base classes — defined here so the test owns them entirely.
    class FakeBaseA(PlatformBase):
        pass

    class FakeBaseB(PlatformBase):
        pass

    # Synthetic package objects; __path__ points to the real temp dirs so
    # pkgutil.iter_modules finds the expected structure.
    synth_pkg = ModuleType("synth_platform")
    synth_pkg.__path__ = [str(tmp_path)]  # type: ignore[attr-defined]
    synth_pkg.__package__ = "synth_platform"

    cpu_pkg = ModuleType("synth_platform.cpu")
    cpu_pkg.__path__ = [str(tmp_path / "cpu")]  # type: ignore[attr-defined]
    cpu_pkg.__package__ = "synth_platform.cpu"

    cuda_pkg = ModuleType("synth_platform.cuda")
    cuda_pkg.__path__ = [str(tmp_path / "cuda")]  # type: ignore[attr-defined]
    cuda_pkg.__package__ = "synth_platform.cuda"

    cpu_impl_mod = ModuleType("synth_platform.cpu.impl")
    cpu_impl_mod.__package__ = "synth_platform.cpu"

    cuda_impl_mod = ModuleType("synth_platform.cuda.impl")
    cuda_impl_mod.__package__ = "synth_platform.cuda"

    # Define the synthetic concrete classes inside their synthetic modules
    # via exec so cls.__module__ matches the module name, which is the
    # condition checked by require_defined_in_module=True (the default).
    cpu_impl_mod.__dict__["FakeBaseA"] = FakeBaseA
    cpu_impl_mod.__dict__["FakeBaseB"] = FakeBaseB
    exec(
        "\n".join(
            [
                "class FakeCpuA(FakeBaseA):",
                "    device_type = 'cpu'",
                "class FakeCpuB(FakeBaseB):",
                "    device_type = 'cpu'",
            ]
        ),
        cpu_impl_mod.__dict__,
    )
    FakeCpuA: type = cpu_impl_mod.__dict__["FakeCpuA"]
    FakeCpuB: type = cpu_impl_mod.__dict__["FakeCpuB"]

    cuda_impl_mod.__dict__["FakeBaseA"] = FakeBaseA
    exec(
        "\n".join(
            [
                "class FakeCudaA(FakeBaseA):",
                "    device_type = 'cuda'",
            ]
        ),
        cuda_impl_mod.__dict__,
    )
    FakeCudaA: type = cuda_impl_mod.__dict__["FakeCudaA"]

    # Register all modules in sys.modules so importlib.import_module returns
    # our pre-built ModuleType objects instead of importing from disk.
    for mod_name, mod_obj in [
        ("synth_platform", synth_pkg),
        ("synth_platform.cpu", cpu_pkg),
        ("synth_platform.cuda", cuda_pkg),
        ("synth_platform.cpu.impl", cpu_impl_mod),
        ("synth_platform.cuda.impl", cuda_impl_mod),
    ]:
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    # --- Assertions ---

    a_impls = set(
        discover_subclasses(synth_pkg, FakeBaseA, levels=[2, 2], include_abstract=False)
    )
    # FakeBaseA must discover exactly its two concrete subclasses.
    assert a_impls == {FakeCpuA, FakeCudaA}
    # FakeCpuB belongs to FakeBaseB, not FakeBaseA.
    assert FakeCpuB not in a_impls
    # The base class itself must not be yielded.
    assert FakeBaseA not in a_impls

    b_impls = set(
        discover_subclasses(synth_pkg, FakeBaseB, levels=[2, 2], include_abstract=False)
    )
    # FakeBaseB must discover exactly its one concrete subclass.
    assert b_impls == {FakeCpuB}
    # FakeCpuA belongs to FakeBaseA, not FakeBaseB.
    assert FakeCpuA not in b_impls
    # The base class itself must not be yielded.
    assert FakeBaseB not in b_impls

    # Building the {device_type: cls} index mirrors _discover_all_once's keying.
    a_index = {cls.device_type: cls for cls in a_impls}  # type: ignore[attr-defined]
    assert a_index == {"cpu": FakeCpuA, "cuda": FakeCudaA}

    b_index = {cls.device_type: cls for cls in b_impls}  # type: ignore[attr-defined]
    assert b_index == {"cpu": FakeCpuB}


def test_registry_lookup_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_impl`` / ``get_all_impls`` perform correct two-dimensional
    ``(base_class, device_type)`` lookups — fully synthetic, no real
    production-class assumptions.

    Properties validated:
    * ``get_all_impls(base)`` returns the complete ``{device_type: cls}``
      mapping for that base.
    * ``get_all_impls`` returns a *copy*: mutating it does not affect the
      registry.
    * ``get_impl(base, device_type)`` returns the correct concrete class.
    * Two different base classes stored in the same registry do not
      cross-contaminate: implementations of one base are unreachable via
      the other.
    * ``get_impl`` raises ``ValueError`` for an unknown ``device_type``.
    * ``get_impl`` raises ``ValueError`` for an unregistered base class.

    The registry is pre-populated via ``restore(...)`` with
    ``discovered=True`` so no real package scan is triggered.
    """

    class FakeBaseA(PlatformBase):
        pass

    class FakeBaseB(PlatformBase):
        pass

    class UnregisteredBase(PlatformBase):
        pass

    class FakeCpuA(FakeBaseA):
        device_type = "cpu"

    class FakeCudaA(FakeBaseA):
        device_type = "cuda"

    class FakeCpuB(FakeBaseB):
        device_type = "cpu"

    saved = platform_registry.snapshot()
    try:
        platform_registry.restore(
            {
                "registry": {
                    FakeBaseA: {"cpu": FakeCpuA, "cuda": FakeCudaA},
                    FakeBaseB: {"cpu": FakeCpuB},
                },
                "availability": {},
                "discovered": True,
            }
        )

        # get_all_impls returns the full mapping for each base.
        assert platform_registry.get_all_impls(FakeBaseA) == {
            "cpu": FakeCpuA,
            "cuda": FakeCudaA,
        }
        assert platform_registry.get_all_impls(FakeBaseB) == {"cpu": FakeCpuB}

        # get_all_impls must return a copy: mutating the result must not
        # change the registry.
        copy = platform_registry.get_all_impls(FakeBaseA)
        copy["cpu"] = UnregisteredBase  # deliberately corrupt the copy
        assert platform_registry.get_impl(FakeBaseA, "cpu") is FakeCpuA

        # get_impl returns the correct concrete class.
        assert platform_registry.get_impl(FakeBaseA, "cuda") is FakeCudaA
        assert platform_registry.get_impl(FakeBaseA, "cpu") is FakeCpuA

        # The two bases do not cross-contaminate.
        assert platform_registry.get_impl(FakeBaseB, "cpu") is FakeCpuB
        with pytest.raises(ValueError):
            # FakeCudaA belongs to FakeBaseA, not FakeBaseB.
            platform_registry.get_impl(FakeBaseB, "cuda")

        # Unregistered device_type raises ValueError.
        with pytest.raises(ValueError):
            platform_registry.get_impl(FakeBaseA, "does_not_exist")

        # Unregistered base class raises ValueError.
        with pytest.raises(ValueError):
            platform_registry.get_impl(UnregisteredBase, "cpu")

    finally:
        platform_registry.restore(saved)


def test_discover_all_once_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_discover_all_once`` applies three keying rules when building the
    registry — validated with purely synthetic classes and monkeypatched
    discovery helpers, so no real production classes are involved.

    Rules validated:
    1. A concrete subclass whose ``device_type`` ClassVar is empty (``""``)
       is silently skipped and does not appear in the index.
    2. When two subclasses claim the same ``(base, device_type)`` pair, the
       first one encountered wins and the second is ignored.
    3. A subclass with ``_is_default_wrapper = False`` is skipped while its
       sibling without that flag is registered normally.
    """

    class FakeBase(PlatformBase):
        pass

    class ImplNamed(FakeBase):
        """Has a valid device_type — should be registered."""

        device_type = "mydev"

    class ImplEmpty(FakeBase):
        """Empty device_type — must be skipped."""

        device_type = ""

    class ImplDuplicate(FakeBase):
        """Same device_type as ImplNamed — second arrival must lose."""

        device_type = "mydev"

    class ImplDefault(FakeBase):
        """Default wrapper sibling — should be registered."""

        device_type = "wrapdev"
        _is_default_wrapper = True

    class ImplNonDefault(FakeBase):
        """Non-default wrapper sibling — must be skipped."""

        device_type = "wrapdev"
        _is_default_wrapper = False

    # The order in which discover_subclasses yields the classes determines
    # which duplicate wins.  We yield ImplNamed before ImplDuplicate.
    discovered_order = [
        ImplNamed,
        ImplEmpty,
        ImplDuplicate,
        ImplDefault,
        ImplNonDefault,
    ]

    def fake_collect_base_classes() -> list[type]:
        return [FakeBase]

    def fake_discover_subclasses(
        _pkg: Any, base_cls: type, **_kwargs: Any
    ) -> Iterator[type]:
        if base_cls is FakeBase:
            return iter(discovered_order)
        return iter([])

    monkeypatch.setattr(
        platform_registry, "_collect_base_classes", fake_collect_base_classes
    )
    # discover_subclasses is imported inside _discover_all_once, so patch
    # it on the subclass_discovery module so the fresh import picks it up.
    # First Party
    from lmcache.v1.utils import subclass_discovery as sd_mod

    monkeypatch.setattr(sd_mod, "discover_subclasses", fake_discover_subclasses)

    saved = platform_registry.snapshot()
    try:
        # Pre-populate _REGISTRY with FakeBase so _discover_all_once can write
        # device_type entries to it.  The real _collect_base_classes does this
        # via _REGISTRY.setdefault(cls, {}); our mock returns the list only, so
        # we replicate the side-effect here via restore().
        platform_registry.restore(
            {"registry": {FakeBase: {}}, "availability": {}, "discovered": False}
        )
        # Trigger _discover_all_once via a public API call.
        all_impls = platform_registry.get_all_impls(FakeBase)

        # Rule 1: empty device_type is skipped.
        assert "" not in all_impls

        # Rule 2: first duplicate wins.
        assert all_impls.get("mydev") is ImplNamed
        assert ImplDuplicate not in all_impls.values()

        # Rule 3: _is_default_wrapper=False is skipped; default sibling kept.
        assert all_impls.get("wrapdev") is ImplDefault
        assert ImplNonDefault not in all_impls.values()

    finally:
        platform_registry.restore(saved)
