# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
from typing import Iterator
import unittest
from unittest.mock import patch


class _FakeExtension:
    def __init__(self, name: str, *args, **kwargs) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs


@contextmanager
def _fake_torch_cpp_extension() -> Iterator[types.SimpleNamespace]:
    """Provide a temporary fake `torch.utils.cpp_extension` module for tests.

    Yields:
        A namespace exposing fake CppExtension/CUDAExtension/SyclExtension and
        BuildExtension symbols used by setup.py helper functions.
    """
    cpp_extension = types.SimpleNamespace(
        CppExtension=_FakeExtension,
        CUDAExtension=_FakeExtension,
        SyclExtension=_FakeExtension,
        BuildExtension=type("_FakeBuildExtension", (), {}),
    )
    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_module.cpp_extension = cpp_extension
    torch_module.utils = torch_utils_module

    module_names = ("torch", "torch.utils", "torch.utils.cpp_extension")
    old_modules = {key: sys.modules.get(key) for key in module_names}
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.cpp_extension"] = cpp_extension
    try:
        yield cpp_extension
    finally:
        for key, value in old_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _load_setup_module() -> types.ModuleType:
    """Load setup.py as a module so helper functions can be tested directly.

    Returns:
        The imported setup.py module object.

    Raises:
        RuntimeError: If setup.py cannot be loaded from disk.
    """
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("lmcache_setup", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupExtensionsTests(unittest.TestCase):
    def test_common_cpp_extensions_excludes_mooncake_by_default(self) -> None:
        module = _load_setup_module()
        with _fake_torch_cpp_extension(), patch.dict("os.environ", {}, clear=False):
            ext_modules, _ = module._common_cpp_extensions([])

        self.assertEqual(
            [ext.name for ext in ext_modules],
            [
                "lmcache.native_storage_ops",
                "lmcache.lmcache_redis",
                "lmcache.lmcache_fs",
            ],
        )

    def test_common_cpp_extensions_can_include_mooncake(self) -> None:
        module = _load_setup_module()
        with _fake_torch_cpp_extension(), patch.dict(
            "os.environ", {"BUILD_MOONCAKE": "1"}, clear=False
        ):
            ext_modules, _ = module._common_cpp_extensions([])

        self.assertIn("lmcache.lmcache_mooncake", [ext.name for ext in ext_modules])

    def test_gpu_extension_builders_only_return_gpu_extensions(self) -> None:
        module = _load_setup_module()
        with _fake_torch_cpp_extension(), patch(
            "shutil.which", return_value="/usr/bin/icpx"
        ), patch.dict("os.environ", {}, clear=False):
            module.hipify_wrapper = lambda: None
            cuda_ext_modules, _ = module.cuda_extension()
            rocm_ext_modules, _ = module.rocm_extension()
            sycl_ext_modules, _ = module.sycl_extension()

        self.assertEqual([ext.name for ext in cuda_ext_modules], ["lmcache.c_ops"])
        self.assertEqual([ext.name for ext in rocm_ext_modules], ["lmcache.c_ops"])
        self.assertEqual([ext.name for ext in sycl_ext_modules], ["lmcache.xpu_ops"])

    def test_collect_extensions_keeps_common_cpp_when_no_cuda_ext(self) -> None:
        module = _load_setup_module()
        with (
            _fake_torch_cpp_extension(),
            patch.dict("os.environ", {}, clear=False),
            patch.object(module, "BUILDING_SDIST", False),
            patch.object(module, "NO_CUDA_EXT", True),
            patch.object(module, "BUILD_WITH_HIP", False),
            patch.object(module, "BUILD_WITH_SYCL", False),
        ):
            ext_modules, _ = module._collect_extensions()

        names = [ext.name for ext in ext_modules]
        self.assertIn("lmcache.native_storage_ops", names)
        self.assertIn("lmcache.lmcache_redis", names)
        self.assertIn("lmcache.lmcache_fs", names)
        self.assertNotIn("lmcache.c_ops", names)
        self.assertNotIn("lmcache.xpu_ops", names)

    def test_collect_extensions_returns_empty_for_sdist(self) -> None:
        module = _load_setup_module()
        module.BUILDING_SDIST = True
        module.NO_CUDA_EXT = False
        ext_modules, cmdclass = module._collect_extensions()
        self.assertEqual(ext_modules, [])
        self.assertEqual(cmdclass, {})

if __name__ == "__main__":
    unittest.main()
