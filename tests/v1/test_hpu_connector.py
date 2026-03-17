# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import math
import sys

# Third Party
import pytest


def _flatten_nested(values):
    if isinstance(values, list):
        if not values:
            return [], (0,)
        flat = []
        child_shape = None
        for item in values:
            child_flat, item_shape = _flatten_nested(item)
            if child_shape is None:
                child_shape = item_shape
            else:
                assert child_shape == item_shape
            flat.extend(child_flat)
        assert child_shape is not None
        return flat, (len(values), *child_shape)
    return [values], ()


class FakeTensor:
    def __init__(self, values, shape=None):
        if shape is None:
            flat, inferred_shape = _flatten_nested(values)
            self._storage = flat
            self.shape = inferred_shape
        else:
            self._storage = values
            self.shape = tuple(shape)

    @property
    def ndim(self):
        return len(self.shape)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        assert math.prod(shape) == len(self._storage)
        return FakeTensor(self._storage, shape)

    def tolist(self):
        def build(offset, shape):
            if not shape:
                return self._storage[offset], offset + 1
            items = []
            next_offset = offset
            stride = math.prod(shape[1:]) if len(shape) > 1 else 1
            for _ in range(shape[0]):
                item, next_offset = build(next_offset, shape[1:])
                items.append(item)
            assert next_offset == offset + shape[0] * stride
            return items, next_offset

        nested, end = build(0, self.shape)
        assert end == len(self._storage)
        return nested


def _make_fake_torch_module():
    torch_module = ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.Size = tuple
    torch_module.device = str
    torch_module.dtype = object
    torch_module.int64 = "int64"
    torch_module.bfloat16 = "bfloat16"
    return torch_module


def _noop_logger(*args, **kwargs):
    return None


def _load_hpu_connector_module():
    fake_torch = _make_fake_torch_module()
    fake_habana_torch = ModuleType("habana_frameworks.torch")
    fake_habana_torch.core = SimpleNamespace(mark_step=lambda: None)

    fake_memory_management = ModuleType("lmcache.v1.memory_management")
    fake_memory_management.MemoryFormat = SimpleNamespace(
        KV_MLA_FMT="KV_MLA_FMT", KV_2LTD="KV_2LTD"
    )
    fake_memory_management.MemoryObj = object

    fake_modules = {
        "torch": fake_torch,
        "habana_frameworks": ModuleType("habana_frameworks"),
        "habana_frameworks.torch": fake_habana_torch,
        "lmcache.logging": ModuleType("lmcache.logging"),
        "lmcache.utils": ModuleType("lmcache.utils"),
        "lmcache.v1.gpu_connector": ModuleType("lmcache.v1.gpu_connector"),
        "lmcache.v1.memory_management": fake_memory_management,
        "lmcache.v1.metadata": ModuleType("lmcache.v1.metadata"),
    }
    fake_modules["lmcache.logging"].init_logger = lambda name: SimpleNamespace(
        error=_noop_logger
    )
    fake_modules["lmcache.utils"]._lmcache_nvtx_annotate = lambda func: func
    fake_modules["lmcache.v1.gpu_connector"].GPUConnectorInterface = object
    fake_modules["lmcache.v1.metadata"].LMCacheMetadata = object

    previous_modules = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    sys.modules["habana_frameworks"].torch = fake_habana_torch

    try:
        module_path = (
            Path(__file__).resolve().parents[2]
            / "lmcache"
            / "v1"
            / "gpu_connector"
            / "hpu_connector.py"
        )
        spec = importlib.util.spec_from_file_location("hpu_connector_under_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture(scope="module")
def hpu_connector_cls():
    return _load_hpu_connector_module().VLLMPagedMemHPUConnectorV2


def test_get_mla_token_major_view_handles_paged_layout(hpu_connector_cls):
    kv_cache = FakeTensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]
    )

    flattened = hpu_connector_cls._get_mla_token_major_view(kv_cache)

    assert flattened.shape == (6, 2)
    assert flattened.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]


def test_get_mla_token_major_view_preserves_legacy_layout(hpu_connector_cls):
    kv_cache = FakeTensor([[0, 1], [2, 3], [4, 5]])

    flattened = hpu_connector_cls._get_mla_token_major_view(kv_cache)

    assert flattened.shape == (3, 2)
    assert flattened.tolist() == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "kv_cache",
    [
        FakeTensor([1, 2, 3]),
        FakeTensor([[[[1], [2]]]]),
    ],
)
def test_get_mla_token_major_view_rejects_invalid_layout(hpu_connector_cls, kv_cache):
    with pytest.raises(ValueError, match="Unsupported MLA KV cache shape"):
        hpu_connector_cls._get_mla_token_major_view(kv_cache)
