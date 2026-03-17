# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import logging
import sys

# Third Party
import pytest
import torch


class _KVCachePair(tuple):
    def __new__(
        cls, kcache: torch.Tensor, vcache: torch.Tensor
    ) -> "_KVCachePair":
        return super().__new__(cls, (kcache, vcache))

    @property
    def device(self) -> torch.device:
        return self[0].device


def _load_hpu_connector_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    target_path = Path(
        "/home/runner/work/LMCache/LMCache/lmcache/v1/gpu_connector/hpu_connector.py"
    )

    habana_frameworks = ModuleType("habana_frameworks")
    habana_frameworks.__path__ = []
    htorch = ModuleType("habana_frameworks.torch")
    htorch.core = SimpleNamespace(mark_step=lambda: None)
    monkeypatch.setitem(sys.modules, "habana_frameworks", habana_frameworks)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", htorch)

    lmcache = ModuleType("lmcache")
    lmcache.__path__ = []
    monkeypatch.setitem(sys.modules, "lmcache", lmcache)

    lmcache_logging = ModuleType("lmcache.logging")
    lmcache_logging.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "lmcache.logging", lmcache_logging)

    lmcache_utils = ModuleType("lmcache.utils")
    lmcache_utils._lmcache_nvtx_annotate = lambda func: func
    monkeypatch.setitem(sys.modules, "lmcache.utils", lmcache_utils)

    lmcache_v1 = ModuleType("lmcache.v1")
    lmcache_v1.__path__ = []
    monkeypatch.setitem(sys.modules, "lmcache.v1", lmcache_v1)

    lmcache_gpu_connector = ModuleType("lmcache.v1.gpu_connector")

    class GPUConnectorInterface:
        def initialize_kvcaches_ptr(self, **kwargs):
            if "kvcaches" in kwargs:
                self.kvcaches = kwargs["kvcaches"]

    lmcache_gpu_connector.GPUConnectorInterface = GPUConnectorInterface
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.gpu_connector", lmcache_gpu_connector
    )

    lmcache_memory_management = ModuleType("lmcache.v1.memory_management")
    memory_format = SimpleNamespace(
        KV_2LTD="kv_2ltd",
        KV_MLA_FMT="kv_mla_fmt",
    )
    lmcache_memory_management.MemoryFormat = memory_format
    lmcache_memory_management.MemoryObj = object
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.memory_management", lmcache_memory_management
    )

    lmcache_metadata = ModuleType("lmcache.v1.metadata")
    lmcache_metadata.LMCacheMetadata = object
    monkeypatch.setitem(sys.modules, "lmcache.v1.metadata", lmcache_metadata)

    spec = importlib.util.spec_from_file_location(
        "test_hpu_connector_target", target_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_memory_obj(tensor: torch.Tensor, fmt) -> SimpleNamespace:
    return SimpleNamespace(
        tensor=tensor,
        metadata=SimpleNamespace(fmt=fmt),
    )


def _make_non_mla_kvcaches(
    num_layers: int,
    page_buffer_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    fill_value: float | None = None,
) -> list[_KVCachePair]:
    ret = []
    for layer_id in range(num_layers):
        if fill_value is None:
            base = torch.arange(
                page_buffer_size * num_heads * head_size, dtype=dtype
            ).reshape(page_buffer_size, num_heads, head_size)
            kcache = base + layer_id * 1000
            vcache = base + layer_id * 1000 + 500
        else:
            kcache = torch.full(
                (page_buffer_size, num_heads, head_size), fill_value, dtype=dtype
            )
            vcache = torch.full(
                (page_buffer_size, num_heads, head_size), fill_value, dtype=dtype
            )
        ret.append(_KVCachePair(kcache, vcache))
    return ret


def _make_mla_kvcaches(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    fill_value: float | None = None,
) -> list[torch.Tensor]:
    ret = []
    for layer_id in range(num_layers):
        if fill_value is None:
            cache = torch.arange(
                num_blocks * block_size * head_size, dtype=dtype
            ).reshape(num_blocks, block_size, head_size)
            cache = cache + layer_id * 1000
        else:
            cache = torch.full(
                (num_blocks, block_size, head_size), fill_value, dtype=dtype
            )
        ret.append(cache)
    return ret


def test_hpu_connector_from_gpu_and_to_gpu_round_trip_without_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hpu_connector_module(monkeypatch)

    num_layers = 3
    page_buffer_size = 8
    num_heads = 2
    head_size = 4
    hidden_dim_size = num_heads * head_size
    dtype = torch.float32
    start = 1
    end = 4
    slot_mapping = torch.tensor([6, 1, 4, 0, 7], dtype=torch.int64)

    src_kvcaches = _make_non_mla_kvcaches(
        num_layers=num_layers,
        page_buffer_size=page_buffer_size,
        num_heads=num_heads,
        head_size=head_size,
        dtype=dtype,
    )
    dst_kvcaches = _make_non_mla_kvcaches(
        num_layers=num_layers,
        page_buffer_size=page_buffer_size,
        num_heads=num_heads,
        head_size=head_size,
        dtype=dtype,
        fill_value=-1.0,
    )
    dst_before = [(k.clone(), v.clone()) for k, v in dst_kvcaches]

    connector = module.VLLMPagedMemHPUConnectorV2(
        hidden_dim_size=hidden_dim_size,
        num_layers=num_layers,
        chunk_size=end - start,
        dtype=dtype,
        device="cpu",
    )
    memory_obj = _make_memory_obj(
        tensor=torch.empty(connector.get_shape(end - start), dtype=dtype),
        fmt=module.MemoryFormat.KV_2LTD,
    )

    connector.from_gpu(
        memory_obj,
        start,
        end,
        kvcaches=src_kvcaches,
        slot_mapping=slot_mapping,
    )

    expected_k = torch.stack(
        [
            src_kvcaches[layer_id][0]
            .view(page_buffer_size, hidden_dim_size)
            .index_select(0, slot_mapping[start:end])
            for layer_id in range(num_layers)
        ]
    )
    expected_v = torch.stack(
        [
            src_kvcaches[layer_id][1]
            .view(page_buffer_size, hidden_dim_size)
            .index_select(0, slot_mapping[start:end])
            for layer_id in range(num_layers)
        ]
    )
    assert torch.equal(memory_obj.tensor[0], expected_k)
    assert torch.equal(memory_obj.tensor[1], expected_v)

    connector.to_gpu(
        memory_obj,
        start,
        end,
        kvcaches=dst_kvcaches,
        slot_mapping=slot_mapping,
    )

    selected_slots = slot_mapping[start:end]
    untouched_slots = torch.tensor(
        [idx for idx in range(page_buffer_size) if idx not in selected_slots.tolist()],
        dtype=torch.int64,
    )
    for layer_id in range(num_layers):
        dst_k = dst_kvcaches[layer_id][0].view(page_buffer_size, hidden_dim_size)
        dst_v = dst_kvcaches[layer_id][1].view(page_buffer_size, hidden_dim_size)
        src_k = src_kvcaches[layer_id][0].view(page_buffer_size, hidden_dim_size)
        src_v = src_kvcaches[layer_id][1].view(page_buffer_size, hidden_dim_size)
        old_k = dst_before[layer_id][0].view(page_buffer_size, hidden_dim_size)
        old_v = dst_before[layer_id][1].view(page_buffer_size, hidden_dim_size)

        assert torch.equal(
            dst_k.index_select(0, selected_slots),
            src_k.index_select(0, selected_slots),
        )
        assert torch.equal(
            dst_v.index_select(0, selected_slots),
            src_v.index_select(0, selected_slots),
        )
        assert torch.equal(
            dst_k.index_select(0, untouched_slots),
            old_k.index_select(0, untouched_slots),
        )
        assert torch.equal(
            dst_v.index_select(0, untouched_slots),
            old_v.index_select(0, untouched_slots),
        )


def test_hpu_connector_from_gpu_and_to_gpu_round_trip_with_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hpu_connector_module(monkeypatch)

    num_layers = 3
    num_blocks = 2
    block_size = 4
    head_size = 5
    total_slots = num_blocks * block_size
    dtype = torch.float32
    start = 0
    end = 3
    slot_mapping = torch.tensor([5, 2, 7, 0], dtype=torch.int64)

    src_kvcaches = _make_mla_kvcaches(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
    )
    dst_kvcaches = _make_mla_kvcaches(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        fill_value=-1.0,
    )
    dst_before = [cache.clone() for cache in dst_kvcaches]

    connector = module.VLLMPagedMemHPUConnectorV2(
        hidden_dim_size=head_size,
        num_layers=num_layers,
        chunk_size=end - start,
        dtype=dtype,
        device="cpu",
        use_mla=True,
    )
    memory_obj = _make_memory_obj(
        tensor=torch.empty(connector.get_shape(end - start), dtype=dtype),
        fmt=module.MemoryFormat.KV_2LTD,
    )

    connector.from_gpu(
        memory_obj,
        start,
        end,
        kvcaches=src_kvcaches,
        slot_mapping=slot_mapping,
    )

    expected = torch.stack(
        [
            src_kvcaches[layer_id]
            .view(total_slots, head_size)
            .index_select(0, slot_mapping[start:end])
            for layer_id in range(num_layers)
        ]
    )
    assert memory_obj.metadata.fmt == module.MemoryFormat.KV_MLA_FMT
    assert torch.equal(memory_obj.tensor[0], expected)

    connector.to_gpu(
        memory_obj,
        start,
        end,
        kvcaches=dst_kvcaches,
        slot_mapping=slot_mapping,
    )

    selected_slots = slot_mapping[start:end]
    untouched_slots = torch.tensor(
        [idx for idx in range(total_slots) if idx not in selected_slots.tolist()],
        dtype=torch.int64,
    )
    for layer_id in range(num_layers):
        dst = dst_kvcaches[layer_id].view(total_slots, head_size)
        src = src_kvcaches[layer_id].view(total_slots, head_size)
        old = dst_before[layer_id].view(total_slots, head_size)

        assert torch.equal(
            dst.index_select(0, selected_slots),
            src.index_select(0, selected_slots),
        )
        assert torch.equal(
            dst.index_select(0, untouched_slots),
            old.index_select(0, untouched_slots),
        )
