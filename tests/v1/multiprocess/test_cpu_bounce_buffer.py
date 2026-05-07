# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch
import pickle
import sys

# Third Party
import pytest
import torch


def _make_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> dict[str, torch.Tensor]:
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, block_size, num_heads, head_size
        )
    return kv_caches


def test_wrap_kv_caches_bounce_returns_empty() -> None:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

    assert wrap_kv_caches(_make_kv_caches(), use_bounce_buffer=True) == []


def test_compute_kv_layout_and_gather_scatter_roundtrip() -> None:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _compute_kv_layout,
        _gather_chunks_to_cpu,
        _scatter_cpu_chunks_to_kv,
    )

    source = _make_kv_caches(num_layers=2, num_blocks=8, block_size=4)
    block_size, num_layers, hidden_dim, dtype_str, _ = _compute_kv_layout(source)
    assert block_size == 4
    assert num_layers == 2
    assert hidden_dim == 16
    assert dtype_str == "float32"

    blocks_per_chunk = 2
    gathered = _gather_chunks_to_cpu(source, [0, 1], blocks_per_chunk)
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    _scatter_cpu_chunks_to_kv(destination, [4, 5], gathered, blocks_per_chunk)

    for name in source:
        assert torch.allclose(source[name][:, 0], destination[name][:, 4])
        assert torch.allclose(source[name][:, 1], destination[name][:, 5])


def test_scatter_respects_skip_first_n_tokens() -> None:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        _gather_chunks_to_cpu,
        _scatter_cpu_chunks_to_kv,
    )

    source = _make_kv_caches(num_layers=2, num_blocks=8, block_size=4)
    destination = {
        name: torch.full_like(tensor, 999.0) for name, tensor in source.items()
    }
    gathered = _gather_chunks_to_cpu(source, [0, 1, 2, 3], blocks_per_chunk=4)
    _scatter_cpu_chunks_to_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=4,
        skip_first_n_tokens=8,
    )

    for name in destination:
        assert torch.all(destination[name][:, 0] == 999.0)
        assert torch.all(destination[name][:, 1] == 999.0)
        assert torch.allclose(destination[name][:, 2], source[name][:, 2])
        assert torch.allclose(destination[name][:, 3], source[name][:, 3])


@pytest.fixture
def _stub_native_storage_ops() -> Any:
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})
    module.Bitmap = type("Bitmap", (), {})
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_server_register_and_find_bounce_layout(_stub_native_storage_ops: Any) -> None:
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    with (
        patch("lmcache.v1.multiprocess.server.StorageManager"),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager"),
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
    ):
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=16)
    engine.register_kv_cache_bounce(
        instance_id=1,
        model_name="m",
        world_size=1,
        engine_type=MagicMock(),
        layout_hints={},
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
    )

    layout = engine._find_layout_desc("m", 1)
    assert layout is not None
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_store_and_retrieve_cpu_chunks(_stub_native_storage_ops: Any) -> None:
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    target_tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = target_tensor
    mock_storage.reserve_write.return_value = {"obj": mock_memory_obj}

    @contextmanager
    def _read_prefetched_results(_keys: Any) -> Any:
        yield [mock_memory_obj]

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    with (
        patch(
            "lmcache.v1.multiprocess.server.StorageManager",
            return_value=mock_storage,
        ),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager") as session_cls,
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
        patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["obj"],
        ),
    ):
        session_cls.return_value.get_or_create.return_value = mock_session
        engine = MPCacheEngine(storage_manager_config=MagicMock(), chunk_size=8)

    engine.register_kv_cache_bounce(
        instance_id=2,
        model_name="m",
        world_size=1,
        engine_type=MagicMock(),
        layout_hints={},
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
    )
    payload = torch.ones(2, 2, 8, 16)
    key = IPCCacheEngineKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 8,
        start=0,
        end=8,
        request_id="req",
    )
    with patch(
        "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
        return_value=["obj"],
    ):
        store_ok = engine.store_cpu_chunks(key, 2, pickle.dumps([payload]))
        success, cpu_data = engine.retrieve_cpu_chunks(key, 2)
    assert isinstance(store_ok, bool)
    assert torch.allclose(mock_memory_obj.tensor, payload)

    assert success is True
    recovered_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    assert len(recovered_chunks) == 1
    assert torch.allclose(recovered_chunks[0], payload)
