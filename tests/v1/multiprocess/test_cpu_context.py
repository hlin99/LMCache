# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import MagicMock, patch
import pickle
import sys

# Third Party
import pytest
import torch

NUM_LAYERS = 2
DEFAULT_NUM_BLOCKS = 6
TEST_NUM_BLOCKS = 8
BLOCK_SIZE = 4
NUM_HEADS = 2
HEAD_SIZE = 8
HIDDEN_SIZE = 16
ROUNDTRIP_BLOCKS_PER_CHUNK = 2
SKIP_TEST_BLOCKS_PER_CHUNK = 4
SENTINEL_VALUE = 999.0
MODEL_NAME = "m"
REGISTER_CHUNK_SIZE = 16
STORE_CHUNK_SIZE = 8


def _make_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_heads: int = NUM_HEADS,
    head_size: int = HEAD_SIZE,
) -> dict[str, torch.Tensor]:
    """Build per-layer NHD KV tensors for CPU cpu context tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, block_size, num_heads, head_size
        )
    return kv_caches


def _make_mla_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[str, torch.Tensor]:
    """Build per-layer MLA KV tensors for CPU cpu context tests.

    Args:
        num_layers: Number of KV layers to generate.
        num_blocks: Number of paged blocks per layer.
        block_size: Number of tokens per block.
        hidden_size: Hidden size per token.

    Returns:
        Mapping from layer name to MLA KV tensor with shape
        ``[num_blocks, block_size, hidden_size]``.
    """
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(num_blocks, block_size, hidden_size)
    return kv_caches


def _make_hnd_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_heads: int = NUM_HEADS,
    head_size: int = HEAD_SIZE,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND KV tensors for CPU cpu context tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            2, num_blocks, num_heads, block_size, head_size
        )
    return kv_caches


def _make_hnd_flashinfer_kv_caches(
    num_layers: int = NUM_LAYERS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_heads: int = NUM_HEADS,
    head_size: int = HEAD_SIZE,
) -> dict[str, torch.Tensor]:
    """Build per-layer HND flash-infer KV tensors for CPU cpu context tests."""
    kv_caches = {}
    for i in range(num_layers):
        kv_caches[f"layer_{i}"] = torch.randn(
            num_blocks, 2, num_heads, block_size, head_size
        )
    return kv_caches


def test_wrap_kv_caches_cpu_context_returns_empty() -> None:
    """Verify wrap_kv_caches returns no IPC wrappers in cpu context mode."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

    assert wrap_kv_caches(_make_kv_caches(), use_cpu_context=True) == []


@pytest.mark.parametrize(
    ("source_builder", "is_mla"),
    [
        (_make_kv_caches, False),
        (_make_mla_kv_caches, True),
    ],
)
def test_compute_kv_layout_and_gather_scatter_roundtrip(
    source_builder: Callable[..., dict[str, torch.Tensor]], is_mla: bool
) -> None:
    """Validate layout extraction and gather/scatter round-trip for NHD and MLA."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        compute_kv_layout,
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )

    if is_mla:
        source = source_builder(
            num_layers=NUM_LAYERS,
            num_blocks=TEST_NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            hidden_size=HIDDEN_SIZE,
        )
    else:
        source = source_builder(
            num_layers=NUM_LAYERS,
            num_blocks=TEST_NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
        )
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
    ) = compute_kv_layout(source)
    assert block_size == BLOCK_SIZE
    assert num_layers == NUM_LAYERS
    assert hidden_dim == HIDDEN_SIZE
    assert dtype_str == "float32"
    assert detected_kv_format is not None

    gathered = gather_chunks_to_cpu(source, [0, 1], ROUNDTRIP_BLOCKS_PER_CHUNK)
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_chunks_to_kv(
        destination,
        [4, 5],
        gathered,
        ROUNDTRIP_BLOCKS_PER_CHUNK,
    )

    for name in source:
        if is_mla:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])
        else:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])


@pytest.mark.parametrize(
    ("hnd_builder", "expected_format"),
    [
        (_make_hnd_kv_caches, "NL_X_TWO_NB_NH_BS_HS"),
        (_make_hnd_flashinfer_kv_caches, "NL_X_NB_TWO_NH_BS_HS"),
    ],
)
def test_gather_scatter_roundtrip_hnd_layout(
    hnd_builder: Callable[[int, int, int, int, int], dict[str, torch.Tensor]],
    expected_format: str,
) -> None:
    """Validate gather/scatter round-trip for HND vLLM KV layout."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        compute_kv_layout,
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )
    import lmcache.c_ops as lmc_ops

    source = hnd_builder(NUM_LAYERS, TEST_NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE)
    layout_hints = {"kv_layout": "HND"}
    (
        block_size,
        num_layers,
        hidden_dim,
        dtype_str,
        detected_kv_format,
    ) = compute_kv_layout(source, layout_hints=layout_hints)
    assert block_size == BLOCK_SIZE
    assert num_layers == NUM_LAYERS
    assert hidden_dim == HIDDEN_SIZE
    assert dtype_str == "float32"
    assert detected_kv_format == getattr(lmc_ops.GPUKVFormat, expected_format)

    gathered = gather_chunks_to_cpu(
        source,
        [0, 1],
        ROUNDTRIP_BLOCKS_PER_CHUNK,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )
    destination = {name: torch.zeros_like(tensor) for name, tensor in source.items()}
    scatter_cpu_chunks_to_kv(
        destination,
        [4, 5],
        gathered,
        ROUNDTRIP_BLOCKS_PER_CHUNK,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )

    for name in source:
        if detected_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
            assert torch.allclose(source[name][:, 0], destination[name][:, 4])
            assert torch.allclose(source[name][:, 1], destination[name][:, 5])
        else:
            assert torch.allclose(source[name][0], destination[name][4])
            assert torch.allclose(source[name][1], destination[name][5])


@pytest.mark.parametrize("skip_first_n_tokens", [8, 5])
def test_scatter_respects_skip_first_n_tokens(skip_first_n_tokens: int) -> None:
    """Ensure NHD scatter honors token skips; non-aligned values round down."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )

    source = _make_kv_caches(
        num_layers=NUM_LAYERS,
        num_blocks=TEST_NUM_BLOCKS,
        block_size=BLOCK_SIZE,
    )
    destination = {
        name: torch.full_like(tensor, SENTINEL_VALUE) for name, tensor in source.items()
    }
    gathered = gather_chunks_to_cpu(
        source, [0, 1, 2, 3], blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK
    )
    scatter_cpu_chunks_to_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK,
        skip_first_n_tokens=skip_first_n_tokens,
    )

    first_written_block = skip_first_n_tokens // BLOCK_SIZE
    for name in destination:
        for block_idx in range(first_written_block):
            assert torch.all(destination[name][:, block_idx] == SENTINEL_VALUE)
        for block_idx in range(first_written_block, SKIP_TEST_BLOCKS_PER_CHUNK):
            assert torch.allclose(
                destination[name][:, block_idx],
                source[name][:, block_idx],
            )


@pytest.mark.parametrize(
    ("hnd_builder", "expected_format"),
    [
        (_make_hnd_kv_caches, "NL_X_TWO_NB_NH_BS_HS"),
        (_make_hnd_flashinfer_kv_caches, "NL_X_NB_TWO_NH_BS_HS"),
    ],
)
def test_scatter_hnd_respects_skip_first_n_tokens(
    hnd_builder: Callable[[int, int, int, int, int], dict[str, torch.Tensor]],
    expected_format: str,
) -> None:
    """Ensure HND/HND-FlashInfer scatter honors skip_first_n_tokens."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        compute_kv_layout,
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )
    import lmcache.c_ops as lmc_ops

    source = hnd_builder(NUM_LAYERS, TEST_NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE)
    layout_hints = {"kv_layout": "HND"}
    _, _, _, _, detected_kv_format = compute_kv_layout(
        source, layout_hints=layout_hints
    )
    assert detected_kv_format == getattr(lmc_ops.GPUKVFormat, expected_format)

    destination = {
        name: torch.full_like(tensor, SENTINEL_VALUE) for name, tensor in source.items()
    }
    gathered = gather_chunks_to_cpu(
        source,
        [0, 1, 2, 3],
        blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )
    scatter_cpu_chunks_to_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK,
        skip_first_n_tokens=8,
        layout_hints=layout_hints,
        gpu_kv_format=detected_kv_format,
    )

    for name in destination:
        if detected_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
            assert torch.all(destination[name][:, 0] == SENTINEL_VALUE)
            assert torch.all(destination[name][:, 1] == SENTINEL_VALUE)
            assert torch.allclose(destination[name][:, 2], source[name][:, 2])
            assert torch.allclose(destination[name][:, 3], source[name][:, 3])
        else:
            assert torch.all(destination[name][0] == SENTINEL_VALUE)
            assert torch.all(destination[name][1] == SENTINEL_VALUE)
            assert torch.allclose(destination[name][2], source[name][2])
            assert torch.allclose(destination[name][3], source[name][3])


def test_compute_kv_layout_empty_raises_value_error() -> None:
    """Ensure compute_kv_layout rejects empty KV cache input."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import compute_kv_layout

    with pytest.raises(ValueError, match="kv_caches is empty"):
        compute_kv_layout({})


def test_scatter_mla_respects_skip_first_n_tokens() -> None:
    """Ensure MLA scatter honors skip_first_n_tokens and preserves skipped blocks."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )

    source = _make_mla_kv_caches(
        num_layers=NUM_LAYERS,
        num_blocks=TEST_NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    destination = {
        name: torch.full_like(tensor, SENTINEL_VALUE) for name, tensor in source.items()
    }
    gathered = gather_chunks_to_cpu(
        source, [0, 1, 2, 3], blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK
    )
    scatter_cpu_chunks_to_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK,
        skip_first_n_tokens=8,
    )

    for name in destination:
        assert torch.all(destination[name][0] == SENTINEL_VALUE)
        assert torch.all(destination[name][1] == SENTINEL_VALUE)
        assert torch.allclose(destination[name][2], source[name][2])
        assert torch.allclose(destination[name][3], source[name][3])


def test_scatter_mla_skip_past_chunk_keeps_destination_unchanged() -> None:
    """Ensure MLA scatter is a no-op when skip_first_n_tokens exceeds chunk tokens."""
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        gather_chunks_to_cpu,
        scatter_cpu_chunks_to_kv,
    )

    source = _make_mla_kv_caches(
        num_layers=NUM_LAYERS,
        num_blocks=TEST_NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    destination = {
        name: torch.full_like(tensor, 123.0) for name, tensor in source.items()
    }
    gathered = gather_chunks_to_cpu(
        source, [0, 1, 2, 3], blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK
    )
    scatter_cpu_chunks_to_kv(
        destination,
        [0, 1, 2, 3],
        gathered,
        blocks_per_chunk=SKIP_TEST_BLOCKS_PER_CHUNK,
        skip_first_n_tokens=40,
    )

    for name in destination:
        assert torch.all(destination[name] == 123.0)


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_server_register_and_find_cpu_context_layout(
    stub_native_storage_ops: Any,
) -> None:
    """Ensure cpu context registration stores metadata and lookup finds its layout."""
    # First Party
    from lmcache.v1.multiprocess.server import MPCacheEngine

    with (
        patch("lmcache.v1.multiprocess.server.StorageManager"),
        patch("lmcache.v1.multiprocess.server.TokenHasher"),
        patch("lmcache.v1.multiprocess.server.SessionManager"),
        patch("lmcache.v1.multiprocess.server.get_event_bus"),
    ):
        engine = MPCacheEngine(
            storage_manager_config=MagicMock(), chunk_size=REGISTER_CHUNK_SIZE
        )
    engine.register_kv_cache_cpu_context(
        instance_id=1,
        model_name=MODEL_NAME,
        world_size=1,
        engine_type=MagicMock(),
        layout_hints={},
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim_size=HIDDEN_SIZE,
        dtype_str="float32",
        use_mla=False,
    )

    layout = engine._find_layout_desc(MODEL_NAME, 1)
    assert layout is not None
    # Shape is [K/V=2, num_layers, chunk_size, hidden_dim_size].
    expected_shape = torch.Size([2, NUM_LAYERS, REGISTER_CHUNK_SIZE, HIDDEN_SIZE])
    assert layout.shapes[0] == expected_shape


def test_server_store_and_retrieve_cpu_chunks(stub_native_storage_ops: Any) -> None:
    """Validate mocked server-side CPU chunk store and retrieve behavior."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
    from lmcache.v1.multiprocess.server import MPCacheEngine

    mock_storage = MagicMock()
    target_tensor = torch.zeros(2, NUM_LAYERS, STORE_CHUNK_SIZE, HIDDEN_SIZE)
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
        engine = MPCacheEngine(
            storage_manager_config=MagicMock(), chunk_size=STORE_CHUNK_SIZE
        )

    engine.register_kv_cache_cpu_context(
        instance_id=2,
        model_name=MODEL_NAME,
        world_size=1,
        engine_type=MagicMock(),
        layout_hints={},
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim_size=HIDDEN_SIZE,
        dtype_str="float32",
        use_mla=False,
    )
    payload = torch.ones(2, NUM_LAYERS, STORE_CHUNK_SIZE, HIDDEN_SIZE)
    key = IPCCacheEngineKey.from_token_ids(
        MODEL_NAME,
        1,
        0,
        [1] * STORE_CHUNK_SIZE,
        start=0,
        end=STORE_CHUNK_SIZE,
        request_id="req",
    )
    with patch(
        "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
        return_value=["obj"],
    ):
        store_ok = engine.store_cpu_chunks(key, 2, pickle.dumps([payload]))
        success, cpu_data = engine.retrieve_cpu_chunks(key, 2)
    assert store_ok is True
    assert torch.allclose(mock_memory_obj.tensor, payload)

    assert success is True
    recovered_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    assert len(recovered_chunks) == 1
    assert torch.allclose(recovered_chunks[0], payload)
