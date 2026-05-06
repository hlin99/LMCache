# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the XPU CPU bounce-buffer MVP path.

These tests run on CPU (simulating XPU) and do NOT require a real XPU device,
CUDA IPC, or a running LMCache server.  They cover:
1. wrap_kv_caches() returns empty list on XPU.
2. _compute_xpu_layout() extracts correct layout from tensor shapes.
3. _xpu_gather_chunks_to_cpu() gathers KV blocks into CPU bytes.
4. _xpu_scatter_cpu_chunks_to_kv() scatters CPU bytes back to paged KV.
5. server MPCacheEngine.register_kv_cache_layout() stores layout metadata.
6. server MPCacheEngine.store_cpu_chunks() / retrieve_cpu_chunks() round-trip.
7. _find_layout_desc() finds XPU layout when no GPU context exists.
"""

# Standard
import sys
from typing import Any
from unittest.mock import MagicMock, patch
import pickle

# Third Party
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures / autouse: mock compiled-extension imports so server.py can load
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_native_extensions():
    """Inject stubs for compiled C++ extensions so server.py can be imported."""
    mocks = {
        "lmcache.native_storage_ops": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_caches_nl_x_two_nb_bs_nh_hs(
    num_layers: int = 4,
    num_blocks: int = 8,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Create fake vLLM paged KV caches in NL_X_TWO_NB_BS_NH_HS format.

    Each layer tensor has shape [2, num_blocks, block_size, num_heads, head_size].
    """
    kv_caches = {}
    for i in range(num_layers):
        # Shape: [2, NB, BS, NH, HS] — vLLM non-MLA flash attention format
        t = torch.randn(2, num_blocks, block_size, num_heads, head_size, dtype=dtype)
        kv_caches[f"layer_{i}"] = t
    return kv_caches


# ---------------------------------------------------------------------------
# Tests: wrap_kv_caches
# ---------------------------------------------------------------------------


class TestWrapKvCaches:
    def test_xpu_returns_empty_list(self):
        """wrap_kv_caches() must return [] on XPU — no CudaIPCWrapper calls."""
        kv_caches = _make_kv_caches_nl_x_two_nb_bs_nh_hs(num_layers=2)

        with patch(
            "lmcache.integration.vllm.vllm_multi_process_adapter.torch_device_type",
            "xpu",
        ):
            # Local
            from lmcache.integration.vllm.vllm_multi_process_adapter import (
                wrap_kv_caches,
            )

            result = wrap_kv_caches(kv_caches)

        assert result == [], "Expected empty list for XPU; got non-empty"

    def test_cuda_returns_cuda_ipc_wrappers(self):
        """wrap_kv_caches() on CUDA should call CudaIPCWrapper (mocked here)."""
        kv_caches = _make_kv_caches_nl_x_two_nb_bs_nh_hs(num_layers=2)

        # Patch torch_device_type to 'cuda' and mock CudaIPCWrapper
        mock_wrapper = MagicMock()
        with (
            patch(
                "lmcache.integration.vllm.vllm_multi_process_adapter.torch_device_type",
                "cuda",
            ),
            patch(
                "lmcache.integration.vllm.vllm_multi_process_adapter.CudaIPCWrapper",
                mock_wrapper,
            ),
        ):
            # Local
            from lmcache.integration.vllm import vllm_multi_process_adapter as adapter

            # Force reload to pick up patches (wrap_kv_caches is a module-level fn)
            result = adapter.wrap_kv_caches(kv_caches)

        assert len(result) == len(kv_caches), "Should wrap each layer tensor"
        assert mock_wrapper.call_count == len(kv_caches)


# ---------------------------------------------------------------------------
# Tests: _compute_xpu_layout
# ---------------------------------------------------------------------------


class TestComputeXpuLayout:
    def test_basic_layout(self):
        """_compute_xpu_layout() should return correct (block_size, num_layers,
        hidden_dim_size, dtype_str) from NL_X_TWO_NB_BS_NH_HS tensors."""
        num_layers = 4
        num_blocks = 8
        block_size = 16
        num_heads = 8
        head_size = 64
        dtype = torch.bfloat16

        kv_caches = _make_kv_caches_nl_x_two_nb_bs_nh_hs(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            dtype=dtype,
        )

        # Local
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _compute_xpu_layout,
        )

        bs, nl, hidden, dtype_str = _compute_xpu_layout(kv_caches)

        assert bs == block_size, f"Expected block_size={block_size}, got {bs}"
        assert nl == num_layers, f"Expected num_layers={num_layers}, got {nl}"
        assert hidden == num_heads * head_size, (
            f"Expected hidden_dim={num_heads * head_size}, got {hidden}"
        )
        assert dtype_str == "bfloat16", f"Expected 'bfloat16', got {dtype_str}"

    def test_empty_raises(self):
        """_compute_xpu_layout() should raise ValueError for empty kv_caches."""
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _compute_xpu_layout,
        )

        with pytest.raises(ValueError, match="empty"):
            _compute_xpu_layout({})


# ---------------------------------------------------------------------------
# Tests: gather and scatter round-trip
# ---------------------------------------------------------------------------


class TestXpuGatherScatterRoundTrip:
    """Tests for _xpu_gather_chunks_to_cpu and _xpu_scatter_cpu_chunks_to_kv."""

    def _create_test_setup(
        self,
        num_layers: int = 2,
        num_blocks: int = 4,
        block_size: int = 16,
        num_heads: int = 4,
        head_size: int = 32,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Create a test KV cache and return (kv_caches, blocks_per_chunk)."""
        kv_caches = _make_kv_caches_nl_x_two_nb_bs_nh_hs(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            dtype=torch.float32,
        )
        blocks_per_chunk = 2  # chunk_size = 2 * block_size
        return kv_caches, blocks_per_chunk

    def test_gather_returns_bytes(self):
        """_xpu_gather_chunks_to_cpu should return non-empty bytes."""
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _xpu_gather_chunks_to_cpu,
        )

        kv_caches, blocks_per_chunk = self._create_test_setup()
        block_ids = [0, 1, 2, 3]  # 2 chunks of 2 blocks each

        cpu_data = _xpu_gather_chunks_to_cpu(kv_caches, block_ids, blocks_per_chunk)

        assert isinstance(cpu_data, bytes), "Should return bytes"
        assert len(cpu_data) > 0, "Should return non-empty bytes"

    def test_gather_shape(self):
        """Gathered chunks should have shape [2, num_layers, chunk_size, hidden_dim]."""
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _xpu_gather_chunks_to_cpu,
        )

        num_layers = 3
        block_size = 8
        num_heads = 4
        head_size = 16
        blocks_per_chunk = 2
        chunk_size = blocks_per_chunk * block_size
        hidden_dim = num_heads * head_size

        kv_caches = _make_kv_caches_nl_x_two_nb_bs_nh_hs(
            num_layers=num_layers,
            num_blocks=6,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            dtype=torch.float32,
        )
        block_ids = [0, 1]  # 1 chunk of 2 blocks

        cpu_data = _xpu_gather_chunks_to_cpu(kv_caches, block_ids, blocks_per_chunk)
        chunks: list[torch.Tensor] = pickle.loads(cpu_data)

        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
        assert chunks[0].shape == torch.Size([2, num_layers, chunk_size, hidden_dim]), (
            f"Expected shape [2, {num_layers}, {chunk_size}, {hidden_dim}], "
            f"got {chunks[0].shape}"
        )

    def test_gather_scatter_round_trip(self):
        """Gather + scatter should reproduce the original KV values in target blocks."""
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _xpu_gather_chunks_to_cpu,
            _xpu_scatter_cpu_chunks_to_kv,
        )

        num_layers = 2
        block_size = 4
        num_heads = 2
        head_size = 8
        blocks_per_chunk = 2

        src_kv = _make_kv_caches_nl_x_two_nb_bs_nh_hs(
            num_layers=num_layers,
            num_blocks=8,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            dtype=torch.float32,
        )
        # Destination (target) starts as zeros
        dst_kv = {name: torch.zeros_like(t) for name, t in src_kv.items()}

        # Gather from blocks 0,1 of src
        block_ids = [0, 1]
        cpu_data = _xpu_gather_chunks_to_cpu(src_kv, block_ids, blocks_per_chunk)

        # Scatter to blocks 4,5 of dst
        target_block_ids = [4, 5]
        _xpu_scatter_cpu_chunks_to_kv(
            dst_kv, target_block_ids, cpu_data, blocks_per_chunk
        )

        # Verify: dst[4,5] should match src[0,1]
        for name in src_kv:
            src_t = src_kv[name]  # [2, NB, BS, NH, HS]
            dst_t = dst_kv[name]

            for k_or_v in range(2):
                src_block0 = src_t[k_or_v, 0]  # [BS, NH, HS]
                src_block1 = src_t[k_or_v, 1]

                dst_block4 = dst_t[k_or_v, 4]
                dst_block5 = dst_t[k_or_v, 5]

                assert torch.allclose(src_block0, dst_block4), (
                    f"Block 0→4 mismatch for k_or_v={k_or_v} in {name}"
                )
                assert torch.allclose(src_block1, dst_block5), (
                    f"Block 1→5 mismatch for k_or_v={k_or_v} in {name}"
                )

    def test_scatter_with_skip_first_n_tokens(self):
        """Scatter with skip_first_n_tokens should not overwrite skipped blocks."""
        from lmcache.integration.vllm.vllm_multi_process_adapter import (
            _xpu_gather_chunks_to_cpu,
            _xpu_scatter_cpu_chunks_to_kv,
        )

        block_size = 4
        blocks_per_chunk = 4
        skip_blocks = 2  # skip first 2 blocks (= skip_first_n_tokens = 2 * block_size)

        src_kv = _make_kv_caches_nl_x_two_nb_bs_nh_hs(
            num_layers=2,
            num_blocks=8,
            block_size=block_size,
            num_heads=2,
            head_size=8,
            dtype=torch.float32,
        )

        # Fill dst with distinct sentinels
        sentinel_value = 999.0
        dst_kv = {
            name: torch.full_like(t, sentinel_value) for name, t in src_kv.items()
        }

        block_ids = [0, 1, 2, 3]  # 1 chunk of 4 blocks
        cpu_data = _xpu_gather_chunks_to_cpu(src_kv, block_ids, blocks_per_chunk)

        skip_first_n_tokens = skip_blocks * block_size
        _xpu_scatter_cpu_chunks_to_kv(
            dst_kv,
            block_ids,
            cpu_data,
            blocks_per_chunk,
            skip_first_n_tokens=skip_first_n_tokens,
        )

        # Skipped blocks (0, 1) must remain as sentinel
        for name in dst_kv:
            dst_t = dst_kv[name]
            for skip_block in range(skip_blocks):
                assert torch.all(dst_t[:, skip_block] == sentinel_value), (
                    f"Block {skip_block} should NOT have been written "
                    f"(skip_first_n_tokens={skip_first_n_tokens})"
                )

        # Non-skipped blocks (2, 3) must be populated from src
        for name in src_kv:
            src_t = src_kv[name]
            dst_t = dst_kv[name]
            for written_block in range(skip_blocks, blocks_per_chunk):
                assert torch.allclose(
                    src_t[:, written_block], dst_t[:, written_block]
                ), f"Block {written_block} should have been written"


# ---------------------------------------------------------------------------
# Tests: server-side register_kv_cache_layout
# ---------------------------------------------------------------------------


class TestServerRegisterKvCacheLayout:
    """Tests for MPCacheEngine.register_kv_cache_layout (layout-only registration)."""

    def _make_engine(self) -> Any:
        """Return an MPCacheEngine with a mocked StorageManager."""
        # We need to mock the storage manager to avoid requiring native extensions
        mock_storage_manager_config = MagicMock()

        with patch(
            "lmcache.v1.multiprocess.server.StorageManager",
        ) as MockSM:
            MockSM.return_value = MagicMock()

            with patch(
                "lmcache.v1.multiprocess.server.TokenHasher",
            ) as MockTH:
                MockTH.return_value = MagicMock()

                with patch(
                    "lmcache.v1.multiprocess.server.SessionManager",
                ) as MockSession:
                    MockSession.return_value = MagicMock()

                    with patch(
                        "lmcache.v1.multiprocess.server.get_event_bus",
                    ) as MockEB:
                        MockEB.return_value = MagicMock()

                        # Local (import late to allow patches to take effect)
                        from lmcache.v1.multiprocess.server import MPCacheEngine

                        engine = MPCacheEngine(
                            storage_manager_config=mock_storage_manager_config,
                            chunk_size=256,
                        )
        return engine

    def test_register_layout_stores_context(self):
        """register_kv_cache_layout() should populate xpu_layout_contexts."""
        engine = self._make_engine()

        engine.register_kv_cache_layout(
            instance_id=1234,
            model_name="test-model",
            world_size=1,
            engine_type=MagicMock(),  # EngineType.VLLM
            layout_hints={},
            block_size=16,
            num_layers=4,
            hidden_dim_size=512,
            dtype_str="float16",
        )

        assert 1234 in engine.xpu_layout_contexts
        assert 1234 in engine.xpu_context_meta
        assert engine.xpu_context_meta[1234] == ("test-model", 1)

        ctx = engine.xpu_layout_contexts[1234]
        assert ctx.block_size == 16
        layout = ctx.layout_desc
        # Shape should be [2, 4, 256, 512] (chunk_size=256 by default)
        assert layout.shapes[0] == torch.Size([2, 4, 256, 512])
        assert layout.dtypes[0] == torch.float16

    def test_register_layout_invalid_dtype_raises(self):
        """register_kv_cache_layout() should raise ValueError for unknown dtype."""
        engine = self._make_engine()

        with pytest.raises(ValueError, match="dtype_str"):
            engine.register_kv_cache_layout(
                instance_id=999,
                model_name="m",
                world_size=1,
                engine_type=MagicMock(),
                layout_hints={},
                block_size=16,
                num_layers=4,
                hidden_dim_size=128,
                dtype_str="not_a_dtype",
            )

    def test_find_layout_desc_finds_xpu(self):
        """_find_layout_desc() should return XPU layout when no GPU context exists."""
        engine = self._make_engine()

        engine.register_kv_cache_layout(
            instance_id=42,
            model_name="my-model",
            world_size=2,
            engine_type=MagicMock(),
            layout_hints={},
            block_size=16,
            num_layers=8,
            hidden_dim_size=256,
            dtype_str="bfloat16",
        )

        result = engine._find_layout_desc("my-model", 2)
        assert result is not None
        assert result.shapes[0] == torch.Size([2, 8, 256, 256])
        assert result.dtypes[0] == torch.bfloat16

    def test_find_layout_desc_returns_none_for_unknown(self):
        """_find_layout_desc() should return None when no context matches."""
        engine = self._make_engine()

        result = engine._find_layout_desc("unknown-model", 1)
        assert result is None

    def test_unregister_removes_xpu_context(self):
        """unregister_kv_cache() should remove XPU layout context."""
        engine = self._make_engine()

        engine.register_kv_cache_layout(
            instance_id=77,
            model_name="m",
            world_size=1,
            engine_type=MagicMock(),
            layout_hints={},
            block_size=16,
            num_layers=2,
            hidden_dim_size=64,
            dtype_str="float32",
        )

        assert 77 in engine.xpu_layout_contexts
        engine.unregister_kv_cache(77)
        assert 77 not in engine.xpu_layout_contexts


# ---------------------------------------------------------------------------
# Tests: server-side store_cpu_chunks / retrieve_cpu_chunks
# ---------------------------------------------------------------------------


class TestServerCpuChunkStoreRetrieve:
    """Mock-based round-trip tests for store_cpu_chunks and retrieve_cpu_chunks."""

    def _build_engine_with_layout(
        self,
        chunk_size: int = 32,
        num_layers: int = 2,
        block_size: int = 8,
        num_heads: int = 2,
        head_size: int = 16,
        instance_id: int = 100,
    ) -> Any:
        """Create an MPCacheEngine with mocked storage and a registered XPU layout."""
        from unittest.mock import patch, MagicMock
        from lmcache.v1.distributed.api import MemoryLayoutDesc

        hidden_dim = num_heads * head_size
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, num_layers, chunk_size, hidden_dim])],
            dtypes=[torch.float32],
        )
        chunk_tensor = torch.zeros(2, num_layers, chunk_size, hidden_dim)

        # Fake MemoryObj with a real tensor
        mock_memory_obj = MagicMock()
        mock_memory_obj.tensor = chunk_tensor.clone()
        mock_memory_obj.get_size.return_value = chunk_tensor.numel() * 4

        # Fake StorageManager
        mock_sm = MagicMock()
        mock_sm.reserve_write.return_value = {"fake_key": mock_memory_obj}

        # For retrieve, simulate the context manager
        from contextlib import contextmanager

        @contextmanager
        def _read_prefetched_ctx(keys):
            yield [mock_memory_obj]

        mock_sm.read_prefetched_results.side_effect = _read_prefetched_ctx
        mock_sm.finish_write = MagicMock()
        mock_sm.finish_read_prefetched = MagicMock()

        with (
            patch(
                "lmcache.v1.multiprocess.server.StorageManager", return_value=mock_sm
            ),
            patch("lmcache.v1.multiprocess.server.TokenHasher"),
            patch("lmcache.v1.multiprocess.server.SessionManager") as MockSession,
            patch("lmcache.v1.multiprocess.server.get_event_bus") as MockEB,
        ):
            mock_session_inst = MagicMock()
            mock_session_inst.get_hashes.return_value = [b"hash1", b"hash2"]
            MockSession.return_value.get_or_create.return_value = mock_session_inst
            MockEB.return_value = MagicMock()

            from lmcache.v1.multiprocess.server import MPCacheEngine

            engine = MPCacheEngine(
                storage_manager_config=MagicMock(),
                chunk_size=chunk_size,
            )

        # Manually set up XPU layout context
        from lmcache.v1.multiprocess.server import XPULayoutContext

        engine.xpu_layout_contexts[instance_id] = XPULayoutContext(
            layout_desc=layout_desc,
            block_size=block_size,
        )
        engine.xpu_context_meta[instance_id] = ("test-model", 1)

        # Give the engine's storage manager the mock
        engine.storage_manager = mock_sm

        # Give engine a real session manager that returns valid hashes
        engine.session_manager = MockSession.return_value

        return engine, mock_memory_obj, chunk_tensor

    def test_store_cpu_chunks_success(self):
        """store_cpu_chunks() should copy data into MemoryObj and call finish_write."""
        chunk_size = 32
        num_layers = 2
        hidden_dim = 32
        instance_id = 100

        engine, mock_mo, _ = self._build_engine_with_layout(
            chunk_size=chunk_size,
            num_layers=num_layers,
            block_size=8,
            num_heads=2,
            head_size=16,
            instance_id=instance_id,
        )

        # Create realistic CPU chunk data
        chunk = torch.randn(2, num_layers, chunk_size, hidden_dim)
        cpu_data = pickle.dumps([chunk])

        mock_key = MagicMock()
        mock_key.worker_id = 42
        mock_key.request_id = "req-1"
        mock_key.token_ids = list(range(64))
        mock_key.start = 0
        mock_key.end = 32

        with patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["fake_key"],
        ):
            result = engine.store_cpu_chunks(
                key=mock_key,
                instance_id=instance_id,
                cpu_data=cpu_data,
            )

        assert result is True
        engine.storage_manager.finish_write.assert_called_once()

    def test_retrieve_cpu_chunks_success(self):
        """retrieve_cpu_chunks() should return (True, non_empty_bytes)."""
        chunk_size = 32
        num_layers = 2
        num_heads = 2
        head_size = 16
        hidden_dim = num_heads * head_size
        instance_id = 100

        engine, mock_mo, ref_tensor = self._build_engine_with_layout(
            chunk_size=chunk_size,
            num_layers=num_layers,
            block_size=8,
            num_heads=num_heads,
            head_size=head_size,
            instance_id=instance_id,
        )

        # Give the MemoryObj tensor a known value
        mock_mo.tensor = torch.ones(2, num_layers, chunk_size, hidden_dim)

        mock_key = MagicMock()
        mock_key.worker_id = 42
        mock_key.request_id = "req-2"
        mock_key.token_ids = list(range(64))
        mock_key.start = 0
        mock_key.end = 32

        with patch(
            "lmcache.v1.multiprocess.server.ipc_key_to_object_keys",
            return_value=["fake_key"],
        ):
            success, cpu_data = engine.retrieve_cpu_chunks(
                key=mock_key,
                instance_id=instance_id,
            )

        assert success is True
        assert isinstance(cpu_data, bytes)
        assert len(cpu_data) > 0

        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
        assert len(chunks) == 1
        assert torch.allclose(chunks[0], torch.ones_like(chunks[0]))
