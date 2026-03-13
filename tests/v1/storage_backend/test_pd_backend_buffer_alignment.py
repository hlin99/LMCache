# SPDX-License-Identifier: Apache-2.0
"""Tests for PD backend buffer-size alignment."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import PDBackend


def create_test_metadata(kv_shape=(4, 2, 256, 8, 128)) -> LMCacheMetadata:
    """Create test metadata with configurable KV shape."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


@patch("lmcache.v1.storage_backend.pd_backend.PagedCpuGpuMemoryAllocator")
def test_buffer_size_exact_alignment(mock_allocator_cls):
    """
    Test that buffer size is aligned before allocator initialization.
    """
    metadata = create_test_metadata(kv_shape=(4, 2, 256, 8, 128))
    expected_chunk_size = 4194304
    expected_aligned_size = 12582912

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        pd_buffer_size=13000000,  # NOT a multiple of 4194304
    )

    allocator = MagicMock()
    mock_allocator_cls.return_value = allocator

    backend = PDBackend.__new__(PDBackend)
    backend.corrected_device = "cpu"

    returned_allocator = PDBackend.initialize_allocator(backend, config, metadata)

    assert returned_allocator is allocator
    allocator.init_gpu_memory_allocator.assert_not_called()
    allocator.init_cpu_memory_allocator.assert_called_once_with(
        expected_aligned_size,
        [torch.Size(metadata.kv_shape)],
        [metadata.kv_dtype],
        MemoryFormat.KV_2LTD,
    )
    assert (
        allocator.init_cpu_memory_allocator.call_args.args[0] % expected_chunk_size == 0
    )
