# SPDX-License-Identifier: Apache-2.0
"""
Test for buffer size alignment in PD backend.

This test verifies that the PDBackend correctly aligns the buffer size
to be a multiple of align_bytes (chunk size).
"""

# Standard
from unittest.mock import MagicMock, patch
import socket

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import PDBackend

logger = init_logger(__name__)


def get_free_port() -> int:
    """Get a free port on localhost by briefly binding to port 0.

    The OS assigns a free ephemeral port; we immediately release it.
    There is a small TOCTOU window, but this is the standard approach
    for test helpers and is safe enough for CI / parallel Docker runs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


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


def _get_allocator(backend, config):
    """Get the appropriate memory allocator based on the buffer device config."""
    if config.pd_buffer_device == "cpu":
        return backend.memory_allocator.cpu_allocator
    return backend.memory_allocator.gpu_allocator


def _cleanup_backend(backend):
    """Clean up PDBackend resources safely.

    Calls the backend's own close() method (which now closes side
    channels before joining threads), then closes the paged allocators
    so that free-block MemoryObj instances are cleaned up immediately.
    """
    try:
        backend.close()
    except Exception as e:
        logger.warning("Error during backend close: %s", e)

    try:
        mem = backend.memory_allocator
        if hasattr(mem, "cpu_allocator"):
            mem.cpu_allocator.close()
        if hasattr(mem, "gpu_allocator"):
            mem.gpu_allocator.close()
    except Exception as e:
        logger.warning("Error closing allocators: %s", e)


@patch.object(PDBackend, "_init_receiver")
@patch(
    "lmcache.v1.storage_backend.pd_backend.CreateTransferChannel",
    return_value=MagicMock(),
)
@patch(
    "lmcache.v1.storage_backend.pd_backend.get_zmq_context", return_value=MagicMock()
)
def test_buffer_size_exact_alignment(_mock_ctx, _mock_channel, _mock_receiver):
    """
    Test that buffer size is correctly aligned to the exact expected size.
    """

    # Create a metadata with KV shape that results in a specific chunk size
    metadata = create_test_metadata(kv_shape=(4, 2, 256, 8, 128))

    # Calculate expected chunk size:
    # 4 * 2 * 256 * 8 * 128 * 2 (bfloat16) = 4194304 bytes
    expected_chunk_size = 4194304

    # Create a config with a buffer size that is NOT a multiple of chunk size
    # Original size: 13000000 bytes
    # Chunks count: 13000000 // 4194304 = 3 chunks
    # Expected aligned size: 3 * 4194304 = 12582912 bytes
    expected_aligned_size = 12582912

    # Dynamically allocate free ports to avoid conflicts when tests run
    # concurrently (e.g., multiple Docker containers on the same host).
    init_port = get_free_port()
    alloc_port = get_free_port()

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        pd_buffer_size=13000000,  # NOT a multiple of 4194304
        pd_buffer_device="cpu",
        pd_role="receiver",
        pd_peer_host="localhost",
        pd_peer_init_port=[init_port],
        pd_peer_alloc_port=[alloc_port],
        transfer_channel="mock_memory",
    )

    backend = PDBackend(config, metadata)
    try:
        # Get the actual buffer size used
        allocator = _get_allocator(backend, config)
        actual_buffer_size = allocator.buffer_size
        align_bytes = allocator.align_bytes

        assert align_bytes == expected_chunk_size, (
            f"Expected chunk size {expected_chunk_size}, got {align_bytes}"
        )

        assert actual_buffer_size == expected_aligned_size, (
            f"Expected exact aligned size {expected_aligned_size}, "
            f"but got {actual_buffer_size}"
        )
    finally:
        _cleanup_backend(backend)
