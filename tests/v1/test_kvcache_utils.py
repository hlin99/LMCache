# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.utils import (
    get_kvcache_hidden_dim,
    get_kvcache_page_buffer_size,
)


@pytest.mark.parametrize(
    "num_blocks,block_size,head_size",
    [
        (4, 16, 128),
        (8, 32, 64),
        (1, 1, 256),
    ],
)
def test_get_kvcache_page_buffer_size_mla(num_blocks, block_size, head_size):
    """Test page buffer size extraction for MLA format kvcaches."""
    num_layers = 3
    # MLA kvcaches: List[Tensor] with shape [num_blocks, block_size, head_size]
    kvcaches = [
        torch.randn(num_blocks, block_size, head_size) for _ in range(num_layers)
    ]
    result = get_kvcache_page_buffer_size(kvcaches, use_mla=True)
    assert result == num_blocks * block_size


@pytest.mark.parametrize(
    "num_blocks,block_size,num_heads,head_size",
    [
        (4, 16, 8, 128),
        (8, 32, 4, 64),
        (1, 1, 1, 256),
    ],
)
def test_get_kvcache_page_buffer_size_non_mla(
    num_blocks, block_size, num_heads, head_size
):
    """Test page buffer size extraction for non-MLA format kvcaches."""
    num_layers = 3
    # Non-MLA kvcaches: List[Tuple[Tensor, Tensor]]
    # each tensor has shape [num_blocks, block_size, num_heads, head_size]
    kvcaches = [
        (
            torch.randn(num_blocks, block_size, num_heads, head_size),
            torch.randn(num_blocks, block_size, num_heads, head_size),
        )
        for _ in range(num_layers)
    ]
    result = get_kvcache_page_buffer_size(kvcaches, use_mla=False)
    assert result == num_blocks * block_size


@pytest.mark.parametrize(
    "num_blocks,block_size,head_size",
    [
        (4, 16, 128),
        (8, 32, 64),
        (1, 1, 256),
    ],
)
def test_get_kvcache_hidden_dim_mla(num_blocks, block_size, head_size):
    """Test hidden dimension extraction for MLA format kvcaches."""
    num_layers = 3
    kvcaches = [
        torch.randn(num_blocks, block_size, head_size) for _ in range(num_layers)
    ]
    result = get_kvcache_hidden_dim(kvcaches, use_mla=True)
    assert result == head_size


@pytest.mark.parametrize(
    "num_blocks,block_size,num_heads,head_size",
    [
        (4, 16, 8, 128),
        (8, 32, 4, 64),
        (1, 1, 1, 256),
    ],
)
def test_get_kvcache_hidden_dim_non_mla(num_blocks, block_size, num_heads, head_size):
    """Test hidden dimension extraction for non-MLA format kvcaches."""
    num_layers = 3
    kvcaches = [
        (
            torch.randn(num_blocks, block_size, num_heads, head_size),
            torch.randn(num_blocks, block_size, num_heads, head_size),
        )
        for _ in range(num_layers)
    ]
    result = get_kvcache_hidden_dim(kvcaches, use_mla=False)
    assert result == num_heads * head_size
