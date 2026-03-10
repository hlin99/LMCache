# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.protocol import (
    ClientMetaMessage,
    RemoteMetadata,
    ServerMetaMessage,
    _pad_shape_to_4d,
)
from lmcache.v1.memory_management import MemoryFormat


def test_pad_shape_to_4d_1d():
    """Test padding a 1D shape to 4D."""
    shape = torch.Size([100])
    padded = _pad_shape_to_4d(shape)
    assert len(padded) == 4
    assert padded == torch.Size([1, 1, 1, 100])


def test_pad_shape_to_4d_2d():
    """Test padding a 2D shape to 4D."""
    shape = torch.Size([100, 200])
    padded = _pad_shape_to_4d(shape)
    assert len(padded) == 4
    assert padded == torch.Size([1, 1, 100, 200])


def test_pad_shape_to_4d_3d():
    """Test padding a 3D shape to 4D (like from GPU connector)."""
    shape = torch.Size([100, 2, 4096])
    padded = _pad_shape_to_4d(shape)
    assert len(padded) == 4
    assert padded == torch.Size([1, 100, 2, 4096])


def test_pad_shape_to_4d_4d():
    """Test that a 4D shape is returned unchanged."""
    shape = torch.Size([2, 32, 100, 4096])
    padded = _pad_shape_to_4d(shape)
    assert len(padded) == 4
    assert padded == torch.Size([2, 32, 100, 4096])


def test_pad_shape_to_4d_5d_raises():
    """Test that shapes with more than 4 dimensions raise an error."""
    shape = torch.Size([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Shape dimension .* is greater than 4"):
        _pad_shape_to_4d(shape)


def test_remote_metadata_with_3d_shape():
    """Test that RemoteMetadata can serialize 3D shapes."""
    # Simulate a 3D shape from a layerwise GPU connector
    shapes = [torch.Size([100, 2, 4096])]
    dtypes = [torch.bfloat16]
    metadata = RemoteMetadata(
        length=100 * 2 * 4096 * 2,  # bfloat16 is 2 bytes
        shapes=shapes,
        dtypes=dtypes,
        fmt=MemoryFormat.KV_T2D,
    )

    # This should not raise an error now
    serialized = metadata.serialize()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_remote_metadata_with_4d_shape():
    """Test that RemoteMetadata still works with 4D shapes."""
    shapes = [torch.Size([2, 32, 100, 4096])]
    dtypes = [torch.bfloat16]
    metadata = RemoteMetadata(
        length=2 * 32 * 100 * 4096 * 2,
        shapes=shapes,
        dtypes=dtypes,
        fmt=MemoryFormat.KV_2LTD,
    )

    serialized = metadata.serialize()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_remote_metadata_multiple_shapes():
    """Test RemoteMetadata with multiple shapes of different dimensions."""
    shapes = [
        torch.Size([100, 2, 4096]),  # 3D
        torch.Size([2, 32, 100, 4096]),  # 4D
        torch.Size([200, 4096]),  # 2D
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    metadata = RemoteMetadata(
        length=1000000,
        shapes=shapes,
        dtypes=dtypes,
        fmt=MemoryFormat.KV_T2D,
    )

    serialized = metadata.serialize()
    assert isinstance(serialized, bytes)
