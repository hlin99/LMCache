# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    RemoteMetadata,
    ServerMetaMessage,
    ServerReturnCode,
    _pad_shape_to_4d,
    _strip_shape_trailing_zeros,
    get_remote_metadata_bytes,
    init_remote_metadata_info,
)


@pytest.mark.parametrize("num_groups", [1, 2, 3])
def test_serialize_and_deserialize(num_groups):
    all_shapes = [
        torch.Size([1, 2, 3, 4]),
        torch.Size([5, 6, 7, 8]),
        torch.Size([9, 10, 11, 12]),
    ]
    all_dtypes = [torch.uint8, torch.float16, torch.float32]

    shapes = all_shapes[:num_groups]
    dtypes = all_dtypes[:num_groups]

    # init remote metadata
    init_remote_metadata_info(num_groups)

    origin_metadata = RemoteMetadata(
        100,
        shapes,
        dtypes,
        MemoryFormat.KV_MLA_FMT,
    )

    meta_bytes = origin_metadata.serialize()
    assert len(meta_bytes) == get_remote_metadata_bytes()
    new_metadata = RemoteMetadata.deserialize(meta_bytes)
    assert origin_metadata.length == new_metadata.length
    assert origin_metadata.shapes == new_metadata.shapes
    assert origin_metadata.dtypes == new_metadata.dtypes
    assert origin_metadata.fmt == new_metadata.fmt


def test_pad_shape_to_4d():
    """Test that shapes shorter than 4D are padded with trailing zeros."""
    assert _pad_shape_to_4d(torch.Size([10, 576])) == (10, 576, 0, 0)
    assert _pad_shape_to_4d(torch.Size([2, 10, 576])) == (2, 10, 576, 0)
    assert _pad_shape_to_4d(torch.Size([1, 27, 10, 576])) == (1, 27, 10, 576)
    assert _pad_shape_to_4d(torch.Size([5])) == (5, 0, 0, 0)
    assert _pad_shape_to_4d(torch.Size([])) == (0, 0, 0, 0)


def test_strip_shape_trailing_zeros():
    """Test that trailing zeros are stripped to recover original shape."""
    assert _strip_shape_trailing_zeros(torch.Size([10, 576, 0, 0])) == torch.Size(
        [10, 576]
    )
    assert _strip_shape_trailing_zeros(torch.Size([2, 10, 576, 0])) == torch.Size(
        [2, 10, 576]
    )
    assert _strip_shape_trailing_zeros(torch.Size([1, 27, 10, 576])) == torch.Size(
        [1, 27, 10, 576]
    )
    assert _strip_shape_trailing_zeros(torch.Size([5, 0, 0, 0])) == torch.Size([5])
    assert _strip_shape_trailing_zeros(torch.Size([0, 0, 0, 0])) == torch.Size([])


def test_pad_strip_roundtrip():
    """Test that pad → strip is an identity for any practical shape."""
    for shape in [
        torch.Size([10, 576]),
        torch.Size([2, 256, 576]),
        torch.Size([1, 27, 256, 576]),
    ]:
        padded = _pad_shape_to_4d(shape)
        recovered = _strip_shape_trailing_zeros(torch.Size(padded))
        assert recovered == shape


def test_client_meta_message_2d_shape():
    """ClientMetaMessage round-trips a 2D MLA layerwise shape."""
    key = CacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=0,
        chunk_hash=12345,
        dtype=torch.bfloat16,
    )
    shape = torch.Size([256, 576])
    msg = ClientMetaMessage(
        ClientCommand.PUT,
        key,
        256 * 576 * 2,
        MemoryFormat.KV_MLA_FMT,
        torch.bfloat16,
        shape,
    )
    data = msg.serialize()
    assert len(data) == ClientMetaMessage.packlength()
    msg2 = ClientMetaMessage.deserialize(data)
    assert msg2.shape == shape
    assert msg2.length == msg.length
    assert msg2.fmt == msg.fmt


def test_server_meta_message_2d_shape():
    """ServerMetaMessage round-trips a 2D MLA layerwise shape."""
    shape = torch.Size([256, 576])
    msg = ServerMetaMessage(
        ServerReturnCode.SUCCESS,
        256 * 576 * 2,
        MemoryFormat.KV_MLA_FMT,
        torch.bfloat16,
        shape,
    )
    data = msg.serialize()
    assert len(data) == ServerMetaMessage.packlength()
    msg2 = ServerMetaMessage.deserialize(data)
    assert msg2.shape == shape
    assert msg2.length == msg.length
    assert msg2.code == msg.code


def test_remote_metadata_2d_shape():
    """RemoteMetadata round-trips a 2D MLA layerwise shape."""
    init_remote_metadata_info(1)
    shape = torch.Size([256, 576])
    meta = RemoteMetadata(
        256 * 576 * 2,
        [shape],
        [torch.bfloat16],
        MemoryFormat.KV_MLA_FMT,
    )
    data = meta.serialize()
    assert len(data) == get_remote_metadata_bytes()
    meta2 = RemoteMetadata.deserialize(data)
    assert meta2.shapes == [shape]
    assert meta2.length == meta.length
    assert meta2.fmt == meta.fmt


def test_remote_metadata_mixed_shapes():
    """RemoteMetadata round-trips when groups have different dimensionalities."""
    init_remote_metadata_info(2)
    shapes = [torch.Size([256, 576]), torch.Size([1, 27, 256, 576])]
    dtypes = [torch.bfloat16, torch.float16]
    meta = RemoteMetadata(
        999,
        shapes,
        dtypes,
        MemoryFormat.KV_T2D,
    )
    data = meta.serialize()
    meta2 = RemoteMetadata.deserialize(data)
    assert meta2.shapes == shapes
