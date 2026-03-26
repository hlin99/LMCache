# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import (
    RemoteMetadata,
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


@pytest.mark.parametrize(
    "shape,expected",
    [
        # 3D layerwise KV_2TD: [2, T, D]
        (torch.Size([2, 256, 8192]), torch.Size([2, 256, 8192])),
        # 3D layerwise KV_T2D: [T, 2, D]
        (torch.Size([256, 2, 8192]), torch.Size([256, 2, 8192])),
        # 2D layerwise MLA: [T, D]
        (torch.Size([256, 8192]), torch.Size([256, 8192])),
        # 4D non-layerwise: unchanged
        (torch.Size([2, 32, 256, 8192]), torch.Size([2, 32, 256, 8192])),
        # 1D edge case
        (torch.Size([1024]), torch.Size([1024])),
    ],
)
def test_serialize_and_deserialize_non_4d(shape, expected):
    """Verify that shapes with fewer than 4 dimensions survive a
    serialize / deserialize round-trip for layerwise remote backends."""
    init_remote_metadata_info(1)

    origin = RemoteMetadata(
        42,
        [shape],
        [torch.float16],
        MemoryFormat.KV_T2D,
    )

    meta_bytes = origin.serialize()
    restored = RemoteMetadata.deserialize(meta_bytes)

    assert restored.length == origin.length
    assert restored.shapes == [expected]
    assert restored.dtypes == origin.dtypes
    assert restored.fmt == origin.fmt


def test_pad_shape_to_4d():
    """Unit-test the padding helper directly."""
    assert RemoteMetadata._pad_shape_to_4d(torch.Size([2, 3, 4, 5])) == [
        2,
        3,
        4,
        5,
    ]
    assert RemoteMetadata._pad_shape_to_4d(torch.Size([2, 256, 8192])) == [
        2,
        256,
        8192,
        0,
    ]
    assert RemoteMetadata._pad_shape_to_4d(torch.Size([256, 8192])) == [
        256,
        8192,
        0,
        0,
    ]
    assert RemoteMetadata._pad_shape_to_4d(torch.Size([1024])) == [1024, 0, 0, 0]


def test_strip_shape_padding():
    """Unit-test the stripping helper directly."""
    assert RemoteMetadata._strip_shape_padding([2, 3, 4, 5]) == torch.Size(
        [2, 3, 4, 5]
    )
    assert RemoteMetadata._strip_shape_padding([2, 256, 8192, 0]) == torch.Size(
        [2, 256, 8192]
    )
    assert RemoteMetadata._strip_shape_padding([256, 8192, 0, 0]) == torch.Size(
        [256, 8192]
    )
    assert RemoteMetadata._strip_shape_padding([1024, 0, 0, 0]) == torch.Size([1024])
    # All zeros: preserve at least one dim
    assert RemoteMetadata._strip_shape_padding([0, 0, 0, 0]) == torch.Size([0])
