# SPDX-License-Identifier: Apache-2.0
"""Tests for the tensor-first MUSA block-transfer adapter."""

import pytest
import torch

from lmcache.v1.platform.musa import device_ops
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)


def _shape_desc() -> PageBufferShapeDesc:
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = 1
    shape_desc.nb = 1
    shape_desc.bs = 1
    shape_desc.nh = 1
    shape_desc.hs = 1
    shape_desc.element_size = 2
    shape_desc.dtype = torch.bfloat16
    return shape_desc


def test_musa_rejects_pointer_tensor_paged_buffer() -> None:
    """Pointer tensors are rejected by the tensor-first MUSA API."""
    with pytest.raises(
        TypeError,
        match='paged_buffer must be a Tensor or list of Tensors, not a pointer tensor',
    ):
        device_ops.MusaDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([101], dtype=torch.int64),
            [torch.zeros((1, 1, 1), dtype=torch.bfloat16)],
            torch.tensor([0], dtype=torch.int64),
            torch.device('cpu'),
            TransferDirection.D2H,
            _shape_desc(),
            1,
            EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )


def test_musa_rejects_pointer_list_objects() -> None:
    """Pointer lists are rejected for LMCache objects."""
    with pytest.raises(
        TypeError,
        match=r'lmcache_objects must be list\[torch.Tensor\], not list\[int\]',
    ):
        device_ops.MusaDeviceOps().multi_layer_block_kv_transfer(
            [torch.zeros((1, 1, 1), dtype=torch.bfloat16)],
            [202],  # type: ignore[list-item]
            torch.tensor([0], dtype=torch.int64),
            torch.device('cpu'),
            TransferDirection.D2H,
            _shape_desc(),
            1,
            EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )
