# SPDX-License-Identifier: Apache-2.0
"""MUSA ops for tensor-first block transfer and ordered publication."""

from __future__ import annotations

from typing import ClassVar

import torch

from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.musa import native_kv_transfer
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)

_MUSA_MP_BLOCK_TRANSFER_FORMATS = {
    int(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS),
    int(EngineKVFormat.NL_X_NB_BS_HS),
}


def _validate_musa_mp_block_transfer_format(
    engine_kv_format: EngineKVFormat,
) -> None:
    """Reject block-transfer layouts unsupported by the torch baseline."""
    try:
        if torch_ops.is_cross_layer(engine_kv_format):
            return
        if torch_ops.is_kv_list(engine_kv_format):
            return
        if torch_ops.is_layer_list(engine_kv_format):
            return
    except ValueError as exc:
        raise ValueError(
            f"Unsupported MUSA block transfer format: {engine_kv_format!r}"
        ) from exc
    raise ValueError(f"Unsupported MUSA block transfer format: {engine_kv_format!r}")


def _synchronize_stream_pointer(stream_ptr: int) -> None:
    """Synchronize a raw MUSA stream pointer through TorchMUSA."""
    if not isinstance(stream_ptr, int):
        raise TypeError("MUSA stream pointer must be an int")
    try:
        import torch_musa

        external_stream = getattr(torch_musa, "ExternalStream", None)
        if not callable(external_stream):
            raise RuntimeError("TorchMUSA ExternalStream is unavailable")
        external_stream(stream_ptr).synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to synchronize MUSA stream pointer {stream_ptr}"
        ) from exc


class TorchMusaBlockTransfer:
    """Execute block transfer with the TorchMUSA-compatible torch backend."""

    def execute(
        self,
        paged_layers: torch.Tensor | list,
        object_tensors: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device,
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Transfer normalized tensor operands through the torch backend."""
        torch_ops.multi_layer_block_kv_transfer(
            paged_layers,
            object_tensors,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )


class NativeMusaBlockTransfer:
    """Try the optional native MUSA block-transfer implementation."""

    def execute_if_supported(
        self,
        paged_layers: torch.Tensor | list,
        object_tensors: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> bool:
        """Run native transfer when enabled and compatible."""
        return native_kv_transfer.try_native_multi_layer_block_kv_transfer(
            paged_layers=paged_layers,
            object_tensors=object_tensors,
            block_ids=block_ids,
            direction=direction,
            shape_desc=shape_desc,
            lmcache_chunk_size=lmcache_chunk_size,
            engine_kv_format=engine_kv_format,
            skip_prefix_n_blocks=skip_prefix_n_blocks,
        )


_TORCH_TRANSFER = TorchMusaBlockTransfer()
_NATIVE_TRANSFER = NativeMusaBlockTransfer()


def _musa_multi_layer_block_kv_transfer(
    paged_layers: torch.Tensor | list,
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Transfer tensor-first MUSA operands through native or torch ops."""
    _validate_musa_mp_block_transfer_format(engine_kv_format)
    if int(engine_kv_format) in _MUSA_MP_BLOCK_TRANSFER_FORMATS:
        if _NATIVE_TRANSFER.execute_if_supported(
            paged_layers,
            object_tensors,
            block_ids,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        ):
            return
    resolved_device = (
        device if isinstance(device, torch.device) else torch.device(device)
    )
    _TORCH_TRANSFER.execute(
        paged_layers,
        object_tensors,
        block_ids,
        resolved_device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


class MusaDeviceOps(DeviceOps):
    """MUSA block-transfer and stream-ordering operations."""

    device_type: ClassVar[str] = "musa"

    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion only after prior MUSA stream work finishes."""
        _synchronize_stream_pointer(stream_ptr)
        super().record_completion_on_stream(0, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Record an event only after prior MUSA stream work finishes."""
        _synchronize_stream_pointer(stream_ptr)
        super().record_event_on_stream(
            0,
            event_type_name,
            session_id,
            str_metadata,
            int_metadata,
        )

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer: torch.Tensor | list,
        lmcache_objects: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Transfer MUSA blocks through native code or the torch baseline.

        Accepts tensor-first inputs. Dispatches to native MUSA code for
        supported formats and falls back to the torch baseline otherwise.
        """
        _musa_multi_layer_block_kv_transfer(
            paged_buffer,
            lmcache_objects,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
