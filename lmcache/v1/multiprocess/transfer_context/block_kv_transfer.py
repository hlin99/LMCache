# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic block KV transfer facade.

This module provides a unified transfer API with CUDA fast-path dispatch and a
Python tensor fallback that preserves gather/scatter semantics.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache


_MAX_CUDA_OBJECTS_PER_CALL = 4


def _as_device(device: torch.device | str | None, fallback: torch.device) -> torch.device:
    if device is None:
        return fallback
    return device if isinstance(device, torch.device) else torch.device(device)


def _is_ptr_tensor(candidate: object) -> bool:
    return isinstance(candidate, torch.Tensor) and candidate.dtype == torch.int64


def _is_tensor_sequence(values: object) -> bool:
    return isinstance(values, Sequence) and all(isinstance(v, torch.Tensor) for v in values)


def _is_int_sequence(values: object) -> bool:
    return isinstance(values, Sequence) and all(isinstance(v, int) for v in values)


def _to_python_block_ids(block_ids: list[int] | torch.Tensor) -> list[int]:
    if isinstance(block_ids, torch.Tensor):
        return [int(x) for x in block_ids.to(dtype=torch.int64).cpu().tolist()]
    return [int(x) for x in block_ids]


def _get_direction_values() -> tuple[object, object]:
    # First Party
    import lmcache.c_ops as lmc_ops

    return lmc_ops.TransferDirection.D2H, lmc_ops.TransferDirection.H2D


def _python_fallback_transfer(
    paged_buffer: "DiscoverableKVCache",
    lmcache_objects: list[torch.Tensor],
    block_ids: list[int] | torch.Tensor,
    direction: object,
    shape_desc: "lmc_ops.PageBufferShapeDesc",
    lmcache_chunk_size: int,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    skip_prefix_n_blocks: int,
) -> None:
    # First Party
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.utils import is_hnd, is_mla

    d2h, h2d = _get_direction_values()
    if direction not in (d2h, h2d):
        raise ValueError(f"Unsupported transfer direction: {direction!r}")

    if not isinstance(paged_buffer, Sequence) or not all(
        isinstance(layer, torch.Tensor) for layer in paged_buffer
    ):
        raise TypeError(
            "Python fallback requires tensor-form paged_buffer (normalized layer tensors)."
        )

    if not _is_tensor_sequence(lmcache_objects):
        raise TypeError(
            "Python fallback requires tensor-form lmcache_objects (list[torch.Tensor])."
        )

    layer_tensors = cast(list[torch.Tensor], list(paged_buffer))
    block_ids_list = _to_python_block_ids(block_ids)
    blocks_per_object = lmcache_chunk_size // int(shape_desc.bs)
    block_size = int(shape_desc.bs)
    use_mla = is_mla(gpu_kv_format)
    use_hnd = is_hnd(gpu_kv_format)

    for object_idx, obj in enumerate(lmcache_objects):
        obj_block_start = object_idx * blocks_per_object
        obj_block_ids = block_ids_list[obj_block_start : obj_block_start + blocks_per_object]
        if not obj_block_ids:
            continue

        skip_blocks_for_object = 0
        if direction == h2d and skip_prefix_n_blocks > 0:
            skip_blocks_for_object = max(skip_prefix_n_blocks - obj_block_start, 0)
            if skip_blocks_for_object >= len(obj_block_ids):
                continue

        effective_block_ids = obj_block_ids[skip_blocks_for_object:]
        skip_tokens = skip_blocks_for_object * block_size

        if direction == d2h:
            if use_mla:
                gathered_layers: list[torch.Tensor] = []
                idx = torch.tensor(effective_block_ids, dtype=torch.long)
                for layer in layer_tensors:
                    layer_blocks = layer[idx]
                    gathered_layers.append(
                        layer_blocks.reshape(
                            len(effective_block_ids) * block_size, layer_blocks.shape[-1]
                        )
                    )
                gathered = torch.stack(gathered_layers, dim=0)
                obj.copy_(gathered, non_blocking=True)
                continue

            k_layers: list[torch.Tensor] = []
            v_layers: list[torch.Tensor] = []
            idx = torch.tensor(effective_block_ids, dtype=torch.long)
            for layer in layer_tensors:
                if use_hnd:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _nb, nh, _bs, hs = k_t.shape
                    k_blocks = k_t[idx]
                    v_blocks = v_t[idx]
                    k_layers.append(
                        k_blocks.permute(0, 2, 1, 3).reshape(
                            len(effective_block_ids) * block_size, nh * hs
                        )
                    )
                    v_layers.append(
                        v_blocks.permute(0, 2, 1, 3).reshape(
                            len(effective_block_ids) * block_size, nh * hs
                        )
                    )
                else:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _nb, _bs, nh, hs = k_t.shape
                    k_blocks = k_t[idx]
                    v_blocks = v_t[idx]
                    k_layers.append(
                        k_blocks.reshape(len(effective_block_ids) * block_size, nh * hs)
                    )
                    v_layers.append(
                        v_blocks.reshape(len(effective_block_ids) * block_size, nh * hs)
                    )

            gathered = torch.stack(
                [torch.stack(k_layers, dim=0), torch.stack(v_layers, dim=0)], dim=0
            )
            obj.copy_(gathered, non_blocking=True)
            continue

        # H2D
        obj_device = obj.to(layer_tensors[0].device)
        if use_mla:
            eff_idx = torch.tensor(effective_block_ids, dtype=torch.long)
            for layer_idx, layer in enumerate(layer_tensors):
                src = obj_device[layer_idx, skip_tokens:]
                hidden_size = layer.shape[-1]
                layer[eff_idx] = src.reshape(len(effective_block_ids), block_size, hidden_size)
            continue

        for layer_idx, layer in enumerate(layer_tensors):
            k_src = obj_device[0, layer_idx, skip_tokens:]
            v_src = obj_device[1, layer_idx, skip_tokens:]
            eff_idx = torch.tensor(effective_block_ids, dtype=torch.long)
            if use_hnd:
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_blocks = k_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                ).permute(0, 2, 1, 3)
                v_blocks = v_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                ).permute(0, 2, 1, 3)
                k_t[eff_idx] = k_blocks
                v_t[eff_idx] = v_blocks
            else:
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _nb, _bs, nh, hs = k_t.shape
                k_t[eff_idx] = k_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                )
                v_t[eff_idx] = v_src.reshape(
                    len(effective_block_ids), block_size, nh, hs
                )


def _cuda_transfer(
    paged_buffer: "DiscoverableKVCache | torch.Tensor",
    lmcache_objects: list[torch.Tensor] | list[int],
    block_ids: list[int] | torch.Tensor,
    device: torch.device,
    direction: object,
    shape_desc: "lmc_ops.PageBufferShapeDesc",
    lmcache_chunk_size: int,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    skip_prefix_n_blocks: int,
) -> None:
    # First Party
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.utils import get_group_data_ptrs

    if _is_ptr_tensor(paged_buffer):
        paged_buffer_ptrs = cast(torch.Tensor, paged_buffer).to(
            device=device, dtype=torch.int64
        )
    else:
        layer_indices = list(range(int(shape_desc.nl)))
        ptrs = get_group_data_ptrs(
            cast("DiscoverableKVCache", paged_buffer), gpu_kv_format, layer_indices
        )
        paged_buffer_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=device)

    if _is_tensor_sequence(lmcache_objects):
        object_ptrs = [obj.data_ptr() for obj in cast(list[torch.Tensor], lmcache_objects)]
    elif _is_int_sequence(lmcache_objects):
        object_ptrs = [int(ptr) for ptr in cast(list[int], lmcache_objects)]
    else:
        raise TypeError(
            "lmcache_objects must be list[torch.Tensor] or list[int] for ptr-capable backends."
        )

    if isinstance(block_ids, torch.Tensor):
        block_ids_gpu = block_ids.to(device=device, dtype=torch.int64)
    else:
        block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)

    blocks_per_object = lmcache_chunk_size // int(shape_desc.bs)
    for i in range(0, len(object_ptrs), _MAX_CUDA_OBJECTS_PER_CALL):
        batch_ptrs = object_ptrs[i : i + _MAX_CUDA_OBJECTS_PER_CALL]
        block_start = i * blocks_per_object
        block_end = (i + len(batch_ptrs)) * blocks_per_object
        batch_block_ids = block_ids_gpu[block_start:block_end]
        if batch_block_ids.numel() == 0:
            continue
        batch_skip_prefix = max(skip_prefix_n_blocks - block_start, 0)
        lmc_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs,
            batch_ptrs,
            batch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            gpu_kv_format,
            batch_skip_prefix,
        )


def multi_layer_block_kv_transfer(
    paged_buffer: "DiscoverableKVCache | torch.Tensor",
    lmcache_objects: list[torch.Tensor] | list[int],
    block_ids: list[int] | torch.Tensor,
    device: torch.device | str,
    direction: "lmc_ops.TransferDirection",
    shape_desc: "lmc_ops.PageBufferShapeDesc",
    lmcache_chunk_size: int,
    gpu_kv_format: "lmc_ops.GPUKVFormat",
    skip_prefix_n_blocks: int = 0,
    *,
    backend: str | None = None,
) -> None:
    """Transfer paged KV blocks between page buffer and LMCache objects.

    The facade accepts tensor-form or ptr-form arguments and dispatches to the
    CUDA kernel when available, otherwise it falls back to Python tensor
    indexing/copy implementation.

    Args:
        paged_buffer: Normalized paged KV tensors or an int64 pointer tensor.
        lmcache_objects: LMCache chunk tensors or LMCache object pointers.
        block_ids: Flattened block ids as a Python list or tensor.
        device: Target backend device.
        direction: Transfer direction enum (D2H or H2D).
        shape_desc: Page buffer shape descriptor for transfer kernels.
        lmcache_chunk_size: Tokens per LMCache object.
        gpu_kv_format: KV format enum describing paged buffer layout.
        skip_prefix_n_blocks: Number of leading destination blocks to skip.
        backend: Optional backend override (``\"cuda\"`` or ``\"python\"``).

    Raises:
        ValueError: If arguments are invalid.
        TypeError: If tensor-only fallback receives pointer-form inputs.
    """
    default_device = (
        paged_buffer.device
        if isinstance(paged_buffer, torch.Tensor)
        else cast(Sequence[torch.Tensor], paged_buffer)[0].device
    )
    resolved_device = _as_device(device, default_device)

    if lmcache_chunk_size <= 0:
        raise ValueError("lmcache_chunk_size must be positive")
    if int(shape_desc.bs) <= 0 or lmcache_chunk_size % int(shape_desc.bs) != 0:
        raise ValueError(
            "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
        )
    if skip_prefix_n_blocks < 0:
        raise ValueError("skip_prefix_n_blocks must be >= 0")
    if backend not in (None, "cuda", "python"):
        raise ValueError("backend must be one of None, 'cuda', or 'python'")

    use_cuda_backend = backend == "cuda" or (
        backend is None and resolved_device.type == "cuda" and torch.cuda.is_available()
    )

    if backend == "python":
        use_cuda_backend = False

    if use_cuda_backend:
        _cuda_transfer(
            paged_buffer,
            lmcache_objects,
            block_ids,
            resolved_device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            gpu_kv_format,
            skip_prefix_n_blocks,
        )
        return

    if _is_ptr_tensor(paged_buffer):
        raise TypeError(
            "Python fallback requires tensor-form paged_buffer, got pointer tensor."
        )
    if _is_int_sequence(lmcache_objects):
        raise TypeError(
            "Python fallback requires tensor-form lmcache_objects, got pointer list."
        )

    _python_fallback_transfer(
        cast("DiscoverableKVCache", paged_buffer),
        cast(list[torch.Tensor], lmcache_objects),
        block_ids,
        direction,
        shape_desc,
        lmcache_chunk_size,
        gpu_kv_format,
        skip_prefix_n_blocks,
    )
