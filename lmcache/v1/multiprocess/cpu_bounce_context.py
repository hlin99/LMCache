# SPDX-License-Identifier: Apache-2.0
"""CPU bounce-buffer helpers and metadata for multiprocess mode."""

# Standard
from dataclasses import dataclass
from typing import Any, cast
import pickle

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc


def device_synchronize(device_type: str | None = None) -> None:
    """Synchronize device work for backends that require explicit barriers.

    Args:
        device_type: Active device type string (for example ``"cuda"``,
            ``"xpu"``, or ``"cpu"``). If None, uses ``lmcache.torch_device_type``.
    """
    dt = device_type or torch_device_type
    if dt == "cuda":
        torch.cuda.synchronize()
    elif dt == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: "Any | None" = None,
) -> tuple[int, int, int, str, Any]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``gpu_kv_format)``.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_hidden_dim_size,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    gpu_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, gpu_kv_format)
    num_layers = get_num_layers(normalized, gpu_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, gpu_kv_format)
    dtype_str = str(tensors[0].dtype).replace("torch.", "")
    return block_size, num_layers, hidden_dim_size, dtype_str, gpu_kv_format


def gather_chunks_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: "Any | None" = None,
    gpu_kv_format: "Any | None" = None,
) -> bytes:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Returns:
        Pickled list of CPU tensors. For non-MLA, each chunk shape is
        ``[2, num_layers, chunk_tokens, hidden_dim]`` where dimension ``0``
        stores ``(K, V)``. For MLA (multi-head latent attention), each chunk
        shape is ``[num_layers, chunk_tokens, hidden_dim]``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        _get_head_size_view,
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)

    block_size = get_block_size(normalized, gpu_kv_format)
    device = tensors[0].device
    num_chunks = len(block_ids) // blocks_per_chunk

    chunks: list[torch.Tensor] = []
    for chunk_idx in range(num_chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        slot_mapping = torch.tensor(
            [b * block_size + t for b in chunk_block_ids for t in range(block_size)],
            dtype=torch.long,
            device=device,
        )
        if use_mla:
            mla_layers: list[torch.Tensor] = []
            for layer in normalized:
                layer_flat = cast(
                    torch.Tensor,
                    _get_head_size_view(
                        layer, use_mla=True, gpu_kv_format=gpu_kv_format
                    ),
                )
                mla_layers.append(layer_flat.index_select(0, slot_mapping))
            chunks.append(torch.stack(mla_layers, dim=0).cpu())
        else:
            k_layers: list[torch.Tensor] = []
            v_layers: list[torch.Tensor] = []
            for layer in normalized:
                k_flat, v_flat = _get_head_size_view(
                    layer, use_mla=False, gpu_kv_format=gpu_kv_format
                )
                k_layers.append(k_flat.index_select(0, slot_mapping))
                v_layers.append(v_flat.index_select(0, slot_mapping))
            k_stacked = torch.stack(k_layers, dim=0)
            v_stacked = torch.stack(v_layers, dim=0)
            chunks.append(torch.stack([k_stacked, v_stacked], dim=0).cpu())
    return pickle.dumps(chunks)


def scatter_cpu_chunks_to_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    cpu_data: bytes,
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: "Any | None" = None,
    gpu_kv_format: "Any | None" = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.
        cpu_data: Serialized CPU chunk list (bytes returned by
            :func:`gather_chunks_to_cpu`, unpickled internally to
            ``list[torch.Tensor]``).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.
    """
    # First Party
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.utils import (
        _get_head_size_view,
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )

    chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    if not chunks:
        return

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)

    block_size = get_block_size(normalized, gpu_kv_format)
    device = tensors[0].device
    is_hnd = gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )

    for chunk_idx, chunk_cpu in enumerate(chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        if not chunk_block_ids:
            continue

        chunk_start_token = chunk_idx * blocks_per_chunk * block_size
        chunk_end_token = chunk_start_token + len(chunk_block_ids) * block_size
        effective_start = max(chunk_start_token, skip_first_n_tokens)
        if effective_start >= chunk_end_token:
            continue

        skip_blocks_in_chunk = (effective_start - chunk_start_token) // block_size
        effective_block_ids = chunk_block_ids[skip_blocks_in_chunk:]
        if not effective_block_ids:
            continue

        slot_mapping = torch.tensor(
            [
                block_id * block_size + token_idx
                for block_id in effective_block_ids
                for token_idx in range(block_size)
            ],
            dtype=torch.long,
            device=device,
        )
        skip_tokens = skip_blocks_in_chunk * block_size
        chunk_device = chunk_cpu.to(device)

        if use_mla:
            for layer_idx, layer in enumerate(normalized):
                mla_src = chunk_device[layer_idx, skip_tokens:]
                dst_flat = cast(
                    torch.Tensor,
                    _get_head_size_view(
                        layer, use_mla=True, gpu_kv_format=gpu_kv_format
                    ),
                )
                dst_flat.index_copy_(0, slot_mapping, mla_src)
        elif is_hnd:
            for layer_idx, layer in enumerate(normalized):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_src_3d = k_src.reshape(-1, nh, hs)
                v_src_3d = v_src.reshape(-1, nh, hs)
                offset = 0
                for block_id in effective_block_ids:
                    k_t[block_id] = k_src_3d[
                        offset : offset + block_size
                    ].permute(1, 0, 2)
                    v_t[block_id] = v_src_3d[
                        offset : offset + block_size
                    ].permute(1, 0, 2)
                    offset += block_size
        else:
            for layer_idx, layer in enumerate(normalized):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                k_flat, v_flat = _get_head_size_view(
                    layer, use_mla=False, gpu_kv_format=gpu_kv_format
                )
                k_flat.index_copy_(0, slot_mapping, k_src)
                v_flat.index_copy_(0, slot_mapping, v_src)


@dataclass
class CPUBounceContext:
    """CPU bounce-buffer layout metadata for non-CUDA workers."""

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool
