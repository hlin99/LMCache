# SPDX-License-Identifier: Apache-2.0
"""CPU bounce-buffer helpers and metadata for multiprocess mode."""

# Standard
from dataclasses import dataclass
from typing import Any, cast

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: Any | None = None,
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


def _gather_chunks_impl(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
) -> list[torch.Tensor]:
    """Core implementation: gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Returns:
        List of CPU tensors. For non-MLA, each chunk shape is
        ``[2, num_layers, chunk_tokens, hidden_dim]`` where dimension ``0``
        stores ``(K, V)``. For MLA (multi-head latent attention), each chunk
        shape is ``[num_layers, chunk_tokens, hidden_dim]``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)
    is_hnd = gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )

    block_size = get_block_size(normalized, gpu_kv_format)
    num_chunks = len(block_ids) // blocks_per_chunk

    # After normalization the structure is always a list of per-layer
    # tensors.  Cast once so all downstream indexing is typed correctly.
    layer_tensors = cast(list[torch.Tensor], normalized)

    chunks: list[torch.Tensor] = []
    for chunk_idx in range(num_chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        if use_mla:
            mla_layers: list[torch.Tensor] = []
            for layer in layer_tensors:
                layer_blocks = layer[torch.tensor(chunk_block_ids, dtype=torch.long)]
                mla_layers.append(
                    layer_blocks.reshape(
                        len(chunk_block_ids) * block_size, layer_blocks.shape[-1]
                    )
                )
            chunks.append(torch.stack(mla_layers, dim=0).cpu())
        else:
            k_layers: list[torch.Tensor] = []
            v_layers: list[torch.Tensor] = []
            for layer in layer_tensors:
                if is_hnd:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _num_blocks, num_heads, _block_size, head_size = k_t.shape
                    k_blocks = k_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    v_blocks = v_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    # HND blocks are [NB, NH, BS, HS]; convert to token-major
                    # [NB, BS, NH, HS] before flattening to [tokens, NH*HS].
                    k_layers.append(
                        k_blocks.permute(0, 2, 1, 3).reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                    v_layers.append(
                        v_blocks.permute(0, 2, 1, 3).reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                else:
                    if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                        k_t = layer[0]
                        v_t = layer[1]
                    else:
                        k_t = layer[:, 0]
                        v_t = layer[:, 1]
                    _num_blocks, _block_size, num_heads, head_size = k_t.shape
                    k_blocks = k_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    v_blocks = v_t[torch.tensor(chunk_block_ids, dtype=torch.long)]
                    k_layers.append(
                        k_blocks.reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
                    v_layers.append(
                        v_blocks.reshape(
                            len(chunk_block_ids) * block_size, num_heads * head_size
                        )
                    )
            k_stacked = torch.stack(k_layers, dim=0)
            v_stacked = torch.stack(v_layers, dim=0)
            chunks.append(torch.stack([k_stacked, v_stacked], dim=0).cpu())
    return chunks


def gather_chunks_to_cpu_tensors(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    Used by the SHM store path where data is written directly to
    shared memory without pickle serialization.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Returns:
        List of CPU tensors. For non-MLA, each chunk shape is
        ``[2, num_layers, chunk_tokens, hidden_dim]``; for MLA,
        ``[num_layers, chunk_tokens, hidden_dim]``.
    """
    return _gather_chunks_impl(
        kv_caches,
        block_ids,
        blocks_per_chunk,
        layout_hints=layout_hints,
        gpu_kv_format=gpu_kv_format,
    )


def _scatter_chunks_impl(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
) -> None:
    """Core implementation: scatter CPU chunk tensors into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.
        chunks: List of CPU chunk tensors.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )
    import lmcache.c_ops as lmc_ops

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

    # After normalization the structure is always a list of per-layer
    # tensors.  Cast once so all downstream indexing is typed correctly.
    layer_tensors = cast(list[torch.Tensor], normalized)

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

        skip_tokens = skip_blocks_in_chunk * block_size
        chunk_device = chunk_cpu.to(device)

        if use_mla:
            for layer_idx, layer in enumerate(layer_tensors):
                mla_src = chunk_device[layer_idx, skip_tokens:]
                hidden_size = layer.shape[-1]
                mla_src_3d = mla_src.reshape(
                    len(effective_block_ids), block_size, hidden_size
                )
                layer[effective_block_ids] = mla_src_3d
        elif is_hnd:
            for layer_idx, layer in enumerate(layer_tensors):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
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
                k_t[effective_block_ids] = k_blocks
                v_t[effective_block_ids] = v_blocks
        else:
            for layer_idx, layer in enumerate(layer_tensors):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _num_blocks, _block_size, num_heads, head_size = k_t.shape
                k_src_4d = k_src.reshape(
                    len(effective_block_ids), block_size, num_heads, head_size
                )
                v_src_4d = v_src.reshape(
                    len(effective_block_ids), block_size, num_heads, head_size
                )
                k_t[effective_block_ids] = k_src_4d
                v_t[effective_block_ids] = v_src_4d


def scatter_tensors_to_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    cpu_tensors: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: Any | None = None,
    gpu_kv_format: Any | None = None,
) -> None:
    """Scatter CPU tensors back into paged KV tensors.

    Used by the SHM retrieve path where tensors are constructed as
    zero-copy views over shared memory.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.
        cpu_tensors: List of CPU chunk tensors (same format as returned
            by :func:`gather_chunks_to_cpu_tensors`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.
    """
    _scatter_chunks_impl(
        kv_caches,
        block_ids,
        cpu_tensors,
        blocks_per_chunk,
        skip_first_n_tokens=skip_first_n_tokens,
        layout_hints=layout_hints,
        gpu_kv_format=gpu_kv_format,
    )


@dataclass
class CPUBounceContext:
    """CPU bounce-buffer layout metadata for non-CUDA workers."""

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool
