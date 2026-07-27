# SPDX-License-Identifier: Apache-2.0
"""Group-aware KV-cache gather/scatter planning for engine-driven hybrid/HMA.

This module provides group-level copy planning that is shared between the
synchronous and asynchronous engine-driven transfer paths. It is
path-neutral: it does not depend on MQ, SHM lifetimes, or any concrete
transfer-context implementation.

Core utilities
--------------
- ``build_group_kv_subset``: extract the sub-dict of KV tensors belonging to
  a single LMCache object group (by layer index).
- ``compute_group_blocks_in_chunk``: derive per-group ``blocks_in_chunk`` from
  the reference chunk size.
- ``plan_group_copy``: build a list of :class:`GroupCopyPlan` entries (one per
  object group) from the registered ``EngineGroupInfo`` list and the
  per-LMCache-group flat block-ID lists.
- ``gather_engine_groups``: multi-group gather (device → CPU) using the above
  plans and the existing single-group ``gather_paged_kv_to_cpu`` primitive.
- ``scatter_engine_groups``: inverse (CPU → device).

Indexing conventions
---------------------
* *engine_group_id* — block-ID address space; dense from 0, matches
  ``EngineGroupInfo.engine_group_id``.
* *LMCache group index* — 0-based position in the ``EngineGroupInfo`` list
  (and in ``block_ids_per_lmcache_group``).  Several LMCache groups may share
  one engine group (same block IDs, different layer split).
* *object group index* — same as LMCache group index for engine-driven paths:
  each LMCache group produces one independent object sequence stored under
  ``ObjectKey.object_group_id == lmcache_group_idx``.

Wire ordering (group-major flat)
---------------------------------
For N chunks and G object groups the flat list sent / received over the wire
is ordered **group-major**:

  [g0_chunk0, g0_chunk1, …, g0_chunkN-1,
   g1_chunk0, g1_chunk1, …, g1_chunkN-1,
   …]

The per-group chunk count comes from ``GroupCopyPlan.num_chunks``.  A
``group_counts`` field in the SHM/pickle context payload encodes these counts
so the worker can reconstruct the grouping on the retrieve side.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints

# Local
from .base import gather_paged_kv_to_cpu, scatter_cpu_to_paged_kv

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_group_kv_subset(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: tuple[int, ...] | list[int],
) -> dict[str, torch.Tensor]:
    """Extract the KV-cache sub-dict for a single object group.

    Args:
        kv_caches: Full KV-cache mapping keyed by layer name, ordered by
            layer registration order (Python 3.7+ dict insertion order).
        layer_indices: Indices into the ordered ``kv_caches`` values that
            belong to the target object group.

    Returns:
        An ordered sub-dict mapping layer names to tensors, preserving the
        relative insertion order of ``kv_caches``.

    Raises:
        ValueError: If any index in ``layer_indices`` is out of range.
    """
    all_items = list(kv_caches.items())
    n = len(all_items)
    idx_set = set(layer_indices)
    for idx in idx_set:
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Layer index {idx} is out of range [0, {n}) "
                "for the registered kv_caches."
            )
    return {k: v for i, (k, v) in enumerate(all_items) if i in idx_set}


def compute_group_blocks_in_chunk(
    ref_blocks_in_chunk: int,
    ref_block_size: int,
    group_block_size: int,
    group_tokens_per_block: int = 0,
) -> int:
    """Derive blocks-per-chunk for a group given a reference chunk size.

    The reference chunk size in tokens is ``ref_blocks_in_chunk * ref_block_size``.
    For the target group, blocks-per-chunk = chunk_tokens / effective_block_size,
    where ``effective_block_size = group_tokens_per_block`` when non-zero, else
    ``group_block_size``.

    Args:
        ref_blocks_in_chunk: Blocks per LMCache chunk for the reference group
            (typically group 0, or the value passed to ``submit_store``).
        ref_block_size: Physical block size of the reference group (tokens per
            paged block as detected from the tensor shape).
        group_block_size: Physical block size of the target group.
        group_tokens_per_block: Logical tokens per block for the target group as
            reported by ``EngineGroupInfo.tokens_per_block``.  ``0`` means
            equal to ``group_block_size``.

    Returns:
        Number of paged blocks per LMCache chunk for the target group.

    Raises:
        ValueError: If the computed chunk-token count is not divisible by the
            effective block size.
    """
    chunk_tokens = ref_blocks_in_chunk * ref_block_size
    effective_block_size = group_tokens_per_block or group_block_size
    if chunk_tokens % effective_block_size != 0:
        raise ValueError(
            f"chunk_tokens={chunk_tokens} is not divisible by effective "
            f"block_size={effective_block_size} for this group"
        )
    return chunk_tokens // effective_block_size


# ---------------------------------------------------------------------------
# GroupCopyPlan
# ---------------------------------------------------------------------------


@dataclass
class GroupCopyPlan:
    """Concrete copy descriptor for one object group.

    Attributes:
        lmcache_group_idx: Index of this group in the LMCache group list
            (also used as ``object_group_id`` in storage).
        engine_group_id: Engine block-ID address space this group draws from.
        flat_block_ids: Flat block-ID list for this group derived from the
            per-LMCache-group ``block_ids`` input.
        blocks_in_chunk: Number of paged blocks per LMCache chunk for this
            group.
        layer_indices: KV-tensor indices assigned to this group.
        kv_subset: Sub-dict of KV-cache tensors for the layers in this group.
        num_chunks: Total number of LMCache chunks to transfer.
    """

    lmcache_group_idx: int
    engine_group_id: int
    flat_block_ids: list[int]
    blocks_in_chunk: int
    layer_indices: tuple[int, ...] | list[int]
    kv_subset: dict[str, torch.Tensor]
    num_chunks: int


def plan_group_copy(
    kv_caches: dict[str, torch.Tensor],
    block_ids_per_lmcache_group: list[list[int]],
    blocks_in_chunk: int,
    engine_group_infos: "list[EngineGroupInfo]",
    group_block_sizes: list[int],
) -> list[GroupCopyPlan]:
    """Build per-group copy plans from registration metadata and block IDs.

    Handles the mapping from LMCache group index → engine group ID → block IDs
    and layer indices, producing one :class:`GroupCopyPlan` per LMCache group.
    If ``engine_group_infos`` is empty the function returns an empty list (the
    caller should fall back to the single-group path).

    Args:
        kv_caches: Full KV-cache mapping keyed by layer name.
        block_ids_per_lmcache_group: Block-ID lists indexed by LMCache group
            index; ``block_ids_per_lmcache_group[g]`` is the flat block-ID list
            for LMCache group ``g``.
        blocks_in_chunk: Reference blocks-per-chunk (for engine group 0, or
            whichever group corresponds to the flat ``blocks_in_chunk`` argument
            passed to ``submit_store``).
        engine_group_infos: Registered ``EngineGroupInfo`` entries, one per
            LMCache group in group-index order.
        group_block_sizes: Per-LMCache-group physical block sizes.  Index ``g``
            is the block size for LMCache group ``g``.

    Returns:
        One :class:`GroupCopyPlan` per LMCache group, in group-index order.
        Empty if ``engine_group_infos`` is empty.
    """
    if not engine_group_infos:
        return []

    ref_block_size = group_block_sizes[0] if group_block_sizes else 0
    # Fall back to inferring from kv_caches when no per-group sizes recorded.
    if not ref_block_size and kv_caches:
        # Shape conventions: (2, NB, BS, ...) or (NB, BS, ...) or similar.
        # Physical block_size is typically dim 1 (NB) or 2 (BS). We use the
        # same heuristic as compute_kv_layout which is called during register().
        # However, we need only the *chunk* token count here and can derive it
        # from blocks_in_chunk * detected_block_size when available.
        # If we have no block size info at all we cannot compute per-group
        # blocks_in_chunk for non-group-0 groups — emit a warning and use the
        # caller-supplied blocks_in_chunk for all groups.
        logger.warning(
            "No per-group block sizes available; using blocks_in_chunk=%d for "
            "all groups. Per-group tokens-per-block override will be skipped.",
            blocks_in_chunk,
        )

    plans: list[GroupCopyPlan] = []
    for lmcache_group_idx, info in enumerate(engine_group_infos):
        flat_ids = block_ids_per_lmcache_group[lmcache_group_idx]

        # Per-group blocks_in_chunk.
        if group_block_sizes and lmcache_group_idx < len(group_block_sizes):
            g_block_size = group_block_sizes[lmcache_group_idx]
            if ref_block_size and g_block_size != ref_block_size:
                g_blocks_in_chunk = compute_group_blocks_in_chunk(
                    blocks_in_chunk,
                    ref_block_size,
                    g_block_size,
                    info.tokens_per_block,
                )
            else:
                g_blocks_in_chunk = blocks_in_chunk
        else:
            g_blocks_in_chunk = blocks_in_chunk

        num_chunks = len(flat_ids) // g_blocks_in_chunk if g_blocks_in_chunk else 0
        kv_subset = (
            build_group_kv_subset(kv_caches, info.layer_indices)
            if info.layer_indices
            else kv_caches
        )

        plans.append(
            GroupCopyPlan(
                lmcache_group_idx=lmcache_group_idx,
                engine_group_id=info.engine_group_id,
                flat_block_ids=flat_ids,
                blocks_in_chunk=g_blocks_in_chunk,
                layer_indices=info.layer_indices,
                kv_subset=kv_subset,
                num_chunks=num_chunks,
            )
        )

    return plans


# ---------------------------------------------------------------------------
# Multi-group gather and scatter
# ---------------------------------------------------------------------------


def gather_engine_groups(
    plans: list[GroupCopyPlan],
    layout_hints: LayoutHints | None = None,
    engine_kv_format: Any = None,
    out_per_group: list[list[torch.Tensor] | None] | None = None,
    chunk_indices_per_group: list[list[int] | None] | None = None,
) -> list[list[torch.Tensor]]:
    """Gather KV data from device to CPU across all object groups.

    Calls :func:`~.base.gather_paged_kv_to_cpu` once per group with the
    group-specific KV subset, block IDs, and blocks-per-chunk, then assembles
    the per-group chunk lists into a group-major result.

    Args:
        plans: Per-group copy plans produced by :func:`plan_group_copy`.
        layout_hints: Optional layout metadata forwarded to each
            ``gather_paged_kv_to_cpu`` call.
        engine_kv_format: Optional engine KV format descriptor forwarded to
            each ``gather_paged_kv_to_cpu`` call.
        out_per_group: Pre-allocated output tensors, one list per group.
            ``None`` at index ``g`` means allocate fresh tensors for group
            ``g`` (pickle mode). Length must equal ``len(plans)`` when given.
        chunk_indices_per_group: Sparse chunk index lists, one per group.
            ``None`` at index ``g`` means all chunks are needed. Length must
            equal ``len(plans)`` when given.

    Returns:
        A group-major list: ``result[g]`` is the list of CPU tensors for
        LMCache group ``g``.  Empty list when ``plans`` is empty.
    """
    if not plans:
        return []

    result: list[list[torch.Tensor]] = []
    for g_idx, plan in enumerate(plans):
        out_g = out_per_group[g_idx] if out_per_group is not None else None
        ci_g = (
            chunk_indices_per_group[g_idx]
            if chunk_indices_per_group is not None
            else None
        )
        chunks = gather_paged_kv_to_cpu(
            plan.kv_subset,
            plan.flat_block_ids,
            plan.blocks_in_chunk,
            layout_hints=layout_hints,
            engine_kv_format=engine_kv_format,
            out=out_g,
            chunk_indices=ci_g,
        )
        result.append(chunks)

    return result


def scatter_engine_groups(
    plans: list[GroupCopyPlan],
    chunks_per_group: list[list[torch.Tensor]],
    layout_hints: LayoutHints | None = None,
    engine_kv_format: Any = None,
    skip_first_n_tokens: int = 0,
) -> None:
    """Scatter KV data from CPU back to device across all object groups.

    Calls :func:`~.base.scatter_cpu_to_paged_kv` once per group with the
    group-specific KV subset, block IDs, and chunks.

    Args:
        plans: Per-group copy plans produced by :func:`plan_group_copy`.
        chunks_per_group: CPU tensors indexed ``[group_idx][chunk_idx]``.
        layout_hints: Optional layout metadata forwarded to each
            ``scatter_cpu_to_paged_kv`` call.
        engine_kv_format: Optional engine KV format descriptor forwarded to
            each ``scatter_cpu_to_paged_kv`` call.
        skip_first_n_tokens: Tokens at the head of the block range to leave
            untouched (forwarded to every group's scatter call).

    Raises:
        ValueError: If ``len(chunks_per_group) != len(plans)``.
    """
    if not plans:
        return
    if len(chunks_per_group) != len(plans):
        raise ValueError(
            f"chunks_per_group has {len(chunks_per_group)} entries "
            f"but plans has {len(plans)} entries"
        )
    for plan, chunks in zip(plans, chunks_per_group, strict=False):
        scatter_cpu_to_paged_kv(
            plan.kv_subset,
            plan.flat_block_ids,
            chunks,
            plan.blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens,
            layout_hints=layout_hints,
            engine_kv_format=engine_kv_format,
        )


# ---------------------------------------------------------------------------
# Wire-format helpers (group-major flat ↔ per-group lists)
# ---------------------------------------------------------------------------


def flatten_chunks_group_major(
    chunks_per_group: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    """Flatten group-major per-group chunks into a single list.

    The wire order is: all group-0 chunks, then all group-1 chunks, etc.

    Args:
        chunks_per_group: ``chunks_per_group[g][c]`` = chunk ``c`` for group ``g``.

    Returns:
        Flat list with all group-0 chunks first, then group-1, etc.
    """
    result: list[torch.Tensor] = []
    for group_chunks in chunks_per_group:
        result.extend(group_chunks)
    return result


def unflatten_chunks_group_major(
    flat_chunks: list[torch.Tensor],
    group_counts: list[int],
) -> list[list[torch.Tensor]]:
    """Reconstruct per-group chunk lists from a group-major flat sequence.

    Args:
        flat_chunks: All chunks in group-major order as produced by
            :func:`flatten_chunks_group_major`.
        group_counts: ``group_counts[g]`` is the number of chunks for group
            ``g``.  Must sum to ``len(flat_chunks)``.

    Returns:
        ``result[g]`` is the list of chunks for group ``g``.

    Raises:
        ValueError: If ``sum(group_counts) != len(flat_chunks)``.
    """
    if sum(group_counts) != len(flat_chunks):
        raise ValueError(
            f"group_counts sum {sum(group_counts)} != len(flat_chunks) "
            f"{len(flat_chunks)}"
        )
    result: list[list[torch.Tensor]] = []
    offset = 0
    for cnt in group_counts:
        result.append(flat_chunks[offset : offset + cnt])
        offset += cnt
    return result
