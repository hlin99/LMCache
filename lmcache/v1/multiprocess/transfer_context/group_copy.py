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
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints

# Local
from .base import gather_paged_kv_to_cpu, scatter_cpu_to_paged_kv

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


@dataclass(frozen=True)
class RegisteredGroup:
    """Authoritative registration metadata for one transfer/object group.

    Attributes:
        object_group_id: Storage object-key namespace, dense from zero.
        engine_group_id: Serving-engine block-ID address space.
        layer_indices: Registered KV tensor indices copied by this group.
        tokens_per_block: Logical tokens represented by one engine block.
        slots_per_block: Physical KV slots stored in one engine block.
        blocks_per_chunk: Engine blocks covering one LMCache logical chunk.
        copy_blocks_per_chunk: Trailing blocks copied for each object after
            Sliding Window subchunk trimming.
        chunk_tokens: Logical tokens represented by one LMCache object.
        shape: Contiguous tensor shape for one stored object.
        dtype: Contiguous tensor dtype.
        engine_kv_format: Native copy format discovered for this group.
        sw_size_tokens: Sliding Window size, or ``-1`` for full attention.
    """

    object_group_id: int
    engine_group_id: int
    layer_indices: tuple[int, ...]
    tokens_per_block: int
    slots_per_block: int
    blocks_per_chunk: int
    copy_blocks_per_chunk: int
    chunk_tokens: int
    shape: torch.Size
    dtype: torch.dtype
    engine_kv_format: Any
    sw_size_tokens: int = -1

    @property
    def objects_in_window(self) -> int | None:
        """Return the number of trailing objects needed by Sliding Window.

        Retrieve can omit older stored objects once they fall outside the
        configured window. Full-attention groups return ``None`` because every
        object remains valid.
        """
        if self.sw_size_tokens < 0:
            return None
        return max(
            1, (self.sw_size_tokens + self.chunk_tokens - 1) // self.chunk_tokens
        )

    def physical_skip(self, logical_skip: int) -> int:
        """Convert a logical token skip to this group's physical slot geometry.

        Args:
            logical_skip: Logical tokens to preserve at the start of the
                selected object range.

        Returns:
            Number of physical slots to skip.

        Raises:
            ValueError: If the logical position cannot be represented exactly
                in the group's compressed physical geometry.

        Example:
            A group with eight logical tokens and two physical slots per block
            converts an eight-token skip to two slots.
        """
        if logical_skip < 0:
            raise ValueError("logical_skip must be non-negative")
        numerator = logical_skip * self.slots_per_block
        if numerator % self.tokens_per_block:
            raise ValueError(
                f"logical skip {logical_skip} is not representable for object "
                f"group {self.object_group_id}: tokens_per_block="
                f"{self.tokens_per_block}, slots_per_block={self.slots_per_block}"
            )
        return numerator // self.tokens_per_block


def validate_registered_groups(
    groups: list[RegisteredGroup], num_registered_layers: int
) -> None:
    """Validate group IDs, geometry, and the registered layer partition.

    Args:
        groups: Transfer groups in deterministic object-group order.
        num_registered_layers: Number of tensors in registered KV order.

    Raises:
        ValueError: If IDs are invalid, geometry is non-positive, or layers are
            duplicated, omitted, or out of range.
    """
    if not groups:
        raise ValueError("at least one registered transfer group is required")
    expected_object_ids = list(range(len(groups)))
    object_ids = [group.object_group_id for group in groups]
    if object_ids != expected_object_ids:
        raise ValueError(
            f"object group IDs must be {expected_object_ids}, got {object_ids}"
        )
    engine_ids = {group.engine_group_id for group in groups}
    if min(engine_ids) < 0 or engine_ids != set(range(max(engine_ids) + 1)):
        raise ValueError(
            "engine group IDs must be non-negative and dense from zero, got "
            f"{sorted(engine_ids)}"
        )

    mapped_layers: list[int] = []
    for group in groups:
        if (
            group.tokens_per_block <= 0
            or group.slots_per_block <= 0
            or group.blocks_per_chunk <= 0
            or group.copy_blocks_per_chunk <= 0
            or group.copy_blocks_per_chunk > group.blocks_per_chunk
            or group.chunk_tokens <= 0
        ):
            raise ValueError(
                f"invalid block geometry for object group {group.object_group_id}"
            )
        if not group.layer_indices:
            raise ValueError(
                f"object group {group.object_group_id} has no registered layers"
            )
        mapped_layers.extend(group.layer_indices)

    invalid_layers = [
        idx for idx in mapped_layers if idx < 0 or idx >= num_registered_layers
    ]
    if invalid_layers:
        raise ValueError(
            f"layer indices {invalid_layers} are outside [0, {num_registered_layers})"
        )
    if len(mapped_layers) != len(set(mapped_layers)):
        raise ValueError("registered layer indices must not be duplicated")
    expected_layers = set(range(num_registered_layers))
    actual_layers = set(mapped_layers)
    if actual_layers != expected_layers:
        raise ValueError(
            "registered layer mapping must cover every KV tensor exactly once; "
            f"missing={sorted(expected_layers - actual_layers)}"
        )


def validate_group_block_ids(
    block_ids: list[list[int]], blocks_per_chunk: list[int]
) -> int:
    """Validate transfer-boundary block IDs and return the logical chunk count.

    Args:
        block_ids: Block IDs indexed by transfer-plan group.
        blocks_per_chunk: Full (pre-window-trim) blocks per logical chunk for
            each transfer-plan group.

    Returns:
        Common logical chunk count covered by every group.

    Raises:
        ValueError: If group counts differ, geometry is non-positive, a list has
            a remainder, or groups cover different logical chunk counts.
    """
    if len(block_ids) != len(blocks_per_chunk):
        raise ValueError(
            f"expected {len(blocks_per_chunk)} block-ID lists, got {len(block_ids)}"
        )
    chunk_counts: list[int] = []
    for group_id, (ids, group_blocks) in enumerate(
        zip(block_ids, blocks_per_chunk, strict=True)
    ):
        if group_blocks <= 0:
            raise ValueError(f"blocks_per_chunk for group {group_id} must be positive")
        remainder = len(ids) % group_blocks
        if remainder:
            raise ValueError(
                f"block_ids[{group_id}] has {len(ids)} entries, not an exact "
                f"multiple of blocks_per_chunk={group_blocks}"
            )
        chunk_counts.append(len(ids) // group_blocks)
    if chunk_counts and len(set(chunk_counts)) != 1:
        raise ValueError(
            f"block-ID groups cover different logical chunk counts: {chunk_counts}"
        )
    return chunk_counts[0] if chunk_counts else 0


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

    group: RegisteredGroup
    flat_block_ids: list[int]
    kv_subset: dict[str, torch.Tensor]
    num_chunks: int
    first_object: int = 0

    @property
    def lmcache_group_idx(self) -> int:
        """Return the protocol-visible transfer group index."""
        return self.group.object_group_id

    @property
    def engine_group_id(self) -> int:
        """Return the serving-engine block address-space ID."""
        return self.group.engine_group_id

    @property
    def blocks_in_chunk(self) -> int:
        """Return copied (post-window-trim) blocks per object."""
        return self.group.copy_blocks_per_chunk

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Return registered tensor indices copied by this group."""
        return self.group.layer_indices


def plan_group_copy(
    kv_caches: dict[str, torch.Tensor],
    block_ids_per_lmcache_group: list[list[int]],
    groups: list[RegisteredGroup],
    *,
    for_retrieve: bool = False,
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
        groups: Authoritative registered transfer groups.
        for_retrieve: Apply Sliding Window object-tail selection when true.

    Returns:
        One :class:`GroupCopyPlan` per LMCache group, in group-index order.
        Empty if ``engine_group_infos`` is empty.
    """
    if not groups:
        return []
    num_chunks = validate_group_block_ids(
        block_ids_per_lmcache_group,
        [group.blocks_per_chunk for group in groups],
    )

    plans: list[GroupCopyPlan] = []
    for group, input_ids in zip(groups, block_ids_per_lmcache_group, strict=True):
        selected_ids: list[int] = []
        for offset in range(0, len(input_ids), group.blocks_per_chunk):
            chunk_ids = input_ids[offset : offset + group.blocks_per_chunk]
            selected_ids.extend(chunk_ids[-group.copy_blocks_per_chunk :])

        first_object = 0
        if for_retrieve and group.objects_in_window is not None:
            first_object = max(0, num_chunks - group.objects_in_window)
            selected_ids = selected_ids[first_object * group.copy_blocks_per_chunk :]
        kv_subset = build_group_kv_subset(kv_caches, group.layer_indices)

        plans.append(
            GroupCopyPlan(
                group=group,
                flat_block_ids=selected_ids,
                kv_subset=kv_subset,
                num_chunks=num_chunks,
                first_object=first_object,
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
            engine_kv_format=plan.group.engine_kv_format,
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
    for plan, chunks in zip(plans, chunks_per_group, strict=True):
        selected_chunks = chunks[plan.first_object :]
        logical_skip = max(
            0,
            skip_first_n_tokens - plan.first_object * plan.group.chunk_tokens,
        )
        scatter_cpu_to_paged_kv(
            plan.kv_subset,
            plan.flat_block_ids,
            selected_chunks,
            plan.blocks_in_chunk,
            skip_first_n_tokens=plan.group.physical_skip(logical_skip),
            layout_hints=layout_hints,
            engine_kv_format=plan.group.engine_kv_format,
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
    if any(count < 0 for count in group_counts):
        raise ValueError("group_counts must not contain negative values")
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
