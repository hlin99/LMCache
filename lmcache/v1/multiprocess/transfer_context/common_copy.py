# SPDX-License-Identifier: Apache-2.0
"""Common registered-group model and request planner for KV-cache transfers.

Both multiprocess transfer paths plan their copies here:

* the **LMCache-driven** server path builds its groups from the
  :class:`~lmcache.v1.kv_layer_groups.KVLayerGroupsManager` it already owns
  (see :func:`registered_groups_from_kv_layer_groups`) and executes the copy on
  the server with IPC-opened KV views and ``MemoryObj`` staging;
* the **Engine-driven** worker path builds its groups from the registered
  engine KV-group metadata (see :func:`registered_groups_from_engine_infos`)
  and executes the copy on the worker with local KV tensors and SHM/pickle
  tensor staging.

This module owns everything that must not be implemented twice:

- the immutable :class:`RegisteredGroup` description of one transfer group;
- exact block-ID validation (:func:`validate_group_block_ids`);
- per-chunk Sliding Window block selection;
- Sliding Window object-tail selection (``first_object``);
- logical-token to physical block/slot skip conversion
  (:meth:`RegisteredGroup.blocks_to_skip`).

It deliberately owns *no* transport, storage, or lifecycle concern: it does not
reserve or release storage, allocate SHM, open IPC handles, or submit MQ
requests. Copy *execution* stays path-specific because the two paths drive
different native entry points (``execute_object_group_transfer`` on the server
against IPC-opened paged KV, ``multi_layer_block_kv_transfer`` on the worker
against local KV tensors); :func:`gather_engine_groups` and
:func:`scatter_engine_groups` are the Engine-driven executors built on the
common plans.

Indexing conventions
---------------------
* *engine_group_id* -- block-ID address space exposed by the serving engine.
  Several transfer groups may share one engine group when their layers differ
  in physical copy identity.
* *kernel_group_id* -- position of the group in registration order; also the
  index of the group's block-ID list in a transfer request.
* *object_group_id* -- storage object-key namespace. The Engine-driven path
  gives every kernel group its own object group; the LMCache-driven path may
  map several kernel groups onto one object group.

Wire ordering (group-major flat)
---------------------------------
For N chunks and G object groups the flat list sent / received over the wire
is ordered **group-major**::

  [g0_chunk0, g0_chunk1, ..., g0_chunkN-1,
   g1_chunk0, g1_chunk1, ..., g1_chunkN-1,
   ...]

The per-group chunk count comes from ``GroupTransferPlan.transfer_objects``. A
``group_counts`` field in the SHM/pickle context payload encodes these counts
so the worker can reconstruct the grouping on the retrieve side.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Any, NamedTuple

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints

# Local
from .base import gather_paged_kv_to_cpu, scatter_cpu_to_paged_kv

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_group_kv_subset(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: tuple[int, ...] | list[int],
) -> dict[str, torch.Tensor]:
    """Extract the KV-cache sub-dict for a single transfer group.

    Args:
        kv_caches: Full KV-cache mapping keyed by layer name, ordered by
            layer registration order (Python 3.7+ dict insertion order).
        layer_indices: Indices into the ordered ``kv_caches`` values that
            belong to the target group.

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


@dataclass(frozen=True)
class RegisteredGroup:
    """Authoritative registration metadata for one transfer group.

    One instance describes one copy-kernel dispatch unit and is shared by both
    transfer paths. It is built once per registration and never mutated.

    Attributes:
        kernel_group_id: Registration order of this group; also the index of
            its block-ID list in a transfer request.
        object_group_id: Storage object-key namespace this group is stored in.
            Several kernel groups may map onto one object group.
        engine_group_id: Serving-engine block-ID address space.
        layer_indices: Registered KV tensor indices copied by this group.
        tokens_per_block: Logical tokens represented by one engine block.
        slots_per_block: Physical KV slots stored in one engine block.
        blocks_per_chunk: Engine blocks covering one LMCache logical chunk.
        copy_blocks_per_chunk: Trailing blocks copied for each object after
            Sliding Window subchunk trimming.
        chunk_tokens: Logical tokens represented by one LMCache object.
        shape: Contiguous tensor shape of this group's part of one object.
        dtype: Contiguous tensor dtype.
        engine_kv_format: Native copy format discovered for this group.
        sw_size_tokens: Sliding Window size, or ``-1`` for full attention.
    """

    kernel_group_id: int
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

    def blocks_to_skip(self, logical_skip: int) -> int:
        """Convert a logical token skip into a copied (post-trim) block count.

        The result counts blocks in the *selected* block sequence, i.e. after
        per-chunk Sliding Window trimming, so it applies directly to
        ``GroupTransferPlan.selected_block_ids``.

        Args:
            logical_skip: Logical tokens to preserve at the start of the
                transferred range. A value that does not land on a block
                boundary is rounded down to the nearest whole block, matching
                the block-granular native copy primitives.

        Returns:
            Number of copied blocks to skip.

        Raises:
            ValueError: If ``logical_skip`` is negative.

        Example:
            With ``blocks_per_chunk=4`` and ``copy_blocks_per_chunk=2`` (a
            sub-chunk window keeping each chunk's last two blocks), a skip of
            three whole blocks becomes one copied block: the first two skipped
            blocks were never selected.
        """
        if logical_skip < 0:
            raise ValueError("logical_skip must be non-negative")
        full_blocks = logical_skip // self.tokens_per_block
        whole_chunks, tail_blocks = divmod(full_blocks, self.blocks_per_chunk)
        trimmed = self.blocks_per_chunk - self.copy_blocks_per_chunk
        return whole_chunks * self.copy_blocks_per_chunk + max(0, tail_blocks - trimmed)

    def physical_skip(self, logical_skip: int) -> int:
        """Convert a logical token skip into copied physical slots.

        Args:
            logical_skip: Logical tokens to preserve at the start of the
                transferred range.

        Returns:
            Number of physical slots to skip: :meth:`blocks_to_skip` scaled by
            ``slots_per_block``. A skip that does not land on a block boundary
            rounds down, so protected tokens are never partially overwritten
            while the copy stays block-granular.

        Raises:
            ValueError: If ``logical_skip`` is negative.
        """
        return self.blocks_to_skip(logical_skip) * self.slots_per_block


class DiscoveredGroupLayout(NamedTuple):
    """Physical layout of one transfer group, discovered from KV tensors.

    Attributes:
        slots_per_block: Physical KV slots stored in one engine block.
        num_layers: KV tensors copied by the group.
        hidden_dim_size: ``num_heads * head_size`` of one slot.
        kv_size: Object planes; ``1`` for MLA and fused-K/V, ``2`` for split
            K/V.
        dtype: Contiguous tensor dtype.
        engine_kv_format: Native copy format discovered for the group.
    """

    slots_per_block: int
    num_layers: int
    hidden_dim_size: int
    kv_size: int
    dtype: torch.dtype
    engine_kv_format: Any


def contiguous_object_shape(
    num_layers: int,
    physical_slots: int,
    hidden_dim_size: int,
    kv_size: int,
) -> torch.Size:
    """Return the contiguous tensor shape of one group's part of an object.

    Args:
        num_layers: KV tensors copied by the group.
        physical_slots: Physical slots stored per object by the group.
        hidden_dim_size: ``num_heads * head_size`` of one slot.
        kv_size: Object planes; ``1`` for MLA and fused-K/V, ``2`` for split
            K/V.

    Returns:
        ``[num_layers, physical_slots, hidden_dim_size]`` for single-plane
        formats, otherwise ``[kv_size, num_layers, physical_slots,
        hidden_dim_size]``.
    """
    if kv_size == 1:
        return torch.Size([num_layers, physical_slots, hidden_dim_size])
    return torch.Size([kv_size, num_layers, physical_slots, hidden_dim_size])


def registered_groups_from_engine_infos(
    engine_group_infos: "Sequence[EngineGroupInfo]",
    layouts: Sequence[DiscoveredGroupLayout],
    lmcache_tokens_per_chunk: int,
) -> list[RegisteredGroup]:
    """Build the common groups for the Engine-driven worker path.

    Every engine KV group becomes one transfer group owning its own object
    group, because the worker stores each group's chunks under its own
    ``object_group_id``.

    Args:
        engine_group_infos: Engine KV-group metadata in protocol order.
        layouts: Layout discovered from the registered KV tensors, in the same
            order as ``engine_group_infos``.
        lmcache_tokens_per_chunk: Authoritative logical tokens per LMCache
            chunk.

    Returns:
        One :class:`RegisteredGroup` per engine KV group, in protocol order.

    Raises:
        ValueError: If the two input sequences differ in length, or the chunk
            size is not divisible by a group's ``tokens_per_block``.
    """
    if len(layouts) != len(engine_group_infos):
        raise ValueError(
            f"got {len(layouts)} layouts for {len(engine_group_infos)} engine groups"
        )
    groups: list[RegisteredGroup] = []
    for group_id, (info, layout) in enumerate(
        zip(engine_group_infos, layouts, strict=True)
    ):
        tokens_per_block = info.tokens_per_block or layout.slots_per_block
        if lmcache_tokens_per_chunk % tokens_per_block:
            raise ValueError(
                f"LMCache chunk size {lmcache_tokens_per_chunk} is not divisible "
                f"by tokens_per_block={tokens_per_block} for object group {group_id}"
            )
        copy_tokens = (
            lmcache_tokens_per_chunk
            if info.sw_size_tokens < 0
            else min(lmcache_tokens_per_chunk, info.sw_size_tokens)
        )
        copy_blocks = max(1, (copy_tokens + tokens_per_block - 1) // tokens_per_block)
        groups.append(
            RegisteredGroup(
                kernel_group_id=group_id,
                object_group_id=group_id,
                engine_group_id=info.engine_group_id,
                layer_indices=tuple(info.layer_indices),
                tokens_per_block=tokens_per_block,
                slots_per_block=layout.slots_per_block,
                blocks_per_chunk=lmcache_tokens_per_chunk // tokens_per_block,
                copy_blocks_per_chunk=copy_blocks,
                chunk_tokens=lmcache_tokens_per_chunk,
                shape=contiguous_object_shape(
                    layout.num_layers,
                    copy_blocks * layout.slots_per_block,
                    layout.hidden_dim_size,
                    layout.kv_size,
                ),
                dtype=layout.dtype,
                engine_kv_format=layout.engine_kv_format,
                sw_size_tokens=info.sw_size_tokens,
            )
        )
    return groups


def registered_groups_from_kv_layer_groups(
    manager: "KVLayerGroupsManager",
    lmcache_tokens_per_chunk: int,
) -> list[RegisteredGroup]:
    """Build the common groups for the LMCache-driven server path.

    This is an adapter over metadata the server already owns: it re-expresses
    the manager's kernel groups, object groups, and window sizes as
    :class:`RegisteredGroup` records so the server can call the common planner.
    It re-discovers nothing.

    Args:
        manager: The server's KV layer groups manager.
        lmcache_tokens_per_chunk: Authoritative logical tokens per LMCache
            chunk (owned by the cache context).

    Returns:
        One :class:`RegisteredGroup` per kernel group, in kernel-group order,
        tagged with the object group that stores it.

    Raises:
        ValueError: If a kernel group has no discovered ``engine_kv_format``
            (a formatless bookkeeping group must never reach the transfer
            path).
    """
    object_group_of_kernel_group: dict[int, int] = {}
    for object_group_id, object_group in enumerate(manager.object_groups):
        for kernel_group_id in object_group.kernel_group_indices:
            object_group_of_kernel_group[kernel_group_id] = object_group_id
    # The manager reports full attention for every object group when it stores
    # full per-chunk KV, so object-tail selection follows the manager rather
    # than a kernel group's raw window size.
    num_chunks_in_sw = manager.get_attn_desc().num_chunks_in_sw

    groups: list[RegisteredGroup] = []
    for kernel_group_id, kernel_group in enumerate(manager.kernel_groups):
        engine_kv_format = kernel_group.engine_kv_format
        if engine_kv_format is None:
            raise ValueError(
                f"kernel group {kernel_group_id} has no engine_kv_format; a "
                "formatless bookkeeping group reached the transfer path"
            )
        object_group_id = object_group_of_kernel_group[kernel_group_id]
        window_chunks = num_chunks_in_sw[object_group_id]
        groups.append(
            RegisteredGroup(
                kernel_group_id=kernel_group_id,
                object_group_id=object_group_id,
                engine_group_id=kernel_group.engine_group_idx,
                layer_indices=tuple(kernel_group.layer_indices),
                tokens_per_block=kernel_group.tokens_per_block,
                slots_per_block=kernel_group.slots_per_block,
                blocks_per_chunk=manager.calculate_num_blocks(
                    kernel_group_id, lmcache_tokens_per_chunk
                ),
                copy_blocks_per_chunk=manager.calculate_num_blocks(
                    kernel_group_id,
                    manager.get_subchunk_sw_size_tokens(kernel_group_id),
                ),
                chunk_tokens=lmcache_tokens_per_chunk,
                shape=contiguous_object_shape(
                    kernel_group.num_layers,
                    manager.get_slots_per_chunk_in_sw(kernel_group_id),
                    kernel_group.hidden_dim_size,
                    kernel_group.shape_desc.kv_size,
                ),
                dtype=kernel_group.dtype,
                engine_kv_format=engine_kv_format,
                sw_size_tokens=(
                    -1
                    if window_chunks < 1
                    else window_chunks * lmcache_tokens_per_chunk
                ),
            )
        )
    return groups


def validate_registered_groups(
    groups: list[RegisteredGroup],
    num_registered_layers: int,
    excluded_layer_indices: AbstractSet[int] = frozenset(),
) -> None:
    """Validate group IDs, geometry, and the registered layer partition.

    Args:
        groups: Transfer groups in registration order.
        num_registered_layers: Number of tensors in registered KV order.
        excluded_layer_indices: Registered cross-layer sharing aliases that
            intentionally have no transfer group.

    Raises:
        ValueError: If IDs are invalid, geometry is non-positive, or layers are
            duplicated, omitted, or out of range.
    """
    if not groups:
        raise ValueError("at least one registered transfer group is required")
    expected_kernel_ids = list(range(len(groups)))
    kernel_ids = [group.kernel_group_id for group in groups]
    if kernel_ids != expected_kernel_ids:
        raise ValueError(
            f"kernel group IDs must be {expected_kernel_ids}, got {kernel_ids}"
        )
    object_ids = [group.object_group_id for group in groups]
    distinct_object_ids = sorted(set(object_ids))
    if distinct_object_ids != list(range(len(distinct_object_ids))):
        raise ValueError(
            f"object group IDs must be dense from zero, got {distinct_object_ids}"
        )
    if object_ids != sorted(object_ids):
        raise ValueError(
            f"object group IDs must not interleave kernel groups, got {object_ids}"
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
                f"invalid block geometry for transfer group {group.kernel_group_id}"
            )
        if not group.layer_indices:
            raise ValueError(
                f"transfer group {group.kernel_group_id} has no registered layers"
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
    actual_layers = set(mapped_layers)
    excluded_layers = set(excluded_layer_indices)
    invalid_excluded = [
        idx for idx in excluded_layers if idx < 0 or idx >= num_registered_layers
    ]
    if invalid_excluded:
        raise ValueError(
            f"excluded layer indices {invalid_excluded} are outside "
            f"[0, {num_registered_layers})"
        )
    excluded_owned = actual_layers & excluded_layers
    if excluded_owned:
        raise ValueError(
            "excluded layer indices must not be assigned to transfer groups; "
            f"owned={sorted(excluded_owned)}"
        )
    expected_layers = set(range(num_registered_layers)) - excluded_layers
    if actual_layers != expected_layers:
        raise ValueError(
            "registered layer mapping must cover every KV tensor exactly once; "
            f"missing={sorted(expected_layers - actual_layers)}"
        )


def validate_group_block_ids(
    block_ids: Sequence[Sequence[int]], blocks_per_chunk: Sequence[int]
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
# Request-level transfer plan
# ---------------------------------------------------------------------------


def sliding_window_first_object(group: RegisteredGroup, total_objects: int) -> int:
    """Return the first object a Sliding Window retrieve still needs.

    Objects older than the group's window are not read back, because the
    engine no longer attends to them.

    Args:
        group: Registered transfer group.
        total_objects: Logical objects covered by the request.

    Returns:
        Number of leading objects to omit; ``0`` for full attention.

    Raises:
        ValueError: If ``total_objects`` is negative.

    Example:
        A group whose window spans three objects omits the first five of an
        eight-object request.
    """
    if total_objects < 0:
        raise ValueError("total_objects must be non-negative")
    objects_in_window = group.objects_in_window
    if objects_in_window is None:
        return 0
    return max(0, total_objects - objects_in_window)


@dataclass(frozen=True)
class GroupTransferPlan:
    """Path-neutral copy plan for one transfer group and one request.

    Attributes:
        group: The immutable registration record this plan was built from.
        selected_block_ids: Block IDs actually copied, after per-chunk Sliding
            Window trimming and object-tail selection.
        total_objects: Logical objects covered by the request block IDs.
        first_object: Leading objects omitted by Sliding Window retrieve.
        transfer_objects: Objects actually transferred
            (``total_objects - first_object``).
        physical_skip: Physical slots to leave untouched at the head of
            ``selected_block_ids`` (APC-shared prefix guard).
    """

    group: RegisteredGroup
    selected_block_ids: tuple[int, ...]
    total_objects: int
    first_object: int
    transfer_objects: int
    physical_skip: int


def build_group_transfer_plans(
    groups: Sequence[RegisteredGroup],
    block_ids: Sequence[Sequence[int]],
    *,
    for_retrieve: bool = False,
    skip_first_n_tokens: int = 0,
) -> list[GroupTransferPlan]:
    """Build one request plan per registered group.

    This is the single implementation of exact block validation, per-chunk
    Sliding Window block selection, Sliding Window object-tail selection, and
    logical-to-physical skip conversion. Both transfer paths call it.

    Args:
        groups: Registered transfer groups in registration order.
        block_ids: Request block IDs, one flat list per group, in the same
            order as ``groups``.
        for_retrieve: Apply Sliding Window object-tail selection. Store keeps
            every object; the storage side decides what to persist.
        skip_first_n_tokens: Logical tokens at the head of the transferred
            range that must not be overwritten (APC-shared blocks).

    Returns:
        One :class:`GroupTransferPlan` per group, in group order. Empty when
        ``groups`` is empty.

    Raises:
        ValueError: If block IDs do not exactly cover the same logical object
            count for every group, or ``skip_first_n_tokens`` is negative.
    """
    if not groups:
        return []
    if skip_first_n_tokens < 0:
        raise ValueError("skip_first_n_tokens must be non-negative")
    total_objects = validate_group_block_ids(
        block_ids, [group.blocks_per_chunk for group in groups]
    )

    plans: list[GroupTransferPlan] = []
    for group, group_block_ids in zip(groups, block_ids, strict=True):
        selected: list[int] = []
        for offset in range(0, len(group_block_ids), group.blocks_per_chunk):
            chunk_ids = group_block_ids[offset : offset + group.blocks_per_chunk]
            selected.extend(chunk_ids[-group.copy_blocks_per_chunk :])

        first_object = (
            sliding_window_first_object(group, total_objects) if for_retrieve else 0
        )
        if first_object:
            selected = selected[first_object * group.copy_blocks_per_chunk :]

        logical_skip = max(0, skip_first_n_tokens - first_object * group.chunk_tokens)
        plans.append(
            GroupTransferPlan(
                group=group,
                selected_block_ids=tuple(selected),
                total_objects=total_objects,
                first_object=first_object,
                transfer_objects=total_objects - first_object,
                physical_skip=group.physical_skip(logical_skip),
            )
        )
    return plans


# ---------------------------------------------------------------------------
# Engine-driven execution over the common plans
# ---------------------------------------------------------------------------


def gather_engine_groups(
    plans: Sequence[GroupTransferPlan],
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
    out_per_group: list[list[torch.Tensor] | None] | None = None,
    chunk_indices_per_group: list[list[int] | None] | None = None,
) -> list[list[torch.Tensor]]:
    """Gather KV data from device to CPU across all transfer groups.

    Calls :func:`~.base.gather_paged_kv_to_cpu` once per group with the group's
    KV subset, planned block IDs, and copied blocks per object, then assembles
    the per-group chunk lists into a group-major result.

    Args:
        plans: Per-group plans from :func:`build_group_transfer_plans`.
        kv_caches: Worker KV-cache tensors keyed by layer name.
        layout_hints: Optional layout metadata forwarded to each
            ``gather_paged_kv_to_cpu`` call.
        out_per_group: Pre-allocated output tensors, one list per group.
            ``None`` at index ``g`` means allocate fresh tensors for group
            ``g`` (pickle mode). Length must equal ``len(plans)`` when given.
        chunk_indices_per_group: Sparse chunk index lists, one per group.
            ``None`` at index ``g`` means all chunks are needed. Length must
            equal ``len(plans)`` when given.

    Returns:
        A group-major list: ``result[g]`` is the list of CPU tensors for group
        ``g``. Empty list when ``plans`` is empty.
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
            build_group_kv_subset(kv_caches, plan.group.layer_indices),
            list(plan.selected_block_ids),
            plan.group.copy_blocks_per_chunk,
            layout_hints=layout_hints,
            engine_kv_format=plan.group.engine_kv_format,
            out=out_g,
            chunk_indices=ci_g,
        )
        result.append(chunks)

    return result


def scatter_engine_groups(
    plans: Sequence[GroupTransferPlan],
    kv_caches: dict[str, torch.Tensor],
    chunks_per_group: Sequence[list[torch.Tensor]],
    layout_hints: LayoutHints | None = None,
) -> None:
    """Scatter KV data from CPU back to device across all transfer groups.

    Calls :func:`~.base.scatter_cpu_to_paged_kv` once per group with the
    group's KV subset, planned block IDs, and chunks. The APC prefix guard
    comes from ``GroupTransferPlan.physical_skip``, resolved once by the common
    planner.

    Args:
        plans: Per-group plans from :func:`build_group_transfer_plans`.
        kv_caches: Worker KV-cache tensors keyed by layer name.
        chunks_per_group: CPU tensors indexed ``[group_idx][chunk_idx]``.
        layout_hints: Optional layout metadata forwarded to each
            ``scatter_cpu_to_paged_kv`` call.

    Raises:
        ValueError: If the number of chunk lists, or the object count of any
            group, does not match its plan.
    """
    if not plans:
        return
    if len(chunks_per_group) != len(plans):
        raise ValueError(
            f"chunks_per_group has {len(chunks_per_group)} entries "
            f"but plans has {len(plans)} entries"
        )
    for plan, chunks in zip(plans, chunks_per_group, strict=True):
        if len(chunks) != plan.transfer_objects:
            raise ValueError(
                f"server returned {len(chunks)} chunks for KV-cache object group "
                f"{plan.group.object_group_id}, expected {plan.transfer_objects} "
                f"(total_objects={plan.total_objects}, "
                f"first_object={plan.first_object})"
            )
        scatter_cpu_to_paged_kv(
            build_group_kv_subset(kv_caches, plan.group.layer_indices),
            list(plan.selected_block_ids),
            chunks,
            plan.group.copy_blocks_per_chunk,
            skip_first_n_tokens=plan.physical_skip,
            layout_hints=layout_hints,
            engine_kv_format=plan.group.engine_kv_format,
        )


# ---------------------------------------------------------------------------
# Wire-format helpers (group-major flat <-> per-group lists)
# ---------------------------------------------------------------------------


def flatten_chunks_group_major(
    chunks_per_group: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    """Flatten group-major per-group chunks into a single list.

    The wire order is: all group-0 chunks, then all group-1 chunks, etc.

    Args:
        chunks_per_group: ``chunks_per_group[g][c]`` = chunk ``c`` for group
            ``g``.

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
            ``g``. Must sum to ``len(flat_chunks)``.

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
