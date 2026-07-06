# SPDX-License-Identifier: Apache-2.0
"""Path-agnostic transfer plan builder for LMCache multiprocess transfers.

This module provides:

- Data classes describing *what* a transfer consists of, with no CUDA or
  execution semantics:
  ``TransferDirection``, ``KernelGroupTransferPlan``,
  ``ObjectGroupTransferPlan``, ``TransferPlan``.
- Pure planning helpers:
  ``recalculate_blocks_to_skip``, ``downsample_block_ids``,
  ``validate_block_ids``, ``build_object_group_layout_desc``.
- ``TransferPlanBuilder``: assembles all metadata into a ``TransferPlan``
  given a :class:`~lmcache.v1.platform.base_cache_context.BaseCacheContext`
  and per-group object keys.

The builder answers:

    What needs to be copied for this transfer?

It does **not** answer:

    How / where / when should the copy be executed?

Specifically, the builder must not contain CUDA streams, IPC events, SHM
slots, pickle serialization, or any GPU copy calls.  In particular, the
following must **not** appear here:

- ``torch_dev.device(...)``
- ``torch_dev.stream(...)``
- ``torch_dev.Event(...)``
- ``event_ipc_handle``
- ``Event.from_ipc_handle(...)``
- ``vllm_event.wait(...)``
- ``cache_context.stream``
- ``cache_context.cupy_stream``
- ``publish_on_stream(...)`` / ``submit_callback_to_stream(...)``
- ``DeviceHostFuncDispatcher``
- ``lmcache_memcpy_async_h2d`` / ``lmcache_memcpy_async_d2h``
- Calls to ``lmc_ops.multi_layer_block_kv_transfer``
- Calls to ``lmc_ops.execute_object_group_transfer``
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base_cache_context import BaseCacheContext
    import lmcache.c_ops as lmc_ops


class TransferDirection(Enum):
    """Direction of a KV cache transfer operation."""

    STORE = "store"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class KernelGroupTransferPlan:
    """Pure-metadata transfer plan for one kernel group.

    Describes what data needs to be transferred for a single kernel group
    within a single transfer operation.  Contains no execution semantics.

    Attributes:
        kernel_group_id: Index of the kernel group within the model's KV
            groups.
        object_group_id: Index of the object group this kernel group belongs
            to.
        blocks_per_chunk: Number of paged blocks per LMCache chunk for this
            group.  May differ from other groups in a hybrid model.
        blocks_per_window: Number of paged blocks per sliding-window frame.
            Equals ``blocks_per_chunk`` for full-attention groups.
        selected_block_ids: Downsampled block IDs for this kernel group.  For
            sliding-window/subchunk groups this is shorter than the raw
            ``gpu_block_ids`` input because only the window's blocks are kept
            per chunk.  Indexed by window position across all chunks.
        skip_blocks: Pre-computed skip-block count for the first written batch
            of this kernel group, derived from ``skip_first_n_tokens``.  Zero
            for subsequent batches (computed by the executor at execution time)
            and for store operations.
        slots_per_chunk: Number of physical KV slots in one LMCache chunk for
            this group (forwarded to the transfer kernel at execution time).
        shape_desc: Physical page-buffer shape for this kernel group
            (forwarded to the transfer kernel at execution time).
        dtype: Torch dtype of the KV data for this group.
        engine_kv_format: Engine KV format for this group (forwarded to the
            transfer kernel at execution time).
    """

    kernel_group_id: int
    object_group_id: int
    blocks_per_chunk: int
    blocks_per_window: int
    selected_block_ids: list[int]
    skip_blocks: int
    slots_per_chunk: int
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    dtype: torch.dtype
    engine_kv_format: "lmc_ops.EngineKVFormat"


@dataclass(frozen=True)
class ObjectGroupTransferPlan:
    """Pure-metadata transfer plan for one object group.

    Aggregates kernel-group plans and per-object-group metadata for one
    LMCache object group.

    Attributes:
        object_group_id: Index of this object group.
        object_keys: Resolved :class:`~lmcache.v1.distributed.api.ObjectKey`
            instances for the chunks in this group, in chunk order.
        layout_desc: Memory layout description for the storage object backing
            this group, describing the shape and dtype of each kernel group's
            contribution.
        kernel_groups: Per-kernel-group plans for this object group, in
            kernel-group declaration order.
        num_chunks: Total number of chunks in this transfer for this group.
        num_objects_to_skip: Number of leading objects (chunks) to skip when
            executing the transfer.  Zero for store operations and for
            full-attention retrieve operations; positive for sliding-window
            retrieve when the prefix exceeds the window size.
    """

    object_group_id: int
    object_keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    kernel_groups: list[KernelGroupTransferPlan]
    num_chunks: int
    num_objects_to_skip: int


@dataclass(frozen=True)
class TransferPlan:
    """Top-level description of one store or retrieve operation.

    Produced by :class:`TransferPlanBuilder` and consumed by the
    LMCache-driven (or engine-driven) executor without any re-planning.

    Attributes:
        direction: ``STORE`` or ``RETRIEVE``.
        request_id: External request identifier (forwarded to observability).
        chunk_size: LMCache chunk size in tokens used when building the plan.
        object_groups: Per-object-group plans, in object-group order.
        selected_block_ids_per_kernel_group: Downsampled block-ID lists
            indexed by kernel group ID, ready to pass to
            ``cache_context.stage_block_ids()``.  This is a convenience
            denormalization of the same data stored in each
            :class:`KernelGroupTransferPlan`; it avoids the need to
            reconstruct the flat list from the nested object-group structure
            at execution time.
    """

    direction: TransferDirection
    request_id: str
    chunk_size: int
    object_groups: list[ObjectGroupTransferPlan]
    selected_block_ids_per_kernel_group: list[list[int]]


# ---------------------------------------------------------------------------
# Pure planning helpers
# ---------------------------------------------------------------------------


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the blocks-to-skip count for a sliding-window kernel group.

    When the sliding-window size is smaller than the LMCache chunk size,
    only a tail portion of each chunk's blocks is stored (the window frame).
    The naive ``blocks_to_skip`` was computed against the full-chunk block
    layout; this function adjusts it to the window-frame layout produced by
    :func:`downsample_block_ids`.

    Args:
        blocks_per_chunk: Total number of paged blocks in one LMCache chunk
            for the current kernel group.
        blocks_per_window: Number of blocks in the sliding-window frame for
            the current kernel group.  Must be ``<= blocks_per_chunk``.
        blocks_to_skip: Skip count computed against the full-chunk layout.

    Returns:
        The adjusted skip count against the window-frame layout.  Equal to
        ``blocks_to_skip`` when ``blocks_per_chunk == blocks_per_window``.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


def downsample_block_ids(
    cache_context: "BaseCacheContext",
    block_ids: list[list[int]],
) -> list[list[int]]:
    """Select/downsample block IDs for sliding-window and subchunk kernel groups.

    For each kernel group, keeps only the trailing ``blocks_per_window``
    blocks from each full-chunk block slice (the window frame).
    Full-attention groups are unaffected
    (``blocks_per_window == blocks_per_chunk``).

    This is the pure-planning half of the original
    ``downsample_and_stage_block_ids`` function.  GPU staging is intentionally
    excluded and must be performed by the caller on the returned list.

    Args:
        cache_context: Cache context supplying kernel-group metadata.
        block_ids: Raw block-ID lists, one per kernel group.  Each list's
            length must be a multiple of ``blocks_per_chunk`` for its group.

    Returns:
        A new list of downsampled block-ID lists, one per kernel group, with
        the same indexing as ``block_ids``.  Groups where
        ``blocks_per_window < blocks_per_chunk`` will have shorter lists.

    Raises:
        ValueError: If any group's block-ID list length is not a multiple of
            its ``blocks_per_chunk``.
    """
    num_kernel_groups = cache_context.kv_layer_groups_manager.num_kernel_groups
    result: list[list[int]] = [list(bids) for bids in block_ids]

    for kg_id in range(num_kernel_groups):
        subchunk_sw_size_tokens = (
            cache_context.kv_layer_groups_manager.get_subchunk_sw_size_tokens(kg_id)
        )
        tokens_per_chunk = min(
            cache_context.lmcache_tokens_per_chunk, subchunk_sw_size_tokens
        )
        keep_blocks_per_chunk = cache_context.calculate_num_blocks(
            tokens_per_chunk, kg_id
        )
        total_blocks_per_chunk = cache_context.calculate_num_blocks(
            cache_context.lmcache_tokens_per_chunk, kg_id
        )

        old_block_ids = result[kg_id]
        if len(old_block_ids) % total_blocks_per_chunk != 0:
            raise ValueError(
                f"len(block_ids[{kg_id}]) should be a multiple "
                f"of total_blocks_per_chunk ({total_blocks_per_chunk}), but got "
                f"{len(old_block_ids)}"
            )

        new_block_ids: list[int] = []
        for i in range(0, len(old_block_ids), total_blocks_per_chunk):
            chunk_block_ids = old_block_ids[i : i + total_blocks_per_chunk]
            new_block_ids.extend(chunk_block_ids[-keep_blocks_per_chunk:])
        result[kg_id] = new_block_ids

    return result


def validate_block_ids(
    block_ids: list[list[int]],
    blocks_per_chunk_per_group: list[int],
    num_chunks: int,
) -> bool:
    """Validate that block IDs cover all chunks for every kernel group.

    Checks the raw (pre-downsampling) block-ID lists: each group must supply
    at least ``num_chunks * blocks_per_chunk`` block IDs.  A shorter list
    would drive the transfer kernel to read or write out-of-bounds GPU memory,
    so the caller should abort the transfer when this returns ``False``.

    Args:
        block_ids: Raw block-ID lists, one per kernel group.
        blocks_per_chunk_per_group: ``blocks_per_chunk`` for each kernel group,
            in the same order as ``block_ids``.
        num_chunks: Number of LMCache chunks expected in this transfer.

    Returns:
        ``True`` if every group has sufficient block IDs; ``False`` if any
        group is under-covered (block-ID underflow).
    """
    return not any(
        len(group_block_ids) < num_chunks * bpc
        for group_block_ids, bpc in zip(
            block_ids, blocks_per_chunk_per_group, strict=True
        )
    )


def build_object_group_layout_desc(
    cache_context: "BaseCacheContext",
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Build the memory layout descriptor for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one ``(shape, dtype)`` entry per kernel group in
    that object group, in the kernel groups' declared layout order.  Kernel
    groups may carry different shapes and dtypes.

    Args:
        cache_context: Cache context providing kernel-group metadata.
        num_tokens: Number of tokens to use when computing tensor shapes.
            Typically the LMCache chunk size
            (``cache_context.lmcache_tokens_per_chunk``).
        object_group_id: Index of the object group to describe.

    Returns:
        A :class:`~lmcache.v1.distributed.api.MemoryLayoutDesc` with one
        shape/dtype entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kg_idx)
        for kg_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TransferPlanBuilder:
    """Build path-agnostic transfer plans from KV cache metadata.

    Assembles :class:`TransferPlan` instances that fully describe *what*
    needs to be transferred, without encoding *how* the execution should
    happen.  The builder is CUDA-free: it uses only pure metadata from the
    :class:`~lmcache.v1.platform.base_cache_context.BaseCacheContext` (group
    geometry, dtype, slot counts) and never touches streams, events, SHM
    slots, or any copy operations.

    Args:
        cache_context: Platform cache context supplying all kernel-group and
            object-group metadata.
    """

    def __init__(self, cache_context: "BaseCacheContext") -> None:
        self._cache_context = cache_context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_blocks_per_chunk_all_groups(self) -> list[int]:
        """Return ``blocks_per_chunk`` for every kernel group.

        Returns:
            A list of block counts indexed by kernel group ID.
        """
        cache_context = self._cache_context
        chunk_size = cache_context.lmcache_tokens_per_chunk
        num_kg = cache_context.kv_layer_groups_manager.num_kernel_groups
        return [
            cache_context.calculate_num_blocks(chunk_size, kg_id)
            for kg_id in range(num_kg)
        ]

    def _build_kernel_group_plan(
        self,
        kernel_group_id: int,
        object_group_id: int,
        downsampled_block_ids: list[int],
        skip_first_n_tokens: int,
        first_batch_start_token: int,
    ) -> KernelGroupTransferPlan:
        """Build a :class:`KernelGroupTransferPlan` for one kernel group.

        Args:
            kernel_group_id: Index of the kernel group.
            object_group_id: Index of the object group this kernel group
                belongs to.
            downsampled_block_ids: Already-downsampled block IDs for this
                group (from :func:`downsample_block_ids`).
            skip_first_n_tokens: Tokens to skip at the start of the retrieve
                range.  Zero for store operations.
            first_batch_start_token: The token offset of the first processed
                batch (``num_objects_to_skip * chunk_size``), used to compute
                ``skip_blocks`` for the first written batch.

        Returns:
            A :class:`KernelGroupTransferPlan` with all planning metadata.
        """
        cache_context = self._cache_context
        chunk_size = cache_context.lmcache_tokens_per_chunk
        kv_groups_manager = cache_context.kv_layer_groups_manager

        blocks_per_chunk = cache_context.calculate_num_blocks(
            chunk_size, kernel_group_id
        )
        tokens_per_window = min(
            chunk_size,
            kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
        )
        blocks_per_window = cache_context.calculate_num_blocks(
            tokens_per_window, kernel_group_id
        )

        # Pre-compute skip_blocks for the first written batch only.
        # Subsequent batches always have skip_blocks == 0, and the executor
        # is responsible for applying that rule at runtime.
        skip_tokens_in_first_chunk = max(
            0, skip_first_n_tokens - first_batch_start_token
        )
        orig_skip_blocks = cache_context.calculate_num_blocks(
            skip_tokens_in_first_chunk, kernel_group_id
        )
        skip_blocks = recalculate_blocks_to_skip(
            blocks_per_chunk, blocks_per_window, orig_skip_blocks
        )

        _, dtype = cache_context.get_kernel_group_shape_dtype(
            chunk_size, kernel_group_id
        )

        return KernelGroupTransferPlan(
            kernel_group_id=kernel_group_id,
            object_group_id=object_group_id,
            blocks_per_chunk=blocks_per_chunk,
            blocks_per_window=blocks_per_window,
            selected_block_ids=downsampled_block_ids,
            skip_blocks=skip_blocks,
            slots_per_chunk=cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
            shape_desc=cache_context.get_shape_desc(kernel_group_id),
            dtype=dtype,
            engine_kv_format=cache_context.get_engine_kv_format(kernel_group_id),
        )

    def _build_object_group_plan(
        self,
        object_group_id: int,
        object_keys: list[ObjectKey],
        downsampled_block_ids: list[list[int]],
        skip_first_n_tokens: int,
        is_retrieve: bool,
    ) -> ObjectGroupTransferPlan:
        """Build an :class:`ObjectGroupTransferPlan` for one object group.

        Args:
            object_group_id: Index of the object group.
            object_keys: Resolved object keys for the chunks in this group,
                in chunk order.
            downsampled_block_ids: Downsampled block IDs indexed by kernel
                group ID (from :func:`downsample_block_ids`).
            skip_first_n_tokens: Tokens to skip at the start of the range.
                Zero for store operations.
            is_retrieve: ``True`` for retrieve operations (enables sliding-
                window skip calculation); ``False`` for store.

        Returns:
            A fully populated :class:`ObjectGroupTransferPlan`.
        """
        cache_context = self._cache_context
        kv_groups_manager = cache_context.kv_layer_groups_manager
        object_group = kv_groups_manager.object_groups[object_group_id]
        num_chunks = len(object_keys)

        # Compute objects to skip for sliding-window retrieve groups.
        num_objects_to_skip = 0
        if is_retrieve:
            attn_desc = kv_groups_manager.get_attn_desc()
            if not attn_desc.is_full_attention(object_group_id):
                sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
                num_objects_to_skip = max(0, num_chunks - sw_size_chunks)

        first_batch_start_token = (
            num_objects_to_skip * cache_context.lmcache_tokens_per_chunk
        )

        layout_desc = build_object_group_layout_desc(
            cache_context, cache_context.lmcache_tokens_per_chunk, object_group_id
        )

        kernel_groups: list[KernelGroupTransferPlan] = []
        for kg_id in object_group.kernel_group_indices:
            kg_plan = self._build_kernel_group_plan(
                kernel_group_id=kg_id,
                object_group_id=object_group_id,
                downsampled_block_ids=downsampled_block_ids[kg_id],
                skip_first_n_tokens=skip_first_n_tokens,
                first_batch_start_token=first_batch_start_token,
            )
            kernel_groups.append(kg_plan)

        return ObjectGroupTransferPlan(
            object_group_id=object_group_id,
            object_keys=object_keys,
            layout_desc=layout_desc,
            kernel_groups=kernel_groups,
            num_chunks=num_chunks,
            num_objects_to_skip=num_objects_to_skip,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_store_plan(
        self,
        request_id: str,
        obj_keys_per_obj_group: list[list[ObjectKey]],
        block_ids: list[list[int]],
    ) -> TransferPlan | None:
        """Build a transfer plan for a store (D2H) operation.

        Validates that ``block_ids`` cover all chunks for every kernel group,
        downsamples block IDs for sliding-window/subchunk groups, and
        assembles the full :class:`TransferPlan`.

        Args:
            request_id: External request identifier.
            obj_keys_per_obj_group: Resolved object keys indexed by object
                group.  ``obj_keys_per_obj_group[g]`` lists the
                :class:`~lmcache.v1.distributed.api.ObjectKey` instances for
                object group ``g``.
            block_ids: Raw (pre-downsampling) block-ID lists indexed by
                kernel group ID.

        Returns:
            A :class:`TransferPlan` on success, or ``None`` if any kernel
            group's block-ID list is too short to cover all chunks (block-ID
            underflow).
        """
        cache_context = self._cache_context
        num_chunks = len(obj_keys_per_obj_group[0])

        blocks_per_chunk = self._compute_blocks_per_chunk_all_groups()
        if not validate_block_ids(block_ids, blocks_per_chunk, num_chunks):
            return None

        downsampled = downsample_block_ids(cache_context, list(block_ids))

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        object_groups: list[ObjectGroupTransferPlan] = []
        for og_id in range(num_object_groups):
            og_plan = self._build_object_group_plan(
                object_group_id=og_id,
                object_keys=obj_keys_per_obj_group[og_id],
                downsampled_block_ids=downsampled,
                skip_first_n_tokens=0,
                is_retrieve=False,
            )
            object_groups.append(og_plan)

        return TransferPlan(
            direction=TransferDirection.STORE,
            request_id=request_id,
            chunk_size=cache_context.lmcache_tokens_per_chunk,
            object_groups=object_groups,
            selected_block_ids_per_kernel_group=downsampled,
        )

    def build_retrieve_plan(
        self,
        request_id: str,
        obj_keys_per_obj_group: list[list[ObjectKey]],
        block_ids: list[list[int]],
        skip_first_n_tokens: int = 0,
    ) -> TransferPlan | None:
        """Build a transfer plan for a retrieve (H2D) operation.

        Validates that ``block_ids`` cover all chunks for every kernel group,
        downsamples block IDs for sliding-window/subchunk groups, and
        assembles the full :class:`TransferPlan`.

        Args:
            request_id: External request identifier.
            obj_keys_per_obj_group: Resolved object keys indexed by object
                group.  ``obj_keys_per_obj_group[g]`` lists the
                :class:`~lmcache.v1.distributed.api.ObjectKey` instances for
                object group ``g``.
            block_ids: Raw (pre-downsampling) block-ID lists indexed by
                kernel group ID.
            skip_first_n_tokens: Number of tokens to skip writing at the
                start of the retrieve range.  Avoids overwriting APC-shared
                blocks that may be concurrently read.  Defaults to 0.

        Returns:
            A :class:`TransferPlan` on success, or ``None`` if any kernel
            group's block-ID list is too short to cover all chunks (block-ID
            underflow).
        """
        cache_context = self._cache_context
        num_chunks = len(obj_keys_per_obj_group[0])

        blocks_per_chunk = self._compute_blocks_per_chunk_all_groups()
        if not validate_block_ids(block_ids, blocks_per_chunk, num_chunks):
            return None

        downsampled = downsample_block_ids(cache_context, list(block_ids))

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        object_groups: list[ObjectGroupTransferPlan] = []
        for og_id in range(num_object_groups):
            og_plan = self._build_object_group_plan(
                object_group_id=og_id,
                object_keys=obj_keys_per_obj_group[og_id],
                downsampled_block_ids=downsampled,
                skip_first_n_tokens=skip_first_n_tokens,
                is_retrieve=True,
            )
            object_groups.append(og_plan)

        return TransferPlan(
            direction=TransferDirection.RETRIEVE,
            request_id=request_id,
            chunk_size=cache_context.lmcache_tokens_per_chunk,
            object_groups=object_groups,
            selected_block_ids_per_kernel_group=downsampled,
        )
