# SPDX-License-Identifier: Apache-2.0
"""Common helpers for MP object-group KV cache transfer preparation and execution.

This module provides reusable logic shared between LMCache-driven and
(in a later step) engine-driven transfer paths:

* Pure geometry helpers: :func:`select_block_ids_for_window`,
  :func:`recalculate_blocks_to_skip`, :func:`batched_iteration_with_skip`,
  :func:`compute_num_objects_to_skip`.
* Transfer-plan builder: :func:`prepare_object_group_transfer` assembles
  the ``KernelGroupSpec`` and ``BatchStep`` lists that
  ``lmc_ops.execute_object_group_transfer`` consumes.
* Thin executor wrapper: :func:`execute_prepared_object_group_transfer`
  calls ``lmc_ops.execute_object_group_transfer`` and is a no-op when the
  batch-step list is empty.

Ownership contract
------------------
Source/destination buffer ownership is **path-specific**.  The common helpers
receive already-prepared :class:`~lmcache.v1.memory_management.MemoryObj`
instances from the caller; they do not allocate or release storage.
"""

# Standard
from itertools import islice
from typing import TYPE_CHECKING, Generator, Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_ops import build_staging_copies
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.base_cache_context import BaseCacheContext
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


def select_block_ids_for_window(
    block_ids: list[int],
    total_blocks_per_chunk: int,
    keep_blocks_per_chunk: int,
) -> list[int]:
    """Return the block IDs needed for a sliding/sub-chunk window transfer.

    For kernel groups whose sliding-window size is smaller than the LMCache
    chunk size, only the *trailing* ``keep_blocks_per_chunk`` blocks of each
    chunk are transferred.  This function performs that selection without any
    GPU or cache-context dependency.

    Args:
        block_ids: The original flat block-ID list for one kernel group,
            covering one or more complete chunks.
        total_blocks_per_chunk: Total number of blocks per LMCache chunk for
            this kernel group (must be >= 1).
        keep_blocks_per_chunk: Number of trailing blocks per chunk to keep
            (must be >= 1 and <= ``total_blocks_per_chunk``).

    Returns:
        A flat list of selected block IDs, one trailing slice per chunk.

    Raises:
        ValueError: If ``total_blocks_per_chunk`` < 1 or
            ``keep_blocks_per_chunk`` is outside ``[1, total_blocks_per_chunk]``,
            or if ``len(block_ids)`` is not a multiple of
            ``total_blocks_per_chunk``.

    Example:
        With ``total_blocks_per_chunk=4`` and ``keep_blocks_per_chunk=2``:

        >>> select_block_ids_for_window([10, 11, 12, 13, 20, 21, 22, 23], 4, 2)
        [12, 13, 22, 23]
    """
    if total_blocks_per_chunk < 1:
        raise ValueError(
            f"total_blocks_per_chunk must be >= 1, got {total_blocks_per_chunk}"
        )
    if keep_blocks_per_chunk < 1 or keep_blocks_per_chunk > total_blocks_per_chunk:
        raise ValueError(
            f"keep_blocks_per_chunk must be in [1, {total_blocks_per_chunk}], "
            f"got {keep_blocks_per_chunk}"
        )
    if len(block_ids) % total_blocks_per_chunk != 0:
        raise ValueError(
            f"len(block_ids) ({len(block_ids)}) must be a multiple of "
            f"total_blocks_per_chunk ({total_blocks_per_chunk})"
        )

    result: list[int] = []
    for i in range(0, len(block_ids), total_blocks_per_chunk):
        chunk = block_ids[i : i + total_blocks_per_chunk]
        result.extend(chunk[-keep_blocks_per_chunk:])
    return result


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip when the window is a sub-chunk.

    When a kernel group's sliding-window size is smaller than the LMCache
    chunk size (i.e. ``blocks_per_window < blocks_per_chunk``), the block-ID
    list passed to the transfer kernel only contains the trailing
    ``blocks_per_window`` entries per chunk.  A raw ``blocks_to_skip`` that was
    originally computed in terms of full-chunk blocks must therefore be
    re-expressed in terms of window-sized blocks.

    If ``blocks_per_chunk == blocks_per_window`` the input is returned
    unchanged.

    Args:
        blocks_per_chunk: Total blocks per LMCache chunk for this kernel group.
        blocks_per_window: Blocks per sliding-window step for this kernel group.
            Must be <= ``blocks_per_chunk``.
        blocks_to_skip: Number of blocks to skip, expressed in full-chunk block
            space.

    Returns:
        The equivalent skip count in window-block space.

    Example:
        With ``blocks_per_chunk=4``, ``blocks_per_window=2``,
        ``blocks_to_skip=6`` (1.5 full chunks):

        * full windows to skip: 6 // 4 = 1
        * tail in original space: 6 % 4 = 2
        * tail offset within window: 2 − (4 − 2) = 0
        * result: 1 * 2 + max(0, 0) = 2
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


def batched_iteration_with_skip(
    sequence: Sequence,
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple], None, None]:
    """Iterate over a sequence in batches, skipping an initial prefix.

    Args:
        sequence: The sequence to iterate over.
        batch_size: The size of each batch.
        skip_count: The number of items to skip at the start of the list.

    Yields:
        Tuples of (batch_start_idx, batch) where batch is a tuple of items
        from the sequence, and batch_start_idx is the original index of the
        first item in the batch (accounting for ``skip_count``).

    Raises:
        ValueError: If ``batch_size`` is less than 1 or ``skip_count`` is
            negative.

    Note:
        ``batch_start_idx`` is the index of the first item *in the original
        list*, not in the post-skip tail.  For example, with
        ``skip_count=10`` and ``batch_size=5``, the first yielded
        ``batch_start_idx`` will be 10.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    it = iter(sequence)
    for _ in range(skip_count):
        next(it, None)
    batch_start_idx = skip_count
    while batch := tuple(islice(it, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def compute_num_objects_to_skip(
    attn_desc: "AttnWindowDesc",
    object_group_id: int,
    num_objects: int,
    is_h2d: bool,
) -> int:
    """Return how many leading objects to skip for a sliding-window group.

    For H2D (retrieve), a sliding-window group only needs the most recent
    ``sw_size_chunks`` objects.  Objects before that window are already present
    in the GPU KV cache and must not be overwritten.

    For D2H (store), or for full-attention groups, no objects are skipped.

    Args:
        attn_desc: Attention window descriptor for the current layout.
        object_group_id: Index of the object group being transferred.
        num_objects: Total number of objects (chunks) in the transfer.
        is_h2d: True for retrieve (H2D), False for store (D2H).

    Returns:
        Number of leading objects to skip (0 if full-attention or D2H).
    """
    if not is_h2d:
        return 0
    if attn_desc.is_full_attention(object_group_id):
        return 0
    sw_size_chunks = attn_desc.num_chunks_in_sw[object_group_id]
    return max(0, num_objects - sw_size_chunks)


# ---------------------------------------------------------------------------
# Transfer-plan builder
# ---------------------------------------------------------------------------


def prepare_object_group_transfer(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmc_ops.TransferDirection",
) -> tuple[list["lmc_ops.KernelGroupSpec"], list["lmc_ops.BatchStep"]]:
    """Build the ``KernelGroupSpec`` and ``BatchStep`` plan for one object group.

    This is the pure-planning counterpart of the execute path: it resolves
    every argument to plain pointers and scalars (the "planner", GIL held
    throughout) and assembles the two data structures that
    :func:`execute_prepared_object_group_transfer` passes directly to
    ``lmc_ops.execute_object_group_transfer``.

    Callers are responsible for providing ``memory_objs``; this function does
    not allocate or release storage.

    Args:
        cache_context: GPU cache context for the registered worker instance.
        block_ids_gpu: GPU block-ID tensors, indexed by LMCache KV group index.
            Produced by
            :func:`~lmcache.v1.multiprocess.modules.lmcache_driven_transfer.downsample_and_stage_block_ids`.
        memory_objs: Memory objects to transfer, one per chunk.  ``None``
            entries are allowed only for D2H (the batch is skipped); H2D with
            a ``None`` entry raises :class:`ValueError`.
        object_group_id: Index of the object group being transferred.
        batch_size: Number of memory objects per batched copy step.
        skip_first_n_tokens: Tokens to skip at the start of the retrieve range
            (APC-shared prefix protection).
        direction: H2D (retrieve) or D2H (store).

    Returns:
        A ``(kernel_group_specs, batch_steps)`` tuple ready for
        :func:`execute_prepared_object_group_transfer`.

    Raises:
        ValueError: If a ``None`` entry is found in a batch when
            ``direction`` is H2D.
    """
    lmcache_chunk_size = cache_context.lmcache_tokens_per_chunk
    kv_groups_manager = cache_context.kv_layer_groups_manager
    object_group = kv_groups_manager.object_groups[object_group_id]
    kernel_group_ids = object_group.kernel_group_indices
    is_h2d = direction == lmc_ops.TransferDirection.H2D
    max_batch_size = cache_context.max_batch_size

    # --- Per-kernel-group invariants, resolved once ---
    kernel_group_specs: list["lmc_ops.KernelGroupSpec"] = []
    spec_index_by_kg: dict[int, int] = {}
    blocks_per_chunk_by_kg: dict[int, int] = {}
    blocks_per_window_by_kg: dict[int, int] = {}
    for kernel_group_id in kernel_group_ids:
        blocks_per_chunk = cache_context.calculate_num_blocks(
            lmcache_chunk_size, kernel_group_id
        )
        tokens_per_window = min(
            lmcache_chunk_size,
            kv_groups_manager.get_subchunk_sw_size_tokens(kernel_group_id),
        )
        blocks_per_window = cache_context.calculate_num_blocks(
            tokens_per_window, kernel_group_id
        )
        blocks_per_chunk_by_kg[kernel_group_id] = blocks_per_chunk
        blocks_per_window_by_kg[kernel_group_id] = blocks_per_window

        paged_ptrs = cache_context.get_kernel_group_kv_pointers(kernel_group_id)
        block_ids_tensor = block_ids_gpu[kernel_group_id]
        temp_buffers = [
            cache_context.get_temp_kernel_group_buffer(slot, kernel_group_id)
            for slot in range(max_batch_size)
        ]

        spec_index_by_kg[kernel_group_id] = len(kernel_group_specs)
        kernel_group_specs.append(
            lmc_ops.KernelGroupSpec(
                paged_ptrs.data_ptr(),
                [buffer.data_ptr() for buffer in temp_buffers],
                cache_context.get_shape_desc(kernel_group_id),
                cache_context.get_slots_per_chunk_in_sw(kernel_group_id),
                cache_context.get_engine_kv_format(kernel_group_id),
                block_ids_tensor.data_ptr(),
                block_ids_tensor.numel(),
            )
        )

    # Temp object-group staging buffers (reused per batch slot).
    object_group_buffers = [
        cache_context.get_temp_object_group_buffer(slot, object_group_id)
        for slot in range(max_batch_size)
    ]

    # Compute how many leading objects to skip for sliding-window retrieval.
    attn_desc = kv_groups_manager.get_attn_desc()
    num_objects_to_skip = compute_num_objects_to_skip(
        attn_desc, object_group_id, len(memory_objs), is_h2d
    )
    if num_objects_to_skip > 0:
        logger.debug(
            "Detected sliding window for object group %d: "
            "skipping the first %d objects in the batch",
            object_group_id,
            num_objects_to_skip,
        )

    # --- Walk the batches, building staging + launch descriptors ---
    batch_steps: list["lmc_ops.BatchStep"] = []
    for start_object_idx, memory_object_batch in batched_iteration_with_skip(
        memory_objs, batch_size, skip_count=num_objects_to_skip
    ):
        if any(mo is None for mo in memory_object_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    "perform H2D copy. memory_object_batch: "
                    f"{memory_object_batch}"
                )
            else:
                continue

        batch_len = len(memory_object_batch)
        batch_start_token = start_object_idx * lmcache_chunk_size
        batch_end_token = batch_start_token + batch_len * lmcache_chunk_size

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        staging = build_staging_copies(
            memory_object_batch,
            object_group_buffers[:batch_len],
            is_h2d,
        )

        launches: list["lmc_ops.LaunchVar"] = []
        for kernel_group_id in kernel_group_ids:
            blocks_per_chunk = blocks_per_chunk_by_kg[kernel_group_id]
            blocks_per_window = blocks_per_window_by_kg[kernel_group_id]

            start_block_pos = start_object_idx * blocks_per_window
            end_block_pos = (start_object_idx + batch_len) * blocks_per_window

            orig_skip_blocks = cache_context.calculate_num_blocks(
                skip_tokens_in_chunk, kernel_group_id
            )
            adjusted_skip_blocks = recalculate_blocks_to_skip(
                blocks_per_chunk,
                blocks_per_window,
                orig_skip_blocks,
            )

            launches.append(
                lmc_ops.LaunchVar(
                    spec_index_by_kg[kernel_group_id],
                    start_block_pos,
                    end_block_pos - start_block_pos,
                    batch_len,
                    adjusted_skip_blocks,
                )
            )

        batch_steps.append(lmc_ops.BatchStep(staging, launches))

    return kernel_group_specs, batch_steps


# ---------------------------------------------------------------------------
# Thin executor wrapper
# ---------------------------------------------------------------------------


def execute_prepared_object_group_transfer(
    direction: "lmc_ops.TransferDirection",
    device: object,
    kernel_group_specs: list["lmc_ops.KernelGroupSpec"],
    batch_steps: list["lmc_ops.BatchStep"],
) -> None:
    """Execute a prepared object-group transfer plan via the native backend.

    This is a thin wrapper around ``lmc_ops.execute_object_group_transfer``
    that no-ops when ``batch_steps`` is empty.

    Args:
        direction: H2D (retrieve) or D2H (store).
        device: CUDA device on which the transfer runs (forwarded unchanged).
        kernel_group_specs: Per-kernel-group layout and pointer descriptors,
            as returned by :func:`prepare_object_group_transfer`.
        batch_steps: Per-batch staging and launch descriptors, as returned by
            :func:`prepare_object_group_transfer`.

    Returns:
        None.
    """
    if not batch_steps:
        return

    lmc_ops.execute_object_group_transfer(
        direction,
        device,
        LazyMemoryAllocator.PIN_CHUNK_SIZE,
        kernel_group_specs,
        batch_steps,
    )
