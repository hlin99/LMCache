# SPDX-License-Identifier: Apache-2.0
"""Path-agnostic MP transfer planning for LMCache KV cache transfers.

This module provides :class:`TransferPlanBuilder` and the associated plan
dataclasses.  The builder decides **what** needs to be copied — which groups,
chunks, block IDs, layouts, and skip offsets — without knowing *how* the copy
will be executed (CUDA IPC, SHM, pickle, etc.).

Design contract:
* ``TransferPlanBuilder`` must not import or reference any CUDA / IPC / stream /
  SHM / pickle execution semantics.
* Executors (:class:`LMCacheDrivenTransferModule`,
  :class:`EngineDrivenTransferModule`) own all transport-layer details and
  consume the plan produced here.

Acceptable contents: MemoryLayoutDesc, torch.dtype, torch.Size,
PageBufferShapeDesc, object keys, block IDs, group IDs, chunk counts,
skip counts, and KVLayerGroupsManager.

Prohibited contents: torch.device/stream/Event, IPC handles, cupy streams,
DeviceHostFuncDispatcher, lmcache_memcpy_async_*, lmc_ops transfer calls.
"""

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


class TransferDirection(str, Enum):
    """Direction of the KV cache transfer.

    This enum is path-agnostic.  Executors map it to their own direction
    constants (e.g. ``lmc_ops.TransferDirection``) internally.
    """

    STORE = "store"
    """GPU→storage (D2H on the lmcache-driven path)."""
    RETRIEVE = "retrieve"
    """Storage→GPU (H2D on the lmcache-driven path)."""


# ---------------------------------------------------------------------------
# Helper: recalculate skip blocks for subchunk sliding-window groups
# ---------------------------------------------------------------------------


def recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip when the SWA window is
    smaller than one LMCache chunk.

    When ``blocks_per_window < blocks_per_chunk``, only the trailing
    ``blocks_per_window`` blocks of each chunk are stored/retrieved.  The
    caller's raw ``blocks_to_skip`` was computed against the *full* chunk
    width; this helper converts it to the reduced-width coordinate system.

    Args:
        blocks_per_chunk: Total blocks in one LMCache chunk for this group.
        blocks_per_window: Blocks in the SWA window (≤ ``blocks_per_chunk``).
        blocks_to_skip: Raw number of blocks to skip (full-chunk coordinates).

    Returns:
        Recalculated skip count in the downsampled (window) coordinate system.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelGroupPlan:
    """Geometric transfer plan for a single kernel group.

    Attributes:
        kernel_group_id: Index of this kernel group in the
            :class:`KVLayerGroupsManager`.
        selected_block_ids: Downsampled block-ID list ready for staging.
            For full-attention groups this equals the trimmed input; for
            subchunk-SWA groups the trailing ``blocks_per_window`` blocks
            of each chunk are kept and the rest are dropped.
        blocks_per_chunk: Blocks per LMCache chunk *before* downsampling
            (i.e. the total blocks in one chunk for this group).
        blocks_per_window: Blocks kept per chunk after SWA downsampling
            (``≤ blocks_per_chunk``).  Equals ``blocks_per_chunk`` for
            full-attention groups.
    """

    kernel_group_id: int
    selected_block_ids: list[int]
    blocks_per_chunk: int
    blocks_per_window: int


@dataclass(frozen=True)
class ObjectGroupPlan:
    """Transfer plan for one object group.

    Attributes:
        object_group_id: Index of this object group.
        object_keys: Resolved :class:`ObjectKey` list; one per chunk.
        layout_desc: Memory-layout descriptor for storage reservation.
        num_objects_to_skip: Number of leading chunks to skip on H2D
            (retrieve) due to the object group's SWA window.  Always ``0``
            for store operations and for full-attention groups.
        kernel_group_ids: Ordered kernel-group IDs that belong to this
            object group (corresponds to
            ``KVLayerGroupsManager.object_groups[i].kernel_group_indices``).
    """

    object_group_id: int
    object_keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    num_objects_to_skip: int
    kernel_group_ids: list[int]


@dataclass(frozen=True)
class TransferPlan:
    """Path-agnostic description of a KV cache transfer operation.

    Executors consume this to drive their transport-specific copy
    primitives without re-deriving transfer geometry.

    Attributes:
        direction: Whether this is a store or retrieve operation.
        chunk_size: LMCache tokens-per-chunk used to build this plan.
        num_chunks: Number of chunks (from the first object group's key count).
        kernel_groups: Per-kernel-group plan in kernel-group-ID order.
        object_groups: Per-object-group plan in object-group-ID order.
        underflow: ``True`` when the supplied block IDs do not cover all
            chunks for at least one group.  Executors should treat this as
            a fail-closed signal and skip the transfer entirely.
    """

    direction: TransferDirection
    chunk_size: int
    num_chunks: int
    kernel_groups: list[KernelGroupPlan] = field(default_factory=list)
    object_groups: list[ObjectGroupPlan] = field(default_factory=list)
    underflow: bool = False

    def selected_block_ids_by_group(self) -> list[list[int]]:
        """Return selected block IDs indexed by kernel-group ID.

        Returns:
            A list whose ``i``-th element is the selected block IDs for
            kernel group ``i``, in kernel-group-ID order.
        """
        return [kg.selected_block_ids for kg in self.kernel_groups]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TransferPlanBuilder:
    """Path-agnostic builder for :class:`TransferPlan`.

    This class extracts the planning logic shared by the LMCache-driven and
    engine-driven transfer paths.  It is intentionally free of CUDA / IPC /
    SHM / pickle execution semantics; all GPU and transport concerns belong
    in the respective executor modules.

    The builder is a pure static utility; no instances need to be created.
    """

    @staticmethod
    def build(
        kv_groups_manager: "KVLayerGroupsManager",
        lmcache_tokens_per_chunk: int,
        block_ids: list[list[int]],
        obj_keys_per_obj_group: list[list[ObjectKey]],
        direction: TransferDirection,
        kernel_group_shape_dtypes: list[tuple[torch.Size, torch.dtype]],
        skip_first_n_tokens: int = 0,
    ) -> TransferPlan:
        """Build a :class:`TransferPlan` from layout and block-ID inputs.

        Performs:

        * Per-kernel-group ``blocks_per_chunk`` / ``blocks_per_window``
          derivation.
        * Block-ID coverage (underflow) validation — fail-closed: if any
          group's block IDs do not cover all chunks, ``TransferPlan.underflow``
          is set and the object-group list is empty.
        * Block-ID downsampling (subchunk-SWA groups drop leading blocks per
          chunk, keeping only the trailing ``blocks_per_window`` blocks).
        * Per-object-group ``MemoryLayoutDesc`` construction from
          ``kernel_group_shape_dtypes``.
        * Per-object-group ``num_objects_to_skip`` for SWA retrieve windows.

        **CUDA-free guarantee**: this method does not touch ``torch.device``,
        ``torch.cuda.Stream``, ``torch.Event``, IPC handles, or any copy
        kernel.  Callers are responsible for staging the resulting
        ``selected_block_ids`` to GPU and executing the copy.

        Args:
            kv_groups_manager: KV layer group manager describing object groups
                and kernel groups.
            lmcache_tokens_per_chunk: LMCache chunk size in tokens.
            block_ids: Raw block IDs indexed by kernel-group ID; one inner
                list per kernel group.  ``block_ids[i]`` must contain at least
                ``num_chunks * blocks_per_chunk[i]`` entries for the plan to be
                valid.
            obj_keys_per_obj_group: Resolved object keys; one inner list per
                object group.  ``obj_keys_per_obj_group[g]`` has one
                :class:`ObjectKey` per chunk.
            direction: Whether this is a store or retrieve operation.
            kernel_group_shape_dtypes: Per-kernel-group (shape, dtype) pairs
                in kernel-group-ID order.  Used to build
                :attr:`ObjectGroupPlan.layout_desc`.
            skip_first_n_tokens: Tokens to skip at the start of the retrieve
                range (APC shared-block protection).  Only relevant for
                retrieve; ignored for store.

        Returns:
            A :class:`TransferPlan`.  If ``underflow`` is ``True`` the callers
            should fail-close and skip the transfer.

        Raises:
            ValueError: If the lengths of ``block_ids``,
                ``kernel_group_shape_dtypes``, or ``obj_keys_per_obj_group``
                are inconsistent with the group manager.
        """
        num_kernel_groups = kv_groups_manager.num_kernel_groups
        num_object_groups = kv_groups_manager.num_object_groups

        if len(block_ids) != num_kernel_groups:
            raise ValueError(
                f"block_ids has {len(block_ids)} entries but "
                f"kv_groups_manager has {num_kernel_groups} kernel groups"
            )
        if len(kernel_group_shape_dtypes) != num_kernel_groups:
            raise ValueError(
                f"kernel_group_shape_dtypes has {len(kernel_group_shape_dtypes)} "
                f"entries but kv_groups_manager has {num_kernel_groups} kernel groups"
            )
        if len(obj_keys_per_obj_group) != num_object_groups:
            raise ValueError(
                f"obj_keys_per_obj_group has {len(obj_keys_per_obj_group)} entries "
                f"but kv_groups_manager has {num_object_groups} object groups"
            )

        # num_chunks is consistent across object groups (each OG has one key per chunk).
        num_chunks = len(obj_keys_per_obj_group[0]) if num_object_groups > 0 else 0

        # ------------------------------------------------------------------
        # Per-kernel-group geometry
        # ------------------------------------------------------------------
        kg_plans: list[KernelGroupPlan] = []
        blocks_per_chunk_by_kg: list[int] = []
        blocks_per_window_by_kg: list[int] = []

        for kg_id in range(num_kernel_groups):
            bpc = kv_groups_manager.calculate_num_blocks(
                kg_id, lmcache_tokens_per_chunk
            )
            tokens_per_window = min(
                lmcache_tokens_per_chunk,
                kv_groups_manager.get_subchunk_sw_size_tokens(kg_id),
            )
            bpw = kv_groups_manager.calculate_num_blocks(kg_id, tokens_per_window)
            blocks_per_chunk_by_kg.append(bpc)
            blocks_per_window_by_kg.append(bpw)

        # ------------------------------------------------------------------
        # Block-ID underflow validation (fail-closed)
        # ------------------------------------------------------------------
        if any(
            len(block_ids[kg_id]) < num_chunks * blocks_per_chunk_by_kg[kg_id]
            for kg_id in range(num_kernel_groups)
        ):
            logger.warning(
                "Transfer plan underflow for direction=%s: "
                "block IDs do not cover all %d chunks "
                "(per-group blocks_per_chunk=%s); failing closed.",
                direction.value,
                num_chunks,
                blocks_per_chunk_by_kg,
            )
            return TransferPlan(
                direction=direction,
                chunk_size=lmcache_tokens_per_chunk,
                num_chunks=num_chunks,
                kernel_groups=[],
                object_groups=[],
                underflow=True,
            )

        # ------------------------------------------------------------------
        # Per-kernel-group block-ID downsampling (subchunk-SWA groups)
        # ------------------------------------------------------------------
        for kg_id in range(num_kernel_groups):
            bpc = blocks_per_chunk_by_kg[kg_id]
            bpw = blocks_per_window_by_kg[kg_id]
            old_ids = block_ids[kg_id]

            if bpc == bpw:
                # Full-attention or standard group: keep all blocks.
                selected = list(old_ids)
            else:
                # Subchunk-SWA: keep only the last bpw blocks per chunk.
                selected = []
                for chunk_start in range(0, len(old_ids), bpc):
                    chunk_ids = old_ids[chunk_start : chunk_start + bpc]
                    selected.extend(chunk_ids[-bpw:])

            kg_plans.append(
                KernelGroupPlan(
                    kernel_group_id=kg_id,
                    selected_block_ids=selected,
                    blocks_per_chunk=bpc,
                    blocks_per_window=bpw,
                )
            )

        # ------------------------------------------------------------------
        # Per-object-group plans
        # ------------------------------------------------------------------
        attn_desc = kv_groups_manager.get_attn_desc()
        is_retrieve = direction is TransferDirection.RETRIEVE
        og_plans: list[ObjectGroupPlan] = []

        for og_id in range(num_object_groups):
            object_group = kv_groups_manager.object_groups[og_id]
            kg_ids_in_og = object_group.kernel_group_indices

            # MemoryLayoutDesc from kernel-group shapes/dtypes.
            shapes_and_dtypes = [
                kernel_group_shape_dtypes[kg_id] for kg_id in kg_ids_in_og
            ]
            shapes, dtypes = zip(*shapes_and_dtypes, strict=True)
            layout_desc = MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))

            # Sliding-window retrieve skip.
            num_objects_to_skip = 0
            if is_retrieve and not attn_desc.is_full_attention(og_id):
                sw_size_chunks = attn_desc.num_chunks_in_sw[og_id]
                num_objects_to_skip = max(0, num_chunks - sw_size_chunks)
                if num_objects_to_skip > 0:
                    logger.debug(
                        "Object group %d SWA retrieve: skipping first %d objects",
                        og_id,
                        num_objects_to_skip,
                    )

            og_plans.append(
                ObjectGroupPlan(
                    object_group_id=og_id,
                    object_keys=list(obj_keys_per_obj_group[og_id]),
                    layout_desc=layout_desc,
                    num_objects_to_skip=num_objects_to_skip,
                    kernel_group_ids=list(kg_ids_in_og),
                )
            )

        return TransferPlan(
            direction=direction,
            chunk_size=lmcache_tokens_per_chunk,
            num_chunks=num_chunks,
            kernel_groups=kg_plans,
            object_groups=og_plans,
            underflow=False,
        )

    @staticmethod
    def build_from_cache_context(
        cache_context: object,
        block_ids: list[list[int]],
        obj_keys_per_obj_group: list[list[ObjectKey]],
        direction: TransferDirection,
        skip_first_n_tokens: int = 0,
    ) -> TransferPlan:
        """Convenience wrapper that extracts builder inputs from a
        :class:`BaseCacheContext`.

        This helper is intended for the LMCache-driven path where a
        ``BaseCacheContext`` is readily available.  It delegates to
        :meth:`build` after extracting the relevant fields.

        Args:
            cache_context: An instance of
                :class:`lmcache.v1.platform.base_cache_context.BaseCacheContext`.
            block_ids: Raw block IDs indexed by kernel-group ID.
            obj_keys_per_obj_group: Resolved object keys per object group.
            direction: Store or retrieve.
            skip_first_n_tokens: Tokens to skip at the start of retrieve.

        Returns:
            A :class:`TransferPlan` produced by :meth:`build`.
        """
        kv_groups_manager = cache_context.kv_layer_groups_manager  # type: ignore[union-attr]
        chunk_size: int = cache_context.lmcache_tokens_per_chunk  # type: ignore[union-attr]
        num_kernel_groups: int = kv_groups_manager.num_kernel_groups
        kernel_group_shape_dtypes = [
            cache_context.get_kernel_group_shape_dtype(chunk_size, kg_id)  # type: ignore[union-attr]
            for kg_id in range(num_kernel_groups)
        ]
        return TransferPlanBuilder.build(
            kv_groups_manager=kv_groups_manager,
            lmcache_tokens_per_chunk=chunk_size,
            block_ids=block_ids,
            obj_keys_per_obj_group=obj_keys_per_obj_group,
            direction=direction,
            kernel_group_shape_dtypes=kernel_group_shape_dtypes,
            skip_first_n_tokens=skip_first_n_tokens,
        )
