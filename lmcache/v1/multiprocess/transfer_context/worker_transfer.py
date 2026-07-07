# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, cast
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    is_mla,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    engine_group_layer_indices,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.object_group_utils import (
    execute_prepared_object_group_transfer,
    has_sufficient_block_ids,
    prepare_object_group_transfer,
    select_block_ids_for_cache_context,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform import get_device_info
from lmcache.v1.platform.base_cache_context import BaseCacheContext
import lmcache.c_ops as lmc_ops
import lmcache.python_ops_fallback as _python_ops_fallback

logger = init_logger(__name__)
_ENGINE_OBJECT_GROUP_BLOCK_IDS_BUFFER_SIZE = 1 << 20
_ENGINE_OBJECT_GROUP_MAX_BATCH_SIZE = 4

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        platform_registry.get_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_info = get_device_info(device_type)
    if device_info and not device_info.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for IPC-capable CUDA events used by transport operations."""

    def ipc_handle(self) -> object:
        """Return an IPC handle consumable by the multiprocess server."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _has_native_object_group_transfer() -> bool:
    """Return whether the native object-group executor is available."""
    return (
        lmc_ops.execute_object_group_transfer
        is not _python_ops_fallback.execute_object_group_transfer
    )


class _EngineObjectGroupCacheContext(BaseCacheContext):
    """Bridge raw engine tensors into shared object-group transfer planning.

    The LMCache-driven path receives a platform cache context from registered
    IPC wrappers.  Engine-driven workers already own the raw vLLM KV tensors, so
    this adapter supplies the same object-group metadata, pointer tensors, block
    ID staging, and temporary buffers directly from those tensors.  Temporary
    buffers are Python-owned and require no explicit cleanup.

    Key responsibilities:
    - ``_group_kv_pointers`` stores native pointer tensors for each kernel group.
    - ``_tmp_buffer`` owns the contiguous temporary staging area used by the
      object-group plan.
    - ``_tmp_chunk_group_offsets`` maps each kernel group into that staging area.

    Instances are created during engine-driven registration only when the native
    object-group executor is available for the active CUDA-backed context.
    """

    device_type = "cuda"

    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: Sequence[EngineGroupInfo],
    ) -> None:
        tensors = list(kv_caches.values())
        kv_caches_norm, engine_kv_formats = normalize_and_discover_per_layer_formats(
            tensors,
            engine_group_layer_indices(engine_group_infos),
            EngineType.VLLM,
            layout_hints,
        )
        if not isinstance(kv_caches_norm, list) or not all(
            isinstance(tensor, torch.Tensor) for tensor in kv_caches_norm
        ):
            raise ValueError(
                "engine-driven object-group transfer requires normalized "
                "per-layer tensor KV caches, got "
                f"{type(kv_caches_norm).__name__}"
            )
        kv_caches_norm = cast(list[torch.Tensor], kv_caches_norm)
        device = get_device(kv_caches_norm)
        kv_layer_groups_manager = KVLayerGroupsManager(
            kv_caches_norm,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=True,
        )
        block_ids_buffer = torch.empty(
            _ENGINE_OBJECT_GROUP_BLOCK_IDS_BUFFER_SIZE,
            dtype=torch.long,
            device=device,
        )
        super().__init__(
            kv_caches=kv_caches_norm,
            device=device,
            num_layers=len(engine_kv_formats),
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self._group_kv_pointers: list[torch.Tensor] = []
        for idx, group in enumerate(self.kv_layer_groups_manager_.kv_layer_groups):
            ptrs = get_group_data_ptrs(
                self.kv_caches_, self.get_engine_kv_format(idx), group.layer_indices
            )
            self._group_kv_pointers.append(
                torch.tensor(ptrs, dtype=torch.long, device=device)
            )
        self._max_batch_size = _ENGINE_OBJECT_GROUP_MAX_BATCH_SIZE
        # The native planner needs two views into the same temporary chunk
        # buffer. Kernel-group offsets map each kernel group to its byte range
        # inside the chunk buffer. Object-group offsets span the concatenated
        # kernel-group ranges that belong to each object group. The running
        # offset advances in object-group order so each object group's staging
        # buffer is a compact byte range while kernel groups can still be
        # addressed directly by index.
        num_kernel_groups = self.kv_layer_groups_manager_.num_kernel_groups
        self._tmp_kernel_group_offsets: list[tuple[int, int]] = [
            (0, 0) for _ in range(num_kernel_groups)
        ]
        self._tmp_object_group_offsets: list[tuple[int, int]] = []
        offset = 0
        for object_group in self.kv_layer_groups_manager_.object_groups:
            object_group_start = offset
            for group_idx in object_group.kernel_group_indices:
                group = self.kv_layer_groups_manager_.kv_layer_groups[group_idx]
                shape = self.get_kv_buffer_shape(lmcache_tokens_per_chunk, group_idx)
                byte_size = shape.numel() * group.dtype.itemsize
                self._tmp_kernel_group_offsets[group_idx] = (
                    offset,
                    offset + byte_size,
                )
                offset += byte_size
            self._tmp_object_group_offsets.append((object_group_start, offset))
        self._tmp_chunk_bytes = offset
        self._tmp_buffer = torch.empty(
            self._tmp_chunk_bytes * self.max_batch_size,
            dtype=torch.uint8,
            device=device,
        )

    @property
    def stream(self) -> object:
        """Return the current device stream placeholder."""
        return torch_dev.current_stream()

    @property
    def cupy_stream(self) -> object:
        """Return the current device stream placeholder."""
        return torch_dev.current_stream()

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of object-group objects per batch."""
        return self._max_batch_size

    def close(self) -> None:
        """No-op: raw engine tensors and temporary buffers are Python-owned."""

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Return the pointer tensor for a kernel group."""
        return self._group_kv_pointers[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return a typed temporary buffer for a batch/kernel group pair."""
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                f"batch_idx {batch_idx} >= max_batch_size {self.max_batch_size}"
            )
        group = self.kv_layer_groups_manager_.kv_layer_groups[kernel_group_idx]
        shape = self.get_kv_buffer_shape(
            self.lmcache_tokens_per_chunk, kernel_group_idx
        )
        group_start, group_end = self._tmp_kernel_group_offsets[kernel_group_idx]
        chunk = self._tmp_chunk_bytes
        return (
            self._tmp_buffer[
                batch_idx * chunk + group_start : batch_idx * chunk + group_end
            ]
            .view(group.dtype)
            .view(shape)
        )

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Return a flat temporary buffer for a batch/object group pair."""
        if batch_idx >= self.max_batch_size:
            raise ValueError(
                f"batch_idx {batch_idx} >= max_batch_size {self.max_batch_size}"
            )
        group_start, group_end = self._tmp_object_group_offsets[object_group_idx]
        chunk = self._tmp_chunk_bytes
        return self._tmp_buffer[
            batch_idx * chunk + group_start : batch_idx * chunk + group_end
        ]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return ``(shape, dtype)`` for a kernel group and token count."""
        group = self.kv_layer_groups_manager_.kv_layer_groups[kernel_group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if num_tokens % compress_ratio != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) is not a multiple of "
                f"compress_ratio ({compress_ratio}) for kernel_group_idx "
                f"{kernel_group_idx}"
            )
        num_slots = num_tokens // compress_ratio
        shape_desc = group.shape_desc
        return (
            torch.Size(
                (
                    shape_desc.kv_size,
                    group.num_layers,
                    num_slots,
                    group.hidden_dim_size,
                )
            ),
            group.dtype,
        )

    def cache_size_per_token(self) -> int:
        """Return cache bytes per logical token across all groups."""
        total = 0
        for group_idx, group in enumerate(
            self.kv_layer_groups_manager_.kv_layer_groups
        ):
            compress_ratio = group.tokens_per_block // group.slots_per_block
            numels = self.get_kv_buffer_shape(compress_ratio, group_idx).numel()
            total += numels * group.dtype.itemsize // compress_ratio
        return total


@dataclass
class _EngineObjectGroupTransferState:
    """Prepared raw-tensor state for engine-driven object-group transfers."""

    cache_context: BaseCacheContext
    object_group_layout_descs: list[MemoryLayoutDesc]
    host_buffer_alignment: int = LazyMemoryAllocator.PIN_CHUNK_SIZE

    def close(self) -> None:
        """Release resources held by the cache context."""
        self.cache_context.close()


def _get_object_group_layout_descs(
    cache_context: BaseCacheContext,
) -> list[MemoryLayoutDesc]:
    """Return per-object-group CPU transport layouts for a cache context.

    Args:
        cache_context: Engine-driven cache context with object-group metadata.

    Returns:
        A list indexed by object group. Each ``MemoryLayoutDesc`` contains one
        shape/dtype entry for every kernel group in that object group. If the
        cache context has no object groups, the returned list is empty.
    """
    layouts: list[MemoryLayoutDesc] = []
    for object_group in cache_context.kv_layer_groups_manager.object_groups:
        if not object_group.kernel_group_indices:
            raise ValueError("engine-driven object groups must not be empty")
        shapes_and_dtypes = [
            cache_context.get_kernel_group_shape_dtype(
                cache_context.lmcache_tokens_per_chunk, kernel_group_id
            )
            for kernel_group_id in object_group.kernel_group_indices
        ]
        shapes, dtypes = zip(*shapes_and_dtypes, strict=True)
        layouts.append(MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes)))
    return layouts


def _layout_desc_num_bytes(layout_desc: MemoryLayoutDesc) -> int:
    """Return the total byte size described by a memory layout descriptor.

    Args:
        layout_desc: Object-group layout descriptor with aligned shapes/dtypes.

    Returns:
        Sum of ``shape.numel() * dtype.itemsize`` for each shape/dtype entry.
    """
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
    )


def _build_engine_object_group_transfer_state(
    kv_caches: dict[str, torch.Tensor],
    lmcache_tokens_per_chunk: int,
    layout_hints: LayoutHints | None,
    engine_group_infos: Sequence[EngineGroupInfo],
) -> _EngineObjectGroupTransferState | None:
    """Build object-group state when the native engine-driven path can use it.

    Args:
        kv_caches: Worker KV tensors keyed by layer name.
        lmcache_tokens_per_chunk: Number of logical tokens in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        engine_group_infos: Engine-provided KV group metadata.

    Returns:
        Prepared raw-tensor object-group transfer state when the native helper
        can run for this engine-driven context; otherwise ``None``.

        ``None`` is returned for empty KV caches, non-CUDA KV tensors, or when
        the native object-group executor is unavailable.  Those cases continue
        through the existing low-level gather/scatter fallback.
    """
    tensors = list(kv_caches.values())
    if (
        not tensors
        or tensors[0].device.type != "cuda"
        or not _has_native_object_group_transfer()
    ):
        return None
    cache_context = _EngineObjectGroupCacheContext(
        kv_caches,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        layout_hints=layout_hints,
        engine_group_infos=engine_group_infos,
    )
    return _EngineObjectGroupTransferState(
        cache_context=cache_context,
        object_group_layout_descs=_get_object_group_layout_descs(cache_context),
    )


def _build_tensor_staging_copies(
    objects: Sequence[torch.Tensor],
    staging_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list["lmc_ops.StagingCopy"]:
    """Build native staging descriptors for tensor-backed transport objects.

    Args:
        objects: CPU tensor transfer objects for one batch.
        staging_buffers: Object-group temporary buffers aligned with
            ``objects``.
        is_h2d: True for retrieve (host-to-device), False for store
            (device-to-host).

    Returns:
        Native staging-copy descriptors consumed by the ``lmc_ops.BatchStep``
        plan structs passed to the native object-group executor.
        ``host_offset`` is computed as the CPU tensor pointer offset within
        ``PIN_CHUNK_SIZE`` because the native copy helper splits pageable host
        buffers at alignment boundaries and needs the offset within that window.

    Raises:
        ValueError: If a transfer tensor and staging buffer differ in size.
    """
    copies: list["lmc_ops.StagingCopy"] = []
    for tensor, staging_buffer in zip(objects, staging_buffers, strict=True):
        if tensor.nbytes != staging_buffer.nbytes:
            raise ValueError(
                f"Size mismatch: tensor nbytes={tensor.nbytes}, "
                f"staging_buffer nbytes={staging_buffer.nbytes}"
            )
        host_ptr = tensor.data_ptr()
        device_ptr = staging_buffer.data_ptr()
        # The native copy helper splits pageable host buffers at alignment
        # boundaries, so it needs the host pointer's offset within that window.
        host_offset = host_ptr % LazyMemoryAllocator.PIN_CHUNK_SIZE
        if is_h2d:
            copies.append(
                lmc_ops.StagingCopy(device_ptr, host_ptr, tensor.nbytes, host_offset)
            )
        else:
            copies.append(
                lmc_ops.StagingCopy(host_ptr, device_ptr, tensor.nbytes, host_offset)
            )
    return copies


def _build_pickle_tensor_staging_copies(
    objects: Sequence[torch.Tensor],
    staging_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list["lmc_ops.StagingCopy"]:
    """Build staging descriptors for pickle tensor payloads.

    This thin wrapper is intentionally separate from the SHM wrapper so call
    sites and tests can assert which transport-specific ownership path is being
    planned, even though both currently share the same tensor descriptor logic.

    Args:
        objects: Pickle payload tensors for one batch.
        staging_buffers: Object-group temporary buffers aligned with
            ``objects``.
        is_h2d: True for retrieve, False for store.

    Returns:
        Native staging-copy descriptors.

    Raises:
        ValueError: If a payload tensor and staging buffer differ in size.
    """
    return _build_tensor_staging_copies(objects, staging_buffers, is_h2d)


def _build_shm_tensor_staging_copies(
    objects: Sequence[torch.Tensor],
    staging_buffers: Sequence[torch.Tensor],
    is_h2d: bool,
) -> list["lmc_ops.StagingCopy"]:
    """Build staging descriptors for SHM tensor views.

    This thin wrapper is intentionally separate from the pickle wrapper so call
    sites and tests can assert which transport-specific ownership path is being
    planned, even though both currently share the same tensor descriptor logic.

    Args:
        objects: SHM-backed tensor views for one batch.
        staging_buffers: Object-group temporary buffers aligned with
            ``objects``.
        is_h2d: True for retrieve, False for store.

    Returns:
        Native staging-copy descriptors.

    Raises:
        ValueError: If a SHM tensor view and staging buffer differ in size.
    """
    return _build_tensor_staging_copies(objects, staging_buffers, is_h2d)


def _is_shm_engine_context(context: EngineDrivenContext) -> bool:
    """Return whether an engine-driven context uses SHM transport.

    Args:
        context: Engine-driven transport context to inspect.

    Returns:
        True when ``context`` is the SHM transport implementation.
    """
    # Local
    from .shm import EngineDrivenContextShm

    return isinstance(context, EngineDrivenContextShm)


def _allocate_pickle_transfer_tensors(
    layout_desc: MemoryLayoutDesc,
    count: int,
) -> list[torch.Tensor]:
    """Allocate pickle transport tensors matching the registered layout.

    Args:
        layout_desc: Registered engine-driven object layout.
        count: Number of tensors to allocate.

    Returns:
        CPU tensors suitable for pickle commit payloads.

    """
    nbytes = _layout_desc_num_bytes(layout_desc)
    return [
        torch.empty(
            (nbytes,),
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=torch_dev.is_available(),
        )
        for _ in range(count)
    ]


def _get_num_chunks_from_block_ids(
    state: _EngineObjectGroupTransferState,
    block_ids: list[list[int]],
) -> int:
    """Return the transfer chunk count implied by the first kernel group.

    All kernel groups in one engine-driven request cover the same number of
    LMCache chunks; only the number of raw blocks per chunk may differ. The
    first kernel group therefore determines the chunk count, and the shared
    sufficiency check later validates every kernel group's raw block IDs
    against that count.

    Args:
        state: Prepared engine-driven object-group transfer state.
        block_ids: Raw block IDs indexed by kernel group.

    Returns:
        Number of LMCache chunks represented by the first kernel group's block
        IDs, or zero when no block IDs are present.
    """
    if not block_ids or not block_ids[0]:
        return 0
    blocks_per_chunk = state.cache_context.calculate_num_blocks(
        state.cache_context.lmcache_tokens_per_chunk, 0
    )
    return len(block_ids[0]) // blocks_per_chunk


def _group_flat_transport_objects(
    flat_objects: Sequence[torch.Tensor],
    num_object_groups: int,
    num_chunks: int,
) -> list[list[torch.Tensor]]:
    """Split object-group-major flat transport objects into per-group lists.

    Args:
        flat_objects: Transport tensors ordered as all chunks for object group
            0, then all chunks for object group 1, and so on.
        num_object_groups: Number of object groups in the transfer.
        num_chunks: Number of chunks per object group.

    Returns:
        Nested object list whose outer index is object group and inner index is
        chunk.

    Raises:
        ValueError: If ``flat_objects`` length is not
            ``num_object_groups * num_chunks``.
    """
    expected = num_object_groups * num_chunks
    if len(flat_objects) != expected:
        raise ValueError(
            f"Expected {expected} engine-driven transport objects, "
            f"got {len(flat_objects)}"
        )
    return [
        list(flat_objects[start : start + num_chunks])
        for start in range(0, expected, num_chunks)
    ]


def _flatten_transport_objects(
    objects_by_group: Sequence[Sequence[torch.Tensor | None]],
) -> list[torch.Tensor]:
    """Flatten object-group-major transport objects after dropping skipped slots.

    Args:
        objects_by_group: Nested transport objects indexed by object group and
            then chunk. ``None`` entries represent skipped SHM store chunks.

    Returns:
        Non-``None`` tensors in object-group-major order: all non-skipped
        chunks for object group 0, then all non-skipped chunks for object group
        1, and so on.
    """
    return [obj for objects in objects_by_group for obj in objects if obj is not None]


def _build_sparse_transport_objects(
    buffers: Sequence[torch.Tensor],
    flat_chunk_indices: Sequence[int],
    num_object_groups: int,
    num_chunks: int,
) -> list[list[torch.Tensor | None]]:
    """Build sparse per-group object lists from SHM flat chunk indices.

    Args:
        buffers: SHM-backed tensors returned by prepare-store.
        flat_chunk_indices: Object-group-major positions for ``buffers``. A
            flat index maps to ``object_group_id = idx // num_chunks`` and
            ``chunk_idx = idx % num_chunks``.
        num_object_groups: Number of object groups in the transfer.
        num_chunks: Number of chunks per object group.

    Returns:
        Nested object list indexed by object group and chunk. Missing chunks
        are represented as ``None``.

    Raises:
        ValueError: If any flat index is outside the transfer range.
    """
    max_flat_idx = num_object_groups * num_chunks
    invalid_indices = [
        idx for idx in flat_chunk_indices if idx < 0 or idx >= max_flat_idx
    ]
    if invalid_indices:
        raise ValueError(
            f"flat chunk indices {invalid_indices} out of range [0, {max_flat_idx})"
        )

    objects_by_group: list[list[torch.Tensor | None]] = [
        [None] * num_chunks for _ in range(num_object_groups)
    ]
    for buffer, flat_idx in zip(buffers, flat_chunk_indices, strict=True):
        object_group_id = flat_idx // num_chunks
        chunk_idx = flat_idx % num_chunks
        objects_by_group[object_group_id][chunk_idx] = buffer
    return objects_by_group


def _execute_engine_object_group_transfer(
    state: _EngineObjectGroupTransferState,
    block_ids: list[list[int]],
    objects_by_group: Sequence[Sequence[torch.Tensor | None]],
    skip_first_n_tokens: int,
    direction: "lmc_ops.TransferDirection",
    staging_copy_builder: Callable[
        [Sequence[torch.Tensor], Sequence[torch.Tensor], bool],
        list["lmc_ops.StagingCopy"],
    ],
    batch_size: int,
) -> None:
    """Plan and execute an engine-driven object-group transfer.

    ``staging_copy_builder`` produces native ``StagingCopy`` descriptors for a
    batch of transport tensors.  It is parameterized so pickle tensors and SHM
    views can keep transport-specific ownership/allocation behavior while still
    sharing the common object-group planner and executor.

    Args:
        state: Prepared object-group transfer state.
        block_ids: Raw block IDs indexed by kernel group.
        objects_by_group: Tensor transport objects indexed by object group and
            chunk, with ``None`` entries allowed for skipped D2H store chunks.
        skip_first_n_tokens: Retrieve prefix to preserve.
        direction: Native transfer direction.
        staging_copy_builder: Builder for native staging-copy descriptors.
        batch_size: Number of objects per planned batch.

    Raises:
        ValueError: If ``block_ids`` do not cover every requested object.
    """
    cache_context = state.cache_context
    if not objects_by_group:
        raise ValueError("objects_by_group must contain at least one object group")
    if len(objects_by_group) != len(
        cache_context.kv_layer_groups_manager.object_groups
    ):
        raise ValueError(
            "objects_by_group length does not match engine-driven object groups"
        )
    num_chunks = len(objects_by_group[0]) if objects_by_group else 0
    if any(len(objects) != num_chunks for objects in objects_by_group):
        raise ValueError("all object groups must contain the same number of chunks")

    raw_blocks_per_chunk = [
        cache_context.calculate_num_blocks(
            cache_context.lmcache_tokens_per_chunk, kernel_group_id
        )
        for kernel_group_id in range(
            cache_context.kv_layer_groups_manager.num_kernel_groups
        )
    ]
    if not has_sufficient_block_ids(
        block_ids,
        raw_blocks_per_chunk,
        num_chunks,
    ):
        raise ValueError("block_ids do not cover all engine-driven transfer objects")
    selected_block_ids = select_block_ids_for_cache_context(cache_context, block_ids)
    block_ids_gpu = cache_context.stage_block_ids(selected_block_ids)
    for object_group_id, objects in enumerate(objects_by_group):
        kernel_group_specs, batch_steps = prepare_object_group_transfer(
            cache_context,
            block_ids_gpu,
            objects,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
            staging_copy_builder,
        )
        execute_prepared_object_group_transfer(
            direction,
            cache_context.device,
            kernel_group_specs,
            batch_steps,
            host_buffer_alignment=state.host_buffer_alignment,
        )


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. CUDA paths return
    CUDA-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (IPC for CUDA,
    SHM wrappers for CPU with CUDA-IPC-like semantics) and the LMCache
    server performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), skip_first_n_tokens],
        ).to_cuda_future()

    def close(self) -> None:
        self._mq_client = None
        self._send_request = None


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for worker-side copy mode.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    CUDA workers use the shared object-group transfer planner when native
    support is available; other workers use the low-level gather/scatter path.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._object_group_transfer_state: _EngineObjectGroupTransferState | None = None

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the engine-driven context server."""
        # TODO: per-group compression (EngineGroupInfo.tokens_per_block vs
        # the tensor-detected slot count, e.g. DeepSeek V4) is only handled
        # on the CUDA path. The non-CUDA path is yet to be implemented.
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format
        self._object_group_transfer_state = _build_engine_object_group_transfer_state(
            kv_caches,
            lmcache_tokens_per_chunk=blocks_in_chunk * block_size,
            layout_hints=layout_hints,
            engine_group_infos=engine_group_infos,
        )

        use_mla_flag = is_mla(engine_kv_format)
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            object_group_layout_descs=(
                self._object_group_transfer_state.object_group_layout_descs
                if self._object_group_transfer_state is not None
                else []
            ),
        )

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    object_group_shapes=[
                        [list(shape) for shape in layout.shapes]
                        for layout in metadata.object_group_layout_descs
                    ],
                    object_group_dtypes=[
                        [str(dtype).split(".")[-1] for dtype in layout.dtypes]
                        for layout in metadata.object_group_layout_descs
                    ],
                    attn_window_num_chunks=(
                        self._object_group_transfer_state.cache_context.kv_layer_groups_manager.get_attn_desc().num_chunks_in_sw
                        if self._object_group_transfer_state is not None
                        else []
                    ),
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            instance_id,
            supported_transfer_mode,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        if self._object_group_transfer_state is not None:
            state = self._object_group_transfer_state
            num_chunks = _get_num_chunks_from_block_ids(state, block_ids)
            num_object_groups = len(
                state.cache_context.kv_layer_groups_manager.object_groups
            )
            if out_buffers is None:
                objects_by_group = [
                    _allocate_pickle_transfer_tensors(layout_desc, num_chunks)
                    for layout_desc in state.object_group_layout_descs
                ]
                _execute_engine_object_group_transfer(
                    state,
                    block_ids,
                    objects_by_group,
                    skip_first_n_tokens=0,
                    direction=lmc_ops.TransferDirection.D2H,
                    staging_copy_builder=_build_pickle_tensor_staging_copies,
                    batch_size=state.cache_context.max_batch_size,
                )
                cpu_chunks = _flatten_transport_objects(objects_by_group)
            else:
                objects_by_group = _build_sparse_transport_objects(
                    out_buffers,
                    chunk_indices or [],
                    num_object_groups,
                    num_chunks,
                )
                _execute_engine_object_group_transfer(
                    state,
                    block_ids,
                    objects_by_group,
                    skip_first_n_tokens=0,
                    direction=lmc_ops.TransferDirection.D2H,
                    staging_copy_builder=_build_shm_tensor_staging_copies,
                    batch_size=1,
                )
                cpu_chunks = out_buffers
            torch_dev.synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

            future = MessagingFuture()
            future.set_result(ok)
            return future
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            _single_group_block_ids(block_ids),
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=out_buffers,
            chunk_indices=chunk_indices,
        )
        if out_buffers is not None:
            # SHM path uses async device->CPU copies; complete them before commit.
            torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                if self._object_group_transfer_state is not None:
                    state = self._object_group_transfer_state
                    num_chunks = _get_num_chunks_from_block_ids(state, block_ids)
                    objects_by_group = _group_flat_transport_objects(
                        src_buffers,
                        len(state.cache_context.kv_layer_groups_manager.object_groups),
                        num_chunks,
                    )
                    staging_copy_builder = (
                        _build_shm_tensor_staging_copies
                        if _is_shm_engine_context(self._engine_driven_context)
                        else _build_pickle_tensor_staging_copies
                    )
                    _execute_engine_object_group_transfer(
                        state,
                        block_ids,
                        objects_by_group,
                        skip_first_n_tokens=skip_first_n_tokens,
                        direction=lmc_ops.TransferDirection.H2D,
                        staging_copy_builder=staging_copy_builder,
                        batch_size=state.cache_context.max_batch_size,
                    )
                else:
                    scatter_cpu_to_paged_kv(
                        kv_caches,
                        _single_group_block_ids(block_ids),
                        src_buffers,
                        blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._object_group_transfer_state is not None:
            self._object_group_transfer_state.close()
            self._object_group_transfer_state = None
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return EngineDrivenTransferContext()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return EngineDrivenTransferContext()
