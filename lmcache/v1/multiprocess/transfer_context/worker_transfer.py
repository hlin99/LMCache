# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import AbstractSet, Any, Callable, Protocol
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    GroupLayoutSpec,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
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
from lmcache.v1.multiprocess.transfer_context.group_copy import (
    GroupCopyPlan,
    RegisteredGroup,
    build_group_kv_subset,
    flatten_chunks_group_major,
    gather_engine_groups,
    plan_group_copy,
    scatter_engine_groups,
    unflatten_chunks_group_major,
    validate_registered_groups,
)
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context() -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext()

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext()


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
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for the legacy single-group path.

    This helper is retained for explicit single-group fallback. The multi-group
    (hybrid/HMA) paths use ``plan_group_copy`` instead.

    Args:
        block_ids: Per-LMCache-group block ID lists.

    Returns:
        The sole group's flat block-ID list.

    Raises:
        RuntimeError: If ``block_ids`` contains more than one group and the
            caller has not routed to the multi-group path first.
    """
    if len(block_ids) != 1:
        raise RuntimeError(
            "single-group fallback called with multi-group block_ids; "
            "ensure engine_group_infos is set before submit_store/retrieve"
        )
    return block_ids[0]


def _split_shm_buffers_by_group(
    out_buffers: list[torch.Tensor] | None,
    chunk_indices: list[int] | None,
    plans: list["GroupCopyPlan"],
    *,
    server_group_counts: list[int] | None = None,
) -> tuple[list[list[torch.Tensor] | None], list[list[int] | None]]:
    """Split flat SHM out_buffers and chunk_indices lists by group.

    The SHM server returns a single flat list of buffers / indices in
    group-major order with lengths equal to ``plan.num_chunks`` for each plan.
    This helper partitions them back into per-group lists for multi-group
    gather calls.

    When ``out_buffers`` is ``None`` (pickle mode) all entries are ``None``.
    This helper is called only for structured group plans. Their SHM responses
    must carry exact ``server_group_counts`` even for one group; ownership is
    never inferred from flat list lengths.

    Args:
        out_buffers: Flat list of SHM-backed output tensors (group-major), or
            ``None`` for pickle mode.
        chunk_indices: Flat list of sparse chunk indices (group-major), or
            ``None`` when all chunks are needed.
        plans: Per-group copy plans (provides ``num_chunks`` per group).
        server_group_counts: Per-group slot counts from the server response, or
            ``None`` / empty list when not available.

    Returns:
        ``(out_per_group, ci_per_group)`` where each list has one entry per
        group.
    """
    if out_buffers is None:
        return [None] * len(plans), [None] * len(plans)
    if server_group_counts is None:
        raise ValueError(
            "structured SHM prepare_store response must include group_counts "
            "with one entry per group, got None"
        )
    has_invalid_group_counts = (
        len(server_group_counts) != len(plans)
        or any(not isinstance(count, int) or count < 0 for count in server_group_counts)
        or sum(server_group_counts) != len(out_buffers)
    )
    if has_invalid_group_counts:
        raise ValueError(
            "structured SHM prepare_store must return one exact slot count "
            f"per group; counts={server_group_counts}, buffers={len(out_buffers)}"
        )
    if chunk_indices is not None and len(chunk_indices) != len(out_buffers):
        raise ValueError("SHM chunk_indices and output buffers must have equal length")

    out_per_group: list[list[torch.Tensor] | None] = []
    ci_per_group: list[list[int] | None] = []
    buf_offset = 0
    ci_offset = 0

    for g_idx, plan in enumerate(plans):
        n = server_group_counts[g_idx]

        g_buffers = out_buffers[buf_offset : buf_offset + n]
        buf_offset += n
        if chunk_indices is not None:
            g_ci = chunk_indices[ci_offset : ci_offset + n]
            ci_offset += n
            if any(idx < 0 or idx >= plan.num_chunks for idx in g_ci):
                raise ValueError(
                    f"group {g_idx} contains out-of-range local chunk indices {g_ci}"
                )
        else:
            g_ci = None

        out_per_group.append(g_buffers if g_buffers else None)
        ci_per_group.append(g_ci if g_ci else None)

    if buf_offset != len(out_buffers) or (
        chunk_indices is not None and ci_offset != len(chunk_indices)
    ):
        raise ValueError("SHM group counts do not consume all prepared buffers")
    return out_per_group, ci_per_group


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return next(iter(kv_caches.values())).device


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        excluded_layer_indices: AbstractSet[int] = frozenset(),
        lmcache_tokens_per_chunk: int | None = None,
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
            excluded_layer_indices: Registered cross-layer sharing aliases
                intentionally omitted from transfer groups. Supply every
                registered tensor that aliases an owner tensor.
            lmcache_tokens_per_chunk: Authoritative logical tokens per LMCache
                chunk. Required when ``engine_group_infos`` is non-empty.

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

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

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
        excluded_layer_indices: AbstractSet[int] = frozenset(),
        lmcache_tokens_per_chunk: int | None = None,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            lmcache_tokens_per_chunk: Authoritative LMCache chunk size
                accepted for interface consistency.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            excluded_layer_indices: Cross-layer aliases accepted for interface
                consistency.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

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
        self._device = device
        self._event_backend = event_backend

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

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
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mq_client = None
        self._send_request = None
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.

    Hybrid/HMA support
    ------------------
    When ``register`` receives a non-empty ``engine_group_infos`` sequence the
    context builds per-group registration metadata and switches
    ``submit_store`` / ``submit_retrieve`` to the multi-group gather/scatter
    path. Single-group (non-hybrid) models continue to use the legacy flat
    path with no overhead.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._registered_groups: list[RegisteredGroup] = []

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
        excluded_layer_indices: AbstractSet[int] = frozenset(),
        lmcache_tokens_per_chunk: int | None = None,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        For single-group (non-hybrid) models the registration mirrors the
        legacy behaviour. When ``engine_group_infos`` is non-empty, per-group
        layout specs are computed and sent to the server so that the server can
        store each group's chunks under the correct ``object_group_id``.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            blocks_in_chunk: Engine blocks per LMCache chunk (reference group).
            lmcache_tokens_per_chunk: Authoritative logical tokens per LMCache
                chunk. Required for hybrid/HMA grouped registration.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata for
                hybrid/HMA models. Empty means single-group legacy mode.
            excluded_layer_indices: Registered cross-layer sharing aliases
                intentionally omitted from transfer groups. Supply every
                registered tensor that aliases an owner tensor.
        """
        layout_source = (
            build_group_kv_subset(kv_caches, engine_group_infos[0].layer_indices)
            if engine_group_infos
            else kv_caches
        )
        (
            block_size,
            _layout_num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(layout_source, layout_hints=layout_hints)
        num_layers = len(kv_caches)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1

        registered_groups: list[RegisteredGroup] = []
        group_layouts: list[GroupLayoutSpec] = []

        if engine_group_infos:
            if lmcache_tokens_per_chunk is None:
                raise ValueError(
                    "lmcache_tokens_per_chunk must be specified when "
                    "engine_group_infos is non-empty; provide the authoritative "
                    "LMCache chunk size in tokens from the engine configuration"
                )
            shapes: list[torch.Size] = []
            dtypes: list[torch.dtype] = []
            chunk_tokens = lmcache_tokens_per_chunk

            for object_group_id, info in enumerate(engine_group_infos):
                group_kv = build_group_kv_subset(kv_caches, info.layer_indices)
                (
                    g_block_size,
                    g_num_layers,
                    g_hidden_dim_size,
                    g_dtype_str,
                    g_engine_kv_format,
                    g_kv_size,
                ) = compute_kv_layout(group_kv, layout_hints=layout_hints)
                g_use_mla = g_kv_size == 1
                g_tokens_per_block = info.tokens_per_block or g_block_size
                if chunk_tokens % g_tokens_per_block:
                    raise ValueError(
                        f"LMCache chunk size {chunk_tokens} is not divisible by "
                        f"tokens_per_block={g_tokens_per_block} for object group "
                        f"{object_group_id}"
                    )
                g_blocks_in_chunk = chunk_tokens // g_tokens_per_block
                copy_tokens = (
                    chunk_tokens
                    if info.sw_size_tokens < 0
                    else min(chunk_tokens, info.sw_size_tokens)
                )
                copy_blocks = max(
                    1,
                    (copy_tokens + g_tokens_per_block - 1) // g_tokens_per_block,
                )
                physical_slots = copy_blocks * g_block_size
                g_shape = (
                    torch.Size(
                        [
                            g_num_layers,
                            physical_slots,
                            g_hidden_dim_size,
                        ]
                    )
                    if g_use_mla
                    else torch.Size(
                        [
                            2,
                            g_num_layers,
                            physical_slots,
                            g_hidden_dim_size,
                        ]
                    )
                )
                g_dtype = getattr(torch, g_dtype_str)
                shapes.append(g_shape)
                dtypes.append(g_dtype)
                registered_group = RegisteredGroup(
                    object_group_id=object_group_id,
                    engine_group_id=info.engine_group_id,
                    layer_indices=tuple(info.layer_indices),
                    tokens_per_block=g_tokens_per_block,
                    slots_per_block=g_block_size,
                    blocks_per_chunk=g_blocks_in_chunk,
                    copy_blocks_per_chunk=copy_blocks,
                    chunk_tokens=chunk_tokens,
                    shape=g_shape,
                    dtype=g_dtype,
                    engine_kv_format=g_engine_kv_format,
                    sw_size_tokens=info.sw_size_tokens,
                )
                registered_groups.append(registered_group)
                group_layouts.append(
                    GroupLayoutSpec(
                        num_layers=g_num_layers,
                        hidden_dim_size=g_hidden_dim_size,
                        dtype_str=g_dtype_str,
                        block_size=g_block_size,
                        use_mla=g_use_mla,
                        tokens_per_block=g_tokens_per_block,
                        engine_group_id=info.engine_group_id,
                        sw_size_tokens=info.sw_size_tokens,
                        object_group_id=object_group_id,
                        layer_indices=tuple(info.layer_indices),
                        shape=tuple(g_shape),
                        engine_kv_format=int(g_engine_kv_format),
                    )
                )

            validate_registered_groups(
                registered_groups,
                len(kv_caches),
                excluded_layer_indices=excluded_layer_indices,
            )
            layout_desc = MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)
        else:
            # Single-group / legacy path.
            shape = (
                torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
                if use_mla_flag
                else torch.Size(
                    [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
                )
            )
            dtype = getattr(torch, dtype_str)
            layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

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
                    group_layouts=group_layouts,
                    excluded_layer_indices=tuple(sorted(excluded_layer_indices)),
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            uses_structured_groups=bool(engine_group_infos),
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        self._registered_groups = registered_groups
        num_groups = len(engine_group_infos) if engine_group_infos else 1
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s, "
            "num_groups=%d)",
            instance_id,
            supported_transfer_mode,
            num_groups,
        )

    def _build_group_plans(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        *,
        for_retrieve: bool = False,
    ) -> list[GroupCopyPlan]:
        """Build per-group copy plans, or return empty list for single-group mode.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name.
            block_ids: Per-LMCache-group block ID lists.

        Returns:
            Non-empty list when hybrid mode is active; empty list otherwise.
        """
        if not self._registered_groups:
            return []
        return plan_group_copy(
            kv_caches,
            block_ids,
            self._registered_groups,
            for_retrieve=for_retrieve,
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
        """Submit a synchronous CPU-side store.

        For single-group models this is identical to the legacy path.  For
        hybrid/HMA models the multi-group gather path is used, assembling a
        group-major flat chunk list before commit.

        Args:
            _request_id: External request identifier (unused).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            block_ids: Per-LMCache-group block ID lists.
            _event: Synchronization event (unused in synchronous path).
            blocks_in_chunk: Engine blocks per LMCache chunk.

        Returns:
            A resolved :class:`MessagingFuture` with the commit result.

        Raises:
            RuntimeError: If ``register()`` was not called first.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        torch_dev.synchronize()

        plans = self._build_group_plans(kv_caches, block_ids)

        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices, server_group_counts = (
            result if result is not None else (None, None, [])
        )
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future

        prepared_shm_store = out_buffers is not None
        try:
            if plans:
                # Multi-group (hybrid/HMA) path: gather per group and flatten.
                out_per_group, ci_per_group = _split_shm_buffers_by_group(
                    out_buffers,
                    chunk_indices,
                    plans,
                    server_group_counts=server_group_counts,
                )
                chunks_per_group = gather_engine_groups(
                    plans,
                    layout_hints=self._layout_hints,
                    out_per_group=out_per_group,
                    chunk_indices_per_group=ci_per_group,
                )
                cpu_chunks = flatten_chunks_group_major(chunks_per_group)
            else:
                # Single-group (legacy) path.
                cpu_chunks = gather_paged_kv_to_cpu(
                    kv_caches,
                    _single_group_block_ids(block_ids),
                    blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                    out=out_buffers,
                    chunk_indices=chunk_indices,
                )
            if prepared_shm_store:
                torch_dev.synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)
        except Exception:
            if prepared_shm_store:
                self._best_effort_abort_store(key, instance_id, "store failure")
            raise

        if not ok and prepared_shm_store:
            self._best_effort_abort_store(key, instance_id, "unsuccessful commit")
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
        """Submit a synchronous CPU-side retrieve.

        For single-group models this is identical to the legacy path.  For
        hybrid/HMA models the multi-group scatter path is used, splitting the
        group-major flat chunk list returned by the server before scattering.

        Args:
            _request_id: External request identifier (unused).
            key: LMCache key for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            block_ids: Per-LMCache-group block ID lists.
            _event: Synchronization event (unused in synchronous path).
            blocks_in_chunk: Engine blocks per LMCache chunk.
            skip_first_n_tokens: Tokens at the head of the range to leave
                untouched (APC-shared block guard).

        Returns:
            A resolved :class:`MessagingFuture` with the retrieve result.

        Raises:
            RuntimeError: If ``register()`` was not called first.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        plans = self._build_group_plans(
            kv_caches, block_ids, for_retrieve=True
        )

        retrieve_result = self._engine_driven_context.prepare_retrieve(key, instance_id)
        group_counts: list[int] = []
        src_buffers: list[torch.Tensor] | None
        if isinstance(retrieve_result, tuple):
            src_buffers, group_counts = retrieve_result
        else:
            src_buffers = retrieve_result
        ok = src_buffers is not None
        try:
            if src_buffers is not None:
                if plans:
                    if len(group_counts) != len(plans):
                        raise ValueError(
                            "multi-group retrieve response omitted exact "
                            "group ownership"
                        )
                    chunks_per_group = unflatten_chunks_group_major(
                        src_buffers, group_counts
                    )
                    scatter_engine_groups(
                        plans,
                        chunks_per_group,
                        layout_hints=self._layout_hints,
                        skip_first_n_tokens=skip_first_n_tokens,
                    )
                else:
                    # Single-group (legacy) path.
                    scatter_cpu_to_paged_kv(
                        kv_caches,
                        _single_group_block_ids(block_ids),
                        src_buffers,
                        blocks_in_chunk,
                        skip_first_n_tokens=skip_first_n_tokens,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
                    )
                torch_dev.synchronize()
        except (RuntimeError, ValueError, TypeError, IndexError):
            logger.exception("Failed to scatter retrieved CPU context chunks")
            ok = False
        finally:
            if ok:
                self._engine_driven_context.commit_retrieve(key, instance_id)
            else:
                self._engine_driven_context.abort_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        """Release resources held by this context."""
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None
        self._registered_groups = []
        self._layout_hints = None
        self._engine_kv_format = None

    def flush_inflight_stores(self) -> None:
        """No-op: synchronous context has no deferred stores."""
        pass

    def _best_effort_abort_store(
        self,
        key: Any,
        instance_id: int,
        failure_context: str,
    ) -> None:
        """Release a failed SHM reservation without raising abort errors.

        Args:
            key: Cache key for the prepared store.
            instance_id: Worker process instance identifier.
            failure_context: Short description included in abort-error logs.

        This side-effect-only method returns no value. Abort errors are logged
        and suppressed so the caller's gather or commit exception propagates
        unmodified.
        """
        if self._engine_driven_context is None:
            return
        try:
            self._engine_driven_context.abort_store(key, instance_id)
        except Exception:
            logger.exception(
                "Failed to abort synchronous SHM store after %s",
                failure_context,
            )


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
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        selected_mode = MPTransferMode.LMCACHE_DRIVEN
        context = _build_lmcache_driven_context(device_type)
    elif resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        selected_mode = MPTransferMode.ENGINE_DRIVEN
        context = _build_engine_driven_context()
    elif device_type == "cuda":
        selected_mode = MPTransferMode.LMCACHE_DRIVEN
        context = LMCacheDrivenTransferContext()
    else:
        selected_mode = MPTransferMode.ENGINE_DRIVEN
        context = _build_engine_driven_context()
    logger.info(
        "LMCache MP transfer context selected: mode=%s device=%s",
        selected_mode.value,
        device_type,
    )
    return context
