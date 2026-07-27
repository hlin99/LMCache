# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, Protocol
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
    flatten_chunks_group_major,
    gather_engine_groups,
    plan_group_copy,
    scatter_engine_groups,
    unflatten_chunks_group_major,
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
    When ``server_group_counts`` is provided (from the ``group_counts`` field
    in the SHM ``prepare_store`` response context), it is used for exact
    per-group slot counts.  Otherwise a proportional heuristic is applied.

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

    out_per_group: list[list[torch.Tensor] | None] = []
    ci_per_group: list[list[int] | None] = []
    buf_offset = 0
    ci_offset = 0

    for g_idx, plan in enumerate(plans):
        # Prefer exact counts from the server (set for hybrid/HMA multi-group
        # SHM responses).  Fall back to heuristic proportional estimate when
        # the server does not supply per-group counts.
        if (
            server_group_counts
            and len(server_group_counts) > g_idx
            and chunk_indices is not None
        ):
            n = server_group_counts[g_idx]
        elif chunk_indices is None:
            n = plan.num_chunks
        else:
            # Proportional heuristic: approximate per-group slot count.
            total_chunks = sum(p.num_chunks for p in plans)
            if total_chunks > 0 and len(chunk_indices) > 0:
                n = round(plan.num_chunks * len(chunk_indices) / total_chunks)
            else:
                n = 0

        g_buffers = out_buffers[buf_offset : buf_offset + n]
        buf_offset += n
        if chunk_indices is not None:
            g_ci = chunk_indices[ci_offset : ci_offset + n]
            ci_offset += n
        else:
            g_ci = None

        out_per_group.append(g_buffers if g_buffers else None)
        ci_per_group.append(g_ci if g_ci else None)

    return out_per_group, ci_per_group


def _group_counts_from_buffers(
    flat_chunks: list[torch.Tensor],
    plans: list["GroupCopyPlan"],
) -> list[int]:
    """Infer per-group chunk counts from the flat retrieved chunk list.

    The retrieve path receives a flat group-major list from the server.  The
    per-group counts are not explicitly carried in the single-group legacy wire
    format, so we reconstruct them from ``plan.num_chunks``.

    When the flat list is shorter than expected (partial cache hit), we
    distribute chunks proportionally and clamp at zero.

    Args:
        flat_chunks: Flat list of CPU tensors returned by prepare_retrieve.
        plans: Per-group copy plans.

    Returns:
        Per-group chunk counts summing to ``len(flat_chunks)``.
    """
    expected = [p.num_chunks for p in plans]
    total_expected = sum(expected)
    total_actual = len(flat_chunks)
    if total_actual == total_expected:
        return expected
    # Partial hit: distribute actual chunks proportionally, ensuring sum matches.
    if total_expected == 0:
        return [0] * len(plans)
    counts: list[int] = []
    remaining = total_actual
    for i, exp in enumerate(expected):
        if i == len(expected) - 1:
            counts.append(remaining)
        else:
            c = min(exp, remaining)
            counts.append(c)
            remaining -= c
    return counts


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
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.

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
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._group_block_sizes: list[int] = []

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
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata for
                hybrid/HMA models. Empty means single-group legacy mode.
        """
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1

        self._engine_group_infos = list(engine_group_infos)
        group_layouts: list[GroupLayoutSpec] = []

        if engine_group_infos:
            # Build per-group layout specs and per-group MemoryLayoutDesc.
            shapes: list[torch.Size] = []
            dtypes: list[torch.dtype] = []
            group_block_sizes: list[int] = []
            group_use_mla_flags: list[bool] = []

            for info in engine_group_infos:
                from lmcache.v1.multiprocess.transfer_context.group_copy import (
                    build_group_kv_subset,
                    compute_group_blocks_in_chunk,
                )

                group_kv = (
                    build_group_kv_subset(kv_caches, info.layer_indices)
                    if info.layer_indices
                    else kv_caches
                )
                (
                    g_block_size,
                    g_num_layers,
                    g_hidden_dim_size,
                    g_dtype_str,
                    _g_engine_kv_format,
                    g_kv_size,
                ) = compute_kv_layout(group_kv, layout_hints=layout_hints)
                g_use_mla = g_kv_size == 1
                g_tokens_per_block = info.tokens_per_block or g_block_size
                try:
                    g_blocks_in_chunk = compute_group_blocks_in_chunk(
                        blocks_in_chunk,
                        block_size,
                        g_block_size,
                        info.tokens_per_block,
                    )
                except ValueError:
                    g_blocks_in_chunk = blocks_in_chunk
                g_shape = (
                    torch.Size(
                        [g_num_layers, g_blocks_in_chunk * g_block_size, g_hidden_dim_size]
                    )
                    if g_use_mla
                    else torch.Size(
                        [2, g_num_layers, g_blocks_in_chunk * g_block_size, g_hidden_dim_size]
                    )
                )
                shapes.append(g_shape)
                dtypes.append(getattr(torch, g_dtype_str))
                group_block_sizes.append(g_block_size)
                group_use_mla_flags.append(g_use_mla)
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
                    )
                )

            layout_desc = MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)
            self._group_block_sizes = group_block_sizes
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
            self._group_block_sizes = [block_size]

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
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        group_use_mla_flags_for_meta: list[bool] = (
            [spec.use_mla for spec in group_layouts]
            if group_layouts
            else [use_mla_flag]
        )
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            group_block_sizes=list(self._group_block_sizes),
            group_use_mla=group_use_mla_flags_for_meta,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
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
        blocks_in_chunk: int,
    ) -> list[GroupCopyPlan]:
        """Build per-group copy plans, or return empty list for single-group mode.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name.
            block_ids: Per-LMCache-group block ID lists.
            blocks_in_chunk: Reference blocks per LMCache chunk (group 0).

        Returns:
            Non-empty list when hybrid mode is active; empty list otherwise.
        """
        if not self._engine_group_infos or len(block_ids) == 1:
            return []
        return plan_group_copy(
            kv_caches,
            block_ids,
            blocks_in_chunk,
            self._engine_group_infos,
            self._group_block_sizes,
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

        plans = self._build_group_plans(kv_caches, block_ids, blocks_in_chunk)

        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices, server_group_counts = (
            result if result is not None else (None, None, [])
        )
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future

        if plans:
            # Multi-group (hybrid/HMA) path: gather per group and flatten.
            out_per_group, ci_per_group = _split_shm_buffers_by_group(
                out_buffers, chunk_indices, plans,
                server_group_counts=server_group_counts,
            )
            chunks_per_group = gather_engine_groups(
                plans,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
                out_per_group=out_per_group,
                chunk_indices_per_group=ci_per_group,
            )
            flat_chunks = flatten_chunks_group_major(chunks_per_group)
            if out_buffers is not None:
                torch_dev.synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, flat_chunks)
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
            if out_buffers is not None:
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

        plans = self._build_group_plans(kv_caches, block_ids, blocks_in_chunk)

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                if plans:
                    # Multi-group: split the flat chunk list by group and scatter.
                    group_counts = _group_counts_from_buffers(src_buffers, plans)
                    chunks_per_group = unflatten_chunks_group_major(
                        src_buffers, group_counts
                    )
                    scatter_engine_groups(
                        plans,
                        chunks_per_group,
                        layout_hints=self._layout_hints,
                        engine_kv_format=self._engine_kv_format,
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
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        """Release resources held by this context."""
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        """No-op: synchronous context has no deferred stores."""
        pass


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
        return _build_engine_driven_context()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return _build_engine_driven_context()
