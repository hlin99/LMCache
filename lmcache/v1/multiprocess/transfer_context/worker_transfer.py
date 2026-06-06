# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Protocol
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints, is_mla
from lmcache.v1.multiprocess.custom_types import RegisterNonGpuContextPayload
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import LMCacheGroupView
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterNonGpuContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    NonGpuContext,
    NonGpuContextMetadata,
    compute_kv_layout,
    create_non_gpu_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform import _registry as platform_registry

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``handle`` /
# ``data``); ``auto`` reproduces the historical device-type-based dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"
DEFAULT_MAX_ASYNC_NON_GPU_STORES = 8
# Number of background threads used to run commit (CPU->server) work for the
# async non-GPU store path. >1 so that a slow gather for one store does not
# block the commit of another store whose gather already finished.
DEFAULT_NON_GPU_COMMIT_WORKERS = 4


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> handle, others
      -> data). Preserves the historical behaviour.
    * ``HANDLE``: force :class:`HandleTransferContext` (IPC / SHM zero-copy
      path). Requires a registered KV-wrapper factory for the device.
    * ``DATA``: force :class:`DataTransferContext` (worker-side gather /
      scatter copy path).
    """

    AUTO = "auto"
    HANDLE = "handle"
    DATA = "data"


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


def _build_handle_context(device_type: str) -> "TransferContext":
    """Build a :class:`HandleTransferContext` after capability check."""
    try:
        platform_registry.get_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'handle' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'data' or 'auto' instead." % device_type
        ) from exc
    return HandleTransferContext()


class IPCEvent(Protocol):
    """Protocol for IPC-capable CUDA events used by transport operations."""

    def ipc_handle(self) -> object:
        """Return an IPC handle consumable by the multiprocess server."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError("non-GPU transfer does not support hybrid KV cache groups")
    return block_ids[0]


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
        group_views: Sequence[LMCacheGroupView] = (),
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
            group_views: LMCache-owned engine KV cache group metadata.

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

    def flush_inflight_gathers(self) -> None:
        """Synchronize any in-flight gather operations.

        The default implementation is a no-op. Non-GPU async save contexts can
        override this to make preemption handling block until deferred reads of
        vLLM paged KV data are complete.
        """
        return None


class HandleTransferContext(TransferContext):
    """Handle-based IPC + MQ future transport context."""

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
        group_views: Sequence[LMCacheGroupView] = (),
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
                list(group_views),
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
                "Handle transfer context is not registered. "
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
                "Handle transfer context is not registered. "
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


class DataTransferContext(TransferContext):
    """Data transfer context for non-CUDA workers.

    Store on the non-GPU path is two-phase and fully async *when the worker
    device supports the required async primitives* (a stream, an event with
    ``record``/``synchronize``/``wait``, and pinned host memory):
    1) gather: enqueue GPU->CPU copies on a dedicated copy stream into
       LMCache-owned pinned staging buffers (ordered behind the per-step event).
    2) commit: wait for gather completion in a background thread, then perform
       commit_store() (pickle or SHM commit) and resolve the returned future.

    When those primitives are not available (e.g. a CPU-only backend without
    streams/events/pinned memory), the context automatically falls back to the
    original synchronous store implementation. This dispatch is internal and
    capability-based; there is no user-facing async/sync flag, and async stays
    the default whenever the device can support it.

    SHM note: SHM slots are generally pageable, so device->SHM DtoH copies may
    implicitly synchronize. To keep gather async, we always gather into pinned
    bounce buffers first, then copy to SHM slots on the commit thread.
    """

    def __init__(
        self,
        max_inflight_stores: int = DEFAULT_MAX_ASYNC_NON_GPU_STORES,
        commit_workers: int = DEFAULT_NON_GPU_COMMIT_WORKERS,
    ) -> None:
        self._non_gpu_context: NonGpuContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._gpu_kv_format: Any = None
        self._max_inflight_stores = max(1, int(max_inflight_stores))
        self._commit_workers = max(1, int(commit_workers))
        # Capability-based dispatch decided in register(); defaults to the
        # synchronous fallback until the device is known to be async-capable.
        self._async_capable = False
        # Async-only resources. Created lazily (never in __init__) and only when
        # the device is async-capable, so backends without Stream/Event/pinned
        # memory never touch these primitives.
        self._copy_stream: Any = None
        self._commit_executor: ThreadPoolExecutor | None = None
        self._inflight_semaphore: threading.BoundedSemaphore | None = None
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        self._inflight_commits: set[ConcurrentFuture[None]] = set()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._is_closing = False

    def _detect_async_capable(self) -> bool:
        """Probe whether the worker device supports the async store primitives.

        Requires a stream, an event exposing ``record``/``synchronize``/
        ``wait``, and pinned (page-locked) host memory. The probe is performed
        once (cached by ``register()``); it never runs per ``submit_store``.
        """
        if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
            return False
        try:
            torch_dev.Stream()
            event = torch_dev.Event()
        except Exception:
            return False
        for attr in ("record", "synchronize", "wait"):
            if not callable(getattr(event, attr, None)):
                return False
        try:
            torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        except (RuntimeError, TypeError):
            return False
        return True

    def _create_async_resources(self) -> None:
        """Create the copy stream / commit executor / backpressure semaphore."""
        self._copy_stream = torch_dev.Stream()
        self._commit_executor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_non_gpu_commit",
        )
        self._inflight_semaphore = threading.BoundedSemaphore(self._max_inflight_stores)

    def _init_async_capability(self) -> None:
        """Detect device async capability and lazily create async resources."""
        self._async_capable = self._detect_async_capable()
        if self._async_capable:
            self._create_async_resources()

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor]:
        key = (tuple(shape), dtype)
        with self._inflight_lock:
            pooled = self._staging_pool.setdefault(key, [])
            staged = [pooled.pop() for _ in range(min(len(pooled), count))]
        if len(staged) == count:
            return staged

        missing = count - len(staged)
        for _ in range(missing):
            try:
                staged.append(
                    torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                )
            except RuntimeError:
                # Graceful fallback for CPU-only / pin-memory-disabled setups.
                logger.warning(
                    "Falling back to non-pinned CPU staging buffer "
                    "(shape=%s, dtype=%s)",
                    tuple(shape),
                    dtype,
                )
                staged.append(torch.empty(shape, dtype=dtype, device="cpu"))
        return staged

    def _release_staging(self, chunks: list[torch.Tensor]) -> None:
        if not chunks:
            return
        key = (tuple(chunks[0].shape), chunks[0].dtype)
        with self._inflight_lock:
            self._staging_pool.setdefault(key, []).extend(chunks)

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
        group_views: Sequence[LMCacheGroupView] = (),
    ) -> None:
        """Register KV caches with the non-GPU context server.

        ``group_views`` is accepted to satisfy the base interface but
        is currently a no-op: the non-GPU transfer path does not support
        hybrid KV cache groups and rejects multi-group transfers at store /
        retrieve time (see ``_single_group_block_ids``).
        """
        # TODO: inference_engine_logical_block_size is currently used by
        # DeepSeek V4 on the CUDA path. The non-CUDA path is yet to be
        # implemented.
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            gpu_kv_format,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._gpu_kv_format = gpu_kv_format

        use_mla_flag = is_mla(gpu_kv_format)
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
            RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
            [
                RegisterNonGpuContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterNonGpuContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = NonGpuContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._non_gpu_context = create_non_gpu_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        self._init_async_capability()
        logger.info(
            "Worker non-GPU transfer context registered "
            "(instance_id=%d, mode=%s, store=%s)",
            instance_id,
            supported_transfer_mode,
            "async" if self._async_capable else "sync",
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
        if self._non_gpu_context is None:
            raise RuntimeError(
                "Data transfer context is not registered. "
                "Call register() before submit_store()."
            )
        if self._async_capable:
            return self._submit_store_async(
                _request_id,
                key,
                instance_id,
                kv_caches,
                block_ids,
                _event,
                blocks_in_chunk,
            )
        return self._submit_store_sync(
            key,
            instance_id,
            kv_caches,
            block_ids,
            blocks_in_chunk,
        )

    def _submit_store_sync(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Original synchronous store path (capability fallback).

        Reproduces the pre-async behaviour exactly: synchronize, prepare,
        gather, (SHM) synchronize, commit, and return an already-resolved
        future.
        """
        assert self._non_gpu_context is not None
        torch_dev.synchronize()
        result = self._non_gpu_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            _single_group_block_ids(block_ids),
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            gpu_kv_format=self._gpu_kv_format,
            out=out_buffers,
            chunk_indices=chunk_indices,
        )
        if out_buffers is not None:
            # SHM path uses async device->CPU copies; complete them before commit.
            torch_dev.synchronize()
        ok = self._non_gpu_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def _submit_store_async(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        completion: MessagingFuture[bool] = MessagingFuture()
        non_gpu_context = self._non_gpu_context
        semaphore = self._inflight_semaphore
        commit_executor = self._commit_executor
        assert non_gpu_context is not None
        assert semaphore is not None
        assert commit_executor is not None

        semaphore.acquire()
        staged_chunks: list[torch.Tensor] = []
        shm_out_buffers: list[torch.Tensor] | None = None
        gather_done: Any | None = None
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    semaphore.release()
                    return completion

            result = non_gpu_context.prepare_store(key, instance_id)
            out_buffers, chunk_indices = result if result is not None else (None, None)
            if chunk_indices is not None and len(chunk_indices) == 0:
                # All chunks are already in cache: no gather, no commit.
                completion.set_result(True)
                semaphore.release()
                return completion

            full_block_ids = _single_group_block_ids(block_ids)
            num_chunks = (
                len(chunk_indices)
                if chunk_indices is not None
                else len(full_block_ids) // blocks_in_chunk
            )
            if not non_gpu_context.layout_desc.shapes:
                raise RuntimeError("non-GPU layout_desc.shapes is empty")
            if not non_gpu_context.layout_desc.dtypes:
                raise RuntimeError("non-GPU layout_desc.dtypes is empty")
            staged_chunks = self._alloc_pinned_staging(
                non_gpu_context.layout_desc.shapes[0],
                non_gpu_context.layout_desc.dtypes[0],
                num_chunks,
            )
            shm_out_buffers = out_buffers
            with torch_dev.stream(self._copy_stream):
                _event.wait(stream=self._copy_stream)
                gather_paged_kv_to_cpu(
                    kv_caches,
                    full_block_ids,
                    blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    gpu_kv_format=self._gpu_kv_format,
                    out=staged_chunks,
                    chunk_indices=chunk_indices,
                )
                gather_done = torch_dev.Event()
                gather_done.record(self._copy_stream)

            with self._inflight_lock:
                if gather_done is not None:
                    self._inflight_gather_events.add(gather_done)

            def _commit_after_gather() -> None:
                ok = False
                try:
                    if gather_done is not None:
                        gather_done.synchronize()
                    if shm_out_buffers is not None:
                        if len(staged_chunks) != len(shm_out_buffers):
                            raise RuntimeError(
                                "SHM staging chunk count mismatch: "
                                f"{len(staged_chunks)} vs {len(shm_out_buffers)} "
                                f"(request_id={_request_id}, instance_id={instance_id})"
                            )
                        for staged, shm_view in zip(
                            staged_chunks, shm_out_buffers, strict=True
                        ):
                            shm_view.copy_(staged)
                        ok = non_gpu_context.commit_store(
                            key, instance_id, shm_out_buffers
                        )
                    else:
                        ok = non_gpu_context.commit_store(
                            key, instance_id, staged_chunks
                        )
                    if not ok:
                        logger.error(
                            "Async non-GPU commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async non-GPU store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    self._release_staging(staged_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                    completion.set_result(ok)
                    semaphore.release()

            # Submitting the commit task is the ownership-transfer point: once it
            # succeeds, the commit task is solely responsible for releasing the
            # semaphore, releasing staging buffers, and resolving the future. The
            # except below therefore only handles failures that occur *before*
            # this submit, so it can never double-release or double-resolve.
            commit_future = commit_executor.submit(_commit_after_gather)
        except Exception:
            logger.exception("Failed to submit async non-GPU store")
            if staged_chunks:
                self._release_staging(staged_chunks)
            if gather_done is not None:
                with self._inflight_lock:
                    self._inflight_gather_events.discard(gather_done)
            completion.set_result(False)
            semaphore.release()
            return completion

        with self._inflight_lock:
            self._inflight_commits.add(commit_future)

        def _drop_commit_future(done_future: ConcurrentFuture[None]) -> None:
            with self._inflight_lock:
                self._inflight_commits.discard(done_future)

        commit_future.add_done_callback(_drop_commit_future)
        return completion

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
        if self._non_gpu_context is None:
            raise RuntimeError(
                "Data transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        src_buffers = self._non_gpu_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    _single_group_block_ids(block_ids),
                    src_buffers,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    gpu_kv_format=self._gpu_kv_format,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._non_gpu_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        # Drain in-flight async work only when async resources were created.
        # In sync (fallback) mode there is no copy stream / executor / inflight
        # state, so guard against touching never-created attributes.
        if self._async_capable:
            with self._inflight_lock:
                self._is_closing = True
                gather_events = list(self._inflight_gather_events)
            for event in gather_events:
                try:
                    event.synchronize()
                except Exception:
                    logger.exception("Failed while draining gather events")
            if self._commit_executor is not None:
                self._commit_executor.shutdown(wait=True, cancel_futures=False)
        if self._non_gpu_context is not None:
            self._non_gpu_context.close()
            self._non_gpu_context = None

    def flush_inflight_gathers(self) -> None:
        # Cheap no-op in sync mode: no copy stream / in-flight gather events.
        if not self._async_capable:
            return
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            event.synchronize()


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
    if resolved_mode is MPTransferMode.HANDLE:
        return _build_handle_context(device_type)
    if resolved_mode is MPTransferMode.DATA:
        return DataTransferContext()
    # AUTO: preserve the historical device-type-based dispatch.
    if device_type == "cuda":
        return HandleTransferContext()
    return DataTransferContext()
