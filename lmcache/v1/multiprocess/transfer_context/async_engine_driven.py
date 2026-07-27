# SPDX-License-Identifier: Apache-2.0
"""Async engine-driven data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.group_copy import (
    flatten_chunks_group_major,
    gather_engine_groups,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    GroupCopyPlan,
    IPCEvent,
    _single_group_block_ids,
    _split_shm_buffers_by_group,
)

logger = init_logger(__name__)

# Number of background threads used to run commit (CPU->server) work for the
# async engine-driven store path. >1 so that a slow gather for one store does
# not block the commit of another store whose gather already finished.
DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 4


# TODO: async retrieve path TBD, but benefit might be very limited
class AsyncEngineDrivenTransferContext(EngineDrivenTransferContext):
    """Fully async engine-driven data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    that resolves only after the deferred gather (GPU->CPU copy) and commit
    (CPU->server) both complete off the forward thread, while
    ``submit_retrieve`` stays synchronous and returns an already-resolved
    future exactly as on the base context.

    Inherits :class:`EngineDrivenTransferContext` and reuses its
    ``register()`` (layout / SHM registration, no stream dependency) and
    ``submit_retrieve()`` (this path does not change retrieve). Only the store
    is made async.

    Hybrid/HMA support
    ------------------
    When the base context registered with ``engine_group_infos``, the async
    store path routes through the multi-group
    :func:`~.group_copy.gather_engine_groups` helper, assembling a group-major
    flat chunk list before commit.  The blocking retrieve path is inherited
    from :class:`EngineDrivenTransferContext` unchanged.

    Store is three-phase, all executed entirely in a background thread:

    1. prepare: call prepare_store() to negotiate buffers with the server
       (the costliest step in pickle mode due to the synchronous RPC round-trip).
    2. gather: wait for the forward event on the copy stream, then enqueue
       GPU->CPU copies. When SHM buffers are available, gather writes directly
       into SHM views (matching the synchronous path). Otherwise, gather
       targets pinned staging buffers.
    3. commit: wait for gather completion (via a recorded CUDA event), then
       perform commit_store() and resolve the returned future.

    ``submit_store`` performs only O(1) work on the forward thread (registration
    check) before submitting all three phases to the background
    ``commit_executor``, so the forward thread is never blocked by the RPC
    round-trip or gather kernel launch latency.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.
    """

    def __init__(
        self,
        commit_workers: int = DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS,
    ) -> None:
        """Initialize the async context and create its async resources.

        Args:
            commit_workers: Number of background threads used to run commit
                (CPU->server) work. >1 so a slow gather for one store does not
                block the commit of another whose gather is already done.
        """
        super().__init__()
        self._commit_workers = max(1, int(commit_workers))
        self._copy_stream: Any = torch_dev.Stream()
        self._commit_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_engine_driven_commit",
        )
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        # Tracks gather tasks that have been submitted to _commit_executor but
        # have not yet recorded their CUDA event. flush_inflight_stores waits
        # on all of these before synchronizing _inflight_gather_events, closing
        # the window where preemption could overwrite paged KV blocks before an
        # in-flight gather has had a chance to record its CUDA event.
        self._pending_stores: set[threading.Event] = set()
        # Serializes commit_store calls across worker threads, since the
        # underlying ZMQ socket is not thread-safe and commit_workers defaults
        # to >1.
        self._commit_lock = threading.Lock()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._is_closing = False

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor]:
        """Allocate pinned (page-locked) staging tensors for GPU->CPU copies.

        Tensors are reused from the pool when available to avoid repeated
        allocations on the hot path.

        Args:
            shape: Tensor shape to allocate.
            dtype: Tensor dtype to allocate.
            count: Number of tensors needed.

        Returns:
            List of ``count`` pinned CPU tensors.
        """
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
        """Return staging tensors to the pool for reuse.

        Args:
            chunks: Tensors previously obtained from :meth:`_alloc_pinned_staging`.
        """
        if not chunks:
            return
        key = (tuple(chunks[0].shape), chunks[0].dtype)
        with self._inflight_lock:
            self._staging_pool.setdefault(key, []).extend(chunks)

    def _alloc_staging_for_plans(
        self,
        plans: list[GroupCopyPlan],
        num_chunks_per_group: list[int],
    ) -> list[list[torch.Tensor]]:
        """Allocate pinned staging tensors for each group in a multi-group store.

        Args:
            plans: Per-group copy plans providing shape/dtype information.
            num_chunks_per_group: Number of chunks to allocate for each group.

        Returns:
            Per-group lists of pinned staging tensors.
        """
        layout_desc = self._engine_driven_context.layout_desc  # type: ignore[union-attr]
        staged_per_group: list[list[torch.Tensor]] = []
        for g_idx, plan in enumerate(plans):
            n = num_chunks_per_group[g_idx]
            if g_idx < len(layout_desc.shapes) and g_idx < len(layout_desc.dtypes):
                shape = layout_desc.shapes[g_idx]
                dtype = layout_desc.dtypes[g_idx]
            elif layout_desc.shapes and layout_desc.dtypes:
                shape = layout_desc.shapes[0]
                dtype = layout_desc.dtypes[0]
            else:
                raise RuntimeError("engine-driven layout_desc.shapes/dtypes is empty")
            staged_per_group.append(self._alloc_pinned_staging(shape, dtype, n))
        return staged_per_group

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
        """Three-phase async store (prepare, gather and commit all in background).

        Performs only O(1) work on the forward thread (registration check),
        then submits all three phases — prepare_store, gather (GPU->CPU), and
        commit — to the background ``commit_executor``.  Returns an unresolved
        future that resolves only after all three phases complete.

        For hybrid/HMA models with multiple engine groups the gather phase runs
        once per group via :func:`~.group_copy.gather_engine_groups`, producing
        a group-major flat chunk list that is committed in a single call.

        Args:
            _request_id: External request identifier (used for logging).
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            _event: Synchronization event; ``wait()`` is called in background.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            An unresolved :class:`MessagingFuture` that resolves to ``True``
            on success, ``False`` on failure.

        Raises:
            RuntimeError: If register() was not called first.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        completion: MessagingFuture[bool] = MessagingFuture()
        engine_driven_context = self._engine_driven_context
        commit_executor = self._commit_executor

        # Build group plans on the forward thread (O(1), dict-slicing only).
        plans = self._build_group_plans(kv_caches, block_ids)
        is_multi_group = bool(plans)

        # For single-group we still flatten block_ids here so the background
        # closure can capture the flat list without holding a reference to the
        # full kv_caches or block_ids.
        flat_block_ids_single: list[int] = []
        if not is_multi_group:
            try:
                flat_block_ids_single = _single_group_block_ids(block_ids)
            except RuntimeError:
                completion.set_result(False)
                return completion

        # Signals when this task has recorded its CUDA event (or exited early),
        # allowing flush_inflight_stores to safely proceed.
        gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            def _prepare_gather_and_commit() -> None:
                gather_done: Any | None = None
                ok = False
                used_shm_direct = False
                prepared_shm_store = False
                abort_attempted = False
                staged_chunks_single: list[torch.Tensor] = []
                staged_per_group: list[list[torch.Tensor]] = []

                def attempt_abort_once() -> None:
                    nonlocal abort_attempted
                    if not prepared_shm_store or abort_attempted:
                        return
                    abort_attempted = True
                    try:
                        engine_driven_context.abort_store(key, instance_id)
                    except Exception:
                        logger.exception(
                            "Failed to abort async SHM store for request_id=%s",
                            _request_id,
                        )

                try:
                    # --- Phase 1: prepare_store ---
                    result = engine_driven_context.prepare_store(key, instance_id)
                    out_buffers, chunk_indices, server_group_counts = (
                        result if result is not None else (None, None, [])
                    )
                    prepared_shm_store = out_buffers is not None

                    if chunk_indices is not None and len(chunk_indices) == 0:
                        ok = True
                        return

                    # --- Phase 2: gather (GPU->CPU on copy stream) ---
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)

                        if is_multi_group:
                            # Multi-group gather.
                            out_per_group, ci_per_group = _split_shm_buffers_by_group(
                                out_buffers,
                                chunk_indices,
                                plans,
                                server_group_counts=server_group_counts,
                            )
                            if out_buffers is not None:
                                used_shm_direct = True
                                gather_target_flat = out_buffers
                            else:
                                # Allocate staging for each group.
                                num_chunks_per_group = [p.num_chunks for p in plans]
                                staged_per_group = self._alloc_staging_for_plans(
                                    plans, num_chunks_per_group
                                )
                                out_per_group = staged_per_group  # type: ignore[assignment]
                                ci_per_group = [None] * len(plans)

                            chunks_per_group = gather_engine_groups(
                                plans,
                                layout_hints=self._layout_hints,
                                out_per_group=out_per_group,
                                chunk_indices_per_group=ci_per_group,
                            )
                            gather_target_flat = flatten_chunks_group_major(
                                chunks_per_group
                            )
                        else:
                            # Single-group gather.
                            num_chunks = (
                                len(chunk_indices)
                                if chunk_indices is not None
                                else len(flat_block_ids_single) // blocks_in_chunk
                            )
                            if out_buffers is not None:
                                gather_target = out_buffers
                                used_shm_direct = True
                            else:
                                layout_desc = engine_driven_context.layout_desc
                                if not layout_desc.shapes or not layout_desc.dtypes:
                                    raise RuntimeError(
                                        "engine-driven layout_desc.shapes/dtypes empty"
                                    )
                                staged_chunks_single = self._alloc_pinned_staging(
                                    layout_desc.shapes[0],
                                    layout_desc.dtypes[0],
                                    num_chunks,
                                )
                                gather_target = staged_chunks_single
                            gather_paged_kv_to_cpu(
                                kv_caches,
                                flat_block_ids_single,
                                blocks_in_chunk,
                                layout_hints=self._layout_hints,
                                engine_kv_format=self._engine_kv_format,
                                out=gather_target,
                                chunk_indices=chunk_indices,
                            )
                            gather_target_flat = gather_target

                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()

                    if gather_done is not None:
                        gather_done.synchronize()

                    # --- Phase 3: commit ---
                    with self._commit_lock:
                        ok = engine_driven_context.commit_store(
                            key, instance_id, gather_target_flat
                        )

                    if not ok:
                        attempt_abort_once()
                        logger.error(
                            "Async engine-driven commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    attempt_abort_once()
                    logger.exception(
                        "Async engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    if not used_shm_direct:
                        self._release_staging(staged_chunks_single)
                        for staged in staged_per_group:
                            self._release_staging(staged)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    completion.set_result(ok)

            commit_executor.submit(_prepare_gather_and_commit)
        except Exception:
            logger.exception("Failed to submit async engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
            gather_launched.set()
            completion.set_result(False)
            return completion

        return completion

    def flush_inflight_stores(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Called at preemption/eviction time so that vLLM cannot overwrite
        paged KV blocks before a deferred gather has finished reading them.

        Waits for all submitted-but-not-yet-launched stores to record their
        CUDA events before synchronizing those events, preventing a race where
        ``flush_inflight_stores`` returns before a background gather has
        started.
        """
        with self._inflight_lock:
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=False)

    def close(self) -> None:
        """Drain in-flight gather/commit work before closing the base context."""
        with self._inflight_lock:
            self._is_closing = True
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=True)
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        super().close()

    def _sync_gather_events(self, suppress_errors: bool = False) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Args:
            suppress_errors: If True, log exceptions instead of propagating.
        """
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                if not suppress_errors:
                    raise
                logger.exception("Failed while draining gather events")
