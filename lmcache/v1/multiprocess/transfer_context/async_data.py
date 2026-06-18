# SPDX-License-Identifier: Apache-2.0
"""Async non-GPU data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.store_timer import StoreTimer
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    DataTransferContext,
    IPCEvent,
    _single_group_block_ids,
)

logger = init_logger(__name__)

DEFAULT_MAX_ASYNC_NON_GPU_STORES = 8
# Number of background threads used to run commit (CPU->server) work for the
# async non-GPU store path. >1 so that a slow gather for one store does not
# block the commit of another store whose gather already finished.
DEFAULT_NON_GPU_COMMIT_WORKERS = 4


class AsyncDataTransferContext(DataTransferContext):
    """Fully async non-GPU data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    that resolves only after the deferred gather (GPU->CPU copy) and commit
    (CPU->server) both complete off the forward thread, while
    ``submit_retrieve`` stays synchronous and returns an already-resolved
    future exactly as on the base context.

    Inherits :class:`DataTransferContext` and reuses its ``register()`` (layout
    / SHM registration, no stream dependency) and ``submit_retrieve()`` (this
    path does not change retrieve). Only the store is made async.

    Store is two-phase, both executed entirely in a background thread:
    1) gather: wait for the forward event on the copy stream, then enqueue
       GPU->CPU copies. When SHM buffers are available, gather writes directly
       into SHM views (matching the synchronous path). Otherwise, gather
       targets pinned staging buffers.
    2) commit: wait for gather completion (via a recorded CUDA event), then
       perform commit_store() and resolve the returned future.

    ``submit_store`` only performs lightweight preparation (prepare_store,
    buffer allocation) on the forward thread and immediately submits all
    GPU/copy work to the background ``commit_executor``, so the forward thread
    is never blocked by gather kernel launch latency.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.
    """

    def __init__(
        self,
        max_inflight_stores: int = DEFAULT_MAX_ASYNC_NON_GPU_STORES,
        commit_workers: int = DEFAULT_NON_GPU_COMMIT_WORKERS,
    ) -> None:
        """Initialize the async context and create its async resources.

        Args:
            max_inflight_stores: Max number of concurrently in-flight async
                stores before ``submit_store`` blocks (backpressure).
            commit_workers: Number of background threads used to run commit
                (CPU->server) work. >1 so a slow gather for one store does not
                block the commit of another whose gather is already done.
        """
        super().__init__()
        self._max_inflight_stores = max(1, int(max_inflight_stores))
        self._commit_workers = max(1, int(commit_workers))
        self._copy_stream: Any = torch_dev.Stream()
        self._commit_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_non_gpu_commit",
        )
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        self._inflight_commits: set[ConcurrentFuture[None]] = set()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._is_closing = False

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
        """Two-phase async store (gather and commit both in background thread).

        Performs lightweight preparation (prepare_store, buffer allocation) on
        the forward thread and immediately submits the gather + commit work to
        the background ``commit_executor``.  Returns an unresolved future that
        resolves only after both gather completion and the commit ACK.
        """
        if self._non_gpu_context is None:
            raise RuntimeError(
                "Data transfer context is not registered. "
                "Call register() before submit_store()."
            )
        _timer = StoreTimer(_request_id, path="data")
        completion: MessagingFuture[bool] = MessagingFuture()
        non_gpu_context = self._non_gpu_context
        commit_executor = self._commit_executor

        staged_chunks: list[torch.Tensor] = []
        # Whether we gathered directly into SHM views (True) or into
        # pinned staging buffers that need to be released later (False).
        used_shm_direct = False
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion

            result = non_gpu_context.prepare_store(key, instance_id)

            out_buffers, chunk_indices = result if result is not None else (None, None)
            if chunk_indices is not None and len(chunk_indices) == 0:
                # All chunks are already in cache: no gather, no commit.
                completion.set_result(True)
                return completion

            full_block_ids = _single_group_block_ids(block_ids)

            num_chunks = (
                len(chunk_indices)
                if chunk_indices is not None
                else len(full_block_ids) // blocks_in_chunk
            )

            # Determine gather target:
            # - SHM path (out_buffers available): gather directly into SHM views
            # - Pickle path (no out_buffers): gather into pinned staging buffers
            if out_buffers is not None:
                # SHM path: gather directly into SHM views, no staging needed.
                gather_target = out_buffers
                used_shm_direct = True
            else:
                # Pickle path: allocate pinned staging buffers.
                if not non_gpu_context.layout_desc.shapes:
                    raise RuntimeError("non-GPU layout_desc.shapes is empty")
                if not non_gpu_context.layout_desc.dtypes:
                    raise RuntimeError("non-GPU layout_desc.dtypes is empty")
                staged_chunks = self._alloc_pinned_staging(
                    non_gpu_context.layout_desc.shapes[0],
                    non_gpu_context.layout_desc.dtypes[0],
                    num_chunks,
                )
                gather_target = staged_chunks

            _timer.set_path("shm" if used_shm_direct else "pickle")

            # Capture variables for the closure
            _used_shm_direct = used_shm_direct
            _gather_target = gather_target

            def _commit_after_gather() -> None:
                gather_done: Any | None = None
                ok = False
                try:
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)

                        gather_paged_kv_to_cpu(
                            kv_caches,
                            full_block_ids,
                            blocks_in_chunk,
                            layout_hints=self._layout_hints,
                            gpu_kv_format=self._gpu_kv_format,
                            out=_gather_target,
                            chunk_indices=chunk_indices,
                        )

                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)

                    # All GPU copies have been enqueued and CPU has returned.
                    _timer.mark("copy_submitted")

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)

                    if gather_done is not None:
                        gather_done.synchronize()
                    _timer.mark("kv_releasable")

                    ok = non_gpu_context.commit_store(key, instance_id, _gather_target)
                    _timer.mark("e2e_complete")
                    _timer.emit()

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
                    if not _used_shm_direct:
                        self._release_staging(staged_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                    completion.set_result(ok)

            # Submitting the commit task is the ownership-transfer point: once it
            # succeeds, the commit task is solely responsible for releasing the
            # staging buffers, and resolving the future. The except below therefore
            # only handles failures that occur *before* this submit, so it can never
            # double-release or double-resolve.
            commit_future = commit_executor.submit(_commit_after_gather)
        except Exception:
            logger.exception("Failed to submit async non-GPU store")
            if staged_chunks:
                self._release_staging(staged_chunks)
            completion.set_result(False)
            return completion

        with self._inflight_lock:
            self._inflight_commits.add(commit_future)

        def _drop_commit_future(done_future: ConcurrentFuture[None]) -> None:
            with self._inflight_lock:
                self._inflight_commits.discard(done_future)

        commit_future.add_done_callback(_drop_commit_future)

        _timer.mark("fwd_return")
        return completion

    def flush_inflight_gathers(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Called at preemption/eviction time (and during ``close``) so that vLLM
        cannot overwrite paged KV blocks before a deferred gather has finished
        reading them. Only gather completion is awaited; commit futures are not
        affected, since commits read from LMCache-owned staging buffers.
        """
        _t0 = time.perf_counter()
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            event.synchronize()
        _t1 = time.perf_counter()
        logger.info(
            "[flush_inflight_gathers] synced %d events in %.3f ms",
            len(gather_events),
            (_t1 - _t0) * 1000,
        )

    def close(self) -> None:
        # Drain in-flight gather/commit work before closing the base context.
        _t0 = time.perf_counter()
        with self._inflight_lock:
            self._is_closing = True
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                logger.exception("Failed while draining gather events")
        _t1 = time.perf_counter()
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        _t2 = time.perf_counter()
        logger.info(
            "[close] gather_drain=%.3f executor_shutdown=%.3f total=%.3f ms",
            (_t1 - _t0) * 1000,
            (_t2 - _t1) * 1000,
            (_t2 - _t0) * 1000,
        )
        super().close()
