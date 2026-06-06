# SPDX-License-Identifier: Apache-2.0
"""Async non-GPU data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
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

    Inherits :class:`DataTransferContext` and reuses its ``register()`` (layout
    / SHM registration, no stream dependency) and ``submit_retrieve()`` (this
    path does not change retrieve). Only the store is made async.

    Store is two-phase:
    1) gather: enqueue GPU->CPU copies on a dedicated copy stream into
       LMCache-owned pinned staging buffers (ordered behind the per-step event).
    2) commit: wait for gather completion in a background thread, then perform
       commit_store() (pickle or SHM commit) and resolve the returned future.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.

    SHM note: SHM slots are generally pageable, so device->SHM DtoH copies may
    implicitly synchronize. To keep gather async, we always gather into pinned
    bounce buffers first, then copy to SHM slots on the commit thread.
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
        self._inflight_semaphore = threading.BoundedSemaphore(self._max_inflight_stores)
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
        """Two-phase async store (gather on copy stream, deferred commit).

        Returns an unresolved future that resolves only after both gather
        completion and the commit ACK.
        """
        if self._non_gpu_context is None:
            raise RuntimeError(
                "Data transfer context is not registered. "
                "Call register() before submit_store()."
            )
        completion: MessagingFuture[bool] = MessagingFuture()
        non_gpu_context = self._non_gpu_context
        semaphore = self._inflight_semaphore
        commit_executor = self._commit_executor

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

    def flush_inflight_gathers(self) -> None:
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            event.synchronize()

    def close(self) -> None:
        # Drain in-flight gather/commit work before closing the base context.
        with self._inflight_lock:
            self._is_closing = True
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                logger.exception("Failed while draining gather events")
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        super().close()
