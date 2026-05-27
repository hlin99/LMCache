# SPDX-License-Identifier: Apache-2.0
"""Lookup / prefetch module for the MP cache engine compositor."""

# Standard
import time
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext, _PrefetchJob
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.modules.gpu_transfer import compute_extra_count
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class LookupModule:
    """Lookup and prefetch module.

    Handles LOOKUP, QUERY_PREFETCH_STATUS, QUERY_PREFETCH_LOOKUP_HITS, and
    FREE_LOOKUP_LOCKS requests.
    """

    def __init__(self, context: MPCacheEngineContext) -> None:
        self._ctx = context
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    def get_handlers(self) -> list[HandlerSpec]:
        return [
            HandlerSpec(RequestType.LOOKUP, self.lookup, ThreadPoolType.CPU),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_STATUS,
                self.query_prefetch_status,
                ThreadPoolType.CPU,
            ),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                self.query_prefetch_lookup_hits,
                ThreadPoolType.CPU,
            ),
            HandlerSpec(
                RequestType.FREE_LOOKUP_LOCKS,
                self.free_lookup_locks,
                ThreadPoolType.CPU,
            ),
        ]

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lookup operations
    # ------------------------------------------------------------------

    def lookup(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        model_name, world_size = key.model_name, key.world_size
        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._ctx.find_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        extra_count = compute_extra_count(tp_size, world_size)

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        if self._ctx._event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx._event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        handle = self._ctx.storage_manager.submit_prefetch_task(
            obj_keys,
            layout_desc,
            extra_count=extra_count,
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=key.world_size,
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                model_name=model_name,
                cache_salt=key.cache_salt,
            )
        )

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self._ctx.storage_manager.query_prefetch_lookup_hits(job.handle)
        if found_count is None:
            return None

        found_count = found_count // job.world_size
        return found_count

    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if found_count is None:
            return None

        found_count = found_count // job.world_size

        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    "hit_tokens": found_count * self._ctx.chunk_size,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)

        return found_count

    def free_lookup_locks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(tp_size, key.world_size)

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys, extra_count=extra_count
        )

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (public interface)."""
        return self._active_prefetch_count()
