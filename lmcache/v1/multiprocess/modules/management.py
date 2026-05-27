# SPDX-License-Identifier: Apache-2.0
"""Session and status management module for the MP cache engine compositor."""

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord, IPCCacheEngineKey
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

logger = init_logger(__name__)


class ManagementModule:
    """Session and server management module.

    Handles PING, GET_CHUNK_SIZE, END_SESSION, CLEAR, NOOP, REPORT_STATUS,
    and REPORT_BLOCK_ALLOCATION requests.
    """

    def __init__(
        self,
        context: MPCacheEngineContext,
        lookup_module: "LookupModule | None" = None,
    ) -> None:
        self._ctx = context
        self._lookup_module = lookup_module

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    def get_handlers(self) -> list[HandlerSpec]:
        return [
            HandlerSpec(RequestType.PING, self.ping, ThreadPoolType.CPU),
            HandlerSpec(
                RequestType.GET_CHUNK_SIZE, self.get_chunk_size, ThreadPoolType.CPU
            ),
            HandlerSpec(
                RequestType.END_SESSION, self.end_session, ThreadPoolType.CPU
            ),
            HandlerSpec(RequestType.CLEAR, self.clear, ThreadPoolType.CPU),
            HandlerSpec(RequestType.NOOP, self.debug, ThreadPoolType.CPU),
            HandlerSpec(
                RequestType.REPORT_BLOCK_ALLOCATION,
                self.report_block_allocations,
                ThreadPoolType.CPU,
            ),
        ]

    def report_status(self) -> dict:
        sm = self._ctx.storage_manager.report_status()
        active_prefetch = 0
        if self._lookup_module is not None:
            active_prefetch = self._lookup_module._active_prefetch_count()
        return {
            "is_healthy": sm["is_healthy"],
            "chunk_size": self._ctx.chunk_size,
            "hash_algorithm": self._ctx.token_hasher.hash_algorithm_name,
            "active_sessions": self._ctx.session_manager.active_count(),
            "active_prefetch_jobs": active_prefetch,
            "storage_manager": sm,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Management operations
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Respond to a ping request."""
        return True

    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations."""
        return self._ctx.chunk_size

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request."""
        from lmcache.v1.distributed.api import ipc_key_to_object_keys

        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self._ctx.session_manager.remove(request_id)
        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.debug("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.debug(
                "Session %s has no lookup ipc key, skipping touch", request_id
            )
            return

        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)
        ]
        obj_keys = ipc_key_to_object_keys(session.lookup_ipc_key, chunk_hashes)
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

    def clear(self) -> None:
        """Clear all stored KV cache data from the storage manager."""
        with self._ctx.lock:
            self._ctx.storage_manager.memcheck()
            self._ctx.storage_manager.clear(force=True)
            self._ctx.storage_manager.memcheck()

    def debug(self) -> str:
        return "OK"

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus."""
        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )
