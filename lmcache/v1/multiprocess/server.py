# SPDX-License-Identifier: Apache-2.0
"""Thin compositor MP cache engine server.

``MPCacheEngine`` orchestrates a set of pluggable
:class:`~lmcache.v1.multiprocess.engine_module.EngineModule` instances built
from :func:`_build_modules`. Each module exposes handlers via
:meth:`~lmcache.v1.multiprocess.engine_module.EngineModule.get_handlers`; the
compositor wires them to the ZMQ message-queue server in
:func:`run_cache_server`.

Delegation methods on ``MPCacheEngine`` forward calls to the appropriate
module so that callers and tests that use the ``MPCacheEngine`` API directly
continue to work without modification.
"""

# Standard
from functools import partial
import argparse
import time

# Third Party
import torch
import zmq

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheEngineKey,
    KVCache,
    RegisterNonGpuContextPayload,
)
from lmcache.v1.multiprocess.engine_context import (
    MPCacheEngineContext,
    RegisteredContext,
    _PrefetchJob,
)
from lmcache.v1.multiprocess.engine_module import (
    EngineModule,
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.modules.non_gpu_transfer import NonGPUTransferModule
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterNonGpuContextResponse,
)

logger = init_logger(__name__)


def _build_modules(
    context: MPCacheEngineContext,
    mp_config: MPServerConfig,
) -> tuple[GPUTransferModule, NonGPUTransferModule, LookupModule, ManagementModule]:
    """Build and return the set of engine modules.

    Args:
        context: Shared engine state context.
        mp_config: MP server configuration.

    Returns:
        Tuple of (gpu_module, non_gpu_module, lookup_module, management_module).
    """
    gpu_module = GPUTransferModule(context)
    non_gpu_module = NonGPUTransferModule(context, shm_name=mp_config.shm_name)
    lookup_module = LookupModule(context)
    management_module = ManagementModule(context, lookup_module=lookup_module)
    return gpu_module, non_gpu_module, lookup_module, management_module


class MPCacheEngine:
    """Thin compositor that orchestrates a set of pluggable engine modules.

    This class holds a :class:`~lmcache.v1.multiprocess.engine_context.MPCacheEngineContext`
    and a list of :class:`~lmcache.v1.multiprocess.engine_module.EngineModule` instances.
    Delegation methods forward calls to the appropriate module so that existing
    callers of the ``MPCacheEngine`` API continue to work.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        mp_config: MPServerConfig | None = None,
    ) -> None:
        if mp_config is None:
            mp_config = MPServerConfig(chunk_size=chunk_size, hash_algorithm=hash_algorithm)

        self._context = MPCacheEngineContext(
            storage_manager_config=storage_manager_config,
            chunk_size=chunk_size,
            hash_algorithm=hash_algorithm,
        )

        (
            self._gpu_module,
            self._non_gpu_module,
            self._lookup_module,
            self._management_module,
        ) = _build_modules(self._context, mp_config)

        self._modules: list[EngineModule] = [
            self._gpu_module,
            self._non_gpu_module,
            self._lookup_module,
            self._management_module,
        ]

        self._setup_metrics()

    # ------------------------------------------------------------------
    # EngineModule aggregation helpers
    # ------------------------------------------------------------------

    def get_all_handlers(self) -> list[HandlerSpec]:
        """Return all handler specs from every module."""
        handlers: list[HandlerSpec] = []
        for module in self._modules:
            handlers.extend(module.get_handlers())
        return handlers

    # ------------------------------------------------------------------
    # Legacy property (backward compat)
    # ------------------------------------------------------------------

    @property
    def gpu_contexts(self) -> dict[int, GPUCacheContext]:
        """Return GPU-only context mapping for backward compatibility."""
        with self._context._contexts_lock:
            contexts = list(self._context.contexts.items())
        return {
            instance_id: ctx.gpu_context
            for instance_id, ctx in contexts
            if ctx.gpu_context is not None
        }

    # ------------------------------------------------------------------
    # Delegation: GPU context management
    # ------------------------------------------------------------------

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
    ) -> None:
        """Register GPU KV cache for a worker instance."""
        self._gpu_module.register_kv_cache(
            instance_id, kv_caches, model_name, world_size, engine_type, layout_hints
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister context for a worker instance (GPU or non-GPU)."""
        with self._context._contexts_lock:
            context = self._context.contexts.get(instance_id)

        if context is None:
            logger.warning(
                "No registered context found for instance ID %d", instance_id
            )
            return

        if context.is_gpu:
            self._gpu_module.unregister_kv_cache(instance_id)
        else:
            with self._context._contexts_lock:
                self._context.contexts.pop(instance_id, None)
            logger.info(
                "Unregistered non-CUDA context for instance ID %d", instance_id
            )
            self._non_gpu_module.cleanup_non_gpu_context(instance_id)

    # ------------------------------------------------------------------
    # Delegation: non-GPU context management
    # ------------------------------------------------------------------

    def register_kv_cache_non_gpu_context(
        self,
        payload: RegisterNonGpuContextPayload,
    ) -> RegisterNonGpuContextResponse:
        """Register non-CUDA KV layout metadata."""
        return self._non_gpu_module.register_kv_cache_non_gpu_context(payload)

    def get_shm_pool_info(self) -> dict:
        """Return SHM pool metadata."""
        return self._non_gpu_module.get_shm_pool_info()

    # ------------------------------------------------------------------
    # Delegation: non-GPU two-phase store/retrieve
    # ------------------------------------------------------------------

    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare writable memory slots for a store operation."""
        return self._non_gpu_module.prepare_store(key, instance_id)

    def commit_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Finalize a store operation."""
        return self._non_gpu_module.commit_store(key, instance_id, cpu_data)

    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Prepare readable memory slots for a retrieve operation."""
        return self._non_gpu_module.prepare_retrieve(key, instance_id)

    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation."""
        return self._non_gpu_module.commit_retrieve(key, instance_id)

    # ------------------------------------------------------------------
    # Delegation: GPU store/retrieve
    # ------------------------------------------------------------------

    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store GPU KV cache blocks to CPU."""
        return self._gpu_module.store(key, instance_id, gpu_block_ids, event_ipc_handle)

    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve CPU KV cache and put into GPU blocks."""
        return self._gpu_module.retrieve(
            key, instance_id, gpu_block_ids, event_ipc_handle, skip_first_n_tokens
        )

    # ------------------------------------------------------------------
    # Delegation: lookup
    # ------------------------------------------------------------------

    def lookup(self, key: IPCCacheEngineKey, tp_size: int) -> None:
        """Submit a prefix lookup."""
        return self._lookup_module.lookup(key, tp_size)

    def query_prefetch_status(self, request_id: str) -> int | None:
        """Poll the status of a prefetch job."""
        return self._lookup_module.query_prefetch_status(request_id)

    def query_prefetch_lookup_hits(self, request_id: str) -> int | None:
        """Query the number of hits for a prefetch request."""
        return self._lookup_module.query_prefetch_lookup_hits(request_id)

    def free_lookup_locks(self, key: IPCCacheEngineKey, tp_size: int) -> None:
        """Release read locks acquired during lookup."""
        return self._lookup_module.free_lookup_locks(key, tp_size)

    # ------------------------------------------------------------------
    # Delegation: management
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Respond to a ping request."""
        return self._management_module.ping()

    def get_chunk_size(self) -> int:
        """Return the chunk size used for KV cache operations."""
        return self._management_module.get_chunk_size()

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request."""
        return self._management_module.end_session(request_id)

    def clear(self) -> None:
        """Clear all stored KV cache data from the storage manager."""
        return self._management_module.clear()

    def debug(self) -> str:
        return self._management_module.debug()

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus."""
        return self._management_module.report_block_allocations(
            instance_id, model_name, records
        )

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for the entire cache engine."""
        status = {
            "engine_type": self.__class__.__name__,
        }
        for module in self._modules:
            status.update(module.report_status())
        return status

    def _find_layout_desc(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Find layout desc from a matching context (backward compat)."""
        return self._context.find_layout_desc(model_name, world_size)

    # ------------------------------------------------------------------
    # Static methods (backward compat for tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_shm_config(config: StorageManagerConfig) -> None:
        """Delegate to MPCacheEngineContext._resolve_shm_config."""
        MPCacheEngineContext._resolve_shm_config(config)

    @staticmethod
    def _compute_shm_pool_info(config: StorageManagerConfig) -> dict:
        """Delegate to MPCacheEngineContext._compute_shm_pool_info."""
        return MPCacheEngineContext._compute_shm_pool_info(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the MPCacheEngine and release all resources."""
        for module in self._modules:
            module.close()
        self._context.close()
        logger.info("MPCacheEngine closed")

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        return self._lookup_module._active_prefetch_count()

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for MP engine metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_engine")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
    start_prometheus_http_server: bool = True,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive
        start_prometheus_http_server: Whether to start a standalone
            Prometheus HTTP server in a background thread.  Set to
            ``False`` when an external HTTP framework already serves
            ``/metrics`` to avoid port conflicts or redundant servers.

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    event_bus = init_observability(
        obs_config, start_prometheus_http_server=start_prometheus_http_server
    )

    # Wire up the trace recorder (no-op when --trace-level is unset).
    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # Apply shm_name override from MP config to storage config.
    if mp_config.shm_name is not None:
        storage_manager_config.l1_manager_config.memory_config.shm_name = (
            mp_config.shm_name
        )

    # Initialize the engine (creates context + modules)
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
        mp_config=mp_config,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
    )

    # Register handlers from engine (delegation methods, same as before)
    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(
        server,
        RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
        engine.register_kv_cache_non_gpu_context,
    )
    add_handler_helper(server, RequestType.PREPARE_STORE, engine.prepare_store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(
        server, RequestType.QUERY_PREFETCH_STATUS, engine.query_prefetch_status
    )
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        engine.query_prefetch_lookup_hits,
    )
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.COMMIT_STORE, engine.commit_store)
    add_handler_helper(server, RequestType.PREPARE_RETRIEVE, engine.prepare_retrieve)
    add_handler_helper(server, RequestType.COMMIT_RETRIEVE, engine.commit_retrieve)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)
    add_handler_helper(
        server,
        RequestType.REPORT_BLOCK_ALLOCATION,
        engine.report_block_allocations,
    )

    # Assign thread pools
    server.add_affinity_thread_pool(
        [
            RequestType.STORE,
            RequestType.RETRIEVE,
            RequestType.PREPARE_STORE,
            RequestType.COMMIT_STORE,
            RequestType.PREPARE_RETRIEVE,
            RequestType.COMMIT_RETRIEVE,
        ],
        max_workers=mp_config.max_gpu_workers,
    )
    server.add_normal_thread_pool(
        [
            RequestType.LOOKUP,
            RequestType.QUERY_PREFETCH_STATUS,
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            RequestType.FREE_LOOKUP_LOCKS,
            RequestType.END_SESSION,
            RequestType.CLEAR,
            RequestType.PING,
            RequestType.REPORT_BLOCK_ALLOCATION,
        ],
        max_workers=mp_config.max_cpu_workers,
    )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    if not hasattr(torch_dev, "init"):
        logger.warning(
            "Backend '%s' does not support init(), skipping device init",
            torch_device_type,
        )
    else:
        torch_dev.init()
    server.start()

    logger.info("LMCache cache server is running...")

    # Return server and engine if requested (for HTTP server integration)
    if return_engine:
        return server, engine

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        event_bus.stop()
        server.close()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
