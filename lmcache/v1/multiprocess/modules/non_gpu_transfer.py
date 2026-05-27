# SPDX-License-Identifier: Apache-2.0
"""Non-GPU transfer module for the MP cache engine compositor.

Integrates with :mod:`~lmcache.v1.multiprocess.server_transfer` to provide
SHM or pickle transport based on :attr:`MPServerConfig.shm_name`.
"""

# Standard
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    RegisterNonGpuContextPayload,
)
from lmcache.v1.multiprocess.engine_context import (
    MPCacheEngineContext,
    RegisteredContext,
)
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterNonGpuContextResponse,
)
from lmcache.v1.multiprocess.server_transfer import (
    PickleTransferStrategy,
    ShmTransferStrategy,
    TransferStrategy,
)
from lmcache.v1.multiprocess.worker_transfer.base import NonGpuContextMetadata

logger = init_logger(__name__)


class NonGPUTransferModule:
    """Non-GPU KV cache transfer module.

    Handles registration of non-GPU contexts and two-phase store/retrieve
    operations. Uses :class:`~lmcache.v1.multiprocess.server_transfer.ShmTransferStrategy`
    when SHM is configured, and falls back to
    :class:`~lmcache.v1.multiprocess.server_transfer.PickleTransferStrategy` otherwise.

    Args:
        context: Shared engine context (storage manager, chunk size, etc.).
        shm_name: Override for the SHM segment name. When ``None`` the storage-
            manager config controls the pool name.  When ``""`` SHM is
            disabled (force pickle). When a non-empty string, use that name.
    """

    def __init__(
        self,
        context: MPCacheEngineContext,
        shm_name: str | None = None,
    ) -> None:
        self._ctx = context

        # Pending SHM transfer tracking, keyed by (instance_id, IPC key).
        self._pending_shm_writes: dict[
            tuple[int, IPCCacheEngineKey], list[ObjectKey]
        ] = {}
        self._pending_shm_reads: dict[
            tuple[int, IPCCacheEngineKey], list[ObjectKey]
        ] = {}
        self._pending_shm_lock = threading.Lock()
        self._strategies: dict[int, TransferStrategy] = {}

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    def get_handlers(self) -> list[HandlerSpec]:
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_NON_GPU_CONTEXT,
                self.register_kv_cache_non_gpu_context,
                ThreadPoolType.CPU,
            ),
            HandlerSpec(
                RequestType.PREPARE_STORE,
                self.prepare_store,
                ThreadPoolType.GPU,
            ),
            HandlerSpec(
                RequestType.COMMIT_STORE,
                self.commit_store,
                ThreadPoolType.GPU,
            ),
            HandlerSpec(
                RequestType.PREPARE_RETRIEVE,
                self.prepare_retrieve,
                ThreadPoolType.GPU,
            ),
            HandlerSpec(
                RequestType.COMMIT_RETRIEVE,
                self.commit_retrieve,
                ThreadPoolType.GPU,
            ),
        ]

    def report_status(self) -> dict:
        non_cuda_context_meta: dict[str, dict] = {}
        registered_non_cuda_ids: list[int] = []

        with self._ctx._contexts_lock:
            contexts = list(self._ctx.contexts.items())
        for instance_id, context in contexts:
            if context.non_cuda_metadata is not None:
                registered_non_cuda_ids.append(instance_id)
                non_cuda_context_meta[str(instance_id)] = {
                    "model_name": context.model_name,
                    "world_size": context.world_size,
                    "block_size": context.non_cuda_metadata.block_size,
                    "use_mla": context.non_cuda_metadata.use_mla,
                }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Non-GPU context management
    # ------------------------------------------------------------------

    def get_shm_pool_info(self) -> dict:
        """Return shared-memory pool metadata for non-GPU SHM transport."""
        return self._ctx.get_shm_pool_info()

    def register_kv_cache_non_gpu_context(
        self,
        payload: RegisterNonGpuContextPayload,
    ) -> RegisterNonGpuContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla).

        Returns:
            Registration response with SHM pool info if SHM is active.

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        with self._ctx._contexts_lock:
            existing_context = self._ctx.contexts.get(payload.instance_id)
            if existing_context is not None:
                logger.warning(
                    "Instance %s's KV cache is already registered, "
                    "skipping the new registration",
                    payload.instance_id,
                )
                return self._build_existing_non_gpu_context_response(existing_context)

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        shape = (
            torch.Size([payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size])
            if payload.use_mla
            else torch.Size(
                [2, payload.num_layers, self._ctx.chunk_size, payload.hidden_dim_size]
            )
        )
        from lmcache.v1.distributed.api import MemoryLayoutDesc

        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        shm_pool_info = self.get_shm_pool_info()
        shm_active = False
        strategy: TransferStrategy = PickleTransferStrategy(self._ctx.storage_manager)
        if not isinstance(shm_pool_info, dict):
            logger.info(
                "Instance %s non-GPU context using pickle transport "
                "(no SHM pool info returned)",
                payload.instance_id,
            )
            response = RegisterNonGpuContextResponse()
        else:
            shm_name = str(shm_pool_info.get("shm_name", ""))
            pool_size = int(shm_pool_info.get("pool_size", 0))
            shm_active = bool(shm_name) and pool_size > 0
            response = RegisterNonGpuContextResponse(
                shm_name=shm_name,
                pool_size=pool_size,
            )
            if shm_active:
                strategy = ShmTransferStrategy(
                    storage_manager=self._ctx.storage_manager,
                    pending_writes=self._pending_shm_writes,
                    pending_reads=self._pending_shm_reads,
                    pending_lock=self._pending_shm_lock,
                    transfer_key_factory=self._make_non_gpu_transfer_key,
                    fallback_strategy=PickleTransferStrategy(self._ctx.storage_manager),
                )
                logger.info(
                    "Instance %s non-GPU context using SHM transport "
                    "(shm_name=%s, pool_size=%d)",
                    payload.instance_id,
                    response.shm_name,
                    response.pool_size,
                )
            else:
                logger.info(
                    "Instance %s non-GPU context using pickle transport",
                    payload.instance_id,
                )

        with self._ctx._contexts_lock:
            existing_context = self._ctx.contexts.get(payload.instance_id)
            if existing_context is not None:
                logger.warning(
                    "Instance %s's KV cache is already registered, "
                    "skipping the new registration",
                    payload.instance_id,
                )
                return self._build_existing_non_gpu_context_response(existing_context)
            self._ctx.contexts[payload.instance_id] = RegisteredContext(
                model_name=payload.model_name,
                world_size=payload.world_size,
                non_cuda_metadata=NonGpuContextMetadata(
                    layout_desc=layout_desc,
                    block_size=payload.block_size,
                    use_mla=payload.use_mla,
                ),
                shm_active=shm_active,
            )
            self._strategies[payload.instance_id] = strategy
        return response

    def _build_existing_non_gpu_context_response(
        self, existing_context: RegisteredContext
    ) -> RegisterNonGpuContextResponse:
        """Build a response for an already-registered non-GPU context."""
        if existing_context.shm_active:
            pool_info = self.get_shm_pool_info()
            if isinstance(pool_info, dict):
                return RegisterNonGpuContextResponse(
                    shm_name=str(pool_info.get("shm_name", "")),
                    pool_size=int(pool_info.get("pool_size", 0)),
                )
        return RegisterNonGpuContextResponse()

    def cleanup_non_gpu_context(self, instance_id: int) -> None:
        """Release SHM resources for a departing non-GPU worker.

        Called from the GPU transfer module's ``unregister_kv_cache`` when the
        departing context is non-GPU, or directly when the server tears down.
        """
        with self._pending_shm_lock:
            stale_writes = {
                k: v
                for k, v in self._pending_shm_writes.items()
                if k[0] == instance_id
            }
            for transfer_key in stale_writes:
                del self._pending_shm_writes[transfer_key]
            stale_reads = {
                k: v
                for k, v in self._pending_shm_reads.items()
                if k[0] == instance_id
            }
            for transfer_key in stale_reads:
                del self._pending_shm_reads[transfer_key]
        for reserved_keys in stale_writes.values():
            if reserved_keys:
                self._ctx.storage_manager.finish_write(reserved_keys)
        for prefetched_keys in stale_reads.values():
            if prefetched_keys:
                self._ctx.storage_manager.finish_read_prefetched(prefetched_keys)
        self._strategies.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Non-GPU transfer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_non_gpu_transfer_key(
        key: IPCCacheEngineKey, instance_id: int
    ) -> tuple[int, IPCCacheEngineKey]:
        """Build a unique key for pending SHM write/read transfer tracking."""
        return (instance_id, key)

    def _get_non_gpu_context_metadata(self, instance_id: int) -> NonGpuContextMetadata:
        """Return non-GPU context metadata for a registered instance."""
        with self._ctx._contexts_lock:
            context = self._ctx.contexts.get(instance_id)
        if context is None or context.non_cuda_metadata is None:
            raise ValueError(
                f"non-CUDA context not registered for instance ID {instance_id}"
            )
        return context.non_cuda_metadata

    def _validate_non_gpu_context_exists(self, instance_id: int) -> None:
        """Validate that a non-GPU context exists for a worker instance."""
        self._get_non_gpu_context_metadata(instance_id)

    def _get_transfer_strategy(self, instance_id: int) -> TransferStrategy:
        """Return the registered transfer strategy for a worker instance."""
        strategy = self._strategies.get(instance_id)
        if strategy is None:
            raise ValueError(
                f"transfer strategy not registered for instance ID {instance_id}"
            )
        return strategy

    # ------------------------------------------------------------------
    # Two-phase non-GPU store/retrieve
    # ------------------------------------------------------------------

    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare writable memory slots on the server side for a store operation."""
        strategy = self._get_transfer_strategy(instance_id)
        context = self._get_non_gpu_context_metadata(instance_id)
        return strategy.prepare_store(
            key=key,
            instance_id=instance_id,
            context=context,
            resolve_obj_keys=self._ctx.resolve_obj_keys,
        )

    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Finalize a store operation after :meth:`prepare_store`."""
        strategy = self._get_transfer_strategy(instance_id)
        context = self._get_non_gpu_context_metadata(instance_id)
        return strategy.commit_store(
            key=key,
            instance_id=instance_id,
            cpu_data=cpu_data,
            context=context,
            resolve_obj_keys=self._ctx.resolve_obj_keys,
        )

    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Prepare readable memory slots on the server side for a retrieve operation."""
        strategy = self._get_transfer_strategy(instance_id)
        self._validate_non_gpu_context_exists(instance_id)
        return strategy.prepare_retrieve(
            key=key,
            instance_id=instance_id,
            resolve_obj_keys=self._ctx.resolve_obj_keys,
        )

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation after :meth:`prepare_retrieve`."""
        strategy = self._get_transfer_strategy(instance_id)
        self._validate_non_gpu_context_exists(instance_id)
        return strategy.commit_retrieve(key=key, instance_id=instance_id)
