# SPDX-License-Identifier: Apache-2.0
"""Shared state context for the MP cache engine compositor."""

# Standard
from dataclasses import dataclass
from functools import partial
import sys
import shutil
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import PrefetchHandle, StorageManager
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.worker_transfer.base import NonGpuContextMetadata

logger = init_logger(__name__)


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 for early-exit paths (no GPU context matches
    # or chunk_hashes is empty).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""


@dataclass
class RegisteredContext:
    """Registered context metadata for a single worker instance.

    At least one of ``gpu_context`` or ``non_cuda_metadata`` is expected to be
    populated for valid registrations.
    """

    model_name: str
    world_size: int
    gpu_context: GPUCacheContext | None = None
    non_cuda_metadata: NonGpuContextMetadata | None = None
    shm_active: bool = False

    @property
    def is_gpu(self) -> bool:
        """Return whether this registration uses a GPU transfer context."""
        return self.gpu_context is not None

    def get_layout_desc(self, chunk_size: int) -> MemoryLayoutDesc:
        """Return the layout descriptor for this registration.

        Args:
            chunk_size: Chunk size in tokens used for GPU layout derivation.

        Returns:
            The resolved memory layout descriptor.

        Raises:
            ValueError: If no GPU context or non-CUDA metadata is configured.
        """
        if self.gpu_context is not None:
            # Inline the layout computation to avoid a circular import with
            # modules.gpu_transfer (which itself imports engine_context).
            gpu_context = self.gpu_context
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            shapes = [
                gpu_context.get_kv_buffer_shape(chunk_size, group_idx)
                for group_idx in range(num_groups)
            ]
            dtypes = [
                gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
                for group_idx in range(num_groups)
            ]
            return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)
        if self.non_cuda_metadata is None:
            raise ValueError(
                "Invalid RegisteredContext: no GPU or non-CUDA metadata configured"
            )
        return self.non_cuda_metadata.layout_desc


class MPCacheEngineContext:
    """Shared state container for MP cache engine modules.

    Holds all resources that are shared across modules:
    - Worker context registry (GPU + non-GPU)
    - Storage manager
    - Token hasher / session manager
    - Event bus
    - Device host function dispatcher (GIL-free GPU callbacks)
    - SHM pool metadata
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ) -> None:
        # Worker instance ID -> registered context metadata
        self.contexts: dict[int, RegisteredContext] = {}
        self._contexts_lock = threading.Lock()

        # Chunk size
        self.chunk_size = chunk_size

        # Lock for clear() to avoid concurrent storage manager mutations
        self.lock = threading.Lock()

        # Storage manager
        self._storage_manager_config = storage_manager_config
        self._resolve_shm_config(self._storage_manager_config)
        self.storage_manager = StorageManager(self._storage_manager_config)
        self._shm_pool_info = self._compute_shm_pool_info(self._storage_manager_config)

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

        # EventBus for observability
        self._event_bus = get_event_bus()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

        self._setup_metrics()

    def resolve_obj_keys(self, key: IPCCacheEngineKey) -> list[ObjectKey]:
        """Resolve object keys from an IPC cache key.

        Args:
            key: IPC cache key describing model/session/token range.

        Returns:
            Resolved object keys for the requested token range.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes)

    def find_layout_desc(
        self,
        model_name: str,
        world_size: int,
    ) -> MemoryLayoutDesc | None:
        """Find layout desc from a matching GPU or CPU context.

        Returns:
            The layout descriptor, or None if no context matches
            ``(model_name, world_size)``. GPU contexts are checked first,
            then CPU contexts.
        """
        with self._contexts_lock:
            contexts = list(self.contexts.values())
        for context in contexts:
            if context.model_name == model_name and context.world_size == world_size:
                return context.get_layout_desc(self.chunk_size)
        return None

    def get_shm_pool_info(self) -> dict:
        """Return shared-memory pool metadata for non-GPU SHM transport."""
        return dict(self._shm_pool_info)

    @staticmethod
    def _resolve_shm_config(config: StorageManagerConfig) -> None:
        """Resolve SHM configuration in place before storage-manager creation.

        Args:
            config: Storage-manager config to mutate in place.

        Notes:
            Clears ``config.l1_manager_config.memory_config.shm_name`` when SHM
            transport should be disabled:
            - lazy allocation is enabled
            - /dev/shm free space is insufficient on Linux
            - /dev/shm capacity cannot be queried on Linux
        """
        mem_cfg = config.l1_manager_config.memory_config
        if not mem_cfg.shm_name or mem_cfg.use_lazy:
            return

        if sys.platform.startswith("linux"):
            try:
                free_bytes = shutil.disk_usage("/dev/shm").free
                if free_bytes < mem_cfg.size_in_bytes:
                    logger.warning(
                        "Insufficient /dev/shm capacity: need %d bytes, have %d bytes. "
                        "Disabling SHM transport.",
                        mem_cfg.size_in_bytes,
                        free_bytes,
                    )
                    mem_cfg.shm_name = ""
            except OSError:
                logger.warning(
                    "Cannot verify /dev/shm capacity required for SHM transport; "
                    "disabling SHM mode.",
                    exc_info=True,
                )
                mem_cfg.shm_name = ""
        else:
            logger.debug(
                "Skipping /dev/shm capacity pre-check on non-Linux platform %s",
                sys.platform,
            )

    @staticmethod
    def _compute_shm_pool_info(config: StorageManagerConfig) -> dict:
        """Compute effective SHM pool metadata from storage manager config.

        Returns:
            A dict with:
            - ``shm_name`` (str): Effective SHM segment name, normalized to include
              ``lmcache_l1_pool_`` prefix when non-empty.
            - ``pool_size`` (int): SHM pool size in bytes.

            Returns ``{"shm_name": "", "pool_size": 0}`` when lazy allocation is
            enabled or SHM is disabled.
        """
        mem_cfg = config.l1_manager_config.memory_config
        shm_name = mem_cfg.shm_name or ""
        if not shm_name or mem_cfg.use_lazy:
            return {"shm_name": "", "pool_size": 0}

        stripped_name = shm_name.lstrip("/")
        if not stripped_name.startswith("lmcache_l1_pool_"):
            shm_name = f"lmcache_l1_pool_{stripped_name}"

        return {"shm_name": shm_name, "pool_size": mem_cfg.size_in_bytes}

    def close(self) -> None:
        """Close the context and release all resources."""
        # Stop the drain thread before storage_manager.close()
        self._device_host_func_dispatcher.stop()
        self.storage_manager.close()
        logger.info("MPCacheEngineContext closed")

        with self._contexts_lock:
            self.contexts.clear()

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for MP engine metrics."""
        pass
