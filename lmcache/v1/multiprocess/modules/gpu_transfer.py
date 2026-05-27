# SPDX-License-Identifier: Apache-2.0
"""GPU transfer module for the MP cache engine compositor."""

# Standard
from functools import partial
from itertools import islice
from typing import Generator
import time
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
    check_interprocess_event_support,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import (
    MPCacheEngineContext,
    RegisteredContext,
)
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream
from lmcache.v1.multiprocess.protocol import RequestType
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def compute_extra_count(
    tp_size: int,
    world_size: int,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


def get_layout_desc(gpu_context: GPUCacheContext, num_tokens: int) -> MemoryLayoutDesc:
    """Get the memory layout description for a given GPU context and number of tokens.

    Supports multiple KV layer groups with different shapes and dtypes.

    Args:
        gpu_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    num_groups = gpu_context.kv_layer_groups_manager.num_groups
    shapes = [
        gpu_context.get_kv_buffer_shape(num_tokens, group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
        for group_idx in range(num_groups)
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def batched_iteration(lst: list, batch_size: int) -> Generator[tuple, None, None]:
    """Utility function to iterate over a list in batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


class GPUTransferModule:
    """GPU KV cache transfer module.

    Handles registration and unregistration of GPU contexts as well as
    STORE and RETRIEVE operations via CUDA IPC.
    """

    def __init__(self, context: MPCacheEngineContext) -> None:
        self._ctx = context

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    def get_handlers(self) -> list[HandlerSpec]:
        return [
            HandlerSpec(RequestType.REGISTER_KV_CACHE, self.register_kv_cache, ThreadPoolType.CPU),
            HandlerSpec(RequestType.UNREGISTER_KV_CACHE, self.unregister_kv_cache, ThreadPoolType.CPU),
            HandlerSpec(RequestType.STORE, self.store, ThreadPoolType.GPU),
            HandlerSpec(RequestType.RETRIEVE, self.retrieve, ThreadPoolType.GPU),
        ]

    def report_status(self) -> dict:
        gpu_context_meta: dict[str, dict] = {}
        registered_gpu_ids: list[int] = []

        with self._ctx._contexts_lock:
            contexts = list(self._ctx.contexts.items())
        for instance_id, context in contexts:
            if context.gpu_context is not None:
                registered_gpu_ids.append(instance_id)
                ctx = context.gpu_context
                entry: dict = {
                    "model_name": context.model_name,
                    "world_size": context.world_size,
                    "kv_cache_layout": {
                        "num_layers": ctx.num_layers,
                        "inference_engine_logical_block_size": (
                            ctx.kv_layer_groups_manager.inference_engine_logical_block_size
                        ),
                        "group_physical_block_sizes": ctx.group_physical_block_sizes,
                        "group_compress_ratios": ctx.group_compress_ratios,
                        "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                        "dtype": str(ctx.dtype),
                        "is_mla": ctx.is_mla,
                        "num_blocks": ctx.num_blocks,
                        "gpu_kv_format": ctx.gpu_kv_format_name,
                        "gpu_kv_shape": ctx.gpu_kv_shape,
                        "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                        "attention_backend": ctx.attention_backend,
                        "cache_size_per_token": ctx.cache_size_per_token(),
                    },
                }
                gpu_context_meta[str(instance_id)] = entry

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "gpu_context_meta": gpu_context_meta,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # GPU-context management
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
        """Register the KV cache tensors for a given GPU instance ID."""
        with self._ctx._contexts_lock:
            if instance_id in self._ctx.contexts:
                logger.warning(
                    "Instance %s's KV cache is already registered, "
                    "skipping the new registration",
                    instance_id,
                )
                return

            gpu_context = GPUCacheContext(
                kv_caches,
                self._ctx.chunk_size,
                layout_hints=layout_hints or None,
                engine_type=engine_type,
            )
            self._ctx.contexts[instance_id] = RegisteredContext(
                model_name=model_name,
                world_size=world_size,
                gpu_context=gpu_context,
            )
        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister context for a given instance ID (GPU path)."""
        with self._ctx._contexts_lock:
            context = self._ctx.contexts.pop(instance_id, None)

        if context is None:
            logger.warning(
                "No registered context found for instance ID %d", instance_id
            )
            return

        if context.is_gpu:
            logger.info("Unregistered KV cache for GPU ID %d", instance_id)
            torch_dev.empty_cache()
        else:
            logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    # ------------------------------------------------------------------
    # GPU STORE
    # ------------------------------------------------------------------

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: The GPU block IDs to store.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            tuple of (event IPC handle, success flag).
        """
        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        with self._ctx._contexts_lock:
            context = self._ctx.contexts.get(instance_id)
        assert context is not None, (
            f"No context registered for instance ID {instance_id}"
        )
        assert context.gpu_context is not None, (
            f"GPU context not registered for instance ID {instance_id}"
        )
        gpu_context = context.gpu_context
        model_name = context.model_name

        blocks_per_chunk = (
            self._ctx.chunk_size
            // gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC event "
                    "handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            self._ctx._event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                )
            )

            self._ctx._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(gpu_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict = {}
            try:
                layout_desc = get_layout_desc(gpu_context, self._ctx.chunk_size)
                reserved_dict = self._ctx.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                num_groups = gpu_context.kv_layer_groups_manager.num_groups
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key in reserved_dict:
                        memory_obj = reserved_dict[obj_key]
                    else:
                        continue

                    chunk_block_ids_gpu = all_block_ids_gpu[
                        idx * blocks_per_chunk : (idx + 1) * blocks_per_chunk
                    ]

                    for group_idx in range(num_groups):
                        tmp_buffer = gpu_context.get_tmp_chunk_gpu_buffer(group_idx)
                        group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                        group_lmcache_chunk_size = gpu_context.get_physical_chunk_size(
                            group_idx
                        )
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.D2H,
                            gpu_context.get_shape_desc(group_idx),
                            group_lmcache_chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )
                    lmcache_memcpy_async_d2h(
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
                    )
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event.record()
                if reserved_dict:
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_write",
                        list(reserved_dict.keys()),
                    )
                total_bytes = (
                    next(iter(reserved_dict.values())).get_size() * len(reserved_dict)
                    if reserved_dict
                    else 0
                )
                self._ctx._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": len(reserved_dict),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                        },
                    ),
                )

        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self._ctx.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True

    # ------------------------------------------------------------------
    # GPU RETRIEVE
    # ------------------------------------------------------------------

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: The GPU block IDs to retrieve into.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip at the start.

        Returns:
            tuple of (event IPC handle, success flag).
        """
        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        with self._ctx._contexts_lock:
            context = self._ctx.contexts.get(instance_id)
        assert context is not None, (
            f"No context registered for instance ID {instance_id}"
        )
        assert context.gpu_context is not None, (
            f"GPU context not registered for instance ID {instance_id}"
        )
        gpu_context = context.gpu_context
        model_name = context.model_name

        self._ctx._event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            )
        )

        self._ctx._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(gpu_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        ie_logical_block_size = (
            gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )
        blocks_per_chunk = self._ctx.chunk_size // ie_logical_block_size

        def _retrieve_loop(
            keys: list[ObjectKey], memory_objs: list[MemoryObj]
        ) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self._ctx.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self._ctx.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self._ctx.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % ie_logical_block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to "
                        "inference_engine_logical_block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        ie_logical_block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // ie_logical_block_size,
                    )
                skip_blocks_in_chunk = skip_tokens_in_chunk // ie_logical_block_size

                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                chunk_block_ids_gpu = all_block_ids_gpu[
                    start_chunk_id * blocks_per_chunk : end_chunk_id * blocks_per_chunk
                ]

                for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                    )
                for group_idx in range(num_groups):
                    tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                    group_lmcache_chunk_size = gpu_context.get_physical_chunk_size(
                        group_idx
                    )

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.get_shape_desc(group_idx),
                        group_lmcache_chunk_size,
                        gpu_context.gpu_kv_format_,
                        skip_blocks_in_chunk,
                    )

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            total_bytes = 0
            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    total_bytes = sum(mo.get_size() for mo in memory_objs)
                    _retrieve_loop(obj_keys, memory_objs)
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                self._ctx._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                        },
                    ),
                )
        tokens_retrieved = len(obj_keys) * self._ctx.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), True
