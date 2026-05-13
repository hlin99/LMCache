# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from typing import Any, Callable

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import is_mla
from lmcache.v1.multiprocess.cpu_context import (
    CPUContext,
    CPUContextMetadata,
    compute_kv_layout,
    create_cpu_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.protocol import RequestType

logger = init_logger(__name__)

SendRequest = Callable[[Any, RequestType, list[Any]], Any]


def _require_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    """Return a required keyword argument or raise ValueError."""
    if key not in kwargs:
        raise ValueError(f"Missing required keyword argument: {key}")
    return kwargs[key]


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer."""

    @abstractmethod
    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: Any,
        mq_timeout: float,
        **kwargs: Any,
    ) -> None:
        """Register KV caches with the server and wait for ACK."""

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        """Submit a store request."""

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        """Submit a retrieve request."""

    @abstractmethod
    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        """Return ``(finished_store_ids, finished_retrieve_ids, error_block_ids)``."""

    @abstractmethod
    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        """Drain all pending requests for unhealthy mode."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class CudaTransferContext(TransferContext):
    """CUDA IPC + MQ future transport context."""

    def __init__(self) -> None:
        self._store_futures: dict[str, Any] = {}
        self._retrieve_futures: dict[str, tuple[Any, list[int]]] = {}

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: Any,
        mq_timeout: float,
        **kwargs: Any,
    ) -> None:
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints

        del blocks_in_chunk
        send_request: SendRequest = _require_kwarg(kwargs, "send_request")
        wrap_kv_caches: Callable[[dict[str, torch.Tensor]], Any] = _require_kwarg(
            kwargs, "wrap_kv_caches"
        )
        layout_hints = vllm_layout_hints()
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
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        del kv_caches, blocks_in_chunk
        send_request: SendRequest = _require_kwarg(kwargs, "send_request")
        future = send_request(
            _require_kwarg(kwargs, "mq_client"),
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self._store_futures[request_id] = future

    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        del kv_caches, blocks_in_chunk
        send_request: SendRequest = _require_kwarg(kwargs, "send_request")
        future = send_request(
            _require_kwarg(kwargs, "mq_client"),
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), skip_first_n_tokens],
        ).to_cuda_future()
        self._retrieve_futures[request_id] = (future, list(block_ids))

    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores: set[str] = set()
        finished_retrieves: set[str] = set()
        error_block_ids: set[int] = set()

        for request_id, s_future in list(self._store_futures.items()):
            if not s_future.query():
                continue
            s_result = s_future.result()
            finished_stores.add(request_id)
            if not s_result:
                logger.error(
                    "Something went wrong when processing the store request "
                    "for request_id=%s",
                    request_id,
                )
            self._store_futures.pop(request_id, None)

        for request_id, (r_future, r_block_ids) in list(self._retrieve_futures.items()):
            if not r_future.query():
                continue
            r_result = r_future.result()
            finished_retrieves.add(request_id)
            if not r_result:
                logger.error(
                    "Something went wrong when processing the retrieve request "
                    "for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )
                error_block_ids.update(r_block_ids)
            self._retrieve_futures.pop(request_id, None)

        return finished_stores, finished_retrieves, error_block_ids

    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores = set(self._store_futures.keys())
        finished_retrieves = set(self._retrieve_futures.keys())
        error_block_ids: set[int] = set()
        for _request_id, (_r_future, block_ids) in self._retrieve_futures.items():
            error_block_ids.update(block_ids)
        self._store_futures.clear()
        self._retrieve_futures.clear()
        return finished_stores, finished_retrieves, error_block_ids

    def close(self) -> None:
        self._store_futures.clear()
        self._retrieve_futures.clear()


class CPUTransferContext(TransferContext):
    """CPU context transport for non-CUDA workers."""

    def __init__(self) -> None:
        self._cpu_context: CPUContext | None = None
        self._layout_hints: Any = None
        self._gpu_kv_format: Any = None
        self._store_done: dict[str, bool] = {}
        self._retrieve_done: dict[str, tuple[bool, list[int]]] = {}

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: Any,
        mq_timeout: float,
        **kwargs: Any,
    ) -> None:
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints

        send_request: SendRequest = _require_kwarg(kwargs, "send_request")
        layout_hints = vllm_layout_hints()
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            gpu_kv_format,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._gpu_kv_format = gpu_kv_format

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_CPU_CONTEXT,
            [
                instance_id,
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                block_size,
                num_layers,
                hidden_dim_size,
                dtype_str,
                is_mla(gpu_kv_format),
            ],
        )

        use_mla_flag = is_mla(gpu_kv_format)
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        metadata = CPUContextMetadata(
            layout_desc=MemoryLayoutDesc(shapes=[shape], dtypes=[dtype]),
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._cpu_context = create_cpu_context(metadata, mq_client, mq_timeout)
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        del kwargs, event
        if self._cpu_context is None:
            raise RuntimeError("CPU transfer context is not registered")
        torch_dev.synchronize()
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            block_ids,
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            gpu_kv_format=self._gpu_kv_format,
        )
        handle = self._cpu_context.prepare_store(key, instance_id, cpu_chunks)
        ok = self._cpu_context.commit_store(handle)
        self._store_done[request_id] = ok

    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: Any,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        del kwargs, event
        if self._cpu_context is None:
            raise RuntimeError("CPU transfer context is not registered")
        handle, chunks = self._cpu_context.prepare_retrieve(key, instance_id)
        ok = chunks is not None
        if chunks is not None:
            try:
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    block_ids,
                    chunks,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    gpu_kv_format=self._gpu_kv_format,
                )
            except Exception:
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
        self._cpu_context.commit_retrieve(handle)
        self._retrieve_done[request_id] = (ok, list(block_ids))

    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores = set(self._store_done.keys())
        finished_retrieves = set(self._retrieve_done.keys())
        error_block_ids: set[int] = set()
        for ok, block_ids in self._retrieve_done.values():
            if not ok:
                error_block_ids.update(block_ids)
        self._store_done.clear()
        self._retrieve_done.clear()
        return finished_stores, finished_retrieves, error_block_ids

    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        return self.poll_finished()

    def close(self) -> None:
        if self._cpu_context is not None:
            self._cpu_context.close()
            self._cpu_context = None
        self._store_done.clear()
        self._retrieve_done.clear()


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    **kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here.
    """
    del kwargs
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    logger.info("Creating transfer context (device_type=%s)", device_type)
    if device_type == "cuda":
        return CudaTransferContext()
    return CPUTransferContext()
