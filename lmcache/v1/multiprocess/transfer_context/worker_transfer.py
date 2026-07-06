# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints, is_mla
from lmcache.v1.multiprocess.custom_types import (
    EngineDrivenKernelGroupMetadata,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_plan import TransferPlanBuilder
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform import get_device_info

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        platform_registry.get_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_info = get_device_info(device_type)
    if device_info and not device_info.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for IPC-capable CUDA events used by transport operations."""

    def ipc_handle(self) -> object:
        """Return an IPC handle consumable by the multiprocess server."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


@dataclass(frozen=True)
class _EngineDrivenKernelGroupPlanMeta:
    """Per-kernel-group metadata needed to build engine-driven transfer plans."""

    kernel_group_id: int
    object_group_id: int
    engine_group_id: int
    layer_indices: tuple[int, ...]
    tokens_per_block: int
    slots_per_block: int
    sw_size_tokens: int
    num_layers: int
    dtype: torch.dtype
    hidden_dim_size: int
    engine_kv_format: Any


def _sw_size_chunks(sw_size_tokens: int, chunk_size: int) -> int:
    """Return the object-group sliding-window size in LMCache chunks."""
    if sw_size_tokens <= 0:
        return -1
    return (sw_size_tokens + chunk_size - 1) // chunk_size


def _shape_for_kernel_group(
    group: _EngineDrivenKernelGroupPlanMeta,
    num_tokens: int,
    *,
    use_mla: bool,
) -> torch.Size:
    """Build the CPU chunk shape for one engine-driven kernel group."""
    num_slots = (
        0
        if num_tokens <= 0
        else num_tokens * group.slots_per_block // group.tokens_per_block
    )
    if use_mla:
        return torch.Size([group.num_layers, num_slots, group.hidden_dim_size])
    return torch.Size([2, group.num_layers, num_slots, group.hidden_dim_size])


def _layout_descs_by_object_group(
    groups: Sequence[_EngineDrivenKernelGroupPlanMeta],
    *,
    chunk_size: int,
    use_mla: bool,
) -> dict[int, MemoryLayoutDesc]:
    """Build per-object-group layout descriptors from kernel group metadata."""
    shapes_by_group: dict[int, list[torch.Size]] = {}
    dtypes_by_group: dict[int, list[torch.dtype]] = {}
    for group in groups:
        shapes_by_group.setdefault(group.object_group_id, []).append(
            _shape_for_kernel_group(group, chunk_size, use_mla=use_mla)
        )
        dtypes_by_group.setdefault(group.object_group_id, []).append(group.dtype)
    return {
        object_group_id: MemoryLayoutDesc(
            shapes=shapes_by_group[object_group_id],
            dtypes=dtypes_by_group[object_group_id],
        )
        for object_group_id in shapes_by_group
    }


def _kernel_group_metadata(
    groups: Sequence[_EngineDrivenKernelGroupPlanMeta],
    *,
    chunk_size: int,
    use_mla: bool,
) -> tuple[EngineDrivenKernelGroupMetadata, ...]:
    """Convert worker planning metadata into wire-safe registration metadata."""
    return tuple(
        EngineDrivenKernelGroupMetadata(
            kernel_group_id=group.kernel_group_id,
            object_group_id=group.object_group_id,
            engine_group_id=group.engine_group_id,
            layer_indices=group.layer_indices,
            tokens_per_block=group.tokens_per_block,
            slots_per_block=group.slots_per_block,
            dtype_str=str(group.dtype).replace("torch.", ""),
            engine_kv_format=str(group.engine_kv_format),
            sw_size_tokens=group.sw_size_tokens,
            shape=tuple(_shape_for_kernel_group(group, chunk_size, use_mla=use_mla)),
        )
        for group in groups
    )


def _build_engine_driven_kernel_groups(
    *,
    num_layers: int,
    block_size: int,
    chunk_size: int,
    dtype: torch.dtype,
    hidden_dim_size: int,
    engine_kv_format: Any,
    engine_group_infos: Sequence[EngineGroupInfo],
) -> list[_EngineDrivenKernelGroupPlanMeta]:
    """Build engine-driven per-kernel-group planning metadata."""
    if not engine_group_infos:
        return [
            _EngineDrivenKernelGroupPlanMeta(
                kernel_group_id=0,
                object_group_id=0,
                engine_group_id=0,
                layer_indices=tuple(range(num_layers)),
                tokens_per_block=block_size,
                slots_per_block=block_size,
                sw_size_tokens=-1,
                num_layers=num_layers,
                dtype=dtype,
                hidden_dim_size=hidden_dim_size,
                engine_kv_format=engine_kv_format,
            )
        ]

    object_group_id_by_sw_chunks: dict[int, int] = {}
    next_object_group_id = 0
    metas: list[_EngineDrivenKernelGroupPlanMeta] = []
    for kg_id, info in enumerate(engine_group_infos):
        sw_size_tokens = info.sw_size_tokens
        sw_size_chunks = _sw_size_chunks(sw_size_tokens, chunk_size)
        if sw_size_chunks not in object_group_id_by_sw_chunks:
            object_group_id_by_sw_chunks[sw_size_chunks] = next_object_group_id
            next_object_group_id += 1
        layer_indices = tuple(info.layer_indices)
        metas.append(
            _EngineDrivenKernelGroupPlanMeta(
                kernel_group_id=kg_id,
                object_group_id=object_group_id_by_sw_chunks[sw_size_chunks],
                engine_group_id=info.engine_group_id,
                layer_indices=layer_indices,
                tokens_per_block=(
                    info.tokens_per_block if info.tokens_per_block > 0 else block_size
                ),
                slots_per_block=block_size,
                sw_size_tokens=sw_size_tokens,
                num_layers=max(1, len(layer_indices)),
                dtype=dtype,
                hidden_dim_size=hidden_dim_size,
                engine_kv_format=engine_kv_format,
            )
        )
    return metas


class _EngineDrivenPlanKVGroupsManager:
    """Minimal kv-layer-groups-manager view required by TransferPlanBuilder."""

    def __init__(
        self, groups: Sequence[_EngineDrivenKernelGroupPlanMeta], chunk_size: int
    ) -> None:
        object_group_map: dict[int, list[int]] = {}
        for group in groups:
            object_group_map.setdefault(group.object_group_id, []).append(
                group.kernel_group_id
            )
        self.object_groups = [
            type(
                "_ObjectGroup",
                (),
                {"kernel_group_indices": list(kernel_group_ids)},
            )()
            for _object_group_id, kernel_group_ids in sorted(object_group_map.items())
        ]
        self.num_kernel_groups = len(groups)
        self.num_object_groups = len(self.object_groups)
        self._groups_by_kg_id = {group.kernel_group_id: group for group in groups}
        self._chunk_size = chunk_size

    def get_subchunk_sw_size_tokens(self, kernel_group_id: int) -> int:
        group = self._groups_by_kg_id[kernel_group_id]
        if group.sw_size_tokens <= 0 or group.sw_size_tokens >= self._chunk_size:
            return self._chunk_size
        return group.sw_size_tokens

    def get_attn_desc(self) -> Any:
        # First Party
        from lmcache.v1.distributed.api import AttnWindowDesc

        sw_chunks = []
        for object_group in self.object_groups:
            first_kg_id = object_group.kernel_group_indices[0]
            sw_size_tokens = self._groups_by_kg_id[first_kg_id].sw_size_tokens
            sw_chunks.append(_sw_size_chunks(sw_size_tokens, self._chunk_size))
        return AttnWindowDesc(num_chunks_in_sw=sw_chunks)


class _EngineDrivenPlanCacheContext:
    """Minimal cache-context adapter required by TransferPlanBuilder."""

    def __init__(
        self,
        *,
        chunk_size: int,
        groups: Sequence[_EngineDrivenKernelGroupPlanMeta],
        use_mla: bool,
    ) -> None:
        self.lmcache_tokens_per_chunk = chunk_size
        self.max_batch_size = 1
        self.kv_layer_groups_manager = _EngineDrivenPlanKVGroupsManager(
            groups, chunk_size
        )
        self._groups_by_kg_id = {group.kernel_group_id: group for group in groups}
        self._use_mla = use_mla

    def calculate_num_blocks(self, num_tokens: int, kernel_group_id: int) -> int:
        group = self._groups_by_kg_id[kernel_group_id]
        if num_tokens <= 0:
            return 0
        num_physical_slots = (
            num_tokens * group.slots_per_block // group.tokens_per_block
        )
        return num_physical_slots // group.slots_per_block

    def get_kernel_group_shape_dtype(
        self, num_tokens: int, kernel_group_id: int
    ) -> tuple[torch.Size, torch.dtype]:
        group = self._groups_by_kg_id[kernel_group_id]
        return (
            _shape_for_kernel_group(group, num_tokens, use_mla=self._use_mla),
            group.dtype,
        )

    def get_slots_per_chunk_in_sw(self, kernel_group_id: int) -> int:
        group = self._groups_by_kg_id[kernel_group_id]
        sw_tokens = self.kv_layer_groups_manager.get_subchunk_sw_size_tokens(
            kernel_group_id
        )
        if sw_tokens > self.lmcache_tokens_per_chunk:
            sw_tokens = self.lmcache_tokens_per_chunk
        num_blocks = self.calculate_num_blocks(sw_tokens, kernel_group_id)
        return num_blocks * group.slots_per_block

    def get_shape_desc(self, _kernel_group_id: int) -> Any:
        return "engine_driven_pickle_shape_desc"

    def get_engine_kv_format(self, kernel_group_id: int) -> Any:
        return self._groups_by_kg_id[kernel_group_id].engine_kv_format


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. CUDA paths return
    CUDA-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (IPC for CUDA,
    SHM wrappers for CPU with CUDA-IPC-like semantics) and the LMCache
    server performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        # First Party
        from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

        self._mq_client = mq_client
        self._send_request = send_request
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
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._mq_client is None or self._send_request is None:
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), skip_first_n_tokens],
        ).to_cuda_future()

    def close(self) -> None:
        self._mq_client = None
        self._send_request = None


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._engine_kernel_groups: list[_EngineDrivenKernelGroupPlanMeta] = []
        self._engine_chunk_size: int = 0
        self._use_mla: bool = False

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register KV caches with the non-GPU context server.

        ``engine_group_infos`` is used to construct a transfer-plan metadata
        view so the non-GPU path can execute grouped pickle transfers.
        """
        # TODO: per-group compression (EngineGroupInfo.tokens_per_block vs
        # the tensor-detected slot count, e.g. DeepSeek V4) is only handled
        # on the CUDA path. The non-CUDA path is yet to be implemented.
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        use_mla_flag = is_mla(engine_kv_format)
        self._use_mla = use_mla_flag
        chunk_size = blocks_in_chunk * block_size
        dtype = getattr(torch, dtype_str)
        self._engine_kernel_groups = _build_engine_driven_kernel_groups(
            num_layers=num_layers,
            block_size=block_size,
            chunk_size=chunk_size,
            dtype=dtype,
            hidden_dim_size=hidden_dim_size,
            engine_kv_format=engine_kv_format,
            engine_group_infos=engine_group_infos,
        )
        self._engine_chunk_size = chunk_size
        shape = (
            torch.Size([num_layers, chunk_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, chunk_size, hidden_dim_size]
            )
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    chunk_size=chunk_size,
                    engine_group_infos=tuple(engine_group_infos),
                    kernel_group_metadata=_kernel_group_metadata(
                        self._engine_kernel_groups,
                        chunk_size=chunk_size,
                        use_mla=use_mla_flag,
                    ),
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
            chunk_size=chunk_size,
            engine_group_infos=tuple(engine_group_infos),
            kernel_group_metadata=_kernel_group_metadata(
                self._engine_kernel_groups,
                chunk_size=chunk_size,
                use_mla=use_mla_flag,
            ),
            layout_descs_by_object_group=_layout_descs_by_object_group(
                self._engine_kernel_groups,
                chunk_size=chunk_size,
                use_mla=use_mla_flag,
            ),
        )
        use_pickle = len(self._engine_kernel_groups) > 1
        if use_pickle and shm_name and pool_size > 0:
            logger.info(
                "Engine-driven multi-group SHM is not implemented; "
                "forcing pickle transport."
            )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
            use_pickle=use_pickle,
        )
        supported_transfer_mode = (
            "pickle"
            if use_pickle
            else ("SHM" if shm_name and pool_size > 0 else "pickle")
        )
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            instance_id,
            supported_transfer_mode,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        store_plan = self._build_transfer_plan(
            key.request_id, block_ids, is_retrieve=False
        )
        if store_plan is None:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(False)
            return future

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        if out_buffers is not None:
            cpu_chunks: list[torch.Tensor] | dict[str, Any] = gather_paged_kv_to_cpu(
                kv_caches,
                store_plan.selected_block_ids_per_kernel_group[0],
                store_plan.object_groups[0].kernel_groups[0].blocks_per_window,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
                out=out_buffers,
                chunk_indices=chunk_indices,
            )
        else:
            cpu_chunks = self._gather_payload_by_plan(kv_caches, store_plan)
        if out_buffers is not None:
            # SHM path uses async device->CPU copies; complete them before commit.
            torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        retrieve_plan = self._build_transfer_plan(
            key.request_id,
            block_ids,
            is_retrieve=True,
            skip_first_n_tokens=skip_first_n_tokens,
        )
        if retrieve_plan is None:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(False)
            return future

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                self._scatter_payload_by_plan(
                    kv_caches,
                    retrieve_plan,
                    src_buffers,
                    skip_first_n_tokens=skip_first_n_tokens,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def _object_group_ids(self) -> list[int]:
        return sorted({group.object_group_id for group in self._engine_kernel_groups})

    def _kernel_group_meta(
        self, kernel_group_id: int
    ) -> _EngineDrivenKernelGroupPlanMeta:
        return self._engine_kernel_groups[kernel_group_id]

    def _slice_kv_caches_for_kernel_group(
        self,
        kv_caches: dict[str, torch.Tensor],
        group: _EngineDrivenKernelGroupPlanMeta,
    ) -> dict[str, torch.Tensor]:
        layer_items = list(kv_caches.items())
        if not group.layer_indices:
            return kv_caches
        try:
            return {
                layer_items[layer_idx][0]: layer_items[layer_idx][1]
                for layer_idx in group.layer_indices
            }
        except IndexError as exc:
            raise ValueError(
                "engine-driven kernel group references a layer index outside "
                "the registered KV cache"
            ) from exc

    def _build_transfer_plan(
        self,
        request_id: str,
        block_ids: list[list[int]],
        *,
        is_retrieve: bool,
        skip_first_n_tokens: int = 0,
    ) -> Any:
        if not self._engine_kernel_groups:
            return None
        cache_context = _EngineDrivenPlanCacheContext(
            chunk_size=self._engine_chunk_size,
            groups=self._engine_kernel_groups,
            use_mla=self._use_mla,
        )
        builder = TransferPlanBuilder(cache_context)
        object_keys_per_group: list[list[Any]] = []
        for object_group_id in self._object_group_ids():
            kernel_group_id = next(
                group.kernel_group_id
                for group in self._engine_kernel_groups
                if group.object_group_id == object_group_id
            )
            blocks_per_chunk = cache_context.calculate_num_blocks(
                cache_context.lmcache_tokens_per_chunk, kernel_group_id
            )
            if blocks_per_chunk <= 0:
                return None
            num_chunks = len(block_ids[kernel_group_id]) // blocks_per_chunk
            object_keys_per_group.append([f"chunk_{idx}" for idx in range(num_chunks)])
        if is_retrieve:
            return builder.build_retrieve_plan(
                request_id,
                object_keys_per_group,
                block_ids,
                skip_first_n_tokens=skip_first_n_tokens,
            )
        return builder.build_store_plan(request_id, object_keys_per_group, block_ids)

    def _gather_payload_by_plan(
        self,
        kv_caches: dict[str, torch.Tensor],
        transfer_plan: Any,
    ) -> list[torch.Tensor] | dict[str, Any]:
        selected_block_ids = transfer_plan.selected_block_ids_per_kernel_group
        if len(transfer_plan.object_groups) == 1 and len(selected_block_ids) == 1:
            kernel_group_plan = transfer_plan.object_groups[0].kernel_groups[0]
            group_meta = self._kernel_group_meta(kernel_group_plan.kernel_group_id)
            return gather_paged_kv_to_cpu(
                self._slice_kv_caches_for_kernel_group(kv_caches, group_meta),
                selected_block_ids[0],
                kernel_group_plan.blocks_per_window,
                layout_hints=self._layout_hints,
                engine_kv_format=group_meta.engine_kv_format,
            )

        payload: dict[str, Any] = {"object_groups": {}}
        for object_group_plan in transfer_plan.object_groups:
            grouped_chunks: list[list[torch.Tensor]] = []
            for kernel_group_plan in object_group_plan.kernel_groups:
                group_meta = self._kernel_group_meta(kernel_group_plan.kernel_group_id)
                grouped_chunks.append(
                    gather_paged_kv_to_cpu(
                        self._slice_kv_caches_for_kernel_group(kv_caches, group_meta),
                        kernel_group_plan.selected_block_ids,
                        kernel_group_plan.blocks_per_window,
                        layout_hints=self._layout_hints,
                        engine_kv_format=group_meta.engine_kv_format,
                    )
                )
            num_chunks = (
                len(grouped_chunks[0])
                if grouped_chunks
                else len(object_group_plan.object_keys)
            )
            chunks = [
                [
                    grouped_chunks[group_idx][chunk_idx]
                    for group_idx in range(len(grouped_chunks))
                ]
                for chunk_idx in range(num_chunks)
            ]
            payload["object_groups"][object_group_plan.object_group_id] = {
                "chunk_indices": list(range(num_chunks)),
                "kernel_group_ids": [
                    kg.kernel_group_id for kg in object_group_plan.kernel_groups
                ],
                "chunks": chunks,
            }
        return payload

    def _scatter_payload_by_plan(
        self,
        kv_caches: dict[str, torch.Tensor],
        transfer_plan: Any,
        payload: list[torch.Tensor] | dict[str, Any],
        *,
        skip_first_n_tokens: int,
    ) -> None:
        if isinstance(payload, list):
            kernel_group_plan = transfer_plan.object_groups[0].kernel_groups[0]
            group_meta = self._kernel_group_meta(kernel_group_plan.kernel_group_id)
            scatter_cpu_to_paged_kv(
                self._slice_kv_caches_for_kernel_group(kv_caches, group_meta),
                transfer_plan.selected_block_ids_per_kernel_group[0],
                payload,
                kernel_group_plan.blocks_per_window,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=group_meta.engine_kv_format,
            )
            return

        object_groups_payload = payload.get("object_groups", {})
        for object_group_plan in transfer_plan.object_groups:
            object_group_payload = object_groups_payload.get(
                object_group_plan.object_group_id
            )
            if object_group_payload is None:
                raise ValueError("missing object group payload")
            chunks = object_group_payload.get("chunks", [])
            for kernel_group_idx, kernel_group_plan in enumerate(
                object_group_plan.kernel_groups
            ):
                group_meta = self._kernel_group_meta(kernel_group_plan.kernel_group_id)
                scatter_cpu_to_paged_kv(
                    self._slice_kv_caches_for_kernel_group(kv_caches, group_meta),
                    kernel_group_plan.selected_block_ids,
                    [chunk[kernel_group_idx] for chunk in chunks],
                    kernel_group_plan.blocks_per_window,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    engine_kv_format=group_meta.engine_kv_format,
                )


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return EngineDrivenTransferContext()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return EngineDrivenTransferContext()
