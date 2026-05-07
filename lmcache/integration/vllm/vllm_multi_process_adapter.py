# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from dataclasses import dataclass
from typing import Any, cast
import os
import pickle
import threading

# Third Party
import torch
import zmq

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.utils import EngineType, _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)

# Timeout (seconds) for blocking MQ requests: initial chunk-size query,
# KV cache registration/unregistration, and other synchronous operations.
DEFAULT_MQ_TIMEOUT: float = 300.0
# Interval (seconds) between periodic heartbeat pings to the server.
DEFAULT_HEARTBEAT_INTERVAL: float = 10.0
def _device_synchronize(device_type: str) -> None:
    """Synchronize device work for backends that require explicit barriers.

    Args:
        device_type: Active device type string (for example ``"cuda"``,
            ``"xpu"``, or ``"cpu"``).
    """
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def _compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: "Any | None" = None,
) -> tuple[int, int, int, str, Any]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``gpu_kv_format)``.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # Local
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_hidden_dim_size,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    gpu_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, gpu_kv_format)
    num_layers = get_num_layers(normalized, gpu_kv_format)
    hidden_dim_size = get_hidden_dim_size(normalized, gpu_kv_format)
    dtype_str = str(tensors[0].dtype).replace("torch.", "")
    return block_size, num_layers, hidden_dim_size, dtype_str, gpu_kv_format


def _gather_chunks_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: "Any | None" = None,
    gpu_kv_format: "Any | None" = None,
) -> bytes:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    Returns:
        Pickled list of CPU tensors. For non-MLA, each chunk shape is
        ``[2, num_layers, chunk_tokens, hidden_dim]`` where dimension ``0``
        stores ``(K, V)``. For MLA (multi-head latent attention), each chunk shape is
        ``[num_layers, chunk_tokens, hidden_dim]``.
    """
    # Local
    from lmcache.v1.gpu_connector.utils import (
        _get_head_size_view,
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)

    block_size = get_block_size(normalized, gpu_kv_format)
    device = tensors[0].device
    num_chunks = len(block_ids) // blocks_per_chunk

    chunks: list[torch.Tensor] = []
    for chunk_idx in range(num_chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        slot_mapping = torch.tensor(
            [b * block_size + t for b in chunk_block_ids for t in range(block_size)],
            dtype=torch.long,
            device=device,
        )
        if use_mla:
            mla_layers: list[torch.Tensor] = []
            for layer in normalized:
                layer_flat = cast(
                    torch.Tensor,
                    _get_head_size_view(
                        layer, use_mla=True, gpu_kv_format=gpu_kv_format
                    ),
                )
                mla_layers.append(layer_flat.index_select(0, slot_mapping))
            chunks.append(torch.stack(mla_layers, dim=0).cpu())
        else:
            k_layers: list[torch.Tensor] = []
            v_layers: list[torch.Tensor] = []
            for layer in normalized:
                k_flat, v_flat = _get_head_size_view(
                    layer, use_mla=False, gpu_kv_format=gpu_kv_format
                )
                k_layers.append(k_flat.index_select(0, slot_mapping))
                v_layers.append(v_flat.index_select(0, slot_mapping))
            k_stacked = torch.stack(k_layers, dim=0)
            v_stacked = torch.stack(v_layers, dim=0)
            chunks.append(torch.stack([k_stacked, v_stacked], dim=0).cpu())
    return pickle.dumps(chunks)


def _scatter_cpu_chunks_to_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    cpu_data: bytes,
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: "Any | None" = None,
    gpu_kv_format: "Any | None" = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.
        cpu_data: Serialized CPU chunk list.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.
        layout_hints: Optional engine layout hints.
        gpu_kv_format: Optional pre-detected KV format.

    """
    # Local
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.utils import (
        _get_head_size_view,
        get_block_size,
        is_mla,
        normalize_kv_and_discover_format,
    )

    chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    if not chunks:
        return

    tensors = list(kv_caches.values())
    fmt, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    if gpu_kv_format is None:
        gpu_kv_format = fmt
    use_mla = is_mla(gpu_kv_format)

    block_size = get_block_size(normalized, gpu_kv_format)
    device = tensors[0].device
    is_hnd = gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    )

    for chunk_idx, chunk_cpu in enumerate(chunks):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        if not chunk_block_ids:
            continue

        chunk_start_token = chunk_idx * blocks_per_chunk * block_size
        chunk_end_token = chunk_start_token + len(chunk_block_ids) * block_size
        effective_start = max(chunk_start_token, skip_first_n_tokens)
        if effective_start >= chunk_end_token:
            continue

        skip_blocks_in_chunk = (effective_start - chunk_start_token) // block_size
        effective_block_ids = chunk_block_ids[skip_blocks_in_chunk:]
        if not effective_block_ids:
            continue

        slot_mapping = torch.tensor(
            [
                block_id * block_size + token_idx
                for block_id in effective_block_ids
                for token_idx in range(block_size)
            ],
            dtype=torch.long,
            device=device,
        )
        skip_tokens = skip_blocks_in_chunk * block_size
        chunk_device = chunk_cpu.to(device)

        if use_mla:
            for layer_idx, layer in enumerate(normalized):
                mla_src = chunk_device[layer_idx, skip_tokens:]
                dst_flat = cast(
                    torch.Tensor,
                    _get_head_size_view(
                        layer, use_mla=True, gpu_kv_format=gpu_kv_format
                    ),
                )
                dst_flat.index_copy_(0, slot_mapping, mla_src)
        elif is_hnd:
            for layer_idx, layer in enumerate(normalized):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                if gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
                    k_t = layer[0]
                    v_t = layer[1]
                else:
                    k_t = layer[:, 0]
                    v_t = layer[:, 1]
                _nb, nh, _bs, hs = k_t.shape
                k_src_3d = k_src.reshape(-1, nh, hs)
                v_src_3d = v_src.reshape(-1, nh, hs)
                offset = 0
                for block_id in effective_block_ids:
                    k_t[block_id] = k_src_3d[
                        offset : offset + block_size
                    ].permute(1, 0, 2)
                    v_t[block_id] = v_src_3d[
                        offset : offset + block_size
                    ].permute(1, 0, 2)
                    offset += block_size
        else:
            for layer_idx, layer in enumerate(normalized):
                k_src = chunk_device[0, layer_idx, skip_tokens:]
                v_src = chunk_device[1, layer_idx, skip_tokens:]
                k_flat, v_flat = _get_head_size_view(
                    layer, use_mla=False, gpu_kv_format=gpu_kv_format
                )
                k_flat.index_copy_(0, slot_mapping, k_src)
                v_flat.index_copy_(0, slot_mapping, v_src)


def wrap_kv_caches(
    kv_caches: dict[str, torch.Tensor], use_bounce_buffer: bool = False
) -> KVCache:
    if use_bounce_buffer:
        return []
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result(timeout=DEFAULT_MQ_TIMEOUT)
    return chunk_size


def send_ping(
    mq_client: MessageQueueClient,
    timeout: float,
) -> bool:
    """Send a PING request and return the result.

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = send_lmcache_request(mq_client, RequestType.PING, [])
        return future.result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("Ping failed with exception", exc_info=True)
        return False


@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    kv_world_size: int
    """
    The kv world size, kv_world_size may not be equal to the actual_world_size, 
    in the case of mla, it will 'exclude' the effect of TP, the value is 
    calculated by `extract_world_size_and_kv_rank` in `lmcache_mp_connector.py`.
    """

    kv_worker_id: int
    """
    The kv worker id of the sub-process, kv_worker_id may not be equal to the 
    actual_worker_id, in the case of mla, it will 'exclude' the effect of TP, 
    the value is calculated by `extract_world_size_and_kv_rank` in 
    `lmcache_mp_connector.py`.
    """

    actual_world_size: int
    """The actual world size."""

    actual_worker_id: int
    """The actual worker id of the sub-process."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""


def _normalize_adapter_init_args(
    vllm_block_size: int,
    parallel_strategy: ParallelStrategy | int,
    legacy_block_size: int | None,
    mq_timeout: float,
) -> tuple[int, ParallelStrategy, float]:
    """Normalize adapter constructor args from old and new vLLM connectors.

    Args:
        vllm_block_size: The vLLM block size for the current connector API, or
            the legacy KV world size when ``parallel_strategy`` is an int.
        parallel_strategy: The current ``ParallelStrategy`` object, or the
            legacy KV worker id from older vLLM MP connectors.
        legacy_block_size: The legacy vLLM block size passed positionally by
            older vLLM MP connectors.
        mq_timeout: Timeout in seconds for synchronous message queue requests.

    Returns:
        A tuple of normalized ``(vllm_block_size, parallel_strategy,
        mq_timeout)``.

    Raises:
        TypeError: If the connector argument shape is not supported.
    """
    if isinstance(parallel_strategy, ParallelStrategy):
        return vllm_block_size, parallel_strategy, mq_timeout
    if not isinstance(parallel_strategy, int) or legacy_block_size is None:
        raise TypeError(
            "parallel_strategy must be ParallelStrategy, or legacy "
            "(kv_world_size, kv_worker_id, block_size) arguments"
        )

    kv_world_size = int(vllm_block_size)
    kv_worker_id = int(parallel_strategy)
    strategy = ParallelStrategy(
        use_mla=False,
        kv_world_size=kv_world_size,
        kv_worker_id=kv_worker_id,
        actual_world_size=kv_world_size,
        actual_worker_id=kv_worker_id,
        tp_size=kv_world_size,
        pp_size=1,
    )
    return int(legacy_block_size), strategy, mq_timeout


class HeartbeatThread(PeriodicThread):
    """Periodically checks server health via PING.

    Manages a threading.Event that adapters use to gate operations.
    When unhealthy, the adapter enters degraded mode; if the server
    recovers, the adapter automatically resumes normal operation.
    """

    def __init__(
        self,
        mq_client: MessageQueueClient,
        health_event: threading.Event,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """
        Args:
            mq_client: The message queue client used to send PING requests.
            health_event: A threading.Event shared with the adapter.
                Set when the server is healthy, cleared when unhealthy.
                Adapters check this event to decide whether to proceed
                with operations or enter degraded mode.
            interval: Seconds between heartbeat pings and ping timeout.
        """
        super().__init__(
            name="lmcache-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._mq_client = mq_client
        self._health_event = health_event
        self._interval = interval

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        healthy = send_ping(self._mq_client, timeout=self._interval)

        if healthy:
            self._health_event.set()
            if not was_healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        else:
            self._health_event.clear()
            if was_healthy:
                logger.warning("LMCache server is unhealthy — entering degraded mode")

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    """Token IDs for the load/store operation"""

    block_ids: list[int]
    """Block ids for the load/store operation"""

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = bool
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `kv_world_size`, `kv_worker_id` and so on. Older vLLM
                connectors pass the KV worker id here.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
            heartbeat_interval: Interval in seconds between heartbeat pings.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = mq_timeout

        # Lookup state tracking:
        # - _pending_lookups: request_ids submitted but not yet resolved
        # - _finished_lookup_results: cached chunk count keyed by request_id,
        #   so that repeated calls to check_lookup_result return the same value
        #   even after the server has already popped the job (exactly-once).
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            self.chunk_size = get_lmcache_chunk_size(self.mq_client)
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within {mq_timeout}s. "
                "Is the server running?"
            ) from None
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first lookup
        # request, by which time vLLM is fully ready.
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        return self.parallel_strategy.tp_size

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.start()

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends a LOOKUP request to the server and blocks until a prefetch
        job ID is returned.  The actual prefetch result can then be polled
        via ``check_lookup_result``.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache
            cache_salt: Per-user isolation salt. Requests with different
                cache_salt values produce separate cache entries.

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()

        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [key, self.tp_size],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LOOKUP request timed out after %ss. Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return
        self._pending_lookups.add(request_id)

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Sends a QUERY_PREFETCH_STATUS request to the server and blocks
        until the server responds.  Returns the matched token count
        when the prefetch is complete, or None if still in progress.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # If we have a cached result, return it to handle repeated calls.
            return self._finished_lookup_results.get(request_id, 0)

        if not self.is_healthy:
            # Server went down — give up on this lookup
            return 0

        if request_id in self._finished_lookup_results:
            # Return cached result if the job is already finished
            return self._finished_lookup_results[request_id]

        try:
            result = send_lmcache_request(
                self.mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss. "
                "Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return 0

        if result is None:
            return None

        token_count = result * self.chunk_size
        self._finished_lookup_results[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup state for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookup_results.pop(request_id, None)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks that
        vLLM has already computed, so they will never be retrieved.  Calling
        this prevents those chunks from holding read locks until TTL expiry.

        Or use this when a request is cancelled or aborted after lookup but
        before retrieve to avoid holding read locks until TTL expiry.

        When ``start`` or ``end`` is not aligned to the chunk size, the
        entire chunk containing start boundary is freed but not end boundary.
        It is caller's responsibility to properly align the boundaries.

        Args:
            token_ids: Token IDs for the key (same as used in lookup).
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
        """
        if not self.is_healthy:
            return

        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()
        send_lmcache_request(
            self.mq_client,
            RequestType.FREE_LOOKUP_LOCKS,
            [key, self.tp_size],
        )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        send_lmcache_request(
            self.mq_client,
            RequestType.END_SESSION,
            [request_id],
        )

    def report_block_allocations(
        self,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Report vLLM GPU block allocation deltas to LMCache server.

        Fire-and-forget: does not wait for a response. If the server
        is unhealthy the report is silently dropped.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        if not self.is_healthy or not records:
            return

        send_lmcache_request(
            self.mq_client,
            RequestType.REPORT_BLOCK_ALLOCATION,
            [os.getpid(), self.model_name, records],
        )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheEngineKey: The constructed key.
        """
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy | int,
        legacy_block_size: int | None = None,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        """Initialize the worker adapter for current or legacy vLLM callers.

        Args:
            server_url: The server URL for the LMCache message queue.
            context: The ZMQ context.
            model_name: The model name used for LMCache keys.
            vllm_block_size: The block size used in vLLM, or legacy KV world
                size when ``parallel_strategy`` is an int.
            parallel_strategy: Current ``ParallelStrategy`` metadata, or the
                legacy KV worker id from older vLLM connectors.
            legacy_block_size: The vLLM block size passed positionally by
                older vLLM connectors.
            mq_timeout: Timeout in seconds for message queue requests.
            heartbeat_interval: Interval in seconds between heartbeat pings.

        Raises:
            TypeError: If the connector argument shape is unsupported.
        """
        vllm_block_size, parallel_strategy, mq_timeout = _normalize_adapter_init_args(
            vllm_block_size,
            parallel_strategy,
            legacy_block_size,
            mq_timeout,
        )
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = mq_timeout

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}
        self._bounce_skip_tokens: dict[str, int] = {}
        self._bounce_layout_hints: Any = None
        self._bounce_gpu_kv_format: Any = None
        self._bounce_block_size: int | None = None
        self._use_bounce_buffer = False
        self._device_type = "cuda"

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs already returned as finished_sending to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        try:
            chunk_size = get_lmcache_chunk_size(self.mq_client)
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within {mq_timeout}s. "
                "Is the server running?"
            ) from None
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first store or retrieve
        # request, by which time vLLM is fully ready (model loaded,
        # KV caches allocated, warmup & CUDA graph capture done).
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

        # request telemetry, used for prefill-decode disagg
        # TODO: pass down the configuration via vLLM connector config
        # instead of env var
        self.request_telemetry = RequestTelemetryFactory.create(
            telemetry_type=os.getenv("LMCACHE_REQUEST_TELEMETRY_TYPE", "noop"),
            config={
                "endpoint": os.getenv(
                    "LMCACHE_REQUEST_TELEMETRY_ENDPOINT",
                    "http://localhost:5768/api/v1/telemetry",
                ),
            },
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """Get the kv worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        return self.parallel_strategy.use_mla

    @property
    def is_first_rank_of_pp_group(self) -> bool:
        """Is the first rank of the pipeline parallel group."""
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints

        layout_hints = vllm_layout_hints()
        self.kv_caches = kv_caches
        if not kv_caches:
            raise ValueError("kv_caches is empty")
        self._device_type = next(iter(kv_caches.values())).device.type
        self._use_bounce_buffer = self._device_type != "cuda"
        logger.info(
            "Registering kv caches (device_type=%s, bounce=%s)",
            self._device_type,
            self._use_bounce_buffer,
        )

        if self._use_bounce_buffer:
            # Local
            from lmcache.v1.gpu_connector.utils import is_mla

            (
                block_size,
                num_layers,
                hidden_dim_size,
                dtype_str,
                gpu_kv_format,
            ) = _compute_kv_layout(kv_caches, layout_hints=layout_hints)
            self._bounce_layout_hints = layout_hints
            self._bounce_gpu_kv_format = gpu_kv_format
            self._bounce_block_size = block_size
            future = send_lmcache_request(
                self.mq_client,
                RequestType.REGISTER_KV_CACHE_BOUNCE,
                [
                    self.instance_id,
                    self.model_name,
                    self.world_size,
                    EngineType.VLLM,
                    layout_hints,
                    block_size,
                    num_layers,
                    hidden_dim_size,
                    dtype_str,
                    is_mla(gpu_kv_format),
                ],
            )
        else:
            future = send_lmcache_request(
                self.mq_client,
                RequestType.REGISTER_KV_CACHE,
                [
                    self.instance_id,
                    wrap_kv_caches(kv_caches, use_bounce_buffer=False),
                    self.model_name,
                    self.world_size,
                    EngineType.VLLM,
                    layout_hints,
                ],
            )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.start()

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: Any,
        cache_salt: str = "",
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        if self._use_bounce_buffer:
            _device_synchronize(self._device_type)
            cpu_data = _gather_chunks_to_cpu(
                self.kv_caches,
                op.block_ids,
                self.blocks_in_chunk,
                layout_hints=self._bounce_layout_hints,
                gpu_kv_format=self._bounce_gpu_kv_format,
            )
            future = send_lmcache_request(
                self.mq_client,
                RequestType.STORE_CPU_CHUNKS,
                [key, self.instance_id, cpu_data],
            )
        else:
            future = send_lmcache_request(
                self.mq_client,
                RequestType.STORE,
                [key, self.instance_id, op.block_ids, event.ipc_handle()],
            ).to_cuda_future()
        self.store_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: Any,
        cache_salt: str = "",
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salt: Per-user isolation salt.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.block_ids)
            return

        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        if self._use_bounce_buffer:
            future = send_lmcache_request(
                self.mq_client,
                RequestType.RETRIEVE_CPU_CHUNKS,
                [key, self.instance_id],
            )
            self._bounce_skip_tokens[request_id] = op.skip_first_n_tokens
        else:
            future = send_lmcache_request(
                self.mq_client,
                RequestType.RETRIEVE,
                [
                    key,
                    self.instance_id,
                    op.block_ids,
                    event.ipc_handle(),
                    op.skip_first_n_tokens,
                ],
            ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, list(op.block_ids))

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be the same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_store_request(request_id, op, event, cache_salt=salt)

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. If None,
                all requests use cache_salt="". The list length should be same as
                request_ids.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        for request_id, op, salt in zip(request_ids, ops, cache_salts, strict=False):
            self.submit_retrieve_request(request_id, op, event, cache_salt=salt)

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(finished_req_ids_from_lmcache)
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self._returned_finished:
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()
            self._bounce_skip_tokens.clear()

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            # A request may have a pending retrieve AND appear in
            # finished_req_ids_from_engine (it ran without loading KV after
            # the server died).  The scheduler processes finished_recving
            # first and deletes the request, so we must not also report it
            # in finished_sending.
            ret_stores -= finished_retrieves
            return ret_stores, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, r_block_ids) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            if self._use_bounce_buffer:
                success, cpu_data = cast(tuple[bool, bytes], r_future.result())
                r_result = success
                if success and cpu_data:
                    try:
                        _scatter_cpu_chunks_to_kv(
                            self.kv_caches,
                            r_block_ids,
                            cpu_data,
                            self.blocks_in_chunk,
                            skip_first_n_tokens=self._bounce_skip_tokens.pop(
                                request_id, 0
                            ),
                            layout_hints=self._bounce_layout_hints,
                            gpu_kv_format=self._bounce_gpu_kv_format,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to scatter retrieved bounce-buffer chunks"
                        )
                        r_result = False
                else:
                    self._bounce_skip_tokens.pop(request_id, None)
            else:
                r_result = r_future.result()
            finished_retrieves.add(request_id)

            if not r_result:
                self.error_block_ids.update(r_block_ids)
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
            self._bounce_skip_tokens.pop(request_id, None)

        # Update the internal states
        ret_stores = self._process_finished_stores(
            finished_stores, finished_req_ids_from_engine
        )

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if ret_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=ret_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Returns the block IDs that failed due to retrieve timeout,
        then clears the internal set.
        """
        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        try:
            send_lmcache_request(
                self.mq_client,
                RequestType.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        self.mq_client.close()
        self.request_telemetry.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheEngineKey: The constructed key.
        """
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
