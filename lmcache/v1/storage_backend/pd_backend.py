# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import threading
import time

# Third Party
import msgspec
import torch
import zmq
import zmq.asyncio

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)

# Maximum seconds to spend retrying a memory allocation before giving up.
_ALLOCATION_TIMEOUT_SECONDS = 5.0
# Polling interval used when waiting on a threading/asyncio Condition; small
# enough to be responsive, large enough not to spin-waste CPU.
_CONDITION_WAIT_INTERVAL = 0.5
# Maximum number of per-request chunk-count entries to retain in the receiver's
# _req_chunk_counts dict.  Entries from completed requests are safe to evict
# (with the serial global sender worker only one req is active at a time, so
# old entries are always for finished requests).  When this limit is exceeded,
# the oldest half of entries are removed to prevent unbounded growth in
# long-running servers.
_MAX_REQ_CHUNK_COUNT_ENTRIES = 1000


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages"""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int
    # req_id is used by the receiver for per-request chunk accounting and
    # fail-fast detection when C_req > max_inflight_chunks.  An empty string
    # means the sender does not provide an identifier (backwards-compatible).
    req_id: str = ""


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

    # Indexes (remote) of allocated memory objects (to be written)
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str  # The request UUID to notify the proxy


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


@dataclass
class _TransferItem:
    """Item placed onto a per-request transfer queue."""

    keys: Sequence[CacheEngineKey]
    memory_objs: List[MemoryObj]
    receiver_id: str
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]]
    transfer_spec: Any


@dataclass
class PDConfig:
    role: str

    peer_host: str
    peer_init_port: int
    peer_alloc_port: int

    proxy_host: str
    proxy_port: int

    buffer_size: int
    buffer_device: str

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        """Convert the LMCacheEngineConfig to PDConfig"""

        role = config.pd_role

        # TODO(Jiayi): Could be both if we want to do dynamic role switch.
        assert role in ["sender", "receiver"], (
            f"Invalid role: {config.pd_role}, must be either sender or receiver"
        )

        assert config.pd_buffer_size is not None
        assert config.pd_buffer_device is not None

        if role == "receiver":
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
        elif role == "sender":
            assert config.pd_proxy_host is not None
            assert config.pd_proxy_port is not None

        corrected_device = get_correct_device(
            config.pd_buffer_device, metadata.worker_id
        )

        if config.pd_peer_alloc_port is not None:
            pd_peer_alloc_port = config.pd_peer_alloc_port[tp_rank]
        else:
            pd_peer_alloc_port = None

        if config.pd_peer_init_port is not None:
            pd_peer_init_port = config.pd_peer_init_port[tp_rank]
        else:
            pd_peer_init_port = None

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=corrected_device,
        )


class PDBackend(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for PD Disaggregation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        self.running = True

        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        self.corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        # TODO(Jiayi): add async zmq context if we want better asynchrony.
        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Initialize transfer channel
        peer_init_url = None
        self.local_id = ""
        # TODO(Jiayi): both sender and receiver have to have
        # peer_init_url if they want to do instance flip.
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        # Create the event loop before the transfer channel so it can be passed
        # into the channel constructor for async_mode initialization.
        if self.pd_config.role == "sender":
            self._sender_loop = asyncio.new_event_loop()
            self._sender_thread = threading.Thread(
                target=self._sender_loop.run_forever,
                daemon=True,
                name="pd-sender-async",
            )
            self._sender_thread.start()
            event_loop = self._sender_loop
        elif self.pd_config.role == "receiver":
            self._recv_loop = asyncio.new_event_loop()
            self._recv_thread = threading.Thread(
                target=self._recv_loop.run_forever,
                daemon=True,
                name="pd-receiver-async",
            )
            self._recv_thread.start()
            event_loop = self._recv_loop
        else:
            raise ValueError("Invalid PD role.")

        allocator = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        self.transfer_channel = CreateTransferChannel(
            async_mode=True,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            buffer_ptr=allocator.buffer_ptr,
            buffer_size=allocator.buffer_size,
            align_bytes=allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
            backends=config.nixl_backends,
            device=self.corrected_device,
            event_loop=event_loop,
        )

        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self._peer_connection_lock = threading.Lock()
            # Separate async ZMQ context for sender coroutines
            self._async_zmq_context = zmq.asyncio.Context()
            self._async_alloc_sockets: dict[str, zmq.asyncio.Socket] = {}
            self._async_alloc_locks: dict[str, asyncio.Lock] = {}
            # Sender staging buffer flow control: block cache_engine.store()
            # (which runs in a vLLM worker thread) when the staging buffer is
            # near-full so that in-flight RDMA transfers can drain before new
            # allocations are allowed.  threading.Condition is required because
            # allocate() is called from a worker thread, not the asyncio loop.
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            self._sender_staging_lock = threading.Lock()
            self._sender_staging_condition = threading.Condition(
                self._sender_staging_lock
            )
            self._sender_inflight_chunks = 0
            # Leave headroom: allow at most 3/4 of total chunks to be in-flight
            self._sender_max_inflight_chunks = max(1, total_chunks * 3 // 4)
            logger.info(
                "PDBackend sender: staging flow control initialized with "
                "max_inflight=%d (total_chunks=%d)",
                self._sender_max_inflight_chunks,
                total_chunks,
            )
            # Per-request transfer queues.  Each queue serialises chunk batches
            # within the same request.  A single global FIFO worker drains
            # requests one at a time (no concurrent multi-request allocation)
            # to prevent decoder-buffer fragmentation and deadlock: the decoder
            # requires all chunks for a request before it can start consuming,
            # so allowing N requests to each partially fill the buffer would
            # leave none of them complete.
            # Only ever accessed from coroutines running on _sender_loop, so
            # no additional lock is needed.
            self._transfer_queues: dict[str, asyncio.Queue] = {}
            # Global FIFO queue of req_ids to be processed by the single worker.
            # _global_req_queue is an asyncio.Queue created on the sender loop.
            self._global_req_queue: asyncio.Queue = asyncio.Queue()
            self._global_req_worker_task: Optional[asyncio.Task] = None
            # Start the single global FIFO worker on the sender event loop.
            future = asyncio.run_coroutine_threadsafe(
                self._start_global_req_worker(),
                self._sender_loop,
            )
            future.result(timeout=5)
        elif self.pd_config.role == "receiver":
            self._init_receiver()
            # Decoder-side flow control: block allocation when buffer is near-full
            assert self._chunk_size_bytes > 0, (
                "chunk_size_bytes must be > 0 for inflight flow control"
            )
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            self._max_inflight_chunks = max(1, total_chunks * 3 // 4)
            self._inflight_chunks = 0
            # The condition must be created on the receiver event loop
            future = asyncio.run_coroutine_threadsafe(
                self._create_inflight_condition(), self._recv_loop
            )
            future.result(timeout=5)
            logger.info(
                "PDBackend receiver: inflight flow control initialized with "
                "max_inflight_chunks=%d (total_chunks=%d, buffer=%d bytes, "
                "chunk=%d bytes)",
                self._max_inflight_chunks,
                total_chunks,
                self._aligned_buffer_size,
                self._chunk_size_bytes,
            )
            # Per-request cumulative chunk counter for fail-fast detection.
            # If a request's total chunks exceed max_inflight_chunks, it can
            # never complete (decoder requires all chunks before consuming).
            self._req_chunk_counts: dict[str, int] = {}

        self.full_chunk_size_bytes = config.chunk_size

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        if self.corrected_device != "cpu":
            logger.info(f"Setting cuda device to {self.corrected_device} ")
            torch.cuda.set_device(self.corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()

        init_func = (
            paged_mem_allocator.init_cpu_memory_allocator
            if self.corrected_device == "cpu"
            else paged_mem_allocator.init_gpu_memory_allocator
        )

        # Calculate the chunk size (align_bytes) and align buffer size
        shapes = [torch.Size(metadata.kv_shape)]
        dtypes = [metadata.kv_dtype]
        chunk_size_bytes = get_size_bytes(shapes, dtypes)
        origin_buffer_size = config.pd_buffer_size
        aligned_buffer_size = origin_buffer_size // chunk_size_bytes * chunk_size_bytes

        if aligned_buffer_size == 0 and origin_buffer_size > 0:
            raise ValueError(
                f"pd_buffer_size ({origin_buffer_size}) is smaller than a "
                f"single chunk ({chunk_size_bytes}), resulting in an aligned "
                f"buffer of size 0. Please increase pd_buffer_size to be at "
                f"least {chunk_size_bytes}."
            )

        if aligned_buffer_size != origin_buffer_size:
            logger.info(
                f"Auto align pd_buffer_size, origin: {origin_buffer_size}, "
                f"aligned: {aligned_buffer_size}, chunk size: {chunk_size_bytes}. "
                f"The remaining {origin_buffer_size - aligned_buffer_size} bytes "
                f"will not be allocated."
            )

        self._chunk_size_bytes = chunk_size_bytes
        self._aligned_buffer_size = aligned_buffer_size

        init_func(
            aligned_buffer_size,
            shapes,
            dtypes,
            MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
        )

        return paged_mem_allocator

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        return self.memory_allocator

    def get_allocator_backend(self):
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # NOTE: no eviction and busy_loop in PD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"

        if self.pd_config.role == "sender":
            # Single unified loop: flow-control check and allocation attempt are
            # combined so a thread can never pass flow control only to have
            # another thread steal the last memory slot before it allocates.
            #
            # The loop body, executed under _sender_staging_condition, does:
            #   1. Check running flag — exit immediately on shutdown.
            #   2. Check flow-control threshold — only attempt allocation when
            #      inflight_chunks < max; otherwise fall through to wait.
            #   3. Attempt allocation — on success, increment the counter and
            #      return; the check + allocate + increment form an atomic unit
            #      protected by the condition lock.
            #   4. Wait — either the threshold is exceeded or allocation failed
            #      (fragmentation / pool exhausted).  In both cases we wait for
            #      _release_sender_staging_chunks to call notify_all().
            #
            # Flow-control waits are not counted against the allocation deadline:
            # they loop back to the top and re-check the threshold, so a long
            # backpressure pause does not consume the 5-second allocation budget.
            # Once the threshold is satisfied we start a fresh deadline for the
            # actual allocation attempts.
            with self._sender_staging_condition:
                # deadline is initialised here and reset whenever backpressure
                # is active so that flow-control waits do not consume the
                # allocation budget.
                deadline: float = time.monotonic() + _ALLOCATION_TIMEOUT_SECONDS
                last_near_full_log = 0.0
                while True:
                    if not self.running:
                        return None

                    at_threshold = (
                        self._sender_inflight_chunks >= self._sender_max_inflight_chunks
                    )

                    if at_threshold:
                        # Log near-full warning at most once per second to avoid
                        # spamming.
                        now = time.monotonic()
                        if now - last_near_full_log >= 1.0:
                            logger.warning(
                                "Sender staging buffer near-full: "
                                "inflight_chunks=%d >= max=%d, waiting for "
                                "transfers to complete...",
                                self._sender_inflight_chunks,
                                self._sender_max_inflight_chunks,
                            )
                            last_near_full_log = now
                        # Reset the allocation deadline so that once
                        # backpressure clears the thread gets a fresh
                        # 5-second window for actual allocation attempts.
                        deadline = time.monotonic() + _ALLOCATION_TIMEOUT_SECONDS
                        self._sender_staging_condition.wait(
                            timeout=_CONDITION_WAIT_INTERVAL
                        )
                        continue

                    # Under threshold: attempt allocation.  deadline is always
                    # a valid float at this point (set above or reset in the
                    # at_threshold branch on a previous iteration).
                    mem_obj = self.memory_allocator.allocate(
                        shapes, dtypes, fmt=fmt, allocator_type=alloc_type
                    )
                    if mem_obj is not None:
                        self._sender_inflight_chunks += 1
                        return mem_obj

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._sender_staging_condition.wait(
                        timeout=min(remaining, _CONDITION_WAIT_INTERVAL)
                    )

            logger.error("Sender staging allocation failed after timeout")
            return None
        else:
            return self.memory_allocator.allocate(
                shapes, dtypes, fmt=fmt, allocator_type=alloc_type
            )

    # TODO(Jiayi): Please implement batched allocate to reduce memory
    # allocation overhead.
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type=alloc_type
        )

    # NOTE(Jiayi): If two requests have overlapped keys, will
    # the later one cause any problems here?
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    ############################################################
    # Prefiller functions
    ############################################################
    def _init_sender(self):
        proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
        self.proxy_side_channel = get_zmq_socket(
            self.zmq_context,
            proxy_url,
            "tcp",
            zmq.PUSH,
            "connect",
        )

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        # Fast path: no lock required if already connected.
        if receiver_id in self.initialized_peers:
            return
        with self._peer_connection_lock:
            # Double-check under the lock to prevent duplicate connections when
            # multiple vLLM worker threads call this concurrently.
            if receiver_id in self.initialized_peers:
                return

            receiver_init_url = f"{receiver_host}:{receiver_init_port}"
            receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

            # Establish the connection with the receiver/decoder.
            # The transfer channel uses an async ZMQ context (async_mode=True), so
            # we must call the async version scheduled on the sender event loop.
            future = asyncio.run_coroutine_threadsafe(
                self.transfer_channel.async_lazy_init_peer_connection(
                    local_id=self.local_id,
                    peer_id=receiver_id,
                    peer_init_url=receiver_init_url,
                ),
                self._sender_loop,
            )
            future.result()  # Block until connection is established

            # Schedule socket creation on the sender event loop to avoid
            # cross-thread issues
            future = asyncio.run_coroutine_threadsafe(
                self._async_create_alloc_socket(receiver_id, receiver_mem_alloc_url),
                self._sender_loop,
            )
            future.result(timeout=10)  # Wait for socket to be created

            self.initialized_peers.add(receiver_id)

    async def _async_create_alloc_socket(
        self, receiver_id: str, receiver_mem_alloc_url: str
    ):
        async_alloc_socket = self._async_zmq_context.socket(zmq.REQ)
        async_alloc_socket.connect(f"tcp://{receiver_mem_alloc_url}")
        self._async_alloc_sockets[receiver_id] = async_alloc_socket

    async def _async_remote_allocate(
        self, receiver_id: str, alloc_request: AllocRequest
    ) -> AllocResponse:
        if receiver_id not in self._async_alloc_locks:
            self._async_alloc_locks[receiver_id] = asyncio.Lock()
        async with self._async_alloc_locks[receiver_id]:
            socket = self._async_alloc_sockets[receiver_id]
            await socket.send(msgspec.msgpack.encode(alloc_request))
            msg = await socket.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)
        return alloc_response

    def _get_remote_alloc_request(
        self,
        keys: Sequence[CacheEngineKey],
        mem_objs: List[MemoryObj],
        req_id: str = "",
    ) -> AllocRequest:
        """
        Get the allocation request given the keys and memory objects.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size_bytes` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = mem_objs[0].meta.fmt
        shape = mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = mem_objs[-1].meta.shape[token_dim]

        str_keys = [key.to_string() for key in keys]

        return AllocRequest(
            keys=str_keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            req_id=req_id,
        )

    async def _async_transfer_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        receiver_id: str,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        transfer_spec: Any = None,
    ) -> None:
        """
        Async coroutine that performs the full KV transfer:
        remote alloc → async_batched_write → ref_count_down → callback.
        Runs in the dedicated sender event loop (_sender_loop).

        ``remote_indexes`` may contain ``-1`` sentinels for keys whose
        allocation failed on the receiver.  Those objects are released
        immediately and excluded from the RDMA write.
        """
        completed_indexes: set[int] = set()
        num_chunks = len(memory_objs)

        # Extract req_id for per-request allocation accounting on the receiver.
        # Using getattr with a default of "" keeps this backwards-compatible with
        # any transfer_spec that pre-dates the req_id field.  An empty string
        # causes the receiver to skip per-request chunk counting (no fail-fast
        # detection), which is acceptable for legacy callers.
        req_id: str = (
            getattr(transfer_spec, "req_id", "") if transfer_spec is not None else ""
        )

        try:
            alloc_request = self._get_remote_alloc_request(
                keys, memory_objs, req_id=req_id
            )
            alloc_response = await self._async_remote_allocate(
                receiver_id, alloc_request
            )
            already_sent_indexes = alloc_response.already_sent_indexes
            remote_indexes = alloc_response.remote_indexes

            mem_objs_to_send: list[MemoryObj] = []
            filtered_remote_indexes: list[int] = []
            # Index into remote_indexes (one entry per non-already-sent key).
            ri = 0
            num_non_already_sent = len(memory_objs) - len(already_sent_indexes)
            if len(remote_indexes) != num_non_already_sent:
                logger.error(
                    "Protocol mismatch: remote_indexes length (%d) != "
                    "non-already-sent objects (%d). Some allocations may "
                    "have been silently dropped by an older receiver.",
                    len(remote_indexes),
                    num_non_already_sent,
                )
            for idx, mem_obj in enumerate(memory_objs):
                if idx in already_sent_indexes:
                    mem_obj.ref_count_down()
                    completed_indexes.add(idx)
                else:
                    remote_addr = remote_indexes[ri] if ri < len(remote_indexes) else -1
                    ri += 1
                    if remote_addr == -1:
                        # Receiver failed to allocate for this key; skip it.
                        logger.warning(
                            "Receiver allocation failed for key %s "
                            "(idx=%d), releasing local memory object.",
                            keys[idx],
                            idx,
                        )
                        mem_obj.ref_count_down()
                        completed_indexes.add(idx)
                    else:
                        mem_objs_to_send.append(mem_obj)
                        filtered_remote_indexes.append(remote_addr)

            if mem_objs_to_send:
                channel_transfer_spec = {
                    "receiver_id": receiver_id,
                    "remote_indexes": filtered_remote_indexes,
                }
                await self.transfer_channel.async_batched_write(
                    objects=mem_objs_to_send,
                    transfer_spec=channel_transfer_spec,
                )
                for idx, mem_obj in enumerate(memory_objs):
                    if idx not in completed_indexes:
                        mem_obj.ref_count_down()
                        completed_indexes.add(idx)
            else:
                logger.debug(
                    "All memory objects have been already sent to the remote peer."
                    " Skipping transfer."
                )

            # Send ProxyNotif if this is the last prefill chunk, BEFORE invoking
            # on_complete_callback.  The worker processes chunks sequentially,
            # so all prior chunks have already completed by the time we reach
            # this point.  Sending here (inside _async_transfer_task) guarantees
            # the notification is observable the moment the callback fires.
            is_last_prefill = transfer_spec is not None and getattr(
                transfer_spec, "is_last_prefill", False
            )
            if is_last_prefill:
                try:
                    notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                    notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self.proxy_side_channel.send, notif_msg_bytes
                    )
                except Exception as e:
                    logger.error(
                        "Failed to send ProxyNotif for req %s: %s",
                        transfer_spec.req_id,
                        e,
                    )

            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            f"on_complete_callback failed for key {key}: {e}"
                        )
        except Exception as e:
            logger.error("Async transfer task failed: %s", str(e))
            # Release ref counts on error to avoid leaks (only those not yet released)
            for idx, mem_obj in enumerate(memory_objs):
                if idx not in completed_indexes:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        pass
        finally:
            # Release sender staging buffer slots so that allocate() waiters
            # can proceed.  num_chunks equals the number of memory objects that
            # were allocated from the staging buffer via allocate().
            self._release_sender_staging_chunks(num_chunks)

    def _release_sender_staging_chunks(self, count: int) -> None:
        """Decrement sender staging inflight counter and notify waiters.

        Called from ``_async_transfer_task`` (asyncio) after all staging
        buffers for a transfer have been freed via ``ref_count_down()``.
        ``threading.Condition.notify_all()`` is non-blocking and safe to call
        from an asyncio coroutine.

        :param count: Number of staging slots to release.
        """
        if self.pd_config.role == "sender" and count > 0:
            with self._sender_staging_condition:
                self._sender_inflight_chunks = max(
                    0, self._sender_inflight_chunks - count
                )
                self._sender_staging_condition.notify_all()

    async def _enqueue_transfer(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        receiver_id: str,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        transfer_spec: Any,
    ) -> None:
        """Enqueue a transfer item onto the per-request asyncio.Queue.

        If no queue exists for the request yet, one is created and the req_id
        is placed onto the global FIFO request queue so the single global
        worker will process it.  This guarantees that chunk batches for the
        same request are transferred in FIFO order and that only one request
        occupies receiver buffer at a time (preventing decoder-buffer
        fragmentation and deadlock).

        Must be called as a coroutine on ``_sender_loop`` (via
        ``asyncio.run_coroutine_threadsafe``).  All dict accesses are therefore
        single-threaded on the event loop.

        :param keys: Cache keys for this transfer batch.
        :param memory_objs: Memory objects to transfer.
        :param receiver_id: Identifier of the remote receiver.
        :param on_complete_callback: Optional per-key completion callback.
        :param transfer_spec: Transfer specification (carries ``req_id`` and
            ``is_last_prefill``).
        """
        req_id: str = (
            getattr(transfer_spec, "req_id", "unknown")
            if transfer_spec is not None
            else "unknown"
        )
        if req_id not in self._transfer_queues:
            q: asyncio.Queue[_TransferItem] = asyncio.Queue()
            self._transfer_queues[req_id] = q
            # Register this req_id with the global FIFO worker so it will be
            # processed after all earlier requests have completed.
            await self._global_req_queue.put(req_id)
        item = _TransferItem(
            keys=keys,
            memory_objs=memory_objs,
            receiver_id=receiver_id,
            on_complete_callback=on_complete_callback,
            transfer_spec=transfer_spec,
        )
        self._transfer_queues[req_id].put_nowait(item)

    async def _drain_transfer_queue(
        self, req_id: str, q: "asyncio.Queue[_TransferItem]"
    ) -> None:
        """Drain remaining items from a per-request transfer queue after failure.

        Releases ref counts on all memory objects to prevent memory leaks and
        releases sender staging chunks to unblock allocate() waiters.

        :param req_id: The request identifier (used for logging).
        :param q: The asyncio.Queue to drain.
        """
        drained = 0
        while not q.empty():
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            for mem_obj in item.memory_objs:
                try:
                    mem_obj.ref_count_down()
                except Exception as e:
                    logger.warning(
                        "ref_count_down() failed during drain for req %s: %s",
                        req_id,
                        e,
                    )
            self._release_sender_staging_chunks(len(item.memory_objs))
            q.task_done()
            drained += 1
        if drained > 0:
            logger.warning(
                "Drained %d remaining transfer items for req %s after failure.",
                drained,
                req_id,
            )

    async def _start_global_req_worker(self) -> None:
        """Create the global FIFO request worker task on the sender event loop.

        Must be called from within the sender event loop (via
        ``asyncio.run_coroutine_threadsafe``) so that the task is bound to the
        correct loop.
        """
        self._global_req_worker_task = asyncio.create_task(self._global_req_worker())

    async def _global_req_worker(self) -> None:
        """Single global FIFO worker: process one request at a time.

        Pulls req_ids from ``_global_req_queue`` in arrival order and drains
        each request's per-request transfer queue serially until the item
        flagged with ``is_last_prefill=True`` is processed.  Only then is the
        next req_id dequeued.

        This prevents receiver-buffer fragmentation: at most one request's
        chunks occupy the decoder buffer at any time, guaranteeing that the
        decoder (which requires all chunks before it can start consuming) can
        always find a complete request available to process.  Without this
        guarantee, N requests each partially filling the buffer would leave
        none of them completable, causing a deadlock.

        If a transfer fails mid-request, remaining items are drained (ref
        counts released) and the worker moves on to the next request.
        """
        while True:
            try:
                req_id: str = await self._global_req_queue.get()
            except asyncio.CancelledError:
                return

            q = self._transfer_queues.get(req_id)
            if q is None:
                # Queue was removed before we could process it — skip.
                continue

            try:
                while True:
                    try:
                        item: _TransferItem = await q.get()
                    except asyncio.CancelledError:
                        return

                    transfer_failed = False
                    try:
                        await self._async_transfer_task(
                            keys=item.keys,
                            memory_objs=item.memory_objs,
                            receiver_id=item.receiver_id,
                            on_complete_callback=item.on_complete_callback,
                            transfer_spec=item.transfer_spec,
                        )
                    except Exception as e:
                        logger.error(
                            "Transfer worker error for req %s: %s. "
                            "Aborting remaining transfers for this request.",
                            req_id,
                            e,
                        )
                        transfer_failed = True
                    finally:
                        q.task_done()

                    is_last = item.transfer_spec is not None and getattr(
                        item.transfer_spec, "is_last_prefill", False
                    )
                    if transfer_failed:
                        # Drain remaining items and release their memory.
                        await self._drain_transfer_queue(req_id, q)
                        break
                    if is_last:
                        break
            finally:
                # Drain any remaining items to prevent memory leaks on
                # unexpected exit (e.g. CancelledError between the try and
                # the is_last check).
                await self._drain_transfer_queue(req_id, q)
                self._transfer_queues.pop(req_id, None)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to transfer KV caches to peer.

        Non-blocking: fires the async transfer coroutine to the background
        event loop and returns immediately. The caller's ref_count_down will
        happen concurrently with the async transfer, so we ref_count_up here
        to keep objects alive until the async task completes.

        :param on_complete_callback: Optional callback invoked once per key
            after the transfer completes. Callback exceptions are caught and logged.
        """
        # Bump ref counts so objects stay alive while async transfer is in-flight.
        # The async task (_async_transfer_task) will call ref_count_down when done.
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host

        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        # Schedule via _enqueue_transfer so the item is placed on the
        # per-request asyncio.Queue.  The dedicated consumer coroutine
        # processes items sequentially, guaranteeing in-order RDMA writes and
        # sending ProxyNotif only after all chunks are complete.
        asyncio.run_coroutine_threadsafe(
            self._enqueue_transfer(
                keys=list(keys),
                memory_objs=list(memory_objs),
                receiver_id=receiver_id,
                on_complete_callback=on_complete_callback,
                transfer_spec=transfer_spec,
            ),
            self._sender_loop,
        )

    ############################################################
    # Prefiller functions end
    ############################################################

    ############################################################
    # Decoder functions
    ############################################################
    async def _create_inflight_condition(self) -> None:
        """Create the asyncio.Condition for inflight chunk flow control.

        Must be called from within the receiver event loop so that the
        Condition is bound to the correct loop.
        """
        self._inflight_condition = asyncio.Condition()

    def _init_receiver(self):
        """
        Launch the async memory allocation server coroutine on the already-running
        receiver event loop (self._recv_loop, created before the transfer channel).
        """
        asyncio.run_coroutine_threadsafe(
            self._async_mem_alloc_server(), self._recv_loop
        )

    async def _async_mem_alloc_server(self):
        """
        Async ZMQ REP server for memory allocation requests.
        Replaces the blocking _mem_alloc_loop / _mem_alloc_thread.
        When _async_allocate_and_put needs to wait for free memory it yields
        via `await asyncio.sleep`, keeping the event loop responsive.
        """
        # Third Party
        import zmq.asyncio as azmq

        async_ctx = azmq.Context()
        socket = async_ctx.socket(zmq.REP)
        alloc_port = self.pd_config.peer_alloc_port
        socket.bind(f"tcp://*:{alloc_port}")
        logger.info(f"Async mem alloc server listening on port {alloc_port}")
        try:
            while self.running:
                try:
                    alloc_req_bytes = await socket.recv()
                    alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=PDMsg)
                    assert isinstance(alloc_req, AllocRequest), (
                        "The request from the remote peer is not an AllocRequest"
                    )
                    # NOTE: it's okay to put the memory objs into the storage backend
                    # first because decode vllm will not be able to see the decode
                    # request until proxy receives the ack.
                    alloc_resp = await self._async_allocate_and_put(alloc_req)
                    await socket.send(msgspec.msgpack.encode(alloc_resp))
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Failed to process async mem alloc: %s", str(e))
                    if self.running:
                        await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            socket.close()
            async_ctx.term()

    async def _async_allocate_and_put(
        self, alloc_request: AllocRequest
    ) -> AllocResponse:
        """
        Async version of _allocate_and_put.
        Uses event-driven waiting on ``_inflight_condition`` instead of
        ``asyncio.sleep`` polling so the event loop stays responsive while
        waiting for free memory.
        pin=False: PDBackend has no eviction; pinning is unnecessary and
        causes ref_count leaks.

        ``remote_indexes`` always contains one entry per non-already-sent key
        so that the sender can match objects to addresses by position.
        A value of ``-1`` signals that allocation failed for that slot.

        Timeout semantics:
        - Backpressure wait (inflight >= max_inflight_chunks): indefinite while
          ``self.running``.  The sender will not submit new chunks until the
          decoder frees some, so this wait is expected to resolve.
        - Allocator wait (allocate() returns None): bounded by
          ``_ALLOCATION_TIMEOUT_SECONDS``.
        """
        total_allocs = len(alloc_request.keys)
        req_id = alloc_request.req_id

        # Fail-fast: detect if this request can never complete because it
        # requires more chunks than the decoder buffer can ever hold at once.
        # The decoder needs all chunks present before it can start consuming,
        # so C_req > max_inflight_chunks is an impossible configuration.
        if req_id:
            prev_count = self._req_chunk_counts.get(req_id, 0)
            new_total = prev_count + total_allocs
            if new_total > self._max_inflight_chunks:
                logger.error(
                    "Request %s requires %d total chunks (already allocated %d, "
                    "new batch %d) but max_inflight_chunks=%d. This request can "
                    "never complete because the decoder requires all chunks before "
                    "it can start consuming. Failing fast. "
                    "To fix: increase pd_buffer_size so that "
                    "max_inflight_chunks >= total chunks needed for the largest "
                    "request, or reduce prompt length / chunk size.",
                    req_id,
                    new_total,
                    prev_count,
                    total_allocs,
                    self._max_inflight_chunks,
                )
                return AllocResponse(
                    already_sent_indexes=[],
                    remote_indexes=[-1] * total_allocs,
                )
            # Update cumulative count for this request.  Evict oldest entries
            # when the dict exceeds _MAX_REQ_CHUNK_COUNT_ENTRIES to prevent
            # unbounded growth in long-running servers.  With the serial FIFO
            # sender worker, old entries are always from completed requests and
            # are safe to remove.
            self._req_chunk_counts[req_id] = new_total
            if len(self._req_chunk_counts) > _MAX_REQ_CHUNK_COUNT_ENTRIES:
                evict_count = _MAX_REQ_CHUNK_COUNT_ENTRIES // 2
                for key_to_evict in list(self._req_chunk_counts.keys())[:evict_count]:
                    del self._req_chunk_counts[key_to_evict]
        else:
            # No req_id provided (legacy sender or untracked call); per-request
            # fail-fast detection is unavailable.  Log at debug level so operators
            # can diagnose potential buffer exhaustion if needed.
            logger.debug(
                "AllocRequest has no req_id — per-request chunk accounting "
                "is disabled for this batch (C_req > T fail-fast will not fire)."
            )

        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)  # copy — we mutate token_dim

        alloc_indexes: list[int] = []
        already_send_indexes: list[int] = []

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self.contains(key, pin=False):
                already_send_indexes.append(idx)
                continue

            if idx == total_allocs - 1:
                token_dim = fmt.token_dim()
                shape[token_dim] = alloc_request.last_chunk_toks

            # Wait until inflight count is below threshold before allocating.
            async with self._inflight_condition:
                while self._inflight_chunks >= self._max_inflight_chunks:
                    logger.warning(
                        "Decoder buffer near-full: inflight_chunks=%d >= max=%d, "
                        "waiting for buffers to be freed...",
                        self._inflight_chunks,
                        self._max_inflight_chunks,
                    )
                    await self._inflight_condition.wait()

            mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
            # Event-driven retry: wait on _inflight_condition for notification
            # instead of asyncio.sleep polling so the coroutine wakes up
            # immediately when _notify_inflight_freed fires.
            deadline = asyncio.get_event_loop().time() + _ALLOCATION_TIMEOUT_SECONDS
            while mem_obj is None:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.error(
                        "Failed to allocate memory for key %s after "
                        "timeout (~%.0fs), aborting.",
                        key,
                        _ALLOCATION_TIMEOUT_SECONDS,
                    )
                    break
                async with self._inflight_condition:
                    try:
                        await asyncio.wait_for(
                            self._inflight_condition.wait(),
                            timeout=min(remaining, _CONDITION_WAIT_INTERVAL),
                        )
                    except asyncio.TimeoutError:
                        pass
                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

            if mem_obj is None:
                logger.warning(
                    "Allocation failed for key %s (idx=%d); "
                    "marking as -1 in remote_indexes.",
                    key,
                    idx,
                )
                # Use -1 sentinel so the sender can match objects to
                # addresses by position and skip failed slots.
                alloc_indexes.append(-1)
                continue
            alloc_indexes.append(mem_obj.meta.address)
            self.put(key, mem_obj)
            # Increment the inflight counter now that a chunk is allocated.
            async with self._inflight_condition:
                self._inflight_chunks += 1

        return AllocResponse(
            already_sent_indexes=already_send_indexes, remote_indexes=alloc_indexes
        )

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        with self.data_lock:
            self.data[key] = mem_obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.data_lock:
            # NOTE(Jiayi): we assume that the key must be in local data
            # because we are using a push-based transfer
            mem_obj = self.data.get(key, None)
            assert mem_obj is not None, f"Key {key} not found in local data."
            return mem_obj

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        """
        Remove the key from the storage backend and free the associated page.

        Unconditionally deletes the key from the data store and calls
        ``ref_count_down()`` so that the underlying paged memory is returned
        to the free-block pool.  The caller (``cache_engine.py``) must NOT
        call ``ref_count_down()`` again for the same object after invoking
        this method — the ``elif`` guard in ``cache_engine.py`` ensures this.

        :param key: The key to remove.
        :param force: Unused; retained for interface compatibility.
        :return: True if the key existed and was removed, False otherwise.
        """
        with self.data_lock:
            mem_obj = self.data.pop(key, None)
            if mem_obj is not None:
                mem_obj.ref_count_down()
                if self.pd_config.role == "receiver":
                    asyncio.run_coroutine_threadsafe(
                        self._notify_inflight_freed(), self._recv_loop
                    )
                return True
            return False

    async def _notify_inflight_freed(self) -> None:
        """Decrement the inflight chunk counter and notify waiting allocations.

        Scheduled on the receiver event loop from ``remove()`` (which runs in
        a vLLM worker thread) so that asyncio.Condition operations are always
        called from within the correct event loop.
        """
        async with self._inflight_condition:
            if self._inflight_chunks == 0:
                logger.warning(
                    "inflight_chunks is already 0 before decrement; "
                    "this indicates a counter synchronization bug."
                )
            else:
                self._inflight_chunks -= 1
            self._inflight_condition.notify_all()

    ############################################################
    # Decoder functions end
    ############################################################

    @staticmethod
    def _shutdown_loop(
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        timeout: float = 5.0,
    ) -> None:
        """Cancel all pending tasks on *loop*, stop it, and join the thread.

        Uses a ``threading.Event`` to synchronize shutdown completion so that
        ``thread.join`` is only called after the loop has actually stopped,
        preventing thread or resource leaks when the loop takes time to drain.

        :param loop: The event loop to shut down.
        :param thread: The thread running the event loop.
        :param timeout: Maximum seconds to wait for shutdown and thread join.
        """
        shutdown_done = threading.Event()

        async def _cancel_and_stop() -> None:
            tasks = [
                t
                for t in asyncio.all_tasks(loop)
                if t is not asyncio.current_task() and not t.done()
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()
            shutdown_done.set()

        if loop.is_running():
            loop.call_soon_threadsafe(loop.create_task, _cancel_and_stop())
            shutdown_done.wait(timeout=timeout)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Event loop thread %s did not terminate within %.1fs timeout.",
                thread.name,
                timeout,
            )

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        # Wake up any threads blocked on the sender staging condition so they
        # can observe running=False and exit cleanly.
        if hasattr(self, "_sender_staging_condition"):
            with self._sender_staging_condition:
                self._sender_staging_condition.notify_all()
        for thread in self.running_threads:
            thread.join()
        # Shut down sender async loop if present
        if hasattr(self, "_sender_loop"):
            # Cancel the global FIFO request worker before stopping the loop.
            # _shutdown_loop also cancels all tasks, but being explicit here
            # ensures the worker sees CancelledError promptly.
            if (
                hasattr(self, "_global_req_worker_task")
                and self._global_req_worker_task is not None
            ):
                self._global_req_worker_task.cancel()
            self._shutdown_loop(self._sender_loop, self._sender_thread)
            # Close async alloc sockets
            for sock in self._async_alloc_sockets.values():
                try:
                    sock.close()
                except Exception:
                    pass
            try:
                self._async_zmq_context.term()
            except Exception:
                pass
        # Shut down receiver async loop if present
        if hasattr(self, "_recv_loop"):
            self._shutdown_loop(self._recv_loop, self._recv_thread)
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True
