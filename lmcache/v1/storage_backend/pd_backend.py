# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import threading

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
            # Separate async ZMQ context for sender coroutines
            self._async_zmq_context = zmq.asyncio.Context()
            self._async_alloc_sockets: dict[str, zmq.asyncio.Socket] = {}
            self._async_alloc_locks: dict[str, asyncio.Lock] = {}
            # Chunk-level semaphore to limit decoder buffer pressure.
            # We allow at most half of the total available chunks to be
            # in-flight at once, leaving headroom for chunks that have been
            # transferred but are still in use by the decoder.
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            max_inflight = max(1, total_chunks // 2)
            self._chunk_semaphore = asyncio.Semaphore(max_inflight)
            logger.info(
                "PDBackend sender: chunk semaphore initialized with "
                "max_inflight=%d (total_chunks=%d, buffer=%d bytes, "
                "chunk=%d bytes)",
                max_inflight,
                total_chunks,
                self._aligned_buffer_size,
                self._chunk_size_bytes,
            )
        elif self.pd_config.role == "receiver":
            self._init_receiver()

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

        # Schedule socket creation on the sender event loop to avoid cross-thread issues
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
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
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

        # Acquire chunk-level permits to limit decoder buffer pressure.
        # Acquired before remote allocation so that the total number of
        # chunks simultaneously occupying decoder buffers is bounded.
        # Track the actual number acquired so the finally block releases
        # only as many as were successfully acquired (guards against
        # cancellation mid-loop).
        acquired_chunks = 0
        for _ in range(num_chunks):
            await self._chunk_semaphore.acquire()
            acquired_chunks += 1

        try:
            alloc_request = self._get_remote_alloc_request(keys, memory_objs)
            # Retry loop: decoder may reject the entire request if its buffer
            # pool is too full. In that case all remote_indexes are -1 and
            # already_sent_indexes is empty. We back off and retry.
            # max_alloc_retries additional retries after the first attempt,
            # for max_alloc_retries+1 total attempts.
            max_alloc_retries = 30
            alloc_retry_sleep = 0.5
            alloc_response = None
            for _retry in range(max_alloc_retries + 1):
                alloc_response = await self._async_remote_allocate(
                    receiver_id, alloc_request
                )
                already_sent_indexes = alloc_response.already_sent_indexes
                remote_indexes = alloc_response.remote_indexes
                all_rejected = (
                    len(already_sent_indexes) == 0
                    and len(remote_indexes) == num_chunks
                    and all(r == -1 for r in remote_indexes)
                )
                if not all_rejected:
                    break
                if _retry < max_alloc_retries:
                    logger.warning(
                        "Decoder rejected entire alloc request (buffer full), "
                        "retry %d/%d in %.1fs.",
                        _retry + 1,
                        max_alloc_retries,
                        alloc_retry_sleep,
                    )
                    await asyncio.sleep(alloc_retry_sleep)
            assert alloc_response is not None
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

            if transfer_spec is not None and getattr(
                transfer_spec, "is_last_prefill", False
            ):
                notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, self.proxy_side_channel.send, notif_msg_bytes
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
            # Always release the permits that were actually acquired so that
            # other requests can proceed (guards against cancellation mid-loop
            # leaving fewer than num_chunks permits held).
            for _ in range(acquired_chunks):
                self._chunk_semaphore.release()

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

        # Fire-and-forget: submit to background asyncio event loop
        asyncio.run_coroutine_threadsafe(
            self._async_transfer_task(
                keys=keys,
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
        Uses `await asyncio.sleep` instead of `time.sleep` so the event loop
        can continue processing while waiting for free memory.
        pin=False: PDBackend has no eviction; pinning is unnecessary and
        causes ref_count leaks.

        ``remote_indexes`` always contains one entry per non-already-sent key
        so that the sender can match objects to addresses by position.
        A value of ``-1`` signals that allocation failed for that slot.

        Decoder-side admission control: if the buffer pool is too full to
        satisfy this request, the entire request is rejected up-front
        (all remote_indexes set to -1) so the sender can retry later.
        This avoids the 500-retry busy-loop inside the for-loop and lets
        the sender back off while the decoder drains its in-use buffers.
        """
        total_allocs = len(alloc_request.keys)

        # Decoder-side admission control: reject the entire request if the
        # buffer pool is too full, so the sender retries with backoff instead
        # of busy-looping here on the decoder event loop.
        allocator = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        free_count = len(allocator.free_blocks)
        total_count = free_count + allocator.num_active_allocations
        if free_count < total_allocs or free_count < total_count * 0.25:
            logger.warning(
                "Decoder buffer pool too full to accept request "
                "(free_blocks=%d, total_allocs=%d, total_count=%d, "
                "threshold=25%%). Rejecting entire request so sender retries.",
                free_count,
                total_allocs,
                total_count,
            )
            return AllocResponse(
                already_sent_indexes=[],
                remote_indexes=[-1] * total_allocs,
            )
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)  # copy — we mutate token_dim

        alloc_indexes: list[int] = []
        already_send_indexes: list[int] = []

        self._log_pool_state("before _async_allocate_and_put")

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self.contains(key, pin=False):
                already_send_indexes.append(idx)
                logger.debug(
                    "Key %s (idx=%d) already exists, skipping allocation.",
                    key,
                    idx,
                )
                continue

            if idx == total_allocs - 1:
                token_dim = fmt.token_dim()
                shape[token_dim] = alloc_request.last_chunk_toks

            mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
            # 500 retries × 10 ms = ~5 s timeout before giving up.
            wait_time = 0.01
            max_retries = 500
            retries = 0
            while mem_obj is None:
                retries += 1
                if retries > max_retries:
                    logger.error(
                        "Failed to allocate memory for key %s after %d retries "
                        "(~%.0f s), aborting. "
                        "Pool state: %d entries in data dict. "
                        "The PD buffer pool may be exhausted.",
                        key,
                        max_retries,
                        wait_time * max_retries,
                        len(self.data),
                    )
                    break
                if retries % 50 == 0:
                    logger.warning(
                        "Allocation retry %d/%d for key %s. "
                        "Pool state: %d entries in data dict.",
                        retries,
                        max_retries,
                        key,
                        len(self.data),
                    )
                await asyncio.sleep(wait_time)
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

            logger.debug(
                "Allocated address %d for key %s (idx=%d).",
                mem_obj.meta.address,
                key,
                idx,
            )
            alloc_indexes.append(mem_obj.meta.address)
            self.put(key, mem_obj)

        self._log_pool_state("after _async_allocate_and_put")

        return AllocResponse(
            already_sent_indexes=already_send_indexes, remote_indexes=alloc_indexes
        )

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        with self.data_lock:
            old = self.data.get(key, None)
            if old is not None:
                logger.warning(
                    "Overwriting existing entry for key %s "
                    "(old address=%s, new address=%s). "
                    "Explicitly freeing old MemoryObj.",
                    key,
                    old.meta.address,
                    mem_obj.meta.address,
                )
                old.ref_count_down()
            self.data[key] = mem_obj
            logger.debug(
                "put key=%s address=%d data_size=%d",
                key,
                mem_obj.meta.address,
                len(self.data),
            )

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
                logger.debug(
                    "remove key=%s address=%d ref_count=%d data_size=%d",
                    key,
                    mem_obj.meta.address,
                    mem_obj.get_ref_count(),
                    len(self.data),
                )
                mem_obj.ref_count_down()
                return True
            logger.debug("remove key=%s not found, data_size=%d", key, len(self.data))
            return False

    ############################################################
    # Decoder functions end
    ############################################################

    @staticmethod
    def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
        """Cancel all pending tasks on *loop*, then stop it."""

        async def _cancel_and_stop():
            tasks = [
                t
                for t in asyncio.all_tasks(loop)
                if t is not asyncio.current_task() and not t.done()
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()

        if loop.is_running():
            loop.call_soon_threadsafe(loop.create_task, _cancel_and_stop())

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        for thread in self.running_threads:
            thread.join()
        # Shut down sender async loop if present
        if hasattr(self, "_sender_loop"):
            self._shutdown_loop(self._sender_loop)
            self._sender_thread.join(timeout=5)
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
            self._shutdown_loop(self._recv_loop)
            self._recv_thread.join(timeout=5)
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    # ------------------------------------------------------------------
    # Debugging helpers
    # ------------------------------------------------------------------

    def _log_pool_state(self, label: str) -> None:
        """Log current pool usage for debugging allocation failures."""
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        allocator = (
            self.memory_allocator.cpu_allocator
            if alloc_type == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        logger.info(
            "[PDBackend %s] %s: "
            "data_entries=%d, "
            "free_blocks=%d, "
            "active_allocs=%d, "
            "total_alloc_bytes=%d",
            self.pd_config.role,
            label,
            len(self.data),
            len(allocator.free_blocks),
            allocator.num_active_allocations,
            allocator.total_allocated_size,
        )
