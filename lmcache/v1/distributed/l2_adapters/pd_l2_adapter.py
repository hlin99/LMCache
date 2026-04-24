# SPDX-License-Identifier: Apache-2.0
"""
PD (Prefetch/Decode) L2 adapter.

Provides ``PdL2AdapterConfig`` for configuration and ``PdL2Adapter`` for
high-performance KV-cache transfer between a sender (prefill) and a
receiver (decode) worker in Multi-process (MP) mode.

Architecture overview
---------------------

* **Sender** — receives ``submit_store_task(keys, objects)`` from the store
  controller.  Each task is dispatched as an async coroutine on a dedicated
  event loop: remote allocation via ZMQ → RDMA write via the transfer
  channel → ``ProxyNotif`` once all batches for a request are done.

* **Receiver** — runs a ZMQ ROUTER server on a dedicated event loop.  Incoming
  ``AllocRequest`` messages allocate staging-buffer slots and ``put()`` them
  into a local data dict.  The store controller on the *sender* side drives
  the RDMA writes; once a key arrives the receiver exposes it through
  ``submit_lookup_and_lock_task`` / ``submit_load_task``.

Reservation-based admission control on the receiver prevents deadlock when
multiple concurrent requests partially fill the buffer.

Reference: ``lmcache/v1/storage_backend/pd_backend_async.py`` on the
``ww16_PR_PD_async_v3`` branch.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union
import asyncio
import os
import threading
import traceback
import uuid

# Third Party
import msgspec
import torch
import zmq
import zmq.asyncio

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import EvictionConfig

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    PersistConfig,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VALID_ROLES: frozenset[str] = frozenset({"sender", "receiver"})
_REQUIRED_FIELDS: tuple[str, ...] = (
    "role",
    "peer_host",
    "peer_init_port",
    "peer_alloc_port",
)


@dataclass(frozen=True)
class PdL2AdapterConfig(L2AdapterConfigBase):
    """
    Configuration for the PD (Prefetch/Decode) L2 adapter.

    Used to set up a high-performance KV-cache transfer channel
    between a sender (prefill) and a receiver (decode) worker in
    Multi-process (MP) mode.

    Required fields:
        role: ``"sender"`` or ``"receiver"``.
        peer_host: Remote peer hostname or IP address.
        peer_init_port: Per-TP-rank NIXL initialisation ports
            (e.g. ``[9051]`` for TP=1).
        peer_alloc_port: Per-TP-rank allocation/notification ports
            (e.g. ``[9052]`` for TP=1).

    Optional fields:
        proxy_host: Proxy notification host; used by sender only
            (default: ``""``).
        proxy_port: Proxy notification port; used by sender only
            (default: ``0``).
        buffer_size: Staging buffer size in bytes per rank
            (default: ``67108864`` = 64 MB).
        buffer_device: Device for the staging buffer, ``"cpu"`` or
            ``"cuda"`` (default: ``"cpu"``).
        transfer_channel: Transfer backend, ``"nixl"`` or
            ``"mock_memory"`` (default: ``"nixl"``).
        nixl_backends: List of NIXL transport backends
            (default: ``["tcp"]``).
        eviction_config: Optional eviction config parsed from the
            ``"eviction"`` sub-dict (default: ``None``).
        persist_config: Persist config parsed from the optional
            ``"persist_enabled"`` key (default: persist enabled).
    """

    role: str
    peer_host: str
    peer_init_port: list[int]
    peer_alloc_port: list[int]
    proxy_host: str = ""
    proxy_port: int = 0
    buffer_size: int = 67108864  # 64 MB
    buffer_device: str = "cpu"
    transfer_channel: str = "nixl"
    nixl_backends: list[str] = field(default_factory=lambda: ["tcp"])
    eviction_config: EvictionConfig | None = field(
        default=None, compare=False, repr=False
    )
    persist_config: PersistConfig = field(
        default_factory=PersistConfig, compare=False, repr=False
    )

    @classmethod
    def from_dict(cls, d: dict) -> "PdL2AdapterConfig":
        """
        Build a PdL2AdapterConfig from a dict (e.g. from parsed JSON).

        Validates all required fields and the ``role`` value, then
        constructs a fully-populated frozen instance.

        Args:
            d: Adapter spec dict. Must include all required keys
                (``role``, ``peer_host``, ``peer_init_port``,
                ``peer_alloc_port``).

        Returns:
            A fully-populated PdL2AdapterConfig instance.

        Raises:
            ValueError: If a required key is missing or ``role`` is
                not ``"sender"`` or ``"receiver"``.
        """
        for key in _REQUIRED_FIELDS:
            if key not in d:
                raise ValueError(
                    f"PdL2AdapterConfig: missing required field {key!r}"
                )

        role = d["role"]
        if role not in _VALID_ROLES:
            raise ValueError(
                f"PdL2AdapterConfig: role must be 'sender' or 'receiver',"
                f" got {role!r}"
            )

        eviction_config = cls._parse_eviction_config(d)
        persist_config = cls._parse_persist_config(d)

        return cls(
            role=role,
            peer_host=d["peer_host"],
            peer_init_port=list(d["peer_init_port"]),
            peer_alloc_port=list(d["peer_alloc_port"]),
            proxy_host=d.get("proxy_host", ""),
            proxy_port=int(d.get("proxy_port", 0)),
            buffer_size=int(d.get("buffer_size", 67108864)),
            buffer_device=d.get("buffer_device", "cpu"),
            transfer_channel=d.get("transfer_channel", "nixl"),
            nixl_backends=list(d.get("nixl_backends", ["tcp"])),
            eviction_config=eviction_config,
            persist_config=persist_config,
        )

    @classmethod
    def help(cls) -> str:
        """
        Return a help string documenting all config fields.

        Returns:
            A multi-line string describing each field, its type,
            default value, and whether it is required or optional.
        """
        return (
            "PD L2 adapter config fields:\n"
            "\n"
            "Required:\n"
            "  role (str)               : 'sender' or 'receiver'\n"
            "  peer_host (str)          : remote peer hostname or IP address\n"
            "  peer_init_port (list[int]): per-TP-rank NIXL init ports"
            " (e.g. [9051] for TP=1)\n"
            "  peer_alloc_port (list[int]): per-TP-rank alloc ports"
            " (e.g. [9052] for TP=1)\n"
            "\n"
            "Optional:\n"
            "  proxy_host (str)         : proxy notification host;"
            " sender only (default: '')\n"
            "  proxy_port (int)         : proxy notification port;"
            " sender only (default: 0)\n"
            "  buffer_size (int)        : staging buffer size in bytes"
            " (default: 67108864 = 64 MB)\n"
            "  buffer_device (str)      : 'cpu' or 'cuda' (default: 'cpu')\n"
            "  transfer_channel (str)   : 'nixl' or 'mock_memory'"
            " (default: 'nixl')\n"
            "  nixl_backends (list[str]): NIXL transport backends"
            " (default: ['tcp'])\n"
        )


# ---------------------------------------------------------------------------
# Wire-protocol messages (msgspec-based, tagged union)
# ---------------------------------------------------------------------------


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages."""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request sent from sender to receiver.

    Attributes:
        keys: Stringified ObjectKeys; ``len(keys)`` == batch size.
        fmt: ``MemoryFormat`` value as int.
        shape: Shape of each memory object as a flat list.
        dtype: String dtype (e.g. ``"bfloat16"``).
        last_chunk_toks: Token count in the last (possibly partial) chunk.
        req_id: Per-request identifier for admission control.  Empty
            string disables per-request chunk accounting.
        is_last_batch: Signals the final batch for *req_id* so the
            receiver can release the reservation.
        total_chunks: Total chunks for this request (for receiver
            reservation on the first batch).
    """

    keys: list[str]
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int
    req_id: str = ""
    is_last_batch: bool = False
    total_chunks: int = 0


class AllocResponse(PDMsgBase):
    """Allocation response sent from receiver to sender.

    Attributes:
        remote_indexes: One index per key in the request; ``-1``
            means allocation failed for that slot.
        already_sent_indexes: Kept for wire-compatibility with the
            sync ``PDBackend``; always empty for the async path.
    """

    remote_indexes: list[int]
    already_sent_indexes: list[int] = []


class ProxyNotif(PDMsgBase):
    """Notification sent from sender to proxy after all RDMA batches
    for a request complete.

    Attributes:
        req_id: The request identifier.
    """

    req_id: str


class CancelNotif(PDMsgBase):
    """Sent from sender to receiver when a request is aborted.

    Attributes:
        req_id: The request identifier.
        keys: Stringified ObjectKeys that the receiver should release.
    """

    req_id: str
    keys: list[str]


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif, CancelNotif]


# ---------------------------------------------------------------------------
# Reservation-based admission control (receiver only)
# ---------------------------------------------------------------------------


class ReservationManager:
    """Manages reservation-based admission control for the receiver PD buffer.

    Prevents deadlock where N concurrent requests each allocate partial
    chunks, fill the buffer, and none can complete.  When a request is
    admitted, its ``total_chunks`` are reserved upfront.  Subsequent
    physical allocations draw from that reservation.

    Used exclusively on the receiver side via asyncio primitives.
    """

    def __init__(
        self,
        total_chunks: int,
        allocation_timeout: float,
        condition_poll_interval: float,
    ) -> None:
        """Initialize the ReservationManager.

        Args:
            total_chunks: Total number of chunks in the buffer.
            allocation_timeout: Max seconds to wait for admission.
            condition_poll_interval: Poll interval for condition waits.
        """
        self._total_chunks = total_chunks
        self._allocation_timeout = allocation_timeout
        self._condition_poll_interval = condition_poll_interval
        self._reservations: dict[str, int] = {}
        self._total_reserved: int = 0
        self._async_admit_condition: asyncio.Condition | None = None

    def init_async_admit_condition(self) -> None:
        """Create ``asyncio.Condition`` bound to the current event loop.

        Must be called from within the target event loop.
        """
        self._async_admit_condition = asyncio.Condition()

    async def async_try_admit(self, req_id: str, total_chunks: int) -> bool:
        """Attempt to reserve *total_chunks* for *req_id*.

        Args:
            req_id: The request identifier to admit.
            total_chunks: Number of chunks to reserve.

        Returns:
            ``True`` if admitted, ``False`` if timed out.
        """
        if self._async_admit_condition is None:
            raise RuntimeError("async_admit_condition not initialized")
        async with self._async_admit_condition:
            deadline = asyncio.get_running_loop().time() + self._allocation_timeout
            while True:
                available = self._total_chunks - self._total_reserved
                if available >= total_chunks:
                    self._reservations[req_id] = total_chunks
                    self._total_reserved += total_chunks
                    logger.debug(
                        "[ADMIT] req=%s admitted, reserved=%d, "
                        "total_reserved=%d/%d",
                        req_id,
                        total_chunks,
                        self._total_reserved,
                        self._total_chunks,
                    )
                    return True
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    logger.warning(
                        "[ADMIT] req=%s admission timed out: need=%d, "
                        "available=%d, total_reserved=%d/%d",
                        req_id,
                        total_chunks,
                        available,
                        self._total_reserved,
                        self._total_chunks,
                    )
                    return False
                try:
                    await asyncio.wait_for(
                        self._async_admit_condition.wait(),
                        timeout=min(remaining, self._condition_poll_interval),
                    )
                except asyncio.TimeoutError:
                    pass

    async def async_release_reservation(self, req_id: str) -> None:
        """Release the reservation held by *req_id*.

        Args:
            req_id: The request identifier whose reservation to release.
        """
        if self._async_admit_condition is None:
            raise RuntimeError("async_admit_condition not initialized")
        async with self._async_admit_condition:
            count = self._reservations.pop(req_id, 0)
            if count > 0:
                self._total_reserved -= count
                logger.debug(
                    "[ADMIT] req=%s reservation released, freed=%d, "
                    "total_reserved=%d/%d",
                    req_id,
                    count,
                    self._total_reserved,
                    self._total_chunks,
                )
                self._async_admit_condition.notify_all()

    def get_total_chunks(self) -> int:
        """Return the total buffer capacity in chunks.

        Returns:
            Total number of chunks this manager was initialized with.
        """
        return self._total_chunks


# ---------------------------------------------------------------------------
# PD L2 Adapter
# ---------------------------------------------------------------------------

# Default timing parameters (can be made configurable later).
_DEFAULT_ALLOCATION_TIMEOUT: float = 30.0
_DEFAULT_SHUTDOWN_TIMEOUT: float = 5.0
_DEFAULT_CONDITION_POLL_INTERVAL: float = 0.1


class PdL2Adapter(L2AdapterInterface):
    """L2 adapter that transfers KV cache between prefill and decode workers.

    The adapter operates in one of two roles:

    * **sender** — the store controller calls ``submit_store_task`` to push
      KV data to a remote receiver via RDMA.
    * **receiver** — a background ZMQ ROUTER server accepts allocation
      requests from senders, registers data in a local dict, and exposes
      it through ``submit_lookup_and_lock_task`` / ``submit_load_task``.

    Args:
        config: A ``PdL2AdapterConfig`` instance.
        l1_memory_desc: Descriptor of the L1 memory buffer (used to
            register L1 memory with the transfer channel for RDMA).
    """

    def __init__(
        self,
        config: PdL2AdapterConfig,
        l1_memory_desc: Optional[L1MemoryDesc] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._l1_memory_desc = l1_memory_desc
        self._running = True

        # Event fds for signalling task completion.
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # Task bookkeeping (thread-safe via _task_lock).
        self._task_lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        # Receiver local data store.
        self._data: dict[ObjectKey, MemoryObj] = {}
        self._data_lock = threading.Lock()
        self._locked_keys: dict[ObjectKey, int] = {}

        # Local identity for ZMQ DEALER sockets.
        self._local_id = ""

        # Timing parameters.
        self._allocation_timeout = _DEFAULT_ALLOCATION_TIMEOUT
        self._shutdown_timeout = _DEFAULT_SHUTDOWN_TIMEOUT
        self._condition_poll_interval = _DEFAULT_CONDITION_POLL_INTERVAL

        # ----- role-specific initialisation -----
        if config.role == "sender":
            self._init_sender_role()
        elif config.role == "receiver":
            self._init_receiver_role()
        else:
            raise ValueError(f"Invalid PD role: {config.role!r}")

    # ------------------------------------------------------------------
    # Event fd interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Return the event fd signalled when a store task completes.

        Returns:
            The store event file descriptor.
        """
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the event fd signalled when a lookup task completes.

        Returns:
            The lookup event file descriptor.
        """
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        """Return the event fd signalled when a load task completes.

        Returns:
            The load event file descriptor.
        """
        return self._load_efd

    # ------------------------------------------------------------------
    # Store interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a store (send) task.

        On the **sender** side this dispatches an async RDMA transfer to
        the remote receiver.  On the **receiver** side this is a no-op
        that completes immediately with success.

        Args:
            keys: Object keys to store.
            objects: Memory objects holding the KV data.

        Returns:
            A task ID for tracking completion.
        """
        task_id = self._alloc_task_id()

        if self._config.role == "sender":
            asyncio.run_coroutine_threadsafe(
                self._sender_store_task(keys, objects, task_id),
                self._sender_loop,
            )
        else:
            # Receiver: store is driven by incoming RDMA, not by the
            # local store controller.  Complete immediately.
            with self._task_lock:
                self._completed_store_tasks[task_id] = True
            self._signal_store_event()

        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Pop all completed store tasks.

        Returns:
            Mapping of task IDs to success flags.
        """
        with self._task_lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and lock interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        """Submit a lookup-and-lock task.

        On the **receiver** side, checks the local data store for keys
        that have been pushed by the sender.  On the **sender** side
        this is a no-op (all bits zero).

        Args:
            keys: Object keys to look up.

        Returns:
            A task ID for tracking completion.
        """
        task_id = self._alloc_task_id()

        bitmap = Bitmap(len(keys))
        if self._config.role == "receiver":
            with self._data_lock:
                for i, key in enumerate(keys):
                    if key in self._data:
                        bitmap.set(i)
                        self._locked_keys[key] = (
                            self._locked_keys.get(key, 0) + 1
                        )

        with self._task_lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._signal_lookup_event()
        return task_id

    def query_lookup_and_lock_result(
        self, task_id: L2TaskId
    ) -> Bitmap | None:
        """Non-blockingly query the result of a lookup task.

        Args:
            task_id: The task to query.

        Returns:
            A ``Bitmap`` on completion, or ``None`` if still pending.
        """
        with self._task_lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Unlock previously locked keys.

        Args:
            keys: Keys to unlock.
        """
        with self._data_lock:
            for key in keys:
                count = self._locked_keys.get(key, 0)
                if count <= 1:
                    self._locked_keys.pop(key, None)
                else:
                    self._locked_keys[key] = count - 1

    # ------------------------------------------------------------------
    # Load interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a load task.

        On the **receiver** side, copies data from the local store into
        the provided *objects*.  On the **sender** side this is a no-op.

        Args:
            keys: Object keys to load.
            objects: Pre-allocated memory objects to write into.

        Returns:
            A task ID for tracking completion.
        """
        task_id = self._alloc_task_id()

        bitmap = Bitmap(len(keys))
        if self._config.role == "receiver":
            with self._data_lock:
                for i, key in enumerate(keys):
                    src = self._data.get(key)
                    if src is None:
                        continue
                    dst_tensor = objects[i].tensor
                    src_tensor = src.tensor
                    if dst_tensor is not None and src_tensor is not None:
                        dst_tensor.copy_(src_tensor)
                        bitmap.set(i)

        with self._task_lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Non-blockingly query the result of a load task.

        Args:
            task_id: The task to query.

        Returns:
            A ``Bitmap`` on completion, or ``None`` if still pending.
        """
        with self._task_lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the adapter and release all resources."""
        self._running = False

        # Wake sender staging waiters.
        if hasattr(self, "_staging_condition"):
            with self._staging_condition:
                self._staging_condition.notify_all()

        # Shut down sender async loop.
        if hasattr(self, "_sender_loop"):
            self._shutdown_loop(
                self._sender_loop,
                self._sender_thread,
                timeout=self._shutdown_timeout,
            )
            for sock in self._async_alloc_sockets.values():
                try:
                    sock.close()
                except Exception:
                    pass
            try:
                self._async_zmq_context.term()
            except Exception:
                pass

        # Shut down receiver async loop.
        if hasattr(self, "_recv_loop"):
            if hasattr(self, "_pending_alloc_tasks"):
                try:

                    async def _wait_pending() -> None:
                        pending = list(self._pending_alloc_tasks)
                        if pending:
                            await asyncio.wait(
                                pending,
                                timeout=self._shutdown_timeout,
                            )

                    future = asyncio.run_coroutine_threadsafe(
                        _wait_pending(), self._recv_loop
                    )
                    future.result(timeout=self._shutdown_timeout + 1)
                except Exception:
                    logger.debug(
                        "Timed out waiting for pending alloc tasks "
                        "during shutdown"
                    )
            self._shutdown_loop(
                self._recv_loop,
                self._recv_thread,
                timeout=self._shutdown_timeout,
            )

        # Close event fds.
        for fd in (self._store_efd, self._lookup_efd, self._load_efd):
            try:
                os.close(fd)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for this adapter.

        Returns:
            A dict with at least ``is_healthy``.
        """
        return {
            "is_healthy": self._running,
            "type": "PdL2Adapter",
            "role": self._config.role,
            "data_count": len(self._data),
        }

    # ==================================================================
    # Sender internals
    # ==================================================================

    def _init_sender_role(self) -> None:
        """Set up the sender event loop, ZMQ context, and proxy socket."""
        # Unique local identity for ZMQ DEALER sockets.
        self._local_id = (
            f"sender-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )

        self._sender_loop = asyncio.new_event_loop()
        self._sender_thread = threading.Thread(
            target=self._sender_loop.run_forever,
            daemon=True,
            name="pd-l2-sender-async",
        )
        self._sender_thread.start()

        self._async_zmq_context = zmq.asyncio.Context()
        self._async_alloc_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._async_alloc_locks: dict[str, asyncio.Lock] = {}
        self._initialized_peers: set[str] = set()
        self._peer_connection_lock = threading.Lock()

        # Staging flow-control.
        self._staging_lock = threading.Lock()
        self._staging_condition = threading.Condition(self._staging_lock)

        # Per-request tracking (accessed only on _sender_loop).
        self._completed_chunks: dict[str, int] = {}
        self._req_has_last: dict[str, bool] = {}
        self._req_total_chunks: dict[str, int] = {}
        self._sent_keys: dict[str, list[str]] = {}
        self._req_receiver: dict[str, str] = {}

        # Proxy socket.
        if self._config.proxy_host and self._config.proxy_port:
            proxy_url = (
                f"{self._config.proxy_host}:{self._config.proxy_port}"
            )
            future = asyncio.run_coroutine_threadsafe(
                self._async_init_proxy_socket(proxy_url),
                self._sender_loop,
            )
            future.result(timeout=10)

        # Transfer channel will be initialized lazily on first
        # _ensure_peer_connection call (requires l1_memory_desc or
        # allocator info that may not be available yet).
        self._transfer_channel: Any = None

    async def _async_init_proxy_socket(self, proxy_url: str) -> None:
        """Create the async ZMQ PUSH socket for ProxyNotif messages.

        Args:
            proxy_url: ``host:port`` string for the proxy.
        """
        self._async_proxy_socket: zmq.asyncio.Socket = (
            self._async_zmq_context.socket(zmq.PUSH)
        )
        self._async_proxy_socket.connect(f"tcp://{proxy_url}")
        self._proxy_send_lock = asyncio.Lock()

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        """Lazily connect to a remote receiver.

        Thread-safe: uses a lock to prevent duplicate connections from
        concurrent vLLM worker threads.

        Args:
            receiver_id: Unique identifier for the receiver.
            receiver_host: Receiver hostname or IP.
            receiver_init_port: NIXL init port for the receiver.
            receiver_alloc_port: ZMQ alloc port for the receiver.
        """
        if receiver_id in self._initialized_peers:
            return
        with self._peer_connection_lock:
            if receiver_id in self._initialized_peers:
                return

            # Transfer channel init (lazy — create on first connection).
            if self._transfer_channel is None:
                self._init_transfer_channel()

            receiver_init_url = f"{receiver_host}:{receiver_init_port}"

            future = asyncio.run_coroutine_threadsafe(
                self._transfer_channel.async_lazy_init_peer_connection(
                    local_id=self._local_id,
                    peer_id=receiver_id,
                    peer_init_url=receiver_init_url,
                ),
                self._sender_loop,
            )
            future.result()

            receiver_alloc_url = f"{receiver_host}:{receiver_alloc_port}"
            future = asyncio.run_coroutine_threadsafe(
                self._async_create_alloc_socket(
                    receiver_id, receiver_alloc_url
                ),
                self._sender_loop,
            )
            future.result(timeout=10)

            self._initialized_peers.add(receiver_id)

    def _init_transfer_channel(self) -> None:
        """Create the transfer channel for RDMA data movement.

        Imports ``CreateTransferChannel`` lazily to avoid pulling in
        heavy dependencies (e.g. NIXL) at module load time.
        """
        # First Party
        from lmcache.v1.transfer_channel import (  # noqa: PLC0415
            CreateTransferChannel,
        )

        peer_init_url: str | None = None
        if self._config.peer_init_port:
            port = self._config.peer_init_port[0]
            peer_init_url = f"{self._config.peer_host}:{port}"
            self._local_id = self._config.peer_host + str(port)

        buffer_ptr = 0
        buffer_size = self._config.buffer_size
        align_bytes = 1
        if self._l1_memory_desc is not None:
            buffer_ptr = self._l1_memory_desc.ptr
            buffer_size = self._l1_memory_desc.size
            align_bytes = self._l1_memory_desc.align_bytes

        self._transfer_channel = CreateTransferChannel(
            async_mode=True,
            channel_type=self._config.transfer_channel,
            role=self._config.role,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=align_bytes,
            tp_rank=0,
            peer_init_url=peer_init_url,
            backends=self._config.nixl_backends,
            device=self._config.buffer_device,
            event_loop=self._sender_loop,
        )

    async def _async_create_alloc_socket(
        self, receiver_id: str, receiver_alloc_url: str
    ) -> None:
        """Create and connect a ZMQ DEALER socket for allocation requests.

        Args:
            receiver_id: Unique identifier for the receiver.
            receiver_alloc_url: ``host:port`` for the receiver's alloc
                ROUTER socket.
        """
        sock = self._async_zmq_context.socket(zmq.DEALER)
        identity = f"{self._local_id}-to-{receiver_id}".encode()
        sock.setsockopt(zmq.IDENTITY, identity)
        sock.connect(f"tcp://{receiver_alloc_url}")
        self._async_alloc_sockets[receiver_id] = sock

    async def _async_remote_allocate(
        self,
        receiver_id: str,
        request: Union[AllocRequest, CancelNotif],
    ) -> AllocResponse:
        """Send an allocation/cancellation request to the remote receiver.

        Args:
            receiver_id: The remote receiver identifier.
            request: The message to send.

        Returns:
            The ``AllocResponse`` from the receiver.
        """
        if receiver_id not in self._async_alloc_locks:
            self._async_alloc_locks[receiver_id] = asyncio.Lock()
        async with self._async_alloc_locks[receiver_id]:
            socket = self._async_alloc_sockets[receiver_id]
            await socket.send_multipart(
                [b"", msgspec.msgpack.encode(request)]
            )
            frames = await socket.recv_multipart()
            msg = frames[-1]
        return msgspec.msgpack.decode(msg, type=PDMsg)  # type: ignore[return-value]

    async def _sender_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        """Execute a single store-and-transfer task on the sender loop.

        Performs: remote alloc → RDMA write → completion bookkeeping.

        Args:
            keys: Object keys for this batch.
            objects: Memory objects holding the KV data.
            task_id: The L2TaskId assigned to this store task.
        """
        success = True
        try:
            # Build alloc request from the first object's metadata.
            if not objects:
                with self._task_lock:
                    self._completed_store_tasks[task_id] = True
                self._signal_store_event()
                return

            fmt = objects[0].meta.fmt
            shape = list(objects[0].meta.shape)
            token_dim = fmt.token_dim()
            last_chunk_toks = objects[-1].meta.shape[token_dim]

            # First Party
            from lmcache.utils import (  # noqa: PLC0415
                TORCH_DTYPE_TO_STR_DTYPE,
            )

            dtype_str = TORCH_DTYPE_TO_STR_DTYPE[objects[0].meta.dtype]
            str_keys = [str(k) for k in keys]

            alloc_request = AllocRequest(
                keys=str_keys,
                fmt=fmt.value,
                shape=shape,
                dtype=dtype_str,
                last_chunk_toks=last_chunk_toks,
            )

            # Determine receiver from config (TP rank 0 for now).
            port = self._config.peer_init_port[0]
            receiver_id = self._config.peer_host + str(port)

            self._ensure_peer_connection(
                receiver_id=receiver_id,
                receiver_host=self._config.peer_host,
                receiver_init_port=port,
                receiver_alloc_port=self._config.peer_alloc_port[0],
            )

            alloc_response = await self._async_remote_allocate(
                receiver_id, alloc_request
            )
            remote_indexes = alloc_response.remote_indexes

            # Check for allocation failure.
            if any(idx == -1 for idx in remote_indexes):
                logger.warning(
                    "Receiver allocation failed for some keys, "
                    "aborting store task %d",
                    task_id,
                )
                success = False
            else:
                # Perform RDMA write.
                if self._transfer_channel is not None and objects:
                    channel_spec = {
                        "receiver_id": receiver_id,
                        "remote_indexes": remote_indexes,
                    }
                    await self._transfer_channel.async_batched_write(
                        objects=objects,
                        transfer_spec=channel_spec,
                    )

        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.error(
                    "Sender store task %d failed: %s\n%s",
                    task_id,
                    exc,
                    traceback.format_exc(),
                )
            success = False
            if isinstance(exc, asyncio.CancelledError):
                raise

        with self._task_lock:
            self._completed_store_tasks[task_id] = success
        self._signal_store_event()

    # ==================================================================
    # Receiver internals
    # ==================================================================

    def _init_receiver_role(self) -> None:
        """Set up the receiver event loop and ZMQ alloc server."""
        self._recv_loop = asyncio.new_event_loop()
        self._recv_thread = threading.Thread(
            target=self._recv_loop.run_forever,
            daemon=True,
            name="pd-l2-receiver-async",
        )
        self._recv_thread.start()

        # Reservation manager (stub chunk count — real value depends on
        # chunk size, which is not known until the first AllocRequest).
        # We initialise with buffer_size / 1 as an upper bound and
        # refine once we know the chunk size.
        self._recv_reservation_mgr = ReservationManager(
            total_chunks=max(self._config.buffer_size, 1),
            allocation_timeout=self._allocation_timeout,
            condition_poll_interval=self._condition_poll_interval,
        )

        # Async primitives (must be created on the receiver loop).
        future = asyncio.run_coroutine_threadsafe(
            self._create_recv_primitives(), self._recv_loop
        )
        future.result(timeout=5)

        # Per-request key tracking for rollback.
        self._req_allocated_keys: dict[str, list[str]] = {}

        # Start the alloc server.
        asyncio.run_coroutine_threadsafe(
            self._async_mem_alloc_server(), self._recv_loop
        )

    async def _create_recv_primitives(self) -> None:
        """Create asyncio primitives bound to the receiver event loop."""
        self._router_send_lock = asyncio.Lock()
        self._pending_alloc_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        self._recv_reservation_mgr.init_async_admit_condition()
        self._alloc_freed_condition: asyncio.Condition = asyncio.Condition()

    async def _async_mem_alloc_server(self) -> None:
        """Async ZMQ ROUTER server for memory allocation requests.

        Uses a ROUTER socket so multiple concurrent senders (xP1D
        topology) can each have requests received and dispatched
        independently.
        """
        async_ctx = zmq.asyncio.Context()
        socket = async_ctx.socket(zmq.ROUTER)
        alloc_port = self._config.peer_alloc_port[0]
        socket.bind(f"tcp://*:{alloc_port}")
        logger.info("PD L2 alloc server listening on port %d", alloc_port)
        try:
            while self._running:
                try:
                    frames = await socket.recv_multipart()
                    identity = frames[0]
                    payload = frames[-1]
                    task = asyncio.create_task(
                        self._handle_alloc_request(
                            socket, identity, payload
                        )
                    )
                    self._pending_alloc_tasks.add(task)
                    task.add_done_callback(
                        self._pending_alloc_tasks.discard
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(
                        "Failed to process alloc request: %s", exc
                    )
                    if self._running:
                        await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            socket.close()
            async_ctx.term()

    async def _handle_alloc_request(
        self,
        socket: zmq.asyncio.Socket,
        identity: bytes,
        payload: bytes,
    ) -> None:
        """Handle a single allocation or cancellation request.

        Args:
            socket: The ROUTER socket for sending the response.
            identity: The sender identity frame.
            payload: The raw msgpack-encoded message bytes.
        """
        n_keys = 0
        try:
            msg = msgspec.msgpack.decode(payload, type=PDMsg)

            if isinstance(msg, CancelNotif):
                for key_str in msg.keys:
                    try:
                        key = ObjectKey.from_string(key_str)
                        self._remove_key(key)
                    except Exception as exc:
                        logger.warning(
                            "Failed to remove key %s during cancel "
                            "for req %s: %s",
                            key_str,
                            msg.req_id,
                            exc,
                        )
                await self._recv_reservation_mgr.async_release_reservation(
                    msg.req_id
                )
                self._req_allocated_keys.pop(msg.req_id, None)
                resp = AllocResponse(remote_indexes=[])
                async with self._router_send_lock:
                    await socket.send_multipart(
                        [identity, b"", msgspec.msgpack.encode(resp)]
                    )
                return

            if not isinstance(msg, AllocRequest):
                raise ValueError(
                    f"Expected AllocRequest, got {type(msg).__name__}"
                )

            n_keys = len(msg.keys)
            alloc_resp = await self._async_allocate_and_put(msg)
            resp_bytes = msgspec.msgpack.encode(alloc_resp)
            async with self._router_send_lock:
                await socket.send_multipart(
                    [identity, b"", resp_bytes]
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to process alloc request from %s: %s",
                identity,
                exc,
            )
            try:
                error_resp = AllocResponse(
                    remote_indexes=[-1] * max(n_keys, 1)
                )
                async with self._router_send_lock:
                    await socket.send_multipart(
                        [
                            identity,
                            b"",
                            msgspec.msgpack.encode(error_resp),
                        ]
                    )
            except Exception:
                logger.warning(
                    "Failed to send error response to %s", identity
                )

    async def _async_allocate_and_put(
        self, alloc_request: AllocRequest
    ) -> AllocResponse:
        """Allocate memory slots and register KV objects on the receiver.

        Uses reservation-based admission: on the first batch for a
        request, reserves ``total_chunks`` upfront.

        Args:
            alloc_request: The allocation request from the sender.

        Returns:
            ``AllocResponse`` with one ``remote_index`` per key.

        Raises:
            RuntimeError: On fail-fast overflow or allocation timeout.
        """
        total_allocs = len(alloc_request.keys)
        req_id = alloc_request.req_id

        # Reservation admission (first batch only).
        is_first_batch = req_id and (
            req_id not in self._req_allocated_keys
        )
        if is_first_batch:
            if alloc_request.total_chunks == 0:
                raise RuntimeError(
                    f"Receiver requires total_chunks > 0 for req "
                    f"{req_id}. Legacy senders (total_chunks=0) are "
                    f"no longer supported."
                )
            admitted = await self._recv_reservation_mgr.async_try_admit(
                req_id, alloc_request.total_chunks
            )
            if not admitted:
                raise RuntimeError(
                    f"Receiver reservation admission timed out for "
                    f"req {req_id} "
                    f"(total_chunks={alloc_request.total_chunks})."
                )

        # Fail-fast: cumulative chunks > declared total.
        if req_id:
            prev = len(self._req_allocated_keys.get(req_id, []))
            if prev + total_allocs > alloc_request.total_chunks:
                for prior_str in self._req_allocated_keys.get(
                    req_id, []
                ):
                    try:
                        self._remove_key(ObjectKey.from_string(prior_str))
                    except Exception as exc:
                        logger.warning(
                            "Rollback failed for key %s: %s",
                            prior_str,
                            exc,
                        )
                self._req_allocated_keys.pop(req_id, None)
                await (
                    self._recv_reservation_mgr.async_release_reservation(
                        req_id
                    )
                )
                raise RuntimeError(
                    f"Request {req_id} protocol violation: declared "
                    f"total_chunks={alloc_request.total_chunks} but "
                    f"attempting {prev + total_allocs} chunks."
                )

        # First Party
        from lmcache.utils import (  # noqa: PLC0415
            STR_DTYPE_TO_TORCH_DTYPE,
        )

        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)

        alloc_indexes: list[int] = []
        current_batch_keys: list[str] = []

        try:
            for idx, key_str in enumerate(alloc_request.keys):
                key = ObjectKey.from_string(key_str)

                if idx == total_allocs - 1:
                    token_dim = fmt.token_dim()
                    shape[token_dim] = alloc_request.last_chunk_toks

                # Allocate memory (with retry + timeout).
                mem_obj = self._receiver_allocate(
                    torch.Size(shape), dtype, fmt
                )
                deadline = (
                    asyncio.get_running_loop().time()
                    + self._allocation_timeout
                )
                while mem_obj is None:
                    remaining = (
                        deadline - asyncio.get_running_loop().time()
                    )
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Failed to allocate memory for key {key} "
                            f"after timeout "
                            f"(~{self._allocation_timeout:.0f}s). "
                            f"req_id={req_id}, "
                            f"key_index={idx}/{total_allocs}."
                        )
                    async with self._alloc_freed_condition:
                        try:
                            await asyncio.wait_for(
                                self._alloc_freed_condition.wait(),
                                timeout=min(
                                    remaining,
                                    self._condition_poll_interval,
                                ),
                            )
                        except asyncio.TimeoutError:
                            pass
                    mem_obj = self._receiver_allocate(
                        torch.Size(shape), dtype, fmt
                    )

                alloc_indexes.append(mem_obj.meta.address)
                self._put(key, mem_obj)
                current_batch_keys.append(key_str)

        except BaseException:
            # Rollback current batch.
            for rk_str in current_batch_keys:
                try:
                    self._remove_key(ObjectKey.from_string(rk_str))
                except Exception as exc:
                    logger.warning(
                        "Rollback remove failed for key %s: %s",
                        rk_str,
                        exc,
                    )
            # Rollback prior batches.
            if req_id:
                for prior_str in self._req_allocated_keys.get(
                    req_id, []
                ):
                    try:
                        self._remove_key(
                            ObjectKey.from_string(prior_str)
                        )
                    except Exception as exc:
                        logger.warning(
                            "Rollback prior key %s: %s", prior_str, exc
                        )
                self._req_allocated_keys.pop(req_id, None)
                await (
                    self._recv_reservation_mgr.async_release_reservation(
                        req_id
                    )
                )
            raise

        # Track allocated keys per request.
        if req_id:
            if req_id not in self._req_allocated_keys:
                self._req_allocated_keys[req_id] = []
            self._req_allocated_keys[req_id].extend(current_batch_keys)
            if alloc_request.is_last_batch:
                self._req_allocated_keys.pop(req_id, None)
                await (
                    self._recv_reservation_mgr.async_release_reservation(
                        req_id
                    )
                )

        return AllocResponse(remote_indexes=alloc_indexes)

    def _receiver_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> MemoryObj | None:
        """Allocate a single memory object on the receiver.

        This is a placeholder that returns ``None`` (allocation failed)
        by default.  A concrete implementation would delegate to the
        underlying paged allocator.  Subclasses or tests can override.

        Args:
            shape: Tensor shape.
            dtype: Tensor data type.
            fmt: Memory format.

        Returns:
            A ``MemoryObj`` or ``None`` if allocation failed.
        """
        # Default no-op; tests and real deployments override via
        # allocator injection.
        return None

    def _put(self, key: ObjectKey, mem_obj: MemoryObj) -> None:
        """Store a memory object in the receiver's local data dict.

        Args:
            key: The object key.
            mem_obj: The memory object.
        """
        with self._data_lock:
            old = self._data.pop(key, None)
            if old is not None:
                logger.debug(
                    "Overwriting existing MemoryObj for key %s", key
                )
                old.ref_count_down()
            self._data[key] = mem_obj

    def _remove_key(self, key: ObjectKey) -> bool:
        """Remove a key from the receiver's local data dict.

        Args:
            key: The object key to remove.

        Returns:
            ``True`` if the key was present and removed.
        """
        with self._data_lock:
            mem_obj = self._data.pop(key, None)
            if mem_obj is not None:
                mem_obj.ref_count_down()
                # Wake allocation retry coroutines.
                if hasattr(self, "_alloc_freed_condition") and hasattr(
                    self, "_recv_loop"
                ):
                    loop = self._recv_loop
                    if loop.is_running():

                        async def _notify_freed() -> None:
                            async with self._alloc_freed_condition:
                                self._alloc_freed_condition.notify_all()

                        asyncio.run_coroutine_threadsafe(
                            _notify_freed(), loop
                        )
                return True
            return False

    def contains(self, key: ObjectKey) -> bool:
        """Check if a key is present in the receiver's local data.

        Args:
            key: The object key to check.

        Returns:
            ``True`` if the key is present.
        """
        with self._data_lock:
            return key in self._data

    # ==================================================================
    # Helpers
    # ==================================================================

    def _alloc_task_id(self) -> L2TaskId:
        """Allocate the next task ID (thread-safe).

        Returns:
            A unique ``L2TaskId``.
        """
        with self._task_lock:
            tid = self._next_task_id
            self._next_task_id += 1
            return tid

    def _signal_store_event(self) -> None:
        """Signal the store event fd."""
        os.eventfd_write(self._store_efd, 1)

    def _signal_lookup_event(self) -> None:
        """Signal the lookup event fd."""
        os.eventfd_write(self._lookup_efd, 1)

    def _signal_load_event(self) -> None:
        """Signal the load event fd."""
        os.eventfd_write(self._load_efd, 1)

    @staticmethod
    def _shutdown_loop(
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        timeout: float = 5.0,
    ) -> None:
        """Cancel all tasks on *loop*, stop it, and join *thread*.

        Args:
            loop: The event loop to shut down.
            thread: The thread running the event loop.
            timeout: Maximum seconds to wait.
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
            loop.call_soon_threadsafe(
                loop.create_task, _cancel_and_stop()
            )
            shutdown_done.wait(timeout=timeout)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Event loop thread %s did not terminate within "
                "%.1fs timeout.",
                thread.name,
                timeout,
            )


# ---------------------------------------------------------------------------
# Factory and registration
# ---------------------------------------------------------------------------


def _create_pd_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create a ``PdL2Adapter`` from config.

    Args:
        config: The adapter config (must be a ``PdL2AdapterConfig``).
        l1_memory_desc: Descriptor of L1 memory (for RDMA registration).

    Returns:
        A new ``PdL2Adapter`` instance.
    """
    if not isinstance(config, PdL2AdapterConfig):
        raise TypeError(
            f"Expected PdL2AdapterConfig, got {type(config).__name__}"
        )
    return PdL2Adapter(config, l1_memory_desc)


register_l2_adapter_type("pd", PdL2AdapterConfig)
register_l2_adapter_factory("pd", _create_pd_adapter)
