# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, no_type_check
import asyncio
import socket

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import (
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def _pad_shape_to_4d(shape: torch.Size, is_layerwise: bool) -> torch.Size:
    """
    Pad a shape to 4D by inserting 1 for missing dimensions.

    This is needed because ClientMetaMessage/ServerMetaMessage protocol
    requires exactly 4 dimensions, but layerwise cache uses 3D shapes.

    Args:
        shape: Input shape (3D or 4D)
        is_layerwise: Whether this is for layerwise cache (determined by key type)

    Returns:
        4D shape with padding if necessary

    Examples:
        With is_layerwise=True:
            [2, 128, 4096] -> [2, 1, 128, 4096]  # pad 3D to 4D
        With is_layerwise=False:
            [2, 32, 128, 4096] -> [2, 32, 128, 4096]  # no change
            [2, 1, 128, 4096] -> [2, 1, 128, 4096]  # no change (valid 4D)
    """
    if len(shape) == 4:
        return shape
    elif len(shape) == 3 and is_layerwise:
        # Insert num_layers=1 at dimension 1 for layerwise cache
        # [2, num_tokens, hidden_dim] -> [2, 1, num_tokens, hidden_dim]
        return torch.Size([shape[0], 1, shape[1], shape[2]])
    else:
        raise ValueError(
            f"Unsupported shape dimension {len(shape)} for "
            f"{'layerwise' if is_layerwise else 'non-layerwise'} cache. "
            f"Expected 3D for layerwise or 4D for non-layerwise, got {shape}"
        )


def _unpad_shape_from_4d(shape: torch.Size, is_layerwise: bool) -> torch.Size:
    """
    Unpad a 4D shape back to 3D if it was padded for layerwise cache.

    Args:
        shape: 4D shape from protocol
        is_layerwise: Whether this is for layerwise cache (determined by key type)

    Returns:
        Original shape (3D if layerwise, 4D if non-layerwise)

    Examples:
        With is_layerwise=True:
            [2, 1, 128, 4096] -> [2, 128, 4096]  # unpad to 3D
        With is_layerwise=False:
            [2, 1, 128, 4096] -> [2, 1, 128, 4096]  # no change (valid 4D)
            [2, 32, 128, 4096] -> [2, 32, 128, 4096]  # no change
    """
    if len(shape) == 4 and is_layerwise and shape[1] == 1:
        # Remove the dummy num_layers dimension for layerwise cache
        return torch.Size([shape[0], shape[2], shape[3]])
    return shape


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        # NOTE(Jiayi): According to Python documentation:
        # https://docs.python.org/3/library/asyncio-eventloop.html
        # In general, protocol implementations that use transport-based APIs
        # such as loop.create_connection() and loop.create_server() are faster
        # than implementations that work with sockets.
        # However, we use socket here as we need to use the socket.recv_into()
        # to reduce memory copy.

        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        # loop.sock_recv_into(sock, buf)

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.async_socket_lock = asyncio.Lock()

    # TODO(Jiayi): This should be an async function
    def receive_all(
        self, meta: ServerMetaMessage, is_layerwise: bool
    ) -> Optional[MemoryObj]:
        received = 0
        n = meta.length

        # Unpad shape from 4D back to original dimensions
        original_shape = _unpad_shape_from_4d(meta.shape, is_layerwise)

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.local_cpu_backend.allocate(
            original_shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        # logger.debug("Call to exists()!")

        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.EXIST,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            response = self.client_socket.recv(ServerMetaMessage.packlength())

        return ServerMetaMessage.deserialize(response).code == ServerReturnCode.SUCCESS

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            res = future.result()
            return res
        except Exception as e:
            logger.warning(f"lm connector failed in exists: {e}")
            return False

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        # logger.debug("Async call to put()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        # Determine if this is layerwise cache by checking key type
        is_layerwise = isinstance(key, LayerCacheEngineKey)

        # Pad shape to 4D for protocol compatibility
        kv_shape_4d = _pad_shape_to_4d(kv_shape, is_layerwise)

        async with self.async_socket_lock:
            await self.loop.sock_sendall(
                self.client_socket,
                ClientMetaMessage(
                    ClientCommand.PUT,
                    key,
                    len(kv_bytes),
                    memory_format,
                    kv_dtype,
                    kv_shape_4d,
                ).serialize(),
            )

            await self.loop.sock_sendall(self.client_socket, kv_bytes)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # NOTE(Jiayi): Not using any await in the following as
        # we don't want to yield control to other tasks which could
        # sacrifice the performance loading to trade the performance of
        # saving

        # Determine if this is layerwise cache by checking key type
        is_layerwise = isinstance(key, LayerCacheEngineKey)

        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    ClientCommand.GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            data = self.client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        if meta.code != ServerReturnCode.SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all(meta, is_layerwise)

        return memory_obj

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        async with self.async_socket_lock:
            self.client_socket.close()
        logger.info("Closed the lmserver connection")
