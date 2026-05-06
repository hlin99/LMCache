# SPDX-License-Identifier: Apache-2.0
"""
Engine protocol definitions for core KV cache operations.

This module defines the protocol for:
- REGISTER_KV_CACHE: Register a KV cache instance with the server
- UNREGISTER_KV_CACHE: Unregister a KV cache instance
- STORE: Store KV cache blocks to the server
- RETRIEVE: Retrieve KV cache blocks from the server
- LOOKUP: Submit a prefix lookup and return a prefetch job ID
- QUERY_PREFETCH_STATUS: Poll a prefetch job for its result
- END_SESSION: End a session and clean up associated resources
"""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "REGISTER_KV_CACHE",
    "UNREGISTER_KV_CACHE",
    "STORE",
    "RETRIEVE",
    "LOOKUP",
    "QUERY_PREFETCH_STATUS",
    "QUERY_PREFETCH_LOOKUP_HITS",
    "FREE_LOOKUP_LOCKS",
    "END_SESSION",
    # XPU / CPU bounce-buffer path
    "REGISTER_KV_CACHE_LAYOUT",
    "STORE_CPU_CHUNKS",
    "RETRIEVE_CPU_CHUNKS",
]

# Type alias for cache keys
KeyType = IPCCacheEngineKey


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for engine operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Register KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the engine instance
        #   - kv_cache: KVCache - The KV cache configuration
        #   - model_name: str - Name of the model associated with the engine
        #   - world_size: int - World size of the engine
        #   - engine_type: EngineType - Which serving engine produced the
        #     caches (vLLM, SGLang, ...). Drives format detection.
        #   - layout_hints: LayoutHints - See custom_types.LayoutHints.
        # Returns: None
        "REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int, KVCache, str, int, EngineType, LayoutHints],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Store KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to store
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs containing the data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "STORE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[int], bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to retrieve
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs to store retrieved data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        #   - skip_first_n_tokens: int - Number of tokens to skip writing at the
        #     start of the retrieve range (to avoid overwriting APC-shared blocks)
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[int], bytes, int],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Submit a prefix lookup; job is tracked server-side by request_id
        # Payload:
        #   - key: KeyType - Cache key to look up
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "LOOKUP": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the status of a prefetch job by request_id
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count when done, None if still in progress
        "QUERY_PREFETCH_STATUS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the lookup hit chunks before the prefetch is done
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count if lookup is done, None if still in progress
        "QUERY_PREFETCH_LOOKUP_HITS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Free locks (release read locks without a full RETRIEVE)
        # Payload:
        #   - key: KeyType - Cache key whose read locks
        #       to release
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "FREE_LOOKUP_LOCKS": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # End session
        # Payload:
        #   - request_id: str - Request ID of the session to end
        # Returns: None
        "END_SESSION": ProtocolDefinition(
            payload_classes=[str],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # XPU/CPU bounce-buffer path: layout-only KV cache registration
        # Payload:
        #   - instance_id: int - Unique identifier for the engine instance
        #   - model_name: str - Name of the model
        #   - world_size: int - World size of the engine
        #   - engine_type: EngineType - Serving engine type
        #   - layout_hints: LayoutHints - Layout hints (e.g., kv_layout)
        #   - block_size: int - vLLM block size (tokens per block)
        #   - num_layers: int - Number of transformer layers
        #   - hidden_dim_size: int - Hidden dim per head (num_heads * head_size)
        #   - dtype_str: str - PyTorch dtype name (e.g. "bfloat16")
        # Returns: None
        "REGISTER_KV_CACHE_LAYOUT": ProtocolDefinition(
            payload_classes=[
                int,
                str,
                int,
                EngineType,
                LayoutHints,
                int,
                int,
                int,
                str,
            ],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # XPU/CPU bounce-buffer path: worker sends gathered CPU KV chunks to server
        # Payload:
        #   - key: KeyType - Cache key identifying the token range
        #   - instance_id: int - Unique identifier for the engine instance
        #   - cpu_data: bytes - Pickled list of CPU tensors, one per chunk,
        #       each with shape [2, num_layers, chunk_size, hidden_dim_size]
        # Returns: bool - True if store succeeded
        "STORE_CPU_CHUNKS": ProtocolDefinition(
            payload_classes=[KeyType, int, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # XPU/CPU bounce-buffer path: worker requests CPU KV chunks from server
        # Payload:
        #   - key: KeyType - Cache key identifying the token range
        #   - instance_id: int - Unique identifier for the engine instance
        # Returns: tuple[bool, bytes]
        #   - bool: True if retrieve succeeded
        #   - bytes: Pickled list of CPU tensors, one per chunk
        "RETRIEVE_CPU_CHUNKS": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=tuple[bool, bytes],
            handler_type=HandlerType.BLOCKING,
        ),
    }
