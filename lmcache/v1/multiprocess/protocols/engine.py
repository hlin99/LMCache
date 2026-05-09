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
- PREPARE_STORE: Reserve SHM slots for a two-phase store
- COMMIT_STORE: Commit a SHM store after worker writes
- PREPARE_RETRIEVE: Acquire read-locks and return SHM slot metadata
- FINISH_READ: Release read-locks after worker reads from SHM
"""

# Third Party
import msgspec

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# ---------- SHM-path msgspec structs ----------


class ShmSlotMetadata(msgspec.Struct):
    """Metadata for a single SHM slot returned by prepare_store/prepare_retrieve.

    Attributes:
        key: Stringified object key for commit/finish correlation.
        shm_name: POSIX shared-memory segment name.
        offset: Byte offset of the slot relative to the pool base.
        length: Byte length of the slot.
        shape: Tensor shape as a list of ints.
        dtype: Torch dtype name (e.g. ``"float16"``).
        chunk_index: Zero-based index of the chunk this slot corresponds
            to within the original request.  Used by the worker to match
            slots to gathered CPU chunks when OOM causes some keys to be
            skipped.
    """

    key: str
    shm_name: str
    offset: int
    length: int
    shape: list[int]
    dtype: str
    chunk_index: int = 0


class PrepareStoreResponse(msgspec.Struct):
    """Response for ``PREPARE_STORE``.

    Contains only the slots that were successfully allocated.
    Keys that hit OOM are silently excluded (consistent with the CUDA path).

    Attributes:
        slots: List of successfully allocated slot metadata.
    """

    slots: list[ShmSlotMetadata]


class PrepareRetrieveResponse(msgspec.Struct):
    """Response for ``PREPARE_RETRIEVE``.

    Attributes:
        success: ``True`` if all requested keys were found and read-locked.
        slots: Slot metadata for the read-locked keys (empty on failure).
    """

    success: bool
    slots: list[ShmSlotMetadata]


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
    "REGISTER_KV_CACHE_BOUNCE",
    "STORE_CPU_CHUNKS",
    "RETRIEVE_CPU_CHUNKS",
    "PREPARE_STORE",
    "COMMIT_STORE",
    "PREPARE_RETRIEVE",
    "FINISH_READ",
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
        "REGISTER_KV_CACHE_BOUNCE": ProtocolDefinition(
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
                bool,
            ],
            response_class=tuple[str, int],
            handler_type=HandlerType.SYNC,
        ),
        "STORE_CPU_CHUNKS": ProtocolDefinition(
            payload_classes=[KeyType, int, bytes],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        "RETRIEVE_CPU_CHUNKS": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=tuple[bool, bytes],
            handler_type=HandlerType.BLOCKING,
        ),
        # Prepare SHM store (two-phase RPC step 1)
        # Payload:
        #   - key: KeyType - Cache key for the token range
        #   - instance_id: int - Worker instance identifier
        # Returns: PrepareStoreResponse - only successfully allocated slots
        "PREPARE_STORE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=PrepareStoreResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        # Commit SHM store (two-phase RPC step 2)
        # Payload:
        #   - key: KeyType - Original cache key from prepare_store
        #   - instance_id: int - Worker instance identifier
        # Returns: bool - True if commit succeeded
        "COMMIT_STORE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # Prepare SHM retrieve (two-phase RPC step 1)
        # Payload:
        #   - key: KeyType - Cache key for the token range
        #   - instance_id: int - Worker instance identifier
        # Returns: PrepareRetrieveResponse - slot metadata with held read-locks
        "PREPARE_RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=PrepareRetrieveResponse,
            handler_type=HandlerType.BLOCKING,
        ),
        # Finish SHM read (two-phase RPC step 2)
        # Payload:
        #   - key: KeyType - Original cache key from prepare_retrieve
        #   - instance_id: int - Worker instance identifier
        # Returns: bool - True if locks released
        "FINISH_READ": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
    }
