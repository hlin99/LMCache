# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import List, Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.protocol import ClientMetaMessage
from lmcache.v1.server.storage_backend.abstract_backend import LMSBackendInterface
from lmcache.v1.server.utils import LMSMemoryObj

logger = init_logger(__name__)


class LMSLocalBackend(LMSBackendInterface):
    def __init__(
        self,
    ) -> None:
        self.dict: OrderedDict[CacheEngineKey, LMSMemoryObj] = OrderedDict()

        self.lock = threading.Lock()

        # TODO(Jiayi): please add evictor

    # TODO
    def list_keys(self) -> List[CacheEngineKey]:
        logger.error(" list_keys +++ key=%s", list(self.dict.keys()))
        with self.lock:
            return list(self.dict.keys())

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:
        logger.error(" contains +++ key=%s", key)
        with self.lock:
            return key in self.dict

    # TODO
    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        logger.error(" remove +++ key=%s", key)
        with self.lock:
            self.dict.pop(key)

    def put(
        self,
        client_meta: ClientMetaMessage,
        kv_chunk_bytes: bytearray,
    ) -> None:
        key_obj = client_meta.key
        hash_str = format(key_obj.chunk_hash, 'x') # key_obj.chunk_hash.hex()
        hash_int = int(hash_str, 16)

        logger.error(f"DEBUG: Putting key to LMSLocalBackend. Hash: {hash_int}")
        with self.lock:
            self.dict[client_meta.key] = LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            )

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[LMSMemoryObj]:
        logger.error(" get +++ key=%s", key)
        with self.lock:
            return self.dict.get(key, None)

    def close(self):
        pass


# TODO(Jiayi): please implement the remote disk backend
# class LMSLocalDiskBackend(LMSBackendInterface):
#    pass
