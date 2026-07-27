# SPDX-License-Identifier: Apache-2.0
"""EngineDrivenContext to store/retrieve a contiguous KV tensor for SDK use."""

# Future
from __future__ import annotations

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContext

logger = init_logger(__name__)


class ContiguousTransferWrapper:
    """Store/retrieve a contiguous KV tensor through an ``EngineDrivenContext``.

    Args:
        context: The engine-driven (SHM or pickle) transport.
        chunk_size: Number of tokens per LMCache chunk.
    """

    def __init__(self, context: EngineDrivenContext, chunk_size: int) -> None:
        self._context = context
        self._chunk_size = chunk_size

    def store(self, key: IPCCacheServerKey, instance_id: int, kv: torch.Tensor) -> bool:
        """Store a contiguous [2, L, T, D] tensor

        Args:
            key: The cache server key.
            instance_id: The cache server instance ID.
            kv: The contiguous KV tensor to store.

        Returns:
            True if the store was successful, False otherwise.
        """
        result = self._context.prepare_store(key, instance_id)
        if result is None:
            # Pickle: chunk the contiguous KV tensor (commit takes list of chunks).
            num_chunks = kv.shape[2] // self._chunk_size
            chunks = [
                kv[
                    :, :, i * self._chunk_size : (i + 1) * self._chunk_size, :
                ].contiguous()
                for i in range(num_chunks)
            ]
        else:
            # SHM: fill missing chunks' slots in place.
            slot_tensors, chunk_indices, _group_counts = result
            for slot, chunk_idx in zip(slot_tensors, chunk_indices, strict=True):
                start = chunk_idx * self._chunk_size
                slot.copy_(kv[:, :, start : start + self._chunk_size, :])
            chunks = []
        return self._context.commit_store(key, instance_id, chunks)

    def retrieve(self, key: IPCCacheServerKey, instance_id: int) -> torch.Tensor | None:
        """Retrieve the KV as a contiguous [2, L, hit_tokens, D] tensor

        Args:
            key: The cache server key.
            instance_id: The cache server instance ID.

        Returns:
            The contiguous KV tensor if found, None otherwise.
        """
        slot_tensors = self._context.prepare_retrieve(key, instance_id)
        if not slot_tensors:
            self._abort_retrieve(key, instance_id)
            return None
        try:
            # Both Pickle and SHM returns list of [2, L, T, D] tensors
            # Concatenate along the token dimension.
            result = torch.cat(slot_tensors, dim=2)
        except Exception:
            self._abort_retrieve(key, instance_id)
            raise
        self._context.commit_retrieve(key, instance_id)
        return result

    def _abort_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> None:
        """Release retrieve resources while preserving the caller's outcome.

        Args:
            key: Cache key for the retrieve range.
            instance_id: Worker process instance identifier.

        Unlike ``commit_retrieve``, this marks the retrieve unsuccessful.
        Transport errors are logged and suppressed so they do not replace the
        miss result or exception already being returned by ``retrieve``.
        """
        try:
            self._context.abort_retrieve(key, instance_id)
        except Exception:
            logger.exception("Failed to abort contiguous retrieve")
