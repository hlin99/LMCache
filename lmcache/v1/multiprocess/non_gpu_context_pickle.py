# SPDX-License-Identifier: Apache-2.0
"""Pickle-based NonGpuContext implementation for multiprocess mode."""

# Standard
from typing import Any
import pickle

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.non_gpu_context import (
    NonGpuContext,
    NonGpuContextMetadata,
)
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


class NonGpuContextPickle(NonGpuContext):
    """Pickle-based implementation of :class:`NonGpuContext`.

    Transport mechanism:
    - **Store**: ``prepare_store`` sends ``PREPARE_STORE`` (returns empty slots
      for pickle mode); ``commit_store`` serialises chunks and sends
      ``COMMIT_STORE``.
    - **Retrieve**: ``prepare_retrieve`` sends ``PREPARE_RETRIEVE`` and
      deserialises the returned bytes; ``commit_retrieve`` sends
      ``COMMIT_RETRIEVE`` (no-op for pickle).
    """

    def __init__(
        self,
        metadata: NonGpuContextMetadata,
        mq_client: Any,
        mq_timeout: float,
    ) -> None:
        super().__init__(metadata, mq_client, mq_timeout)

    def prepare_store(
        self, key: Any, instance_id: int, block_ids: list[int], blocks_per_chunk: int
    ) -> tuple[Any, list[torch.Tensor] | None]:
        """Send PREPARE_STORE RPC. For pickle, returns no pre-allocated buffers.

        Returns:
            ``((key, instance_id), None)`` — gather will allocate its own tensors.
        """
        future = self.mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass
        return ((key, instance_id), None)

    def commit_store(self, handle: Any, chunks: list[torch.Tensor]) -> bool:
        """Serialize chunks and send via COMMIT_STORE.

        Returns:
            ``True`` on success, ``False`` on failure or timeout.
        """
        key, instance_id = handle
        serialised = pickle.dumps(chunks)
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, serialised],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve(
        self, key: Any, instance_id: int
    ) -> tuple[Any, list[torch.Tensor] | None]:
        """Send PREPARE_RETRIEVE and deserialize the response data.

        Returns:
            ``((key, instance_id), chunks)`` on hit, or
            ``((key, instance_id), None)`` on miss/timeout.
        """
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return ((key, instance_id), None)
        if not response.success or not response.data:
            return ((key, instance_id), None)
        chunks: list[torch.Tensor] = pickle.loads(response.data)
        return ((key, instance_id), chunks)

    def commit_retrieve(self, handle: Any) -> None:
        """Send COMMIT_RETRIEVE (no-op for pickle path)."""
        key, instance_id = handle
        future = self.mq_client.submit_request(
            RequestType.COMMIT_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.COMMIT_RETRIEVE),
        )
        try:
            future.result(timeout=self.mq_timeout)
        except TimeoutError:
            pass

    def close(self) -> None:
        """No-op: the pickle path holds no persistent resources."""
