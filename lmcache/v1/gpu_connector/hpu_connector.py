# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import List, Optional

# Third Party
import habana_frameworks.torch as htorch
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

lmc_ops = None


logger = init_logger(__name__)


class VLLMPagedMemHPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]
    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        use_gpu = True
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemHPUConnectorV2":
        """Create a connector from LMCacheMetadata.
        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
        Returns:
            A new instance of VLLMPagedMemHPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".
        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)
        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! to_gpu")
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        if isinstance(self.kvcaches, torch.Tensor):
            logger.error("kvcaches is Tensor: shape=%s, dtype=%s", self.kvcaches.shape, self.kvcaches.dtype)
        elif isinstance(self.kvcaches, (list, tuple)):
            logger.error("kvcaches is %s, len=%s", type(self.kvcaches).__name__, len(self.kvcaches))
            for i, item in enumerate(self.kvcaches):
                if isinstance(item, torch.Tensor):
                    logger.error("  kvcaches[%s]: shape=%s, dtype=%s", i, item.shape, item.dtype)
                elif isinstance(item, (list, tuple)):
                    logger.error("  kvcaches[%s]: %s, len=%s", i, type(item).__name__, len(item))
                    for j, t in enumerate(item):
                        if isinstance(t, torch.Tensor):
                            logger.error("    kvcaches[%s][%s]: shape=%s, dtype=%s", i, j, t.shape, t.dtype)
                        else:
                            logger.error("    kvcaches[%s][%s]: type=%s", i, j, type(t).__name__)
                else:
                    logger.error("  kvcaches[%s]: type=%s", i, type(item).__name__)
        else:
            logger.error("kvcaches is unknown type: %s", type(self.kvcaches).__name__)

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        if lmc_ops is None:
            if self.gpu_buffer is not None:
                assert self.gpu_buffer.device == self.kvcaches[0][0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                layers = range(len(self.kvcaches))

                htorch.core.mark_step()
                if self.use_mla:
                    # MLA: only fused KV in slot 0
                    tmp_gpu_buffer[0] = memory_obj.tensor[0].to(slot_mapping.device)
                    htorch.core.mark_step()

                    for i in layers:
                        self._get_mla_token_major_view(self.kvcaches[i][0]).index_copy_(
                            0,
                            slot_mapping[start:end],
                            tmp_gpu_buffer[0][i],
                        )
                    # IMPORTANT: do NOT touch kvcaches[i][1]

                else:
                    # non-MLA: real K / V
                    tmp_gpu_buffer[0] = memory_obj.tensor[0].to(slot_mapping.device)
                    tmp_gpu_buffer[1] = memory_obj.tensor[1].to(slot_mapping.device)
                    htorch.core.mark_step()

                    b, h, d = self.kvcaches[0][0].shape
                    hd = h * d

                    for i in layers:
                        self.kvcaches[i][0].view(b, hd).index_copy_(
                            0,
                            slot_mapping[start:end],
                            tmp_gpu_buffer[0][i],
                        )
                        self.kvcaches[i][1].view(b, hd).index_copy_(
                            0,
                            slot_mapping[start:end],
                            tmp_gpu_buffer[1][i],
                        )
                htorch.core.mark_step()

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".
        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.
        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)
        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! from_gpu ")
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if isinstance(self.kvcaches, torch.Tensor):
            logger.error("kvcaches is Tensor: shape=%s, dtype=%s", self.kvcaches.shape, self.kvcaches.dtype)
        elif isinstance(self.kvcaches, (list, tuple)):
            logger.error("kvcaches is %s, len=%s", type(self.kvcaches).__name__, len(self.kvcaches))
            for i, item in enumerate(self.kvcaches):
                if isinstance(item, torch.Tensor):
                    logger.error("  kvcaches[%s]: shape=%s, dtype=%s", i, item.shape, item.dtype)
                elif isinstance(item, (list, tuple)):
                    logger.error("  kvcaches[%s]: %s, len=%s", i, type(item).__name__, len(item))
                    for j, t in enumerate(item):
                        if isinstance(t, torch.Tensor):
                            logger.error("    kvcaches[%s][%s]: shape=%s, dtype=%s", i, j, t.shape, t.dtype)
                        else:
                            logger.error("    kvcaches[%s][%s]: type=%s", i, j, type(t).__name__)
                else:
                    logger.error("  kvcaches[%s]: type=%s", i, type(item).__name__)
        else:
            logger.error("kvcaches is unknown type: %s", type(self.kvcaches).__name__)

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        htorch.core.mark_step()

        if lmc_ops is None:
            if self.gpu_buffer is not None:
                assert self.gpu_buffer.device == self.kvcaches[0][0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                layers = range(len(self.kvcaches))

                if self.use_mla:
                    tmp_gpu_buffer[0] = torch.stack(
                        tuple(
                            self._get_mla_token_major_view(self.kvcaches[i][0]).index_select(
                                0, slot_mapping[start:end]
                            )
                            for i in layers
                        ),
                        dim=0,
                    )
                else:
                    b, h, d = self.kvcaches[0][0].shape
                    hd_shape = h * d
                    tmp_gpu_buffer[0] = torch.stack(
                        tuple(
                            self.kvcaches[i][0]
                            .view(b, hd_shape)
                            .index_select(0, slot_mapping[start:end])
                            for i in layers
                        ),
                        dim=0,
                    )
                    tmp_gpu_buffer[1] = torch.stack(
                        tuple(
                            self.kvcaches[i][1]
                            .view(b, hd_shape)
                            .index_select(0, slot_mapping[start:end])
                            for i in layers
                        ),
                        dim=0,
                    )

                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)
        htorch.core.mark_step()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    @staticmethod
    def _get_mla_token_major_view(kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Return a token-major view for MLA KV caches.

        Accepts both the legacy `[page_buffer_size, head_size]` layout and the
        current `[num_blocks, block_size, head_size]` layout.
        """
        if kv_cache.ndim == 2:
            return kv_cache
        if kv_cache.ndim == 3:
            num_blocks, block_size, head_size = kv_cache.shape
            return kv_cache.view(num_blocks * block_size, head_size)
        raise ValueError(
            "Unsupported MLA KV cache shape. Expected "
            "[page_buffer_size, head_size] or [num_blocks, block_size, head_size], "
            f"but got {tuple(kv_cache.shape)}."
        )
