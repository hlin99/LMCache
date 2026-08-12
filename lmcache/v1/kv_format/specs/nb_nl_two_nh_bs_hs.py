# SPDX-License-Identifier: Apache-2.0
"""Cross-layer, HND: a single bare tensor ``[NB, NL, 2, NH, BS, HS]``.

All layers are packed along dim-1, heads before block-size. Produced e.g. by
TRT-LLM.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.specs.base import KVFormatSpec
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class NB_NL_TWO_NH_BS_HS_Spec(KVFormatSpec):
    engine_type = EngineType.TRTLLM
    engine_kv_format = lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS
    attention_backends = ("TRT-LLM cross-layer (HND layout)",)
    is_cross_layer = True
    is_hnd = True

    def num_layers(self) -> int:
        return self.kv_caches.shape[1]

    def num_blocks(self) -> int:
        return self.kv_caches.shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4]

    def page_buffer_size(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[4]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3] * self.kv_caches.shape[5]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[5]

    def tokens_per_layer(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[4]

    def elements_per_layer(self) -> int:
        t = self.kv_caches
        return t.shape[0] * 2 * t.shape[3] * t.shape[4] * t.shape[5]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches.dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        tensor = cast(torch.Tensor, self.kv_caches)
        return [tensor.data_ptr()]

    @classmethod
    def try_normalize(
        cls,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache | None:
        if isinstance(kv_caches, list) and len(kv_caches) == 1:
            kv_caches = kv_caches[0]
        if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 4:
            num_kv_heads = layout_hints.get("num_kv_heads")
            tokens_per_block = layout_hints.get("tokens_per_block")
            head_dim = layout_hints.get("head_dim")
            if num_kv_heads is None or tokens_per_block is None or head_dim is None:
                raise ValueError(
                    "TRT-LLM discovery needs layout_hints with "
                    "num_kv_heads, tokens_per_block, head_dim"
                )
            num_blocks, num_layers, kv_size, flat = kv_caches.shape
            if flat != num_kv_heads * tokens_per_block * head_dim:
                raise ValueError(
                    f"TRT-LLM 4-D flat dim {flat} != num_kv_heads ({num_kv_heads}) "
                    f"* tokens_per_block ({tokens_per_block}) * head_dim ({head_dim})"
                )
            kv_caches = kv_caches.view(
                num_blocks,
                num_layers,
                kv_size,
                num_kv_heads,
                tokens_per_block,
                head_dim,
            )
        if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 6:
            return kv_caches
        return None
