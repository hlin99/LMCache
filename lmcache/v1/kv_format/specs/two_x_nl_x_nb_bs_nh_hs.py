# SPDX-License-Identifier: Apache-2.0
"""Two-list MHA, split NB/BS: ``2 x NL x [NB, BS, NH, HS]`` (SGLang MP daemon).

``[K_layers, V_layers]``, each a ``list[NL]`` of a 4-D tensor that keeps
num_blocks and block_size as separate axes.
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
from lmcache.v1.kv_format.probes.base import KVFormatProbe
from lmcache.v1.kv_format.specs.base import KVFormatSpec
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class TWO_X_NL_X_NB_BS_NH_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
    attention_backends = ("SGLang MHA via MP daemon (4-D inner)",)
    is_kv_list = True

    def num_layers(self) -> int:
        return len(self.kv_caches[0])

    def num_blocks(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][0].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[0][layer_idx]
        return inner.shape[2] * inner.shape[3]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[0][layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        k, v = cast(list[list[torch.Tensor]], self.kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]


class TWO_X_NL_X_NB_BS_NH_HS_Probe(KVFormatProbe):
    engine_type = EngineType.SGLANG
    format_spec = TWO_X_NL_X_NB_BS_NH_HS_Spec

    @classmethod
    def probe(
        cls,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache | None:
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], list)
            and kv_caches[0]
            and isinstance(kv_caches[0][0], torch.Tensor)
            and kv_caches[0][0].dim() == 4
        ):
            return kv_caches
        if not (
            isinstance(kv_caches, list)
            and len(kv_caches) > 0
            and len(kv_caches) % 2 == 0
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 3
            and kv_caches[0].shape[1] > 1
            and "tokens_per_block" in layout_hints
        ):
            return None

        layers_list = cast(list[torch.Tensor], kv_caches)
        block_size = layout_hints["tokens_per_block"]
        half = len(layers_list) // 2
        regrouped: list[DiscoverableKVCache] = []
        for layers in (layers_list[:half], layers_list[half:]):
            reshaped: list[DiscoverableKVCache] = []
            for layer in layers:
                page_buffer_size = layer.shape[0]
                if page_buffer_size % block_size != 0:
                    raise ValueError(
                        f"SGLang page_buffer_size {page_buffer_size} not "
                        f"divisible by tokens_per_block {block_size}"
                    )
                num_blocks = page_buffer_size // block_size
                reshaped.append(layer.view(num_blocks, block_size, *layer.shape[1:]))
            regrouped.append(reshaped)
        return regrouped
