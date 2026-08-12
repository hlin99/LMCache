# SPDX-License-Identifier: Apache-2.0
"""DSA indexer k-cache: ``NL x [NB, BS, 132]`` uint8, blocked scales.

Logically identical to the per-layer MLA layout (one plane, ``num_heads ==
1``), which is what vLLM's ``get_kv_cache_shape`` reports and how the tensor
registers. The PHYSICAL page layout differs (vllm
``indexer_k_quant_and_cache``): per block, all tokens' 128-byte fp8 values
first, then all tokens' 4-byte fp32 scales —
``[BS x 128 vals][BS x 4B scales]``. Only the transfer kernels care (they
address values and scales separately); every geometry accessor here matches
the MLA spec.
"""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.specs.nl_x_nb_bs_hs import (
    NL_X_NB_BS_HS_Spec,
)
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class NL_X_NB_BSV_BSS_Spec(NL_X_NB_BS_HS_Spec):
    engine_type = EngineType.VLLM
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS
    attention_backends = ("vLLM MLA",)
    is_layer_list = True
    is_mla = True

    @classmethod
    def try_normalize(
        cls,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache | None:
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 3
            and kv_caches[0].dtype == torch.uint8
            and int(kv_caches[0].shape[-1]) == 132
        ):
            return kv_caches
        return None
