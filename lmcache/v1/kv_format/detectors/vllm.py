# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.kv_format.detectors.base import EngineDetector
from lmcache.v1.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.lmcache_native as lmcache_native


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[lmcache_native.EngineKVFormat | None, DiscoverableKVCache]":
        """Resolve vLLM's layout hint before dispatching detectable specs."""
        resolved_hints = layout_hints.copy()
        kv_layout = resolved_hints.get("kv_layout")
        if torch_device_type == "cpu":
            kv_layout = "HND"
        elif kv_layout is None:
            kv_layout = "NHD"
        resolved_hints["kv_layout"] = kv_layout
        return super().discover(kv_caches, resolved_hints)
