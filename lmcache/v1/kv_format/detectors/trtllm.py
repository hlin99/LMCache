# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM KV cache discovery."""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.detectors.base import EngineDetector


class TRTLLM_Detector(EngineDetector):
    engine_type = EngineType.TRTLLM
