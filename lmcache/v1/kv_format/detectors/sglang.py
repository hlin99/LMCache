# SPDX-License-Identifier: Apache-2.0
"""SGLang KV cache discovery."""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.kv_format.detectors.base import EngineDetector


class SGLANG_Detector(EngineDetector):
    engine_type = EngineType.SGLANG
