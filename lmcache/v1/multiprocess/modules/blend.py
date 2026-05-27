# SPDX-License-Identifier: Apache-2.0
"""Blend module stub for the MP cache engine compositor."""

# First Party
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec


class BlendModule:
    """Blend module placeholder.

    The full blend implementation lives in
    :mod:`~lmcache.v1.multiprocess.blend_server_v2`. This stub satisfies the
    EngineModule protocol and can be used when blend functionality is not
    required.
    """

    def __init__(self, context: MPCacheEngineContext) -> None:
        self._ctx = context

    def get_handlers(self) -> list[HandlerSpec]:
        return []

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        pass
