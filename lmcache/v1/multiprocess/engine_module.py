# SPDX-License-Identifier: Apache-2.0
"""Engine module protocol, HandlerSpec, and ThreadPoolType for compositor pattern."""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

# First Party
from lmcache.v1.multiprocess.protocol import RequestType


class ThreadPoolType(Enum):
    """Thread pool affinity for a request handler."""

    GPU = "gpu"
    """Affinity pool (single CUDA device thread)."""

    CPU = "cpu"
    """Normal CPU thread pool."""


@dataclass
class HandlerSpec:
    """Binding of a request type to a handler callable and its thread pool.

    Attributes:
        request_type: The ZMQ message request type this handler serves.
        handler: Callable invoked when a matching request arrives.
        pool_type: Which thread pool should execute the handler.
    """

    request_type: RequestType
    handler: Callable[..., Any]
    pool_type: ThreadPoolType


class EngineModule(Protocol):
    """Protocol that every pluggable cache-engine module must satisfy.

    Modules are assembled by :func:`~lmcache.v1.multiprocess.server._build_modules`
    and collectively provide all handler logic for ``MPCacheEngine``.
    """

    def get_handlers(self) -> list[HandlerSpec]:
        """Return the list of request handlers provided by this module."""
        ...

    def report_status(self) -> dict:
        """Return a status dict fragment contributed by this module."""
        ...

    def close(self) -> None:
        """Release any resources held by this module."""
        ...
