# SPDX-License-Identifier: Apache-2.0
"""Lightweight store-path timing utility for multiprocess transfer contexts."""

# Standard
import logging
import time

# First Party
from lmcache.utils import init_logger

logger = init_logger(__name__)


class StoreTimer:
    """Lightweight store timing utility.

    Only records timestamps when debug logging is enabled — zero overhead in
    production.  All four store paths (GPU IPC C++ ops, GPU IPC Python ops,
    Async SHM, Async Pickle) use this to emit a single unified
    ``[STORE-TIMING]`` log line with four standardized metrics.

    Usage::

        timer = StoreTimer(request_id, path="shm")
        # ... do work ...
        timer.mark("fwd_return")        # ① vLLM forward thread unblocked
        # ... do work ...
        timer.mark("copy_submitted")    # ② all GPU copies enqueued, CPU returned
        # ... do work ...
        timer.mark("kv_releasable")     # ③ GPU finished reading paged KV
        # ... do work ...
        timer.mark("e2e_complete")      # ④ data committed, retrievable
        timer.emit()

    Args:
        request_id: The request identifier string used in the log line.
        path: Short label for the store path (e.g. ``"gpu_ipc"``, ``"shm"``,
            ``"pickle"``).
    """

    __slots__ = ("_enabled", "_request_id", "_path", "_t0", "_marks")

    def __init__(self, request_id: str, path: str) -> None:
        """Initialize the timer and capture the entry timestamp if debug is on.

        Args:
            request_id: The request identifier string used in the log line.
            path: Short label for the store path (``"gpu_ipc"``, ``"shm"``,
                or ``"pickle"``).
        """
        self._enabled = logger.isEnabledFor(logging.DEBUG)
        self._request_id = request_id
        self._path = path
        self._t0 = time.perf_counter() if self._enabled else 0.0
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record a named timestamp relative to construction time.

        No-op when debug logging is disabled.

        Args:
            name: Metric name (e.g. ``"fwd_return"``, ``"copy_submitted"``,
                ``"kv_releasable"``, ``"e2e_complete"``).
        """
        if self._enabled:
            self._marks[name] = time.perf_counter()

    def emit(self) -> None:
        """Emit a single ``[STORE-TIMING]`` debug log line.

        No-op when debug logging is disabled.  Unrecorded metrics are
        represented as ``—`` in the output.
        """
        if not self._enabled:
            return
        t0 = self._t0

        def _ms(name: str) -> str:
            t = self._marks.get(name)
            if t is None:
                return "\u2014"
            return "%.3f" % ((t - t0) * 1000)

        logger.debug(
            "[STORE-TIMING] req=%s path=%s "
            "fwd_return=%s copy_submitted=%s "
            "kv_releasable=%s e2e_complete=%s ms",
            self._request_id,
            self._path,
            _ms("fwd_return"),
            _ms("copy_submitted"),
            _ms("kv_releasable"),
            _ms("e2e_complete"),
        )
