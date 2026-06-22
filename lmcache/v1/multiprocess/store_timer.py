# SPDX-License-Identifier: Apache-2.0
"""Lightweight store-path timing utility for multiprocess transfer contexts."""

# Standard
import logging
import threading
import time
from typing import Optional

# First Party
from lmcache.utils import init_logger

logger = init_logger(__name__)


class StoreTimer:
    """Thread-safe timing utility supporting multiple named groups.

    Only records timestamps when debug logging is enabled — zero overhead in
    production.  A single timer instance can track multiple independent
    *names* (e.g. different store paths or sub-operations), each with its
    own ordered sequence of (step, time) entries.  Safe for concurrent use
    from multiple threads.

    Usage::

        timer = StoreTimer(prefix="req-42")

        # Thread A — GPU IPC path
        timer.mark("gpu_ipc", "copy_start")
        timer.mark("gpu_ipc", "copy_done")

        # Thread B — SHM path (can run concurrently)
        timer.mark("shm", "serialize_start")
        timer.mark("shm", "serialize_done")
        timer.mark("shm", "write_complete")

        timer.emit()  # outputs one line per name

    Args:
        prefix: Optional prefix prepended to all log lines for easier
            grep / filtering (e.g. request id).
    """

    __slots__ = ("_enabled", "_prefix", "_t0", "_groups", "_lock")

    def __init__(self, prefix: str = "") -> None:
        """Initialize the timer and capture the entry timestamp if debug is on.

        Args:
            prefix: Optional prefix for log output (e.g. request id).
        """
        self._enabled = logger.isEnabledFor(logging.DEBUG)
        self._prefix = prefix
        self._t0 = time.perf_counter() if self._enabled else 0.0
        self._groups: dict[str, list[tuple[str, float]]] = {}
        self._lock = threading.Lock()

    @property
    def prefix(self) -> str:
        """Return the timer prefix."""
        return self._prefix

    def mark(self, name: str, step: str) -> None:
        """Record a step under a named group.

        Thread-safe.  The same (name, step) pair can be recorded multiple
        times (e.g. reentrant calls) — each occurrence is kept with its own
        timestamp.

        Args:
            name: Group name identifying the operation or path (e.g.
                ``"gpu_ipc"``, ``"shm"``, ``"pickle"``).
            step: Step name describing what just completed (e.g.
                ``"copy_start"``, ``"copy_done"``).
        """
        if not self._enabled:
            return
        t = time.perf_counter()
        with self._lock:
            self._groups.setdefault(name, []).append((step, t))

    def elapsed_ms(self, name: str, step: str) -> Optional[float]:
        """Return elapsed ms from construction to the first occurrence of step.

        Thread-safe.

        Args:
            name: The group name.
            step: The step name to look up.

        Returns:
            Elapsed time in milliseconds, or None if name/step not found.
        """
        with self._lock:
            steps = self._groups.get(name)
        if steps is None:
            return None
        for s, t in steps:
            if s == step:
                return (t - self._t0) * 1000
        return None

    def names(self) -> list[str]:
        """Return the list of recorded group names (insertion order)."""
        with self._lock:
            return list(self._groups.keys())

    def steps(self, name: str) -> list[tuple[str, float]]:
        """Return all (step, elapsed_ms) pairs for a given name.

        Args:
            name: The group name.

        Returns:
            Ordered list of (step_name, elapsed_ms) tuples.
        """
        with self._lock:
            entries = self._groups.get(name, [])
        t0 = self._t0
        return [(s, (t - t0) * 1000) for s, t in entries]

    def emit(self) -> None:
        """Emit ``[STORE-TIMING]`` debug log lines, one per name group.

        No-op when debug logging is disabled.
        """
        if not self._enabled:
            return
        t0 = self._t0
        with self._lock:
            snapshot = {k: list(v) for k, v in self._groups.items()}

        for name, entries in snapshot.items():
            parts = " ".join(
                f"{step}={((t - t0) * 1000):.3f}" for step, t in entries
            )
            logger.debug(
                "[STORE-TIMING] %sname=%s %s ms",
                f"prefix={self._prefix} " if self._prefix else "",
                name,
                parts,
            )
