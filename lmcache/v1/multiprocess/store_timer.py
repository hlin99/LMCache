# SPDX-License-Identifier: Apache-2.0
"""Lightweight store-path timing utility for multiprocess transfer contexts."""

# Standard
import logging
import time
from typing import Optional

# First Party
from lmcache.utils import init_logger

logger = init_logger(__name__)


class StoreTimer:
    """Lightweight timing utility with named steps.

    Only records timestamps when debug logging is enabled — zero overhead in
    production.  Each timer instance has a unique *name* that distinguishes it
    in logs, and records an ordered sequence of (step, time) entries.

    Usage::

        timer = StoreTimer(name="store:req-42:shm")
        # ... do work ...
        timer.mark("fwd_return")        # ① vLLM forward thread unblocked
        # ... do work ...
        timer.mark("copy_submitted")    # ② all GPU copies enqueued
        # ... do work ...
        timer.mark("kv_releasable")     # ③ GPU finished reading paged KV
        # ... do work ...
        timer.mark("e2e_complete")      # ④ data committed, retrievable
        timer.emit()

    Args:
        name: Unique identifier for this timer instance.  Used to
            distinguish different timers in the log output (e.g.
            ``"store:req-42:shm"``, ``"store:req-7:gpu_ipc"``).
    """

    __slots__ = ("_enabled", "_name", "_t0", "_steps")

    def __init__(self, name: str) -> None:
        """Initialize the timer and capture the entry timestamp if debug is on.

        Args:
            name: Unique identifier for this timer instance.
        """
        self._enabled = logger.isEnabledFor(logging.DEBUG)
        self._name = name
        self._t0 = time.perf_counter() if self._enabled else 0.0
        self._steps: list[tuple[str, float]] = []

    @property
    def name(self) -> str:
        """Return the timer name."""
        return self._name

    def mark(self, step: str) -> None:
        """Record a named step with its timestamp.

        No-op when debug logging is disabled.

        Args:
            step: Step name describing what just completed (e.g.
                ``"fwd_return"``, ``"copy_submitted"``).
        """
        if self._enabled:
            self._steps.append((step, time.perf_counter()))

    def elapsed_ms(self, step: str) -> Optional[float]:
        """Return elapsed milliseconds from construction to the given step.

        Args:
            step: The step name to look up.

        Returns:
            Elapsed time in milliseconds, or None if the step was never
            recorded.
        """
        for s, t in self._steps:
            if s == step:
                return (t - self._t0) * 1000
        return None

    def step_names(self) -> list[str]:
        """Return an ordered list of recorded step names."""
        return [s for s, _ in self._steps]

    def emit(self) -> None:
        """Emit a single ``[STORE-TIMING]`` debug log line.

        No-op when debug logging is disabled.  Each recorded step is shown
        as ``step=<elapsed_ms>ms`` in the order it was marked.
        """
        if not self._enabled:
            return
        t0 = self._t0
        parts = " ".join(
            f"{step}={((t - t0) * 1000):.3f}" for step, t in self._steps
        )
        logger.debug(
            "[STORE-TIMING] name=%s %s ms",
            self._name,
            parts,
        )
