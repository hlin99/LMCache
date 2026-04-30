# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic stream wrapper with launch_host_func support.

This module provides :class:`StreamWrapper`, a thin adapter around a
``torch.Stream`` that exposes a ``launch_host_func`` method compatible with
the ``cupy.cuda.ExternalStream`` API.

On CUDA devices the wrapper delegates directly to CuPy's
``ExternalStream.launch_host_func`` for zero-overhead host-function
scheduling.  On non-CUDA backends (XPU, HPU, …) a portable fallback is used:
a device event is recorded on the stream, and a daemon thread waits for that
event to complete before invoking the callback.
"""

# Standard
from typing import Any, Callable
import logging
import queue
import threading

# Third Party
import torch

_logger = logging.getLogger(__name__)


class StreamWrapper:
    """Wraps a torch stream and exposes ``launch_host_func``.

    On CUDA the implementation delegates to
    ``cupy.cuda.ExternalStream.launch_host_func`` (imported lazily so that
    ``cupy`` is never touched on non-CUDA systems).  On other backends a
    single persistent daemon thread drains a queue of ``(event, func, args)``
    entries in order, waiting on each event before invoking the callback.
    This guarantees sequential execution of callbacks while avoiding the
    overhead of creating a new thread per call.

    Args:
        stream: The ``torch.Stream`` (or backend-specific subclass) to wrap.
        device: The ``torch.device`` on which *stream* was created.
    """

    def __init__(self, stream: torch.Stream, device: torch.device) -> None:
        self._stream = stream
        self._device = device
        self._use_cupy = device.type == "cuda"
        if self._use_cupy:
            # Third Party
            import cupy

            self._cupy_stream = cupy.cuda.ExternalStream(
                stream.cuda_stream, device.index
            )
        else:
            self._queue: queue.Queue[
                tuple[Any, Callable[..., Any], tuple[Any, ...]] | None
            ] = queue.Queue()
            self._worker = threading.Thread(
                target=self._run, daemon=True, name="StreamWrapper-worker"
            )
            self._worker.start()

    def _run(self) -> None:
        """Worker loop: drain the callback queue sequentially."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            event, func, args = item
            event.synchronize()
            try:
                func(*args)
            except Exception:
                _logger.exception("Exception in stream host callback %s", func)

    def launch_host_func(self, func: Callable[..., Any], *args: Any) -> None:
        """Schedule *func* to be called after all work on the stream finishes.

        On CUDA this uses ``cudaLaunchHostFunc`` via CuPy, which is
        efficient and does not block.  On other backends the event and
        callback are enqueued and processed in order by a single worker
        thread.

        Args:
            func: Callable to invoke after stream completion.
            *args: Positional arguments forwarded to *func*.
        """
        if self._use_cupy:
            self._cupy_stream.launch_host_func(func, *args)
        else:
            event = self._stream.record_event()
            self._queue.put((event, func, args))
