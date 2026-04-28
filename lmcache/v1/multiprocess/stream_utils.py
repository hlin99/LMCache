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
import threading

# Third Party
import torch


class StreamWrapper:
    """Wraps a torch stream and exposes ``launch_host_func``.

    On CUDA the implementation delegates to
    ``cupy.cuda.ExternalStream.launch_host_func`` (imported lazily so that
    ``cupy`` is never touched on non-CUDA systems).  On other backends a
    background thread polls the stream via a recorded event and then calls the
    callback.

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

    def launch_host_func(self, func: Callable[..., Any], *args: Any) -> None:
        """Schedule *func* to be called after all work on the stream finishes.

        On CUDA this uses ``cudaLaunchHostFunc`` via CuPy, which is
        efficient and does not block.  On other backends a daemon thread is
        started that waits for a recorded stream event and then calls
        ``func(*args)``.

        Args:
            func: Callable to invoke after stream completion.
            *args: Positional arguments forwarded to *func*.
        """
        if self._use_cupy:
            self._cupy_stream.launch_host_func(func, *args)
        else:
            event = self._stream.record_event()

            def _wait_and_call() -> None:
                event.synchronize()
                func(*args)

            threading.Thread(target=_wait_and_call, daemon=True).start()
