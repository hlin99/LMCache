# SPDX-License-Identifier: Apache-2.0
"""CPU backend for reconstructing tensors from raw pointers."""

# Standard
from typing import ClassVar
import ctypes
import math

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.tensor_from_ptr import TensorFromPtrBackend


class CpuTensorFromPtrBackend(TensorFromPtrBackend):
    """CPU tensor-from-pointer backend using ``torch.frombuffer`` zero-copy."""

    device_type: ClassVar[str] = "cpu"
    impl_key: ClassVar[str] = "default"

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a CPU tensor view over a raw host pointer.

        Args:
            ptr: Raw host-memory pointer.
            shape: Desired tensor shape.
            dtype: Tensor dtype matching the host-memory layout.
            device: Target device. Must be CPU.

        Returns:
            A CPU tensor that shares memory with the pointed-to buffer.

        Raises:
            ValueError: If ``device`` is not a CPU device.
        """
        if device.type != "cpu":
            raise ValueError("CpuTensorFromPtrBackend requires a CPU device")

        numel = math.prod(int(dim) for dim in shape)
        total_bytes = numel * torch.empty((), dtype=dtype).element_size()
        buffer_type = ctypes.c_uint8 * total_bytes
        buffer = buffer_type.from_address(ptr)
        return torch.frombuffer(buffer, dtype=dtype).view(*shape)
