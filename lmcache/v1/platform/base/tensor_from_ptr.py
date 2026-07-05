# SPDX-License-Identifier: Apache-2.0
"""Platform-abstraction base class for pointer-to-tensor reconstruction."""

# Standard
import abc

# Third Party
import torch


class TensorFromPtrBackend(abc.ABC):
    """Base class for reconstructing tensors from raw pointers.

    Concrete backends provide the device-specific logic needed to build a
    :class:`torch.Tensor` from a raw pointer, shape, dtype, and target device.
    This capability is required, so callers resolve implementations via
    :func:`lmcache.v1.platform._registry.resolve_impl`, which raises when no
    backend is registered for the requested device type.
    """

    @abc.abstractmethod
    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Reconstruct a tensor from a raw pointer on the given device.

        Args:
            ptr: Raw memory pointer as an integer. Must remain valid for the
                lifetime required by the returned tensor.
            shape: Desired tensor shape.
            dtype: Tensor dtype matching the pointed-to memory layout.
            device: Device where the pointer lives.

        Returns:
            A tensor backed by the provided pointer, either zero-copy or by the
            backend's documented fallback strategy.

        Raises:
            RuntimeError: If the backend cannot materialize a tensor for the
                pointer on the requested device.
            ValueError: If the backend does not support the requested input.
        """
        raise NotImplementedError
