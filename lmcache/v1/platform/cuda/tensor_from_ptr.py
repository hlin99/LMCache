# SPDX-License-Identifier: Apache-2.0
"""CUDA backend for reconstructing tensors from raw pointers."""

# Standard
from typing import ClassVar
import ctypes
import ctypes.util
import math

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.tensor_from_ptr import TensorFromPtrBackend

_COPY_LIB_NOT_LOADED = object()
_copy_lib: ctypes.CDLL | None | object = _COPY_LIB_NOT_LOADED


def get_cuda_copy_lib() -> ctypes.CDLL | None:
    """Return the cached CUDA or ROCm runtime library for device-to-device copies."""
    global _copy_lib
    if _copy_lib is _COPY_LIB_NOT_LOADED:
        for name, fallback in [
            ("cudart", "libcudart.so"),
            ("amdhip64", "libamdhip64.so"),
        ]:
            try:
                path = ctypes.util.find_library(name)
                _copy_lib = ctypes.CDLL(path) if path else ctypes.CDLL(fallback)
                break
            except OSError:
                continue
        else:
            _copy_lib = None

    if isinstance(_copy_lib, ctypes.CDLL):
        return _copy_lib
    return None


class CudaTensorFromPtrBackend(TensorFromPtrBackend):
    """CUDA tensor-from-pointer backend with zero-copy and memcpy fallback."""

    device_type: ClassVar[str] = "cuda"
    impl_key: ClassVar[str] = "default"

    def tensor_from_ptr(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a CUDA tensor from a raw device pointer.

        Args:
            ptr: Raw CUDA device pointer.
            shape: Desired tensor shape.
            dtype: Tensor dtype matching the device-memory layout.
            device: Target CUDA device, including device index when present.

        Returns:
            A CUDA tensor backed by the pointer. The backend first attempts a
            zero-copy ``__cuda_array_interface__`` reconstruction and falls back
            to ``cudaMemcpy`` device-to-device copy when needed.

        Raises:
            RuntimeError: If neither reconstruction strategy is available.
            ValueError: If ``device`` is not a CUDA device.
        """
        if device.type != "cuda":
            raise ValueError("CudaTensorFromPtrBackend requires a CUDA device")

        numel = math.prod(int(dim) for dim in shape)
        total_bytes = numel * torch.empty((), dtype=dtype).element_size()

        try:
            dtype_to_typestr = {
                torch.float16: "<f2",
                torch.float32: "<f4",
                torch.float64: "<f8",
                torch.int8: "|i1",
                torch.int16: "<i2",
                torch.int32: "<i4",
                torch.int64: "<i8",
                torch.uint8: "|u1",
                torch.bool: "|b1",
            }
            is_bfloat16 = dtype == torch.bfloat16
            typestr = "<i2" if is_bfloat16 else dtype_to_typestr.get(dtype, "|u1")

            class _CudaArrayWrapper:
                def __init__(
                    self, ptr_int: int, shape_tuple: tuple[int, ...], type_str: str
                ) -> None:
                    self.__cuda_array_interface__ = {
                        "data": (ptr_int, False),
                        "shape": shape_tuple,
                        "typestr": type_str,
                        "version": 3,
                    }

            tensor = torch.as_tensor(
                _CudaArrayWrapper(ptr, (numel,), typestr), device=device
            )
            if is_bfloat16:
                tensor = tensor.view(torch.bfloat16)

            return tensor.view(*shape)
        except Exception:
            pass

        copy_lib = get_cuda_copy_lib()
        if copy_lib is None:
            raise RuntimeError("Failed to load libcudart/libamdhip")

        cuda_memcpy = copy_lib.cudaMemcpy
        cuda_memcpy.restype = ctypes.c_int
        cuda_memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]

        dst = torch.empty(numel, dtype=dtype, device=device)
        cuda_memcpy_d2d = 3
        err = cuda_memcpy(
            ctypes.c_void_p(dst.data_ptr()),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(total_bytes),
            ctypes.c_int(cuda_memcpy_d2d),
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2D failed with error code {err}.")

        return dst.view(*shape)
