# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Standard
import ctypes
import ctypes.util
from enum import Enum, IntEnum
from multiprocessing import shared_memory
from typing import Sequence

# Third Party
import torch

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}

_copy_lib_not_loaded = object()
_copy_lib: ctypes.CDLL | None | object = _copy_lib_not_loaded


class TransferDirection(Enum):
    H2D = 0
    D2H = 1


class GPUKVFormat(IntEnum):
    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5


def _get_copy_lib() -> ctypes.CDLL | None:
    """Lazily load libcudart for raw pointer copies.

    Returns:
        Loaded libcudart handle, or None when CUDA runtime is unavailable.
    """
    global _copy_lib
    if _copy_lib is _copy_lib_not_loaded:
        try:
            libcudart_path = ctypes.util.find_library("cudart")
            _copy_lib = (
                ctypes.CDLL(libcudart_path)
                if libcudart_path
                else ctypes.CDLL("libcudart.so")
            )
        except OSError:
            _copy_lib = None
    return _copy_lib


def _cuda_memcpy(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Copy bytes between two raw pointers.

    Uses cudaMemcpy(cudaMemcpyDefault) when libcudart is available.
    Falls back to ctypes.memmove for CPU-only environments.
    """
    if nbytes <= 0:
        return

    copy_lib = _get_copy_lib()
    if copy_lib is None:
        ctypes.memmove(dst_ptr, src_ptr, nbytes)
        return

    ret = copy_lib.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(4),  # cudaMemcpyDefault
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy failed with error code {ret}")


def _is_pointer_seq(values: Sequence[object]) -> bool:
    """Return True when all sequence elements are integer pointers."""
    return all(isinstance(value, int) for value in values)


def alloc_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory with NUMA awareness.
    Note: NUMA and pinned memory are not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)

    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()

    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Non-CUDA equivalent of freeing a previously allocated NUMA pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_pinned_ptr(size: int, device_id: int = 0) -> int:
    """Non-CUDA equivalent of allocating pinned memory and returning pointer
    to it. Note: Pinned memory is not supported on non-CUDA."""

    # Create a 1D uint8 CPU tensor, as uint8 == 1 byte
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=False)

    # First-touch initialization (forces physical allocation)
    tensor.fill_(0)

    # Get a pointer to the start of the tensor object as this is what is
    # returned by the CUDA equivalent function
    ptr = tensor.data_ptr()

    # Store the tensor so it can be accessed outide this function scope
    _tensor_registry[ptr] = tensor

    return ptr


def free_pinned_ptr(ptr: int) -> None:
    """Non-CUDA equivalent of freeing a previously allocated pinned pointer."""

    # Release the tensor object for that pointer reference
    _tensor_registry.pop(ptr, None)


def alloc_shm_pinned_ptr(size: int, shm_name: str = "") -> int:
    """Non-CUDA equivalent of allocating shared memory pinned pointer.
    Uses multiprocessing.shared_memory for cross-platform POSIX shm."""

    # Strip leading '/' for SharedMemory name
    name = shm_name.lstrip("/") if shm_name else None

    # Clean up stale shm segment if it exists
    if name:
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=size)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_buffer(shm.buf)
    ptr = ctypes.addressof(buf)

    # Store references to keep them alive
    tensor = torch.frombuffer(buf, dtype=torch.uint8)
    _tensor_registry[ptr] = tensor
    _buf_registry[ptr] = buf
    _shm_registry[ptr] = shm
    return ptr


def free_shm_pinned_ptr(ptr: int, size: int = 0, shm_name: str = "") -> None:
    """Non-CUDA equivalent of freeing a shared memory
    pinned pointer."""

    # Release in order: tensor -> ctypes buf -> shm
    _tensor_registry.pop(ptr, None)
    _buf_registry.pop(ptr, None)
    shm = _shm_registry.pop(ptr, None)
    if shm is not None:
        shm.close()
        shm.unlink()


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor] | list[int],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    block_size: int = 0,
    skip_prefix_n_tokens: int = 0,
) -> None:
    """Fallback multi-layer KV transfer for pointer-based paged buffers.

    This implementation supports pointer-form `key_value_ptrs` and performs
    row-wise copies via cudaMemcpy semantics for both H2D and D2H directions.
    """
    del paged_memory_device
    del skip_prefix_n_tokens
    num_layers = key_value.size(1)
    num_tokens = key_value.size(2)
    hidden_size = key_value.size(3)
    row_nbytes = hidden_size * key_value.element_size()

    is_mla = gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    kv_size = 1 if is_mla else 2
    if gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS and block_size <= 0:
        raise ValueError("block_size must be positive for NL_X_NB_TWO_BS_NH_HS")

    if isinstance(key_value_ptrs, torch.Tensor):
        pointer_values = [int(ptr) for ptr in key_value_ptrs.tolist()]
    elif isinstance(key_value_ptrs, list) and _is_pointer_seq(key_value_ptrs):
        pointer_values = [int(ptr) for ptr in key_value_ptrs]
    else:
        pointer_values = None

    if pointer_values is None:
        raise TypeError(
            "non-pointer key_value_ptrs are not supported in non-CUDA fallback"
        )

    base_ptr = key_value.data_ptr()

    for token_idx in range(num_tokens):
        slot_idx = int(slot_mapping[token_idx].item())
        if slot_idx < 0:
            continue
        for layer_idx in range(num_layers):
            layer_ptr = pointer_values[layer_idx]
            for kv_idx in range(kv_size):
                lmc_offset = (
                    # key_value layout: [kv_size, num_layers, num_tokens, hidden_size]
                    ((kv_idx * num_layers + layer_idx) * num_tokens + token_idx)
                    * row_nbytes
                )
                lmc_ptr = base_ptr + lmc_offset

                if gpu_kv_format in (
                    GPUKVFormat.NB_NL_TWO_BS_NH_HS,
                    GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
                    GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS,
                ):
                    paged_ptr = layer_ptr + (
                        (kv_idx * page_buffer_size + slot_idx) * row_nbytes
                    )
                elif gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                    block_idx = slot_idx // block_size
                    block_offset = slot_idx % block_size
                    paged_ptr = layer_ptr + (
                        ((block_idx * 2 + kv_idx) * block_size + block_offset)
                        * row_nbytes
                    )
                elif gpu_kv_format in (
                    GPUKVFormat.NL_X_NB_BS_HS,
                    GPUKVFormat.NL_X_NBBS_ONE_HS,
                ):
                    paged_ptr = layer_ptr + slot_idx * row_nbytes
                else:
                    raise ValueError(f"Unsupported GPUKVFormat: {gpu_kv_format}")

                if direction == TransferDirection.H2D:
                    src_ptr, dst_ptr = lmc_ptr, paged_ptr
                else:
                    src_ptr, dst_ptr = paged_ptr, lmc_ptr
                _cuda_memcpy(dst_ptr, src_ptr, row_nbytes)


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor | list[torch.Tensor] | list[int],
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
) -> None:
    """Fallback unilateral multi-layer KV transfer for pointer buffers.

    For MLA formats it delegates to `multi_layer_kv_transfer`. For non-MLA
    formats, it handles separate key/value pointer arrays.
    """
    if gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    ):
        multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            direction,
            gpu_kv_format,
        )
        return

    del paged_memory_device
    num_layers = key_value.size(1)
    num_tokens = key_value.size(2)
    hidden_size = key_value.size(3)
    row_nbytes = hidden_size * key_value.element_size()

    if isinstance(key_value_ptrs, torch.Tensor):
        pointer_values = [int(ptr) for ptr in key_value_ptrs.tolist()]
    elif isinstance(key_value_ptrs, list) and _is_pointer_seq(key_value_ptrs):
        pointer_values = [int(ptr) for ptr in key_value_ptrs]
    else:
        pointer_values = None

    if pointer_values is None:
        raise TypeError(
            "non-pointer key_value_ptrs are not supported in non-CUDA fallback"
        )

    base_ptr = key_value.data_ptr()

    for token_idx in range(num_tokens):
        slot_idx = int(slot_mapping[token_idx].item())
        if slot_idx < 0:
            continue
        for layer_idx in range(num_layers):
            for kv_idx in range(2):
                lmc_offset = (
                    # key_value layout: [2, num_layers, num_tokens, hidden_size]
                    ((kv_idx * num_layers + layer_idx) * num_tokens + token_idx)
                    * row_nbytes
                )
                lmc_ptr = base_ptr + lmc_offset
                paged_base = pointer_values[layer_idx + kv_idx * num_layers]
                paged_ptr = paged_base + slot_idx * row_nbytes
                if direction == TransferDirection.H2D:
                    _cuda_memcpy(paged_ptr, lmc_ptr, row_nbytes)
                else:
                    _cuda_memcpy(lmc_ptr, paged_ptr, row_nbytes)
