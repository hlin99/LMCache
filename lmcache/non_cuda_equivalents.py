# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Standard
from multiprocessing import shared_memory
import ctypes
import ctypes.util
import enum

# Third Party
import torch

# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}


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


class TransferDirection(enum.IntEnum):
    """Transfer direction enum matching the C++ TransferDirection."""

    H2D = 0  # Host (LMCache) to Device (PagedBuffer)
    D2H = 1  # Device (PagedBuffer) to Host (LMCache)


class GPUKVFormat(enum.IntEnum):
    """GPU KV cache format enum matching the C++ GPUKVFormat."""

    NB_NL_TWO_BS_NH_HS = 0  # vLLM CROSS_LAYER mode
    NL_X_TWO_NB_BS_NH_HS = 1  # vLLM non-MLA flash attention
    NL_X_NB_TWO_BS_NH_HS = 2  # vLLM non-MLA flash infer
    NL_X_NB_BS_HS = 3  # vLLM MLA
    TWO_X_NL_X_NBBS_NH_HS = 4  # SGLang MHA
    NL_X_NBBS_ONE_HS = 5  # SGLang MLA


# cudaMemcpyDefault lets the CUDA runtime infer direction automatically
_CUDA_MEMCPY_DEFAULT = 4

# Lazy-loaded CUDA runtime library handle
_cudart_handle = None


def _get_cudart() -> ctypes.CDLL:
    """Load and return the CUDA runtime shared library (libcudart)."""
    global _cudart_handle
    if _cudart_handle is None:
        lib_name = ctypes.util.find_library("cudart")
        if lib_name is None:
            raise RuntimeError("CUDA runtime library (libcudart) not found.")
        _cudart_handle = ctypes.CDLL(lib_name)
    return _cudart_handle


def _cuda_memcpy(dst: int, src: int, count: int) -> None:
    """Call cudaMemcpy with cudaMemcpyDefault to copy *count* bytes.

    Args:
        dst: Destination address (CUDA device or host pointer).
        src: Source address (CUDA device or host pointer).
        count: Number of bytes to copy.
    """
    cudart = _get_cudart()
    ret = cudart.cudaMemcpy(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(count),
        ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy failed with error code {ret}")


def _is_mla(gpu_kv_format: GPUKVFormat) -> bool:
    """Return True if *gpu_kv_format* is an MLA (Multi-Head Latent Attention)
    format."""
    return gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )


def _page_buffer_byte_offset(
    gpu_kv_format: GPUKVFormat,
    k_or_v: int,
    slot_id: int,
    bytes_per_token: int,
    page_buffer_size: int,
    block_size: int,
) -> int:
    """Compute the byte offset into a paged buffer for the given format.

    Args:
        gpu_kv_format: The GPU KV cache layout format.
        k_or_v: 0 for key, 1 for value.
        slot_id: Slot index from the slot mapping.
        bytes_per_token: Number of bytes per token
            (num_elements * element_size).
        page_buffer_size: Total number of slots in the paged buffer.
        block_size: Block size (used only by NL_X_NB_TWO_BS_NH_HS).

    Returns:
        Byte offset into the paged buffer.
    """
    if gpu_kv_format in (
        GPUKVFormat.NB_NL_TWO_BS_NH_HS,
        GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
    ):
        return (k_or_v * page_buffer_size + slot_id) * bytes_per_token
    elif gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        block_idx = slot_id // block_size
        block_offset = slot_id % block_size
        return (
            block_idx * 2 * block_size + k_or_v * block_size + block_offset
        ) * bytes_per_token
    elif gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    ):
        # MLA formats – no separate k/v dimension
        return slot_id * bytes_per_token
    else:
        raise ValueError(f"Unsupported GPUKVFormat: {gpu_kv_format}")


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
    block_size: int = 0,
    skip_prefix_n_tokens: int = 0,
) -> None:
    """Transfer KV cache data between the LMCache tensor and paged GPU
    buffers using ``cudaMemcpy``.

    This is a non-CUDA-kernel equivalent of the C++ ``multi_layer_kv_transfer``
    function.  Because *key_value_ptrs* contains raw device/host pointers,
    ``cudaMemcpy`` (with ``cudaMemcpyDefault``) is used for each per-token copy
    so the CUDA runtime can handle any combination of device and host memory.

    Args:
        key_value: Contiguous LMCache tensor with shape
            ``[k_or_v_size, num_layers, num_tokens, num_elements]``.
        key_value_ptrs: 1-D ``int64`` tensor of length ``num_layers`` whose
            elements are raw pointers to per-layer paged buffers.
        slot_mapping: 1-D ``int64`` tensor of length ``num_tokens`` mapping
            token indices to slot indices in the paged buffers.
        paged_memory_device: ``torch.device`` where paged buffers reside.
        page_buffer_size: Number of slots in each paged buffer.
        direction: ``TransferDirection.H2D`` copies *from* the LMCache tensor
            *to* the paged buffers; ``D2H`` copies in the opposite direction.
        gpu_kv_format: Layout format of the GPU KV caches.
        block_size: Block size used by ``NL_X_NB_TWO_BS_NH_HS`` format.
        skip_prefix_n_tokens: Number of leading tokens to skip during the
            transfer.
    """
    k_or_v_size = key_value.size(0)
    num_layers = key_value.size(1)
    num_tokens = key_value.size(2)
    num_elements = key_value.size(3)
    element_size = key_value.element_size()
    bytes_per_token = num_elements * element_size

    num_transfer_tokens = num_tokens - skip_prefix_n_tokens

    # Read pointer values and slot mapping on CPU for iteration
    ptrs = key_value_ptrs.cpu().tolist()
    slots = slot_mapping.cpu().tolist()

    kv_base = key_value.data_ptr()

    for token_idx in range(num_transfer_tokens):
        kv_token_idx = token_idx + skip_prefix_n_tokens
        slot_id = int(slots[kv_token_idx])
        if slot_id < 0:
            continue

        for layer_idx in range(num_layers):
            pb_base = int(ptrs[layer_idx])

            for kv in range(k_or_v_size):
                kv_offset = (
                    kv * num_layers * num_tokens + layer_idx * num_tokens + kv_token_idx
                ) * bytes_per_token

                pb_offset = _page_buffer_byte_offset(
                    gpu_kv_format,
                    kv,
                    slot_id,
                    bytes_per_token,
                    page_buffer_size,
                    block_size,
                )

                kv_addr = kv_base + kv_offset
                pb_addr = pb_base + pb_offset

                if direction == TransferDirection.H2D:
                    _cuda_memcpy(pb_addr, kv_addr, bytes_per_token)
                else:
                    _cuda_memcpy(kv_addr, pb_addr, bytes_per_token)


def multi_layer_kv_transfer_unilateral(
    key_value: torch.Tensor,
    key_value_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    gpu_kv_format: GPUKVFormat,
) -> None:
    """Transfer KV cache data between the LMCache tensor and separate K/V
    paged buffers using ``cudaMemcpy``.

    For MLA formats this delegates to :func:`multi_layer_kv_transfer`.
    For non-MLA formats (e.g. SGLang MHA), the pointer array stores
    ``num_layers`` key pointers followed by ``num_layers`` value pointers,
    each pointing to a flat ``[page_buffer_size, scalars_per_token]`` buffer.

    Args:
        key_value: Contiguous LMCache tensor with shape
            ``[2, num_layers, num_tokens, num_elements]`` (non-MLA) or
            ``[1, num_layers, num_tokens, aligned_head_size]`` (MLA).
        key_value_ptrs: 1-D ``int64`` tensor of length ``num_layers * 2``
            containing raw pointers – first ``num_layers`` for keys, next
            ``num_layers`` for values.
        slot_mapping: 1-D ``int64`` tensor of length ``num_tokens``.
        paged_memory_device: ``torch.device`` where paged buffers reside.
        page_buffer_size: Number of slots in each paged buffer.
        direction: ``TransferDirection.H2D`` or ``D2H``.
        gpu_kv_format: Layout format of the GPU KV caches.
    """
    if _is_mla(gpu_kv_format):
        return multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            direction,
            gpu_kv_format,
        )

    num_layers = key_value.size(1)
    num_tokens = slot_mapping.size(0)
    num_elements = key_value.size(3)
    element_size = key_value.element_size()
    bytes_per_token = num_elements * element_size

    # Read pointer values and slot mapping on CPU for iteration
    ptrs = key_value_ptrs.cpu().tolist()
    slots = slot_mapping.cpu().tolist()

    kv_base = key_value.data_ptr()

    for token_idx in range(num_tokens):
        slot_id = int(slots[token_idx])
        if slot_id < 0:
            continue

        for layer_idx in range(num_layers):
            key_ptr = int(ptrs[layer_idx])
            value_ptr = int(ptrs[layer_idx + num_layers])

            for kv in range(2):
                kv_offset = (
                    kv * num_layers * num_tokens + layer_idx * num_tokens + token_idx
                ) * bytes_per_token

                # Unilateral: flat [page_buffer_size, scalars_per_token]
                pb_offset = slot_id * bytes_per_token

                kv_addr = kv_base + kv_offset
                pb_addr = (key_ptr if kv == 0 else value_ptr) + pb_offset

                if direction == TransferDirection.H2D:
                    _cuda_memcpy(pb_addr, kv_addr, bytes_per_token)
                else:
                    _cuda_memcpy(kv_addr, pb_addr, bytes_per_token)
