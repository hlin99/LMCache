# SPDX-License-Identifier: Apache-2.0
#
# This file contains Python non-CUDA fallback implementations for
# CUDA-specific operations.
#
# Standard
from multiprocessing import shared_memory
import ctypes

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class TransferDirection:
    """Python fallback for lmcache.c_ops.TransferDirection."""

    H2D = 0
    D2H = 1


class GPUKVFormat:
    """Python fallback for lmcache.c_ops.GPUKVFormat.

    Each value mirrors the corresponding C++ enum value so that code
    comparing against lmc_ops.GPUKVFormat constants works unchanged.
    """

    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5


# Store the tensor objects in memory so that they can be accessed
# outside the scope of this file
_tensor_registry: dict[int, torch.Tensor] = {}
_shm_registry: dict[int, shared_memory.SharedMemory] = {}
_buf_registry: dict[int, ctypes.Array] = {}


def register_tensor(tensor: torch.Tensor) -> None:
    """Register a tensor in the tensor registry by its data pointer.

    This is required before passing the tensor's pointer to
    :func:`multi_layer_kv_transfer` so that the Python fallback can
    reconstruct the tensor from the raw integer pointer stored in
    ``key_value_ptrs``.

    :param torch.Tensor tensor: The tensor to register.
    """
    _tensor_registry[tensor.data_ptr()] = tensor


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
    key_value_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: int,
    gpu_kv_format: int,
    block_size: int = 0,
    skip_prefix_n_tokens: int = 0,
) -> None:
    """Python fallback for lmcache.c_ops.multi_layer_kv_transfer.

    Transfers KV-cache data between a flat LMCache buffer (``key_value``) and
    per-layer paged GPU/CPU buffers, supporting both H2D and D2H directions
    across all supported :class:`GPUKVFormat` layouts.

    Each paged tensor must have been registered with :func:`register_tensor`
    before calling this function so that the Python fallback can look it up
    from the raw integer pointer stored in ``key_value_ptrs``.

    :param torch.Tensor key_value: Flat LMCache buffer with shape
        ``[2, num_layers, num_tokens, hidden_size]`` for non-MLA formats, or
        ``[1, num_layers, num_tokens, hidden_size]`` for MLA formats.
        May reside on any device (CPU, CUDA, or CPU-pinned).
    :param torch.Tensor key_value_ptrs: 1-D int64 tensor of length
        ``num_layers`` holding the ``data_ptr()`` of each per-layer paged
        tensor.  Every referenced tensor must be present in the module-level
        ``_tensor_registry`` (populated via :func:`register_tensor`).
    :param torch.Tensor slot_mapping: 1-D int64 tensor of length
        ``num_tokens`` mapping logical token positions to physical paged slots.
        Entries ``< 0`` indicate prefix / invalid tokens and are skipped.
    :param torch.device paged_memory_device: Device on which the paged tensors
        reside (used to move index tensors to the correct device).
    :param int page_buffer_size: Total number of slots in the paged buffer
        (``num_blocks * block_size``).
    :param int direction: Transfer direction.  Use
        :attr:`TransferDirection.H2D` (``0``) to copy from ``key_value`` into
        the paged buffers, or :attr:`TransferDirection.D2H` (``1``) to copy
        from the paged buffers into ``key_value``.
    :param int gpu_kv_format: Layout of the paged buffers.  Must be one of the
        :class:`GPUKVFormat` constants.
    :param int block_size: Block / page size used by the
        ``NL_X_NB_TWO_BS_NH_HS`` flash-infer format.  Ignored for other
        formats.
    :param int skip_prefix_n_tokens: Number of prefix tokens at the beginning
        of ``slot_mapping`` (and the corresponding positions in ``key_value``)
        that should be skipped.
    :raises KeyError: If a pointer in ``key_value_ptrs`` is not found in the
        tensor registry.  Call :func:`register_tensor` first.
    """
    num_layers = key_value.shape[1]
    hidden_size = key_value.shape[3]
    kv_device = key_value.device

    # Bring slot_mapping to CPU so it can index both CPU and CUDA tensors.
    slot_mapping_cpu = slot_mapping.cpu() if slot_mapping.is_cuda else slot_mapping

    # Only the tokens after the prefix are transferred.
    slot_mapping_active = slot_mapping_cpu[skip_prefix_n_tokens:]

    valid_mask_kv = slot_mapping_active >= 0
    valid_slots_cpu = slot_mapping_active[valid_mask_kv]

    # Move the index tensor to the device that hosts the paged buffers so that
    # index_select / advanced indexing operates on the correct device.
    valid_slots_paged = valid_slots_cpu.to(paged_memory_device)

    # Compute absolute token indices in key_value for the valid (non-prefix,
    # non-negative-slot) positions.  Using integer indices instead of a
    # boolean mask avoids chained advanced-indexing pitfalls when writing
    # back into key_value.
    valid_token_indices = skip_prefix_n_tokens + torch.where(valid_mask_kv)[0]

    is_mla = gpu_kv_format in (
        GPUKVFormat.NL_X_NB_BS_HS,
        GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    is_flash_infer = gpu_kv_format == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS

    for layer_id in range(num_layers):
        ptr = int(key_value_ptrs[layer_id])
        paged_tensor = _tensor_registry[ptr]

        if is_mla:
            # MLA paged layout: [num_blocks, block_size, head_size] or
            # [page_buffer_size, 1, head_size].  Flatten to
            # [page_buffer_size, hidden_size] for uniform access.
            paged_flat = paged_tensor.reshape(page_buffer_size, hidden_size)

            if direction == TransferDirection.H2D:
                src = key_value[0, layer_id, valid_token_indices, :].to(
                    paged_memory_device
                )
                paged_flat[valid_slots_paged] = src
            else:  # D2H
                gathered = paged_flat.index_select(0, valid_slots_paged)
                key_value[0, layer_id, valid_token_indices, :] = gathered.to(
                    kv_device, non_blocking=False
                )

        elif is_flash_infer:
            # Flash-infer paged layout:
            # [num_blocks, 2, block_size, num_heads, head_size].
            # Reshape to [num_blocks, 2, block_size, hidden_size].
            num_blocks = page_buffer_size // block_size
            paged_reshaped = paged_tensor.reshape(
                num_blocks, 2, block_size, hidden_size
            )
            block_indices = (valid_slots_paged // block_size).long()
            block_offsets = (valid_slots_paged % block_size).long()

            if direction == TransferDirection.H2D:
                # key_value slice: [2, num_valid, hidden_size]
                # transpose to [num_valid, 2, hidden_size] to match paged layout
                src = (
                    key_value[:, layer_id, valid_token_indices, :]
                    .to(paged_memory_device)
                    .transpose(0, 1)
                )
                paged_reshaped[block_indices, :, block_offsets, :] = src
            else:  # D2H
                gathered = paged_reshaped[
                    block_indices, :, block_offsets, :
                ]  # [num_valid, 2, hidden_size]
                key_value[:, layer_id, valid_token_indices, :] = gathered.to(
                    kv_device, non_blocking=False
                ).transpose(0, 1)

        else:
            # Non-MLA, non-flash-infer layouts
            # (NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS).
            # Both are viewed flat as [2, page_buffer_size, hidden_size].
            paged_flat = paged_tensor.reshape(2, page_buffer_size, hidden_size)

            if direction == TransferDirection.H2D:
                src = key_value[:, layer_id, valid_token_indices, :].to(
                    paged_memory_device
                )  # [2, num_valid, hidden_size]
                paged_flat[:, valid_slots_paged, :] = src
            else:  # D2H
                gathered = paged_flat.index_select(1, valid_slots_paged)
                key_value[:, layer_id, valid_token_indices, :] = gathered.to(
                    kv_device, non_blocking=False
                )
