# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: explicit native adapters over the torch baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    def __init__(self) -> None:
        self._native: object | None = None
        try:
            import lmcache.c_ops as native

            if getattr(native, "__file__", None) is not None:
                self._native = native
        except ImportError:
            logger.warning(
                "lmcache.c_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops."
            )

    def __getattr__(self, name: str) -> object:
        if self._native is not None:
            try:
                return getattr(self._native, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        if self._native is not None:
            names.update(name for name in dir(self._native) if not name.startswith("_"))
        return sorted(names)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer,
        lmcache_objects,
        block_ids,
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
    ):
        """Transfer KV blocks using the native CUDA extension.

        Accepts tensor-first inputs, converts to pointer form for C++,
        and falls back to the torch baseline when the native module is absent.
        """
        import torch

        if self._native is None:
            return super().multi_layer_block_kv_transfer(
                paged_buffer,
                lmcache_objects,
                block_ids,
                device,
                direction,
                shape_desc,
                lmcache_chunk_size,
                engine_kv_format,
                skip_prefix_n_blocks,
            )
        ptr_list = self._flatten_paged_buffer_ptrs(paged_buffer)
        dev = device if isinstance(device, torch.device) else torch.device(device)
        paged_ptr_tensor = torch.tensor(ptr_list, dtype=torch.int64, device=dev)
        obj_ptrs = [int(t.data_ptr()) for t in lmcache_objects]
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64, device=dev)
        return self._native.multi_layer_block_kv_transfer(
            paged_ptr_tensor,
            obj_ptrs,
            block_ids,
            dev,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )

    def batched_memcpy(self, src_ptrs, dst_ptrs, sizes):
        if self._native is not None:
            return self._native.batched_memcpy(src_ptrs, dst_ptrs, sizes)
        return super().batched_memcpy(src_ptrs, dst_ptrs, sizes)

    def lmcache_memcpy_async(self, *args, **kwargs):
        if self._native is not None:
            return self._native.lmcache_memcpy_async(*args, **kwargs)
        return super().lmcache_memcpy_async(*args, **kwargs)

    def record_completion_on_stream(self, *args, **kwargs):
        if self._native is not None:
            return self._native.record_completion_on_stream(*args, **kwargs)
        return super().record_completion_on_stream(*args, **kwargs)

    def record_event_on_stream(self, *args, **kwargs):
        if self._native is not None:
            return self._native.record_event_on_stream(*args, **kwargs)
        return super().record_event_on_stream(*args, **kwargs)

    def drain_recorded_completions(self):
        if self._native is not None:
            return self._native.drain_recorded_completions()
        return super().drain_recorded_completions()

    def drain_recorded_events(self):
        if self._native is not None:
            return self._native.drain_recorded_events()
        return super().drain_recorded_events()

    def alloc_pinned_ptr(self, size, device_id=0):
        if self._native is not None:
            return self._native.alloc_pinned_ptr(size, device_id)
        return super().alloc_pinned_ptr(size, device_id)

    def free_pinned_ptr(self, ptr):
        if self._native is not None:
            return self._native.free_pinned_ptr(ptr)
        return super().free_pinned_ptr(ptr)

    def alloc_hugepage_pinned_ptr(self, size, device_id=0):
        if self._native is not None:
            return self._native.alloc_hugepage_pinned_ptr(size, device_id)
        return super().alloc_hugepage_pinned_ptr(size, device_id)

    def get_gpu_pci_bus_id(self, device_id=0):
        if self._native is not None:
            return self._native.get_gpu_pci_bus_id(device_id)
        return super().get_gpu_pci_bus_id(device_id)

    def rotary_embedding_k_fused(self, *args, **kwargs):
        if self._native is not None:
            return self._native.rotary_embedding_k_fused(*args, **kwargs)
        return super().rotary_embedding_k_fused(*args, **kwargs)

    def calculate_cdf(self, input_tensor, num_bins):
        if self._native is not None:
            return self._native.calculate_cdf(input_tensor, num_bins)
        return super().calculate_cdf(input_tensor, num_bins)

    def encode_fast_new(self, cdf, input_sym, output_buffer, output_lengths):
        if self._native is not None:
            return self._native.encode_fast_new(
                cdf,
                input_sym,
                output_buffer,
                output_lengths,
            )
        return super().encode_fast_new(cdf, input_sym, output_buffer, output_lengths)

    def decode_fast_new(self, cdf, bytestreams, lengths, output):
        if self._native is not None:
            return self._native.decode_fast_new(cdf, bytestreams, lengths, output)
        return super().decode_fast_new(cdf, bytestreams, lengths, output)

    def decode_fast_prefsum(self, cdf, bytestreams, lengths_prefsum, output):
        if self._native is not None:
            return self._native.decode_fast_prefsum(
                cdf,
                bytestreams,
                lengths_prefsum,
                output,
            )
        return super().decode_fast_prefsum(cdf, bytestreams, lengths_prefsum, output)

    def load_and_reshape_flash(self, *args, **kwargs):
        if self._native is not None:
            return self._native.load_and_reshape_flash(*args, **kwargs)
        return super().load_and_reshape_flash(*args, **kwargs)

    def reshape_and_cache_back_flash(self, *args, **kwargs):
        if self._native is not None:
            return self._native.reshape_and_cache_back_flash(*args, **kwargs)
        return super().reshape_and_cache_back_flash(*args, **kwargs)

    @staticmethod
    def _flatten_paged_buffer_ptrs(paged_buffer) -> list[int]:
        import torch

        if isinstance(paged_buffer, torch.Tensor):
            return [paged_buffer.data_ptr()]
        if isinstance(paged_buffer, list):
            flat: list[torch.Tensor] = []
            for item in paged_buffer:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return [tensor.data_ptr() for tensor in flat]
        return [paged_buffer.data_ptr()]
