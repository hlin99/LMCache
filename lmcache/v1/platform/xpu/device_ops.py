# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: explicit native adapters over the torch baseline."""

from __future__ import annotations

from typing import ClassVar

from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def __init__(self) -> None:
        self._native: object | None = None
        try:
            import lmcache.xpu_ops as sycl

            self._native = sycl
        except ImportError:
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
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
        """Transfer KV blocks using native XPU SYCL extension."""
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

    def lmcache_memcpy_async(self, *args, **kwargs):
        if self._native is not None:
            return self._native.lmcache_memcpy_async(*args, **kwargs)
        return super().lmcache_memcpy_async(*args, **kwargs)

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
