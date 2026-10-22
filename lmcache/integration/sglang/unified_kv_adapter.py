# SPDX-License-Identifier: Apache-2.0
"""SGLang KV pool adapter for the unified LMCache connector."""

# Future
from __future__ import annotations

# Standard
from typing import Any, Optional

# Third Party
from sglang.srt.configs.model_config import AttentionArch, is_deepseek_v4
from sglang.srt.mem_cache.unified_cache.components import ComponentType
import torch

# First Party
from lmcache.integration.sglang.lmcache_mp_metadata import (
    SGLangKVComponentGroup,
)


class SGLangUnifiedKVAdapter:
    """Map SGLang unified KV components to LMCache groups and block IDs."""

    def __init__(
        self,
        *,
        token_to_kv_pool_allocator: Any,
        req_to_token_pool: Any,
        tree_components: tuple[ComponentType, ...],
        mamba_component: Any,
        page_size: int,
        sliding_window_size: Optional[int],
        vllm_fused_kv: bool = False,
        vllm_mamba_block_size: int = 0,
    ) -> None:
        # check tree components
        if len(tree_components) <= 0:
            raise ValueError("tree_components must not be empty")
        if tree_components[0] is not ComponentType.FULL:
            raise ValueError("tree_components must start with FULL")
        if ComponentType.C128 in tree_components:
            raise NotImplementedError("C128 is not supported by LMCache")

        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_components = tree_components
        self.mamba_component = mamba_component
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size
        if vllm_mamba_block_size < 0:
            raise ValueError("vllm_mamba_block_size must be nonnegative")
        if vllm_mamba_block_size and (
            not vllm_fused_kv
            or tree_components != (ComponentType.FULL, ComponentType.MAMBA)
        ):
            raise ValueError("Experimental GDN packing requires fused FULL plus MAMBA")
        if (
            vllm_fused_kv
            and tree_components != (ComponentType.FULL,)
            and not vllm_mamba_block_size
        ):
            raise ValueError("Experimental vLLM fused KV requires FULL attention only")
        self.vllm_fused_kv = vllm_fused_kv
        self.vllm_mamba_block_size = vllm_mamba_block_size
        self._fused_kv_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._gdn_state_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def is_mla_enabled(self, model_config: Any) -> bool:
        """Return whether KV is TP-replicated for the MLA optimization."""
        return bool(
            model_config.attention_arch == AttentionArch.MLA
            or is_deepseek_v4(model_config.hf_config)
        )

    def reset_mamba_checkpoint_metadata(self, indices: torch.Tensor) -> None:
        """Reset ReplaySSM cursors that are not persisted by LMCache."""
        mamba_pool = self.req_to_token_pool.mamba_pool
        for name in (
            "replayssm_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
        ):
            value = getattr(mamba_pool, name, None)
            if value is not None:
                value[indices] = 0

    def resolve_registered_groups(self) -> list[SGLangKVComponentGroup]:
        """Map Unified tree components to LMCache engine KV groups."""
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        # Third Party
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
            return self._resolve_dsv4_registered_groups(kv_pool)

        groups: list[SGLangKVComponentGroup] = []
        for component_type in self.tree_components:
            if component_type is ComponentType.FULL:
                component_pool = getattr(kv_pool, "full_kv_pool", kv_pool)
                tensors = self._resolve_pool_tensors(component_pool)
                if self.vllm_fused_kv:
                    if getattr(component_pool, "kv_buffer", None) is not None:
                        raise ValueError(
                            "Experimental vLLM fused KV does not support MLA"
                        )
                    half = len(tensors) // 2
                    self._fused_kv_pairs = []
                    for key, value in zip(tensors[:half], tensors[half:], strict=True):
                        if (
                            key.ndim != 3
                            or key.shape != value.shape
                            or key.dtype != value.dtype
                            or key.dtype not in (torch.bfloat16, torch.float16)
                        ):
                            raise ValueError(
                                "Experimental vLLM fused KV requires matching "
                                "3D BF16/FP16 K and V tensors"
                            )
                        fused = torch.empty(
                            (*key.shape[:-1], 2 * key.shape[-1]),
                            dtype=key.dtype,
                            device=key.device,
                        )
                        self._fused_kv_pairs.append((key, value, fused))
                    tensors = tuple(fused for _, _, fused in self._fused_kv_pairs)
                tensor_rows_per_block = (self.page_size,) * len(tensors)
                dsa_tensors = self._resolve_dsa_indexer_tensors(component_pool)
                if dsa_tensors:
                    tensors = (*tensors, *dsa_tensors)
                    tensor_rows_per_block += (1,) * len(dsa_tensors)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=-1,
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=tensor_rows_per_block,
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.SWA:
                component_pool = getattr(kv_pool, "swa_kv_pool", None)
                if component_pool is None:
                    raise NotImplementedError(
                        f"SWA component requires an SWA KV sub-pool, got "
                        f"{type(kv_pool).__name__}"
                    )
                tensors = self._resolve_pool_tensors(component_pool)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=self.aligned_swa_window_size(),
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=(self.page_size,) * len(tensors),
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.MAMBA:
                mamba_pool = self.req_to_token_pool.mamba_pool
                checkpoint_grid = self.mamba_component.mamba_checkpoint_grid
                state_tensors = (
                    self._resolve_vllm_gdn_rows(mamba_pool)
                    if self.vllm_mamba_block_size
                    else self._resolve_mamba_pool_tensors(mamba_pool)
                )
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tuple(
                            tensor.view(tensor.shape[0], 1, -1)
                            for tensor in state_tensors
                        ),
                        sliding_window_size=checkpoint_grid,
                        tokens_per_block=checkpoint_grid,
                        slots_per_block=1,
                        recurrent_state=True,
                    )
                )
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        if self.vllm_mamba_block_size:
            # Qwen3.5's vLLM registration puts recurrent objects before FULL KV.
            groups.reverse()
        return groups

    def pack_for_store(self, indices: torch.Tensor) -> None:
        """Pack selected SGLang slots into per-head ``[K, V]`` transfer buffers.

        Args:
            indices: Token-slot indices in the FULL pool's address space.

        Notes:
            Only the experimental fused mode owns these additional buffers.
            Call on the producer stream before recording the STORE event.
        """
        for key, value, fused in self._fused_kv_pairs:
            slots = indices.to(device=key.device, dtype=torch.int64)
            packed = torch.cat(
                (key.index_select(0, slots), value.index_select(0, slots)), dim=-1
            )
            fused.index_copy_(0, slots, packed)

    def unpack_after_load(self, indices: torch.Tensor) -> None:
        """Restore selected fused transfer slots into SGLang's K/V buffers.

        Args:
            indices: Fresh FULL token slots populated by RETRIEVE.

        Notes:
            The caller must order this stream after RETRIEVE and wait for
            unpacking before publishing or releasing the destination slots.
        """
        for key, value, fused in self._fused_kv_pairs:
            slots = indices.to(device=key.device, dtype=torch.int64)
            packed = fused.index_select(0, slots)
            width = key.shape[-1]
            key.index_copy_(0, slots, packed[..., :width])
            value.index_copy_(0, slots, packed[..., width:])

    def pack_recurrent_for_store(self, indices: torch.Tensor) -> None:
        """Pack GDN conv then recurrent bytes into vLLM-compatible opaque rows.

        Args:
            indices: Checkpoint-slot vector; zero entries are absent history.

        Raises:
            ValueError: No real checkpoint is present.
        """
        slots = indices[indices != 0].unique().to(dtype=torch.int64)
        if not slots.numel():
            raise ValueError("GDN STORE requires a resident nonzero checkpoint slot")
        for conv, temporal, packed in self._gdn_state_rows:
            slots = slots.to(device=conv.device)
            conv_bytes = conv[0].numel() * conv.element_size()
            temporal_bytes = temporal[0].numel() * temporal.element_size()
            raw = packed.view(torch.uint8).flatten(1)
            raw[:, :conv_bytes].index_copy_(
                0, slots, conv.index_select(0, slots).view(torch.uint8).flatten(1)
            )
            raw[:, conv_bytes : conv_bytes + temporal_bytes].index_copy_(
                0, slots, temporal.index_select(0, slots).view(torch.uint8).flatten(1)
            )

    def unpack_recurrent_after_load(self, indices: torch.Tensor) -> None:
        """Restore opaque GDN rows into the selected SGLang checkpoint slots.

        Args:
            indices: Checkpoint-slot vector, ignoring absent zero history.

        Raises:
            ValueError: No destination checkpoint is present.
        """
        slots = indices[indices != 0].unique().to(dtype=torch.int64)
        if not slots.numel():
            raise ValueError("GDN RETRIEVE requires a nonzero checkpoint slot")
        for conv, temporal, packed in self._gdn_state_rows:
            slots = slots.to(device=conv.device)
            raw = packed.index_select(0, slots).view(torch.uint8).flatten(1)
            conv_bytes = conv[0].numel() * conv.element_size()
            temporal_bytes = temporal[0].numel() * temporal.element_size()
            conv.index_copy_(
                0,
                slots,
                raw[:, :conv_bytes]
                .contiguous()
                .view(conv.dtype)
                .reshape(-1, *conv.shape[1:]),
            )
            temporal.index_copy_(
                0,
                slots,
                raw[:, conv_bytes : conv_bytes + temporal_bytes]
                .contiguous()
                .view(temporal.dtype)
                .reshape(-1, *temporal.shape[1:]),
            )

    def aligned_swa_window_size(self) -> int:
        assert self.sliding_window_size is not None
        return (
            (self.sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )

    def device_indices_by_group(
        self,
        full_indices: torch.Tensor,
        *,
        mamba_value: Optional[torch.Tensor] = None,
        mamba_transfer_tokens: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """Translate tree FULL ids into each component's physical address space."""
        allocator = self.token_to_kv_pool_allocator
        result: list[torch.Tensor] = []
        for component_type in self.tree_components:
            if component_type is ComponentType.FULL:
                result.append(allocator.translate_kv_indices_for_transfer(full_indices))
            elif component_type is ComponentType.SWA:
                result.append(allocator.translate_loc_from_full_to_swa(full_indices))
            elif component_type is ComponentType.MAMBA:
                checkpoint_grid = self.mamba_component.mamba_checkpoint_grid
                logical_tokens = (
                    len(full_indices)
                    if mamba_transfer_tokens is None
                    else mamba_transfer_tokens
                )
                if logical_tokens % checkpoint_grid:
                    raise ValueError(
                        "LMCache Mamba transfer range must align to checkpoint "
                        f"grid {checkpoint_grid}, got {logical_tokens} tokens"
                    )
                checkpoint_slots = torch.zeros(
                    logical_tokens // checkpoint_grid,
                    dtype=torch.int64,
                    device=full_indices.device,
                )
                if mamba_value is not None and checkpoint_slots.numel():
                    physical = self.req_to_token_pool.translate_mamba_indices(
                        mamba_value.view(-1)
                    )
                    if physical.numel() != 1:
                        raise ValueError(
                            "LMCache Mamba transfer expects exactly one state slot"
                        )
                    checkpoint_slots[-1] = physical[0]
                result.append(checkpoint_slots)
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        if self.vllm_mamba_block_size:
            result.reverse()
        return result

    def _resolve_vllm_gdn_rows(self, mamba_pool: Any) -> tuple[torch.Tensor, ...]:
        """Reproduce Qwen3.5 vLLM's grouped, padded conv/recurrent byte rows."""
        state = mamba_pool.mamba_cache
        if len(state.conv) != 1:
            raise ValueError("Experimental GDN packing requires one QKV conv tensor")
        conv = state.conv[0]
        temporal = state.temporal
        layer_ids = mamba_pool.mamba_layer_ids
        if (
            conv.ndim != 4
            or temporal.ndim != 5
            or conv.shape[:2] != temporal.shape[:2]
            or conv.shape[0] != len(layer_ids)
            or conv.dtype != torch.bfloat16
            or temporal.dtype != torch.float32
            or not conv.is_contiguous()
            or not temporal.is_contiguous()
            or list(layer_ids) != sorted(set(layer_ids))
        ):
            raise ValueError(
                "Experimental GDN packing requires BF16 DS conv/FP32 state"
            )
        if conv.shape[2] != temporal.shape[2] * (
            2 * temporal.shape[4] + temporal.shape[3]
        ):
            raise ValueError(
                "Experimental GDN packing requires DS QKV conv state "
                "with equal key/value head counts"
            )
        block_size = self.vllm_mamba_block_size
        self._gdn_state_rows = []
        for index in range(len(layer_ids)):
            conv_layer, temporal_layer = conv[index], temporal[index]
            payload_bytes = (
                conv_layer[0].numel() * conv_layer.element_size()
                + temporal_layer[0].numel() * temporal_layer.element_size()
            )
            # Match _MambaUnifiedViewEdit's 16-byte aligned per-token width.
            token_bytes = (
                (payload_bytes + block_size * 16 - 1) // (block_size * 16) * 16
            )
            packed = torch.zeros(
                (conv.shape[1], 1, block_size * token_bytes),
                dtype=torch.int8,
                device=conv.device,
            )
            self._gdn_state_rows.append((conv_layer, temporal_layer, packed))
        return tuple(packed for _, _, packed in self._gdn_state_rows)

    def _resolve_pool_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        kv_buffer = getattr(kv_pool, "kv_buffer", None)
        if kv_buffer is not None:
            if not kv_buffer:
                raise NotImplementedError("LMCache cannot register an empty MLA pool")
            tensors = tuple(
                tensor
                for tensor in kv_buffer
                if tensor.numel() > 0 or not getattr(kv_pool, "use_dsa", False)
            )
            if not tensors:
                raise NotImplementedError("DSA pool has no locally owned KV buffers")
            return tensors

        k_buffer = getattr(kv_pool, "k_buffer", None)
        v_buffer = getattr(kv_pool, "v_buffer", None)
        if not k_buffer or not v_buffer or len(k_buffer) != len(v_buffer):
            raise NotImplementedError(
                f"Unsupported SGLang KV pool for LMCache MP: {type(kv_pool).__name__}"
            )
        if getattr(kv_pool, "kv_cache_layout", "nhd") != "nhd":
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache currently supports NHD MHA pools only"
            )
        return tuple([*k_buffer, *v_buffer])

    def _resolve_dsa_indexer_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Return page-native DSA indexers that share FULL block IDs."""
        if not getattr(kv_pool, "use_dsa", False):
            return ()
        buffers = getattr(kv_pool, "index_k_with_scale_buffer", None)
        if buffers is None:
            raise NotImplementedError(
                f"DSA pool {type(kv_pool).__name__} has no indexer buffers"
            )
        tensors = tuple(tensor for tensor in buffers if tensor.numel() > 0)
        if not tensors:
            raise NotImplementedError("DSA pool has no locally owned indexer buffers")
        if any(tensor.dim() != 2 for tensor in tensors):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise NotImplementedError(
                f"LMCache MP expects page-native 2-D DSA indexer buffers, got {shapes}"
            )
        return tensors

    def _validate_page_native_tensors(
        self, pool_name: str, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Validate tensors whose leading row is one complete SGLang page."""
        if not tensors:
            return tensors
        shapes = [tuple(tensor.shape) for tensor in tensors]
        if any(tensor.dim() != 2 for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP expects 2-D page-native {pool_name} buffers, got {shapes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP requires contiguous page-native {pool_name} buffers"
            )
        block_counts = {tensor.shape[0] for tensor in tensors}
        if len(block_counts) != 1:
            raise ValueError(
                f"DeepSeek V4 {pool_name} buffers expose different page counts: "
                f"{sorted(block_counts)}"
            )
        return tensors

    def _resolve_dsv4_full_page_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4-family sidecars that share FULL block IDs."""
        if getattr(kv_pool, "_unified_kv", False):
            raise NotImplementedError(
                "LMCache DeepSeek V4 does not yet support ROCm unified_kv_triton; "
                "its request-scoped SWA ring has no content-stable block-id space"
            )

        full_size = kv_pool.full_size
        if full_size is None or full_size % self.page_size != 0:
            raise ValueError(
                "DeepSeek V4 requires a page-aligned FULL pool size, got "
                f"full_size={full_size}, page_size={self.page_size}"
            )
        full_block_count = full_size // self.page_size + 1

        tensors: list[torch.Tensor] = []
        # Preserve V4's existing wire order, then append V4.1 source pools.
        for ratio in (4, 128, 1, 2):
            compressed_pool = kv_pool.kv_pools.get(ratio)
            if compressed_pool is not None:
                if ratio == 4 and hasattr(
                    compressed_pool, "full_to_hisparse_device_index_mapping"
                ):
                    raise NotImplementedError(
                        "LMCache DeepSeek V4 does not yet support HiSparse C4 remapping"
                    )
                for tensor in compressed_pool.kv_buffer:
                    if tensor.numel() > 0:
                        tensors.append(
                            self._fold_dsv4_full_page_rows(
                                f"C{ratio}",
                                tensor,
                                rows_per_full_block=1,
                                full_block_count=full_block_count,
                            )
                        )

            indexer_pool = kv_pool.index_pools.get(ratio)
            if indexer_pool is None:
                continue
            slots_per_full_block = self.page_size // ratio
            if slots_per_full_block % indexer_pool.page_size:
                raise ValueError(
                    f"DeepSeek V4 C{ratio} index pages of "
                    f"{indexer_pool.page_size} slots do not tile a FULL page "
                    f"of {slots_per_full_block} slots"
                )
            rows_per_full_block = slots_per_full_block // indexer_pool.page_size
            for tensor in indexer_pool.contiguous_page_row_buffers():
                if tensor.numel() > 0:
                    tensors.append(
                        self._fold_dsv4_full_page_rows(
                            f"C{ratio} indexer",
                            tensor,
                            rows_per_full_block=rows_per_full_block,
                            full_block_count=full_block_count,
                        )
                    )

        resolved = self._validate_page_native_tensors("FULL sidecar", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned compressed/indexer buffers"
            )
        return resolved

    def _fold_dsv4_full_page_rows(
        self,
        pool_name: str,
        tensor: torch.Tensor,
        *,
        rows_per_full_block: int,
        full_block_count: int,
    ) -> torch.Tensor:
        """Expose physical sidecar rows in the logical FULL-page address space."""
        if tensor.dim() != 2 or not tensor.is_contiguous():
            raise NotImplementedError(
                f"LMCache MP requires a contiguous 2-D {pool_name} buffer, got "
                f"shape={tuple(tensor.shape)}, contiguous={tensor.is_contiguous()}"
            )
        if rows_per_full_block <= 0:
            raise ValueError(
                f"DeepSeek V4 {pool_name} has invalid rows per FULL block: "
                f"{rows_per_full_block}"
            )
        required_rows = full_block_count * rows_per_full_block
        if tensor.shape[0] < required_rows:
            raise ValueError(
                f"DeepSeek V4 {pool_name} buffer is shorter than the logical FULL "
                f"block space: rows={tensor.shape[0]}, required={required_rows}"
            )
        return tensor[:required_rows].view(torch.uint8).reshape(full_block_count, -1)

    def _resolve_dsv4_swa_page_tensors(self, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4 SWA KV and C4 state with shared SWA block IDs."""
        swa_pool = getattr(kv_pool, "swa_kv_pool", None)
        if swa_pool is None:
            raise NotImplementedError(
                "LMCache DeepSeek V4 requires the standard page-addressed SWA pool"
            )

        swa_tensors = tuple(
            tensor
            for tensor in getattr(swa_pool, "kv_buffer", ())
            if tensor.numel() > 0
        )
        resolved_swa = self._validate_page_native_tensors("SWA", swa_tensors)
        if not resolved_swa:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA buffers"
            )
        swa_page_count = resolved_swa[0].shape[0]
        tensors: list[torch.Tensor] = list(resolved_swa)
        for state_pools in (
            getattr(kv_pool, "compress_state_pools", ()),
            getattr(kv_pool, "indexer_compress_state_pools", ()),
        ):
            for state_pool in state_pools:
                if state_pool is None or state_pool.ratio != 4:
                    continue
                state = state_pool.kv_score_buffer.kv_score
                if not state.is_contiguous():
                    raise NotImplementedError(
                        "LMCache DeepSeek V4 requires contiguous C4 state buffers"
                    )
                ring_size = int(state_pool.ring_size)
                if ring_size <= 0:
                    raise ValueError(
                        f"DeepSeek V4 C4 state has invalid ring size {ring_size}"
                    )
                usable_rows = state.shape[0] // ring_size * ring_size
                state_bytes = state.view(torch.uint8).reshape(state.shape[0], -1)
                state_pages = state_bytes[:usable_rows].reshape(
                    usable_rows // ring_size, -1
                )
                if state_pages.shape[0] < swa_page_count:
                    raise ValueError(
                        "DeepSeek V4 C4 state exposes fewer pages than its SWA "
                        f"pool: {state_pages.shape[0]} < {swa_page_count}"
                    )
                tensors.append(state_pages[:swa_page_count])

        resolved = self._validate_page_native_tensors("SWA/state", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA/state buffers"
            )
        return resolved

    def _resolve_dsv4_registered_groups(self, kv_pool) -> list[SGLangKVComponentGroup]:
        if tuple(self.tree_components) != (ComponentType.FULL, ComponentType.SWA):
            names = [component.name for component in self.tree_components]
            raise NotImplementedError(
                f"LMCache DeepSeek V4 expects FULL/SWA tree components, got {names}"
            )

        full_tensors = self._resolve_dsv4_full_page_tensors(kv_pool)
        swa_tensors = self._resolve_dsv4_swa_page_tensors(kv_pool)
        return [
            SGLangKVComponentGroup(
                name="full",
                kv_tensors=full_tensors,
                sliding_window_size=-1,
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(full_tensors),
            ),
            SGLangKVComponentGroup(
                name="swa",
                kv_tensors=swa_tensors,
                sliding_window_size=self.aligned_swa_window_size(),
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(swa_tensors),
            ),
        ]

    def _resolve_mamba_pool_tensors(self, mamba_pool) -> tuple[torch.Tensor, ...]:
        """Expose Mamba state as zero-copy LMCache tensors."""
        unified_buffer = getattr(mamba_pool, "_unified_buffer", None)
        sub_pool_name = getattr(mamba_pool, "_sub_pool_name", None)
        if unified_buffer is not None and sub_pool_name is not None:
            if unified_buffer.anchor_bytes(sub_pool_name) != 0:
                raise NotImplementedError(
                    "LMCache requires the unified Mamba pool to start at its "
                    "backing buffer base"
                )
            entry_bytes = unified_buffer.mamba_spec(sub_pool_name).entry_bytes()
            num_slots = mamba_pool._max_size + 1
            raw = unified_buffer._raw[: num_slots * entry_bytes]
            return (raw.view(num_slots, 1, entry_bytes),)

        tensors: list[torch.Tensor] = []
        for (
            field,
            layer_tensor,
            _slice_axis,
            layer_id,
        ) in mamba_pool._iter_transfer_state_entries():
            # SGLang always exposes these tensors as [slot, ...]. slice_axis
            # describes TP slicing within one slot and does not affect LMCache's
            # same-TP full-slot store/retrieve layout.
            if not layer_tensor.is_contiguous():
                raise NotImplementedError(
                    f"LMCache MP requires contiguous {field} state for "
                    f"Mamba layer {layer_id}"
                )
            tensors.append(layer_tensor)
        return tuple(tensors)
