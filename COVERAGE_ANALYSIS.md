# Test Coverage Analysis for test_non_cuda_equivalents.py

## Summary

This document analyzes the test coverage for `tests/v1/test_non_cuda_equivalents.py` against the C++ functions exposed in `csrc/pybind.cpp`.

## Analysis Date
2026-03-04

## Coverage Analysis

### Functions Exposed in pybind.cpp

The following functions and enums are exposed to Python via pybind11:

1. **Enums:**
   - `TransferDirection` (H2D, D2H) ✅ Covered
   - `GPUKVFormat` (6 formats) - See detailed analysis below

2. **Functions:**
   - `multi_layer_kv_transfer` ✅ Covered
   - `multi_layer_kv_transfer_unilateral` ✅ Covered
   - `single_layer_kv_transfer` ✅ Covered
   - `single_layer_kv_transfer_sgl` ✅ Covered
   - `load_and_reshape_flash` ✅ Covered
   - `reshape_and_cache_back_flash` ✅ Covered
   - `lmcache_memcpy_async` ✅ Covered
   - `encode_fast_new` ✅ Covered
   - `decode_fast_new` ✅ Covered
   - `decode_fast_prefsum` ✅ Covered
   - `calculate_cdf` ✅ Covered
   - `rotary_embedding_k_fused` ✅ Covered
   - `alloc_pinned_ptr` / `free_pinned_ptr` ✅ Covered
   - `alloc_pinned_numa_ptr` / `free_pinned_numa_ptr` ✅ Covered
   - `alloc_numa_ptr` / `free_numa_ptr` ✅ Covered
   - `alloc_shm_pinned_ptr` / `free_shm_pinned_ptr` ✅ Covered
   - `get_gpu_pci_bus_id` ✅ Covered

### GPUKVFormat Enum Coverage

The `GPUKVFormat` enum defines 6 formats:

1. **NB_NL_TWO_BS_NH_HS** (vLLM cross-layer format)
   - Status: ❌ **NOT COVERED** (before this PR)
   - Status: ✅ **NOW COVERED** (after this PR)
   - Used in: `multi_layer_kv_transfer`
   - Layout: `[2, page_buffer_size, hidden_size]`

2. **NL_X_TWO_NB_BS_NH_HS** (vLLM flash attention)
   - Status: ✅ Already covered
   - Used in: `multi_layer_kv_transfer`, `single_layer_kv_transfer`
   - Layout: `[2, num_blocks, block_size, num_heads, head_size]`

3. **NL_X_NB_TWO_BS_NH_HS** (vLLM flash infer)
   - Status: ✅ Already covered
   - Used in: `multi_layer_kv_transfer`, `single_layer_kv_transfer`
   - Layout: `[num_blocks, 2, block_size, num_heads, head_size]`

4. **NL_X_NB_BS_HS** (vLLM MLA)
   - Status: ✅ Already covered
   - Used in: `multi_layer_kv_transfer`, `single_layer_kv_transfer`
   - Layout: `[num_blocks, block_size, head_size]`

5. **TWO_X_NL_X_NBBS_NH_HS** (SGLang MHA)
   - Status: ⚠️ **Not applicable for multi_layer_kv_transfer**
   - Note: This format is NOT handled in `multi_layer_kv_transfer` CUDA kernels
   - It's designed for use with SGLang's separate K/V cache structure
   - Uses different code path (not through multi_layer_kv_transfer)
   - Layout: `[[k_list], [v_list]]` where each is `[page_buffer_size, num_heads, head_size]`

6. **NL_X_NBBS_ONE_HS** (SGLang MLA)
   - Status: ✅ Already covered
   - Used in: `multi_layer_kv_transfer`, `multi_layer_kv_transfer_unilateral`
   - Layout: `[page_buffer_size, head_size]`

## Changes Made

### Added Coverage for NB_NL_TWO_BS_NH_HS

**File:** `tests/v1/test_non_cuda_equivalents.py`

**Changes:**
1. Added `NB_NL_TWO_BS_NH_HS` to the `format_cases` list in `scenario_multi_layer_kv_transfer()` (line 1100)
2. Updated comments to clarify that the existing code path handles both `NB_NL_TWO_BS_NH_HS` and `NL_X_TWO_NB_BS_NH_HS` formats (lines 1147, 1170, 1213)

**Rationale:**
- Both `NB_NL_TWO_BS_NH_HS` and `NL_X_TWO_NB_BS_NH_HS` use the same paged buffer layout: `[2, page_buffer_size, hidden_size]`
- The CUDA kernel implementation handles both formats identically (see `csrc/mem_kernels.cu:200-207`)
- The Python fallback implementation also handles both formats with the same code path
- The test now ensures both formats are verified

## Why TWO_X_NL_X_NBBS_NH_HS is Not Covered in multi_layer_kv_transfer

The `TWO_X_NL_X_NBBS_NH_HS` format is intentionally not tested with `multi_layer_kv_transfer` because:

1. **Not implemented in CUDA kernel:** The format is not present in the switch statement in `multi_layer_kv_transfer_templated` (csrc/mem_kernels.cu:478-508)

2. **Different architecture:** SGLang MHA uses a separate K/V cache structure that requires special handling

3. **Uses different code path:** This format is handled through SGLang-specific connector code in `lmcache/v1/gpu_connector/gpu_connectors.py`

4. **Would fail if tested:** Attempting to use this format with `multi_layer_kv_transfer` would throw a "Unsupported GPUKVFormat" runtime error

## Test Execution

The test suite executes scenarios in multiple modes:
- **CUDA_OPS with GPU visible** - Tests CUDA implementation
- **NON_CUDA with GPU visible** - Tests Python fallback with GPU available
- **NON_CUDA without GPU visible** - Tests Python fallback CPU-only mode

Each scenario saves results and compares them across backends to ensure equivalence.

## Conclusion

After this PR:
- ✅ All applicable GPUKVFormat enums are now covered in tests
- ✅ All functions exposed via pybind11 are covered
- ✅ The `NB_NL_TWO_BS_NH_HS` format is now tested
- ⚠️ `TWO_X_NL_X_NBBS_NH_HS` is intentionally not covered for `multi_layer_kv_transfer` as it uses a different code path

**Test coverage is now complete for all applicable operations.**
