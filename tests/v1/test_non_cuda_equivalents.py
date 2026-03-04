# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import ctypes
import os
import shutil
import subprocess
import sys

# Third Party
import pytest
import torch

RESULTS_DIR = Path("test_results")

_is_child = "LMC_TEST_MODE" in os.environ

# Skip entire module if no CUDA hardware (only check in top-level process)
if not _is_child and not torch.cuda.is_available():
    pytest.skip(
        "CUDA is not available, skipping entire test module", allow_module_level=True
    )


# ==========================================
# 1. Core Logic
# ==========================================


def get_test_context():
    mode = os.getenv("LMC_TEST_MODE", "NON_CUDA")
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")

    cuda_status = "cuda_ready" if cuda_visible != "" else "no_cuda"
    backend = "cuda_ops" if mode == "CUDA_OPS" else "non_cuda"

    if backend == "cuda_ops":
        print(f">>> Importing lmcache.c_ops as ops (Mode: {mode})")
        # First Party
        import lmcache.c_ops as ops
    else:
        print(f">>> Importing lmcache.non_cuda_equivalents as ops (Mode: {mode})")
        # First Party
        import lmcache.non_cuda_equivalents as ops

    return ops, f"{backend}_{cuda_status}"


def save_result(func_name, data):
    _, scene = get_test_context()
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(data, RESULTS_DIR / f"{func_name}@{scene}.pt")


# ==========================================
# 2. Scenario functions
# ==========================================


def scenario_get_gpu_pci_bus_id():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    res = ops.get_gpu_pci_bus_id(0)

    if is_cuda_backend and torch.cuda.is_available():
        assert res is not None, "get_gpu_pci_bus_id returned None"
        assert isinstance(res, str) and len(res) > 0

    # Save 1 = PASS (call succeeded without crash)
    save_result(
        "get_gpu_pci_bus_id",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_calculate_cdf():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    num_bins_list = [1, 2, 5, 11, 15, 31, 32, 63]

    for num_bins in num_bins_list:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        input_tensor = torch.randint(0, num_bins, (1, 1000, 1), dtype=torch.int8)

        if is_cuda_backend:
            target_dev = f"cuda:{torch.cuda.current_device()}"
            input_tensor = input_tensor.to(target_dev)

        raw_output = ops.calculate_cdf(input_tensor, num_bins)
        out_cpu = raw_output.flatten().cpu()

        if is_cuda_backend:
            out_int32 = out_cpu.to(torch.int32)
            out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
            final_result = out_uint16.float() / 65536.0
        else:
            final_result = out_cpu.float()

        save_result(f"calculate_cdf_bins{num_bins}", final_result)


def scenario_rotary_embedding_k_fused():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Setup Dimensions
    num_tokens = 128
    num_kv_heads = 32
    head_size = 128
    max_position = 2048
    rotary_dim = head_size

    # 2. Generate Inputs
    old_positions = torch.randint(0, 1000, (num_tokens,), dtype=torch.long)
    new_positions = old_positions + 1

    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32)
    cos_sin_cache = torch.randn(max_position, rotary_dim, dtype=torch.float32)
    is_neox = True

    if is_cuda_backend:
        target_dev = f"cuda:{torch.cuda.current_device()}"
        old_positions = old_positions.to(target_dev)
        new_positions = new_positions.to(target_dev)
        key = key.to(target_dev)
        cos_sin_cache = cos_sin_cache.to(target_dev)

    # 3. Execute (in-place update on key)
    ops.rotary_embedding_k_fused(
        old_positions,
        new_positions,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )

    # 4. Save
    save_result("rotary_embedding_k_fused", key.cpu())


def scenario_lmcache_memcpy_async():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Setup dimensions and mock data (4KB)
    nbytes = 1024 * 4
    src_host = torch.randint(1, 255, (nbytes,), dtype=torch.uint8)
    gpu_buffer = torch.zeros(nbytes, dtype=torch.uint8)

    if torch.cuda.is_available():
        dst_host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    else:
        dst_host = torch.zeros(nbytes, dtype=torch.uint8)

    # 2. Assign directions and device locations
    if is_cuda_backend:
        gpu_buffer = gpu_buffer.to(f"cuda:{torch.cuda.current_device()}")

    h2d_dir = ops.TransferDirection.H2D
    d2h_dir = ops.TransferDirection.D2H

    # --- PART A: H2D (Host to Device) ---
    ops.lmcache_memcpy_async(
        gpu_buffer.data_ptr(),
        src_host.data_ptr(),
        nbytes,
        h2d_dir,
        0,
        16,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # --- PART B: D2H (Device to Host) ---
    ops.lmcache_memcpy_async(
        dst_host.data_ptr(),
        gpu_buffer.data_ptr(),
        nbytes,
        d2h_dir,
        0,
        16,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 3. Internal sanity check
    final_result = dst_host.cpu()
    assert torch.equal(final_result, src_host), (
        f"Data corrupted during H2D→D2H loop in {scene_info}, "
        f"max diff = {(final_result.float() - src_host.float()).abs().max().item()}"
    )

    # 4. Save
    save_result("lmcache_memcpy_async", final_result)


def scenario_load_and_reshape_flash():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Standard Params
    src_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"
    dst_device = "cpu"

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Setup Data (Deterministic Pattern)
    total_elements = num_blocks * block_size * num_heads * head_size

    kv_cache_cpu = []
    for i in range(num_layers):
        base_tensor = torch.linspace(i, i + 1, total_elements, dtype=torch.float32)
        base_tensor = base_tensor.reshape(
            num_blocks, block_size, num_heads, head_size
        ).to(dtype)
        k = base_tensor
        v = base_tensor + 0.5
        kv_cache_cpu.append([k, v])

    kv_cache = [
        [layer[0].to(src_device), layer[1].to(src_device)] for layer in kv_cache_cpu
    ]

    # Slot mapping: deterministic strided selection
    step = (num_blocks * block_size) // num_tokens
    slot_indices = list(range(0, num_blocks * block_size, step))[:num_tokens]
    slot_mapping = torch.tensor(slot_indices, device=src_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 3. Extract (to CPU pinned)
    extracted_chunks = []
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = (2, num_layers, len(slot_mapping_temp), num_heads * head_size)
        mem_obj_tensor = torch.zeros(mem_obj_shape, dtype=dtype, device=dst_device)

        if is_cuda_backend:
            mem_obj_tensor = mem_obj_tensor.pin_memory()

        for layer_id in range(num_layers):
            ops.load_and_reshape_flash(
                mem_obj_tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        extracted_chunks.append(mem_obj_tensor)

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Verify: compare extracted data against original kv_cache
    #    mem_obj_tensor layout:
    #       [2, num_layers, num_tokens_in_chunk, num_heads * head_size]
    #    dim 0: K=0, V=1
    #    Original kv_cache layout: [num_blocks, block_size, num_heads, head_size]
    #    slot_mapping tells us which (block, offset) each token comes from
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        slots = slot_mapping_temp.cpu()
        extracted = extracted_chunks[chunk_id].cpu()

        for layer_id in range(num_layers):
            orig_k = kv_cache_cpu[layer_id][
                0
            ]  # [num_blocks, block_size, num_heads, head_size]
            orig_v = kv_cache_cpu[layer_id][1]

            for tok_idx, slot in enumerate(slots):
                block_idx = slot.item() // block_size
                offset = slot.item() % block_size

                # Expected: flattened [num_heads * head_size]
                expected_k = orig_k[block_idx, offset].reshape(-1)
                expected_v = orig_v[block_idx, offset].reshape(-1)

                # Extracted
                got_k = extracted[0, layer_id, tok_idx]
                got_v = extracted[1, layer_id, tok_idx]

                k_diff = (got_k.float() - expected_k.float()).abs().max().item()
                assert torch.equal(got_k, expected_k), (
                    f"K mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={k_diff}"
                )

                v_diff = (got_v.float() - expected_v.float()).abs().max().item()
                assert torch.equal(got_v, expected_v), (
                    f"V mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={v_diff}"
                )

    # 5. Save extracted data for cross-scenario comparison
    save_result("load_and_reshape_flash", extracted_chunks[0].cpu())


def scenario_reshape_and_cache_back_flash():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Environment Setup
    src_device = "cpu"
    dst_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Prepare Source Data (CPU Buffer)
    # Shape: [2, num_layers, num_tokens, num_heads * head_size]
    mem_obj_shape = (2, num_layers, num_tokens, num_heads * head_size)
    src_buffer = torch.zeros(mem_obj_shape, dtype=dtype, device=src_device)

    # Data Pattern: Odd numbers (1.0, 3.0, 5.0, ...)
    for i in range(num_tokens):
        val = 1.0 + (i * 2.0)
        src_buffer[0, :, i, :] = val  # Key
        src_buffer[1, :, i, :] = val + 0.5  # Value

    if is_cuda_backend:
        src_buffer = src_buffer.pin_memory()

    # 3. Prepare Destination (Empty Cache)
    kv_cache = [
        [
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
        ]
        for _ in range(num_layers)
    ]

    # 4. Slot Mapping (Continuous: Token 0 → Slot 0, Token 1 → Slot 1, ...)
    slot_indices = list(range(num_tokens))
    slot_mapping = torch.tensor(slot_indices, device=dst_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 5. Execute Operator (Load Back)
    current_token_offset = 0
    for chunk_id, slot_chunk in enumerate(slot_mapping_chunked):
        chunk_len = len(slot_chunk)

        buffer_chunk = src_buffer[
            :, :, current_token_offset : current_token_offset + chunk_len, :
        ]
        if not buffer_chunk.is_contiguous():
            buffer_chunk = buffer_chunk.contiguous()

        for layer_id in range(num_layers):
            ops.reshape_and_cache_back_flash(
                buffer_chunk,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_chunk,
                layer_id,
            )
        current_token_offset += chunk_len

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 6. Verify: check written values against source pattern
    for layer_id in range(num_layers):
        k_cache = kv_cache[layer_id][
            0
        ].cpu()  # [num_blocks, block_size, num_heads, head_size]
        v_cache = kv_cache[layer_id][1].cpu()

        for tok_idx, slot in enumerate(slot_indices):
            block_idx = slot // block_size
            offset = slot % block_size

            expected_k_val = 1.0 + (tok_idx * 2.0)
            expected_v_val = expected_k_val + 0.5

            got_k = k_cache[block_idx, offset]
            got_v = v_cache[block_idx, offset]

            expected_k = torch.full_like(got_k, expected_k_val)
            expected_v = torch.full_like(got_v, expected_v_val)

            assert torch.allclose(got_k.float(), expected_k.float(), atol=0.1), (
                f"K mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_k_val}, got={got_k[0, 0].item()}"
            )
            assert torch.allclose(got_v.float(), expected_v.float(), atol=0.1), (
                f"V mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_v_val}, got={got_v[0, 0].item()}"
            )

    # 7. Save first block of layer 0 key cache for cross-scenario comparison
    save_result("reshape_and_cache_back_flash", kv_cache[0][0][0].cpu())


def scenario_encode_fast_new():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Hyperparameters
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    src_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Construct Data
    # A. CDF: uniform distribution, strictly increasing
    step = 100 // alphabet_size
    base_cdf = torch.arange(0, 100, step, dtype=torch.int32)
    base_cdf = base_cdf[:alphabet_size]

    cdf_cpu = (
        base_cdf.unsqueeze(0).unsqueeze(0).expand(nlayers, nchannels, -1).contiguous()
    )
    cdf = cdf_cpu.to(dtype=torch.int16, device=src_device)

    # B. Input symbols: cycling 0..14
    total_syms = nlayers * ntokens * nchannels
    input_cpu = torch.arange(total_syms, dtype=torch.float32)
    input_cpu = (input_cpu % (alphabet_size - 1)).to(torch.int8)
    input_cpu = input_cpu.reshape(nlayers, ntokens, nchannels)
    input_sym = input_cpu.to(device=src_device)

    # 3. Prepare Outputs
    output_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=src_device,
    )
    output_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=src_device,
    )

    # 4. Execute
    ops.encode_fast_new(
        cdf,
        input_sym,
        output_buffer,
        output_lengths,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 5. Verify
    lengths_cpu = output_lengths.cpu()

    assert (lengths_cpu > 0).all(), "Encoding produced zero-length output!"
    assert (lengths_cpu <= max_buf_len).all(), "Buffer overflow detected!"

    # 6. Save: first 20 bytes of layer 0, channel 0
    valid_len = lengths_cpu[0, 0].item()
    res = output_buffer[0, 0, : min(valid_len, 20)].cpu()
    save_result("encode_fast_new", res)


def scenario_decode_fast_new():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Config
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Data Generation
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).to(device).to(torch.int16)

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode first (need encoded data to test decode)
    encoded_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    encoded_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )

    ops.encode_fast_new(
        cdf,
        input_sym,
        encoded_buffer,
        encoded_lengths,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Decode
    decoded_sym = torch.zeros_like(input_sym, dtype=torch.uint8)

    ops.decode_fast_new(
        cdf,
        encoded_buffer,
        encoded_lengths,
        decoded_sym,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 5. Verify: decoded must match original
    input_uint8 = input_sym.to(torch.uint8)
    mismatch = (input_uint8 != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_uint8 != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Decode mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_uint8[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 6. Save decoded slice for cross-scenario comparison
    save_result("decode_fast_new", decoded_sym[0, :20, 0].cpu())


def scenario_decode_fast_prefsum():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Configuration
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Data Generation (Normalized CDF)
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).float()
    cdf = (cdf / cdf[..., -1:] * 65536).to(torch.int32)
    cdf[..., -1] = 65536
    cdf = cdf.to(device).to(torch.int16).contiguous()

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode to get variable lengths
    tmp_buf = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    tmp_len = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )
    ops.encode_fast_new(cdf, input_sym, tmp_buf, tmp_len)
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Pack into 1D dense bytestream
    lens_flat = tmp_len.cpu().flatten().tolist()
    bufs_flat = tmp_buf.cpu().reshape(-1, max_buf_len).numpy()
    all_bytes = []
    for i, length in enumerate(lens_flat):
        all_bytes.extend(bufs_flat[i, :length].tolist())

    bytestream_1d = torch.tensor(
        all_bytes,
        dtype=torch.uint8,
        device=device,
    ).contiguous()

    # 5. Offsets (end-position via cumsum)
    lengths_prefsum = (
        tmp_len.flatten().cumsum(0).reshape(tmp_len.shape).to(torch.int64).to(device)
    ).contiguous()

    # 6. Decode
    decoded_sym = (
        torch.zeros_like(
            input_sym,
            dtype=torch.uint8,
        )
        .to(device)
        .contiguous()
    )

    ops.decode_fast_prefsum(
        cdf,
        bytestream_1d,
        lengths_prefsum,
        decoded_sym,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 7. Verify roundtrip
    input_ref = input_sym.to(torch.uint8)
    mismatch = (input_ref != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_ref != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Prefsum mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_ref[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 8. Save
    save_result(
        "decode_fast_prefsum",
        decoded_sym[0, :20, 0].cpu(),
    )


def scenario_single_layer_kv_transfer():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_tokens = 64
    num_blocks = 256
    block_size = 16
    num_heads = 12
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 2,
        2,
        device=device,
    ).to(torch.int64)

    # (gpu_kv_format, is_mla, token_major, direction)
    # direction: False = LMC→vLLM (H2D), True = vLLM→LMC (D2H)
    test_cases = [
        # flash attn: [2, NB, BS, NH, HS] — two_major
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, False),
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, False, False),
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, True),
        # flash infer: [NB, 2, BS, NH, HS]
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, False),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, False, False),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, True),
        # vLLM MLA: [NB, BS, HS]
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, True, False),
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, True, True),
    ]

    for gpu_kv_format, is_mla, token_major, direction in test_cases:
        dir_tag = "v2l" if direction else "l2v"
        is_two_major = gpu_kv_format == ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        case_desc = (
            f"fmt={gpu_kv_format}, MLA={is_mla}, TM={token_major}, Dir={dir_tag}"
        )

        # ── 1. Setup Shapes ──
        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
        else:
            lmc_shape = (
                (num_tokens, 2, hidden_size)
                if token_major
                else (2, num_tokens, hidden_size)
            )
            if is_two_major:
                # flash attn: [2, num_blocks, block_size, num_heads, head_size]
                vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            else:
                # flash infer: [num_blocks, 2, block_size, num_heads, head_size]
                vllm_shape = (num_blocks, 2, block_size, num_heads, head_size)

        # ── 2. Deterministic Data ──
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        # ── 3. Golden Reference ──
        lmc_ref = lmc_tensor.clone()
        vllm_ref = vllm_tensor.clone()
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → vLLM
            if is_mla:
                vllm_ref[block_indices, block_offsets, :] = lmc_ref
            else:
                src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
                src = src.view(num_tokens, 2, num_heads, head_size)
                if is_two_major:
                    # [2, NB, BS, NH, HS]
                    vllm_ref[0, block_indices, block_offsets] = src[:, 0, :, :]
                    vllm_ref[1, block_indices, block_offsets] = src[:, 1, :, :]
                else:
                    # [NB, 2, BS, NH, HS]
                    vllm_ref[block_indices, 0, block_offsets] = src[:, 0, :, :]
                    vllm_ref[block_indices, 1, block_offsets] = src[:, 1, :, :]
        else:  # vLLM → LMC
            if is_mla:
                lmc_ref = vllm_ref[block_indices, block_offsets, :]
            else:
                if is_two_major:
                    k = vllm_ref[0, block_indices, block_offsets]
                    v = vllm_ref[1, block_indices, block_offsets]
                else:
                    k = vllm_ref[block_indices, 0, block_offsets]
                    v = vllm_ref[block_indices, 1, block_offsets]
                combined = torch.stack(
                    [k, v],
                    dim=1,
                ).view(num_tokens, 2, hidden_size)
                lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # ── 4. Execute ──
        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            gpu_kv_format,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # ── 5. Verify ──
        if not direction:
            torch.testing.assert_close(
                vllm_tensor,
                vllm_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

    # ── 6. Save canonical results for cross-backend comparison ──
    # Use flash attn (two_major) format to match original file names
    canonical_cases = [
        (False, True, False),  # l2v, non-MLA
        (False, True, True),  # v2l, non-MLA
        (True, True, False),  # l2v, MLA
        (True, True, True),  # v2l, MLA
    ]

    for is_mla, token_major, direction in canonical_cases:
        dir_tag = "v2l" if direction else "l2v"

        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
            fmt = ops.GPUKVFormat.NL_X_NB_BS_HS
        else:
            lmc_shape = (num_tokens, 2, hidden_size)
            vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            fmt = ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            fmt,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        result = lmc_tensor.cpu() if direction else vllm_tensor.cpu()
        save_result(
            f"single_layer_kv_transfer_{dir_tag}_mla_{is_mla}",
            result,
        )


def scenario_single_layer_kv_transfer_sgl():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_tokens = 32
    num_blocks = 128
    block_size = 16
    num_heads = 8
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 3,
        3,
        device=device,
    ).to(torch.int64)

    # (token_major, direction)
    # direction: False = LMC→SGL, True = SGL→LMC
    test_cases = [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]

    for token_major, direction in test_cases:
        dir_tag = "s2l" if direction else "l2s"

        # 1. Setup Shapes
        lmc_shape = (
            (num_tokens, 2, hidden_size)
            if token_major
            else (2, num_tokens, hidden_size)
        )
        sgl_shape = (
            num_blocks,
            block_size,
            num_heads,
            head_size,
        )

        # 2. Deterministic Data
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        sgl_size = 1
        for s in sgl_shape:
            sgl_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 500)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        sgl_k_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 500)
            .to(torch.float16)
            .reshape(sgl_shape)
        )
        sgl_v_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 1000)
            .to(torch.float16)
            .reshape(sgl_shape)
        )

        # 3. Golden Reference
        lmc_ref = lmc_tensor.clone()
        sgl_k_ref = sgl_k_tensor.clone()
        sgl_v_ref = sgl_v_tensor.clone()

        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → SGL
            src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
            src_k = src[:, 0, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            src_v = src[:, 1, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            sgl_k_ref[block_indices, block_offsets] = src_k
            sgl_v_ref[block_indices, block_offsets] = src_v
        else:  # SGL → LMC
            k_data = sgl_k_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )
            v_data = sgl_v_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )

            combined = torch.stack(
                [k_data, v_data],
                dim=1,
            )  # [N, 2, H]
            lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # 4. Execute
        ops.single_layer_kv_transfer_sgl(
            lmc_tensor,
            sgl_k_tensor,
            sgl_v_tensor,
            slot_mapping,
            ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # 5. Verify
        case_desc = f"TM={token_major}, Dir={dir_tag}"
        if not direction:
            torch.testing.assert_close(
                sgl_k_tensor,
                sgl_k_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"K mismatch in {case_desc}",
            )
            torch.testing.assert_close(
                sgl_v_tensor,
                sgl_v_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"V mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

        # 6. Save each case separately
        result = lmc_tensor.cpu() if direction else sgl_k_tensor.cpu()
        save_result(
            f"single_layer_kv_transfer_sgl_{dir_tag}_tm_{token_major}",
            result,
        )


def scenario_multi_layer_kv_transfer():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    block_size = 5
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [0, 2, 5, 9],
        dtype=torch.int64,
        device=device,
    )

    # ── Format-specific test cases ──
    # Each: (gpu_kv_format, is_mla, block_size_arg)
    format_cases = [
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, 1),  # flash attn
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, block_size),  # flash infer
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, 1),  # vLLM MLA
        (ops.GPUKVFormat.NL_X_NBBS_ONE_HS, True, 1),  # SGLang MLA
    ]

    for gpu_kv_format, is_mla, bs_arg in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "paged2lmc" if direction else "lmc2paged"
            fmt_name = str(gpu_kv_format).split(".")[-1]

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            key_value[kv, ly, t] = val

            # ── 2. Paged Buffers (one per layer) ──
            page_buffers = []
            for ly in range(num_layers):
                if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                    num_blocks = page_buffer_size // bs_arg
                    pb = torch.zeros(
                        (num_blocks, 2, bs_arg, head_size),
                        dtype=dtype,
                        device=device,
                    )
                elif is_mla:
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                else:
                    pb = torch.zeros(
                        (2, page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )

                if direction:  # Paged → LMC
                    for s in range(page_buffer_size):
                        for kv in range(k_or_v_size):
                            val = (
                                kv * 7000
                                + ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                                blk_idx = s // bs_arg
                                blk_off = s % bs_arg
                                pb[blk_idx, kv, blk_off] = val
                            elif is_mla:
                                pb[s] = val
                            else:
                                pb[kv, s] = val

                page_buffers.append(pb)

            # ── 3. Pointer Tensor ──
            key_value_ptrs = torch.tensor(
                [pb.data_ptr() for pb in page_buffers],
                dtype=torch.int64,
                device=device,
            )

            # ── 4. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer(
                key_value,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                gpu_kv_format,
                bs_arg,
            )
            if is_cuda_backend:
                torch.cuda.synchronize()

            # ── 5. Verify (internal, per-format) ──
            for t_id in range(num_tokens):
                s_idx = slot_mapping[t_id].item()
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = key_value[kv, ly, t_id]

                        if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                            blk_idx = s_idx // bs_arg
                            blk_off = s_idx % bs_arg
                            paged_val = page_buffers[ly][blk_idx, kv, blk_off]
                        elif is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            paged_val = page_buffers[ly][kv, s_idx]

                        torch.testing.assert_close(
                            lmc_val,
                            paged_val,
                            msg=(
                                f"Mismatch: {fmt_name} {dir_tag}, "
                                f"kv={kv}, layer={ly}, token={t_id}"
                            ),
                        )

    # ── 6. Save ONE canonical result for cross-backend comparison ──
    # Use flash attn format (NL_X_TWO_NB_BS_NH_HS), direction=True (paged→lmc)
    # Re-run the canonical case to get a clean result
    canonical_format = ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    for direction in [True, False]:
        dir_tag = "paged2lmc" if direction else "lmc2paged"
        lmc_shape = (2, num_layers, num_tokens, head_size)
        key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for ly in range(num_layers):
                for t in range(num_tokens):
                    val = (
                        ly * 1000 + t * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    key_value[0, ly, t] = val
                    key_value[1, ly, t] = val + 500

        page_buffers = []
        for ly in range(num_layers):
            pb = torch.zeros(
                (2, page_buffer_size, head_size),
                dtype=dtype,
                device=device,
            )
            if direction:
                for s in range(page_buffer_size):
                    val = (
                        ly * 2000 + s * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    pb[0, s] = val
                    pb[1, s] = val + 700
            page_buffers.append(pb)

        key_value_ptrs = torch.tensor(
            [pb.data_ptr() for pb in page_buffers],
            dtype=torch.int64,
            device=device,
        )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            canonical_format,
            1,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        save_result(
            f"multi_layer_kv_transfer_{dir_tag}",
            key_value.cpu(),
        )


def scenario_multi_layer_kv_transfer_unilateral():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [1, 3, 4, 7],
        dtype=torch.int64,
        device=device,
    )

    # ── Test cases: (gpu_kv_format, is_mla) ──
    format_cases = [
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False),  # SGLang MHA (unilateral path)
        (
            ops.GPUKVFormat.NL_X_NB_BS_HS,
            True,
        ),  # vLLM MLA (delegates to multi_layer_kv_transfer)
        (
            ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
            True,
        ),  # SGLang MLA (delegates to multi_layer_kv_transfer)
    ]

    for gpu_kv_format, is_mla in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "p2l" if direction else "l2p"

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            lmc_tensor[kv, ly, t] = val

            # ── 2. Paged Buffers ──
            if is_mla:
                # MLA delegates to multi_layer_kv_transfer
                # ptrs: [layer0, layer1, ...], each -> [page_buffer_size, head_size]
                page_buffers = []
                for ly in range(num_layers):
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                    if direction:  # Paged → LMC
                        for s in range(page_buffer_size):
                            val = (
                                ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            pb[s] = val
                    page_buffers.append(pb)

                key_value_ptrs = torch.tensor(
                    [pb.data_ptr() for pb in page_buffers],
                    dtype=torch.int64,
                    device=device,
                )
            else:
                # Non-MLA unilateral: separate K/V buffers
                # ptrs: [K_l0, K_l1, ..., V_l0, V_l1, ...]
                # each -> [page_buffer_size, head_size]
                buffers = {}
                for kv in range(2):
                    for ly in range(num_layers):
                        pb = torch.zeros(
                            (page_buffer_size, head_size),
                            dtype=dtype,
                            device=device,
                        )
                        if direction:  # Paged → LMC
                            for s in range(page_buffer_size):
                                val = (
                                    kv * 7000
                                    + ly * 2000
                                    + s * 10
                                    + torch.arange(head_size, device=device)
                                ).to(dtype)
                                pb[s] = val
                        buffers[(kv, ly)] = pb

                ptr_list = []
                for ly in range(num_layers):
                    ptr_list.append(buffers[(0, ly)].data_ptr())
                for ly in range(num_layers):
                    ptr_list.append(buffers[(1, ly)].data_ptr())

                key_value_ptrs = torch.tensor(
                    ptr_list,
                    dtype=torch.int64,
                    device=device,
                ).contiguous()

            # ── 3. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer_unilateral(
                lmc_tensor,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                gpu_kv_format,
            )
            if is_cuda_backend:
                torch.cuda.synchronize()

            # ── 4. Verify ──
            for t_id in range(num_tokens):
                s_idx = slot_mapping[t_id].item()
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = lmc_tensor[kv, ly, t_id]

                        if is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            paged_val = buffers[(kv, ly)][s_idx]

                        torch.testing.assert_close(
                            lmc_val,
                            paged_val,
                            msg=(
                                f"Mismatch: {gpu_kv_format} {dir_tag}, "
                                f"KV={kv}, layer={ly}, "
                                f"token={t_id}, slot={s_idx}"
                            ),
                        )

    # ── 5. Save canonical result for cross-backend comparison ──
    # Use non-MLA unilateral (the primary use case of this function)
    for direction in [True, False]:
        dir_tag = "p2l" if direction else "l2p"

        lmc_shape = (2, num_layers, num_tokens, head_size)
        lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for kv in range(2):
                for ly in range(num_layers):
                    for t in range(num_tokens):
                        val = (
                            kv * 5000
                            + ly * 1000
                            + t * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        lmc_tensor[kv, ly, t] = val

        buffers = {}
        for kv in range(2):
            for ly in range(num_layers):
                pb = torch.zeros(
                    (page_buffer_size, head_size),
                    dtype=dtype,
                    device=device,
                )
                if direction:
                    for s in range(page_buffer_size):
                        val = (
                            kv * 7000
                            + ly * 2000
                            + s * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        pb[s] = val
                buffers[(kv, ly)] = pb

        ptr_list = []
        for ly in range(num_layers):
            ptr_list.append(buffers[(0, ly)].data_ptr())
        for ly in range(num_layers):
            ptr_list.append(buffers[(1, ly)].data_ptr())

        key_value_ptrs = torch.tensor(
            ptr_list,
            dtype=torch.int64,
            device=device,
        ).contiguous()

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer_unilateral(
            lmc_tensor,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        save_result(
            f"multi_layer_kv_transfer_unilateral_{dir_tag}",
            lmc_tensor.cpu(),
        )


def scenario_alloc_free_pinned_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    flags = 0  # cudaHostAllocDefault

    # 1. Allocate
    ptr = ops.alloc_pinned_ptr(alloc_size, flags)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_ptr returned null"

    # 2. Free
    ops.free_pinned_ptr(ptr)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_pinned_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_free_numa_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    node = 0  # NUMA node 0 (always exists)

    # 1. Allocate
    ptr = ops.alloc_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_numa_ptr returned null"

    # 2. Free (must pass same size as alloc)
    ops.free_numa_ptr(ptr, alloc_size)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_numa_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_free_pinned_numa_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    node = 0  # NUMA node 0

    # 1. Allocate (NUMA + cudaHostRegister)
    ptr = ops.alloc_pinned_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_numa_ptr returned null"

    # 2. Free (cudaHostUnregister + munmap)
    ops.free_pinned_numa_ptr(ptr, alloc_size)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_pinned_numa_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_transfer_direction_enum():
    ops, scene_info = get_test_context()

    # 1. Verify enum members exist
    td = ops.TransferDirection
    assert hasattr(td, "H2D"), "Missing TransferDirection.H2D"
    assert hasattr(td, "D2H"), "Missing TransferDirection.D2H"

    # 2. Verify values are distinct
    assert td.H2D != td.D2H, "H2D and D2H should be distinct"

    # 3. Extract int value (compatible with both
    #    pybind11 enum and Python enum)
    h2d = td.H2D
    d2h = td.D2H
    h2d_val = h2d.value if hasattr(h2d, "value") else int(h2d)
    d2h_val = d2h.value if hasattr(d2h, "value") else int(d2h)

    # 4. Save for cross-backend comparison
    save_result(
        "transfer_direction_enum",
        torch.tensor(
            [h2d_val, d2h_val],
            dtype=torch.int32,
        ),
    )


def scenario_alloc_pinned_ptr_data_readwrite():
    ops, scene_info = get_test_context()

    for size in [64, 4096, 1024 * 1024]:
        ptr = ops.alloc_pinned_ptr(size, 0)
        assert isinstance(ptr, int)
        assert ptr != 0, f"alloc_pinned_ptr({size}) returned null"

        # Write a pattern
        pattern = bytes(range(256)) * (size // 256 + 1)
        pattern = pattern[:size]
        src_buf = (ctypes.c_uint8 * size)(*pattern)
        ctypes.memmove(ctypes.c_void_p(ptr), src_buf, size)

        # Read back
        dst_buf = (ctypes.c_uint8 * size)()
        ctypes.memmove(dst_buf, ctypes.c_void_p(ptr), size)
        result_bytes = bytes(dst_buf)
        assert result_bytes == pattern[:size], (
            f"Data mismatch for size={size}"
        )

        ops.free_pinned_ptr(ptr)

    save_result(
        "alloc_pinned_ptr_data_readwrite",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_pinned_ptr_device_id():
    ops, scene_info = get_test_context()

    for device_id in [0, 1]:
        ptr = ops.alloc_pinned_ptr(4096, device_id)
        assert isinstance(ptr, int)
        assert ptr != 0, f"alloc_pinned_ptr(device_id={device_id}) returned null"
        ops.free_pinned_ptr(ptr)

    save_result(
        "alloc_pinned_ptr_device_id",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_pinned_numa_ptr_numa_id():
    ops, scene_info = get_test_context()

    for numa_id in [0, 1]:
        ptr = ops.alloc_pinned_numa_ptr(4096, numa_id)
        assert isinstance(ptr, int)
        assert ptr != 0, f"alloc_pinned_numa_ptr(numa_id={numa_id}) returned null"
        ops.free_pinned_numa_ptr(ptr, 4096)

    save_result(
        "alloc_pinned_numa_ptr_numa_id",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_free_idempotency():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    if not is_cuda_backend:
        # alloc_pinned_ptr / free_pinned_ptr
        ptr = ops.alloc_pinned_ptr(4096, 0)
        ops.free_pinned_ptr(ptr)
        ops.free_pinned_ptr(ptr)  # double-free
        ops.free_pinned_ptr(0)  # null free
        ops.free_pinned_ptr(0xDEADBEEF)  # arbitrary address

        # alloc_pinned_numa_ptr / free_pinned_numa_ptr
        ptr2 = ops.alloc_pinned_numa_ptr(4096, 0)
        ops.free_pinned_numa_ptr(ptr2, 4096)
        ops.free_pinned_numa_ptr(ptr2, 4096)  # double-free
        ops.free_pinned_numa_ptr(0, 0)
        ops.free_pinned_numa_ptr(0xDEADBEEF, 4096)

        # alloc_numa_ptr / free_numa_ptr
        ptr3 = ops.alloc_numa_ptr(4096, 0)
        ops.free_numa_ptr(ptr3, 4096)
        ops.free_numa_ptr(ptr3, 4096)  # double-free
        ops.free_numa_ptr(0, 0)
        ops.free_numa_ptr(0xDEADBEEF, 4096)

    save_result(
        "free_idempotency",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_multi_layer_kv_transfer_neg_slots():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    torch.manual_seed(7)

    num_layers = 2
    num_tokens = 4
    head_size = 8
    page_buffer_size = 12
    dtype = torch.float32

    mixed_slots = [-1, 2, -1, 9]
    all_neg_slots = [-1, -1, -1, -1]

    def _run_neg_slot_test(gpu_kv_format, is_mla, slots_list, direction, tag):
        k_or_v_size = 1 if is_mla else 2
        slot_mapping = torch.tensor(slots_list, dtype=torch.int64, device=device)

        lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
        key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if direction == ops.TransferDirection.H2D:
            # Fill key_value with known values for H2D
            for kv in range(k_or_v_size):
                for ly in range(num_layers):
                    for t in range(num_tokens):
                        key_value[kv, ly, t] = (
                            kv * 1000 + ly * 100 + t * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)

        page_buffers = []
        for ly in range(num_layers):
            if is_mla:
                pb = torch.zeros(
                    (page_buffer_size, head_size), dtype=dtype, device=device
                )
            else:
                pb = torch.zeros(
                    (2, page_buffer_size, head_size), dtype=dtype, device=device
                )
            if direction == ops.TransferDirection.D2H:
                # Fill page_buffers with known values for D2H
                for s in range(page_buffer_size):
                    for kv in range(k_or_v_size):
                        val = (
                            kv * 2000 + ly * 500 + s * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        if is_mla:
                            pb[s] = val
                        else:
                            pb[kv, s] = val
            page_buffers.append(pb)

        key_value_ptrs = torch.tensor(
            [pb.data_ptr() for pb in page_buffers],
            dtype=torch.int64,
            device=device,
        )

        ops.multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            direction,
            gpu_kv_format,
            1,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # Verify negative slots are untouched for D2H
        if direction == ops.TransferDirection.D2H:
            for t_id, slot_idx in enumerate(slots_list):
                if slot_idx < 0:
                    assert torch.all(key_value[:, :, t_id, :] == 0), (
                        f"{tag}: token {t_id} (neg slot) should be zero"
                    )

        save_result(f"multi_layer_kv_transfer_neg_slots_{tag}", key_value.cpu())

    # NL_X_TWO_NB_BS_NH_HS (non-MLA), mixed slots, D2H
    _run_neg_slot_test(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, mixed_slots,
        ops.TransferDirection.D2H, "nonmla_mixed_d2h",
    )
    # NL_X_TWO_NB_BS_NH_HS (non-MLA), mixed slots, H2D
    _run_neg_slot_test(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, mixed_slots,
        ops.TransferDirection.H2D, "nonmla_mixed_h2d",
    )
    # NL_X_TWO_NB_BS_NH_HS (non-MLA), all-negative, D2H
    _run_neg_slot_test(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, all_neg_slots,
        ops.TransferDirection.D2H, "nonmla_allneg_d2h",
    )
    # NL_X_NB_BS_HS (MLA), mixed slots, D2H
    _run_neg_slot_test(
        ops.GPUKVFormat.NL_X_NB_BS_HS, True, mixed_slots,
        ops.TransferDirection.D2H, "mla_mixed_d2h",
    )
    # NL_X_NB_BS_HS (MLA), mixed slots, H2D
    _run_neg_slot_test(
        ops.GPUKVFormat.NL_X_NB_BS_HS, True, mixed_slots,
        ops.TransferDirection.H2D, "mla_mixed_h2d",
    )


def scenario_multi_layer_kv_transfer_unilateral_neg_slots():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    torch.manual_seed(11)

    num_layers = 2
    num_tokens = 4
    head_size = 8
    page_buffer_size = 12
    dtype = torch.float32

    mixed_slots = [-1, 3, 7, -1]

    for direction in [ops.TransferDirection.D2H, ops.TransferDirection.H2D]:
        dir_tag = "d2h" if direction == ops.TransferDirection.D2H else "h2d"

        slot_mapping = torch.tensor(mixed_slots, dtype=torch.int64, device=device)

        lmc_shape = (2, num_layers, num_tokens, head_size)
        lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if direction == ops.TransferDirection.H2D:
            for kv in range(2):
                for ly in range(num_layers):
                    for t in range(num_tokens):
                        lmc_tensor[kv, ly, t] = (
                            kv * 1000 + ly * 100 + t * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)

        # Non-MLA unilateral: separate K/V buffers
        buffers = {}
        for kv in range(2):
            for ly in range(num_layers):
                pb = torch.zeros(
                    (page_buffer_size, head_size), dtype=dtype, device=device
                )
                if direction == ops.TransferDirection.D2H:
                    for s in range(page_buffer_size):
                        val = (
                            kv * 2000 + ly * 500 + s * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        pb[s] = val
                buffers[(kv, ly)] = pb

        ptr_list = []
        for ly in range(num_layers):
            ptr_list.append(buffers[(0, ly)].data_ptr())
        for ly in range(num_layers):
            ptr_list.append(buffers[(1, ly)].data_ptr())

        key_value_ptrs = torch.tensor(
            ptr_list, dtype=torch.int64, device=device
        ).contiguous()

        ops.multi_layer_kv_transfer_unilateral(
            lmc_tensor,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            direction,
            ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # Verify negative slots untouched for D2H
        if direction == ops.TransferDirection.D2H:
            for t_id, slot_idx in enumerate(mixed_slots):
                if slot_idx < 0:
                    assert torch.all(lmc_tensor[:, :, t_id, :] == 0), (
                        f"token {t_id} (neg slot) should remain zero"
                    )

        save_result(
            f"multi_layer_kv_transfer_unilateral_neg_slots_{dir_tag}",
            lmc_tensor.cpu(),
        )


def scenario_single_layer_kv_transfer_extra():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_blocks = 64
    block_size = 16
    num_heads = 4
    head_size = 32
    hidden_size = num_heads * head_size

    # ── Extra combos: NL_X_TWO_NB_BS_NH_HS, non-MLA, token_major=False, D2H ──
    extra_cases = [
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, False, ops.TransferDirection.D2H),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, ops.TransferDirection.D2H),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, False, ops.TransferDirection.D2H),
    ]

    def _run_case(
        gpu_kv_format, is_mla, token_major, direction, num_tokens, slot_mapping, tag
    ):
        is_two_major = gpu_kv_format == ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
        else:
            lmc_shape = (
                (num_tokens, 2, hidden_size)
                if token_major
                else (2, num_tokens, hidden_size)
            )
            if is_two_major:
                vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            else:
                vllm_shape = (num_blocks, 2, block_size, num_heads, head_size)

        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 500)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 500 + 100)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            direction,
            gpu_kv_format,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        result = (
            lmc_tensor.cpu()
            if direction == ops.TransferDirection.D2H
            else vllm_tensor.cpu()
        )
        save_result(f"single_layer_kv_transfer_extra_{tag}", result)

    # Standard num_tokens cases
    num_tokens_std = 16
    slot_mapping_std = torch.arange(
        0, num_tokens_std * 2, 2, device=device, dtype=torch.int64
    )

    for i, (fmt, is_mla, tm, direction) in enumerate(extra_cases):
        dir_tag = "d2h" if direction == ops.TransferDirection.D2H else "h2d"
        fmt_tag = str(fmt).split(".")[-1].lower()
        tag = f"{fmt_tag}_tm{tm}_{dir_tag}"
        _run_case(fmt, is_mla, tm, direction, num_tokens_std, slot_mapping_std, tag)

    # Single-token edge case for non-MLA
    slot_mapping_1 = torch.tensor([5], device=device, dtype=torch.int64)
    _run_case(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True,
        ops.TransferDirection.H2D, 1, slot_mapping_1,
        "nonmla_1tok_h2d",
    )
    _run_case(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True,
        ops.TransferDirection.D2H, 1, slot_mapping_1,
        "nonmla_1tok_d2h",
    )

    # Single-token edge case for MLA
    _run_case(
        ops.GPUKVFormat.NL_X_NB_BS_HS, True, True,
        ops.TransferDirection.H2D, 1, slot_mapping_1,
        "mla_1tok_h2d",
    )
    _run_case(
        ops.GPUKVFormat.NL_X_NB_BS_HS, True, True,
        ops.TransferDirection.D2H, 1, slot_mapping_1,
        "mla_1tok_d2h",
    )

    # Non-contiguous (gapped, step=11) slot mapping
    num_tokens_gap = 8
    max_slot = num_tokens_gap * 11
    if max_slot > num_blocks * block_size:
        max_slot = num_blocks * block_size - 1
    slot_mapping_gap = torch.arange(
        0, min(num_tokens_gap * 11, num_blocks * block_size),
        11, device=device, dtype=torch.int64,
    )[:num_tokens_gap]

    _run_case(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True,
        ops.TransferDirection.H2D, len(slot_mapping_gap), slot_mapping_gap,
        "nonmla_gapped_h2d",
    )
    _run_case(
        ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True,
        ops.TransferDirection.D2H, len(slot_mapping_gap), slot_mapping_gap,
        "nonmla_gapped_d2h",
    )


def scenario_single_layer_kv_transfer_sgl_extra():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_blocks = 32
    block_size = 16
    num_heads = 4
    head_size = 32
    hidden_size = num_heads * head_size

    sgl_k_shape = (num_blocks, block_size, num_heads, head_size)
    sgl_k_size = num_blocks * block_size * num_heads * head_size
    sgl_v_size = sgl_k_size

    # Single-slot [7] cases across token_major=True/False and both directions
    slot_mapping_1 = torch.tensor([7], device=device, dtype=torch.int64)

    for token_major in [True, False]:
        for direction in [ops.TransferDirection.H2D, ops.TransferDirection.D2H]:
            dir_tag = "h2d" if direction == ops.TransferDirection.H2D else "d2h"
            tm_tag = "tm" if token_major else "ntm"
            tag = f"sgl_1tok_{tm_tag}_{dir_tag}"

            lmc_shape = (1, 2, hidden_size) if token_major else (2, 1, hidden_size)
            lmc_size = 2 * hidden_size

            lmc_tensor = (
                (torch.arange(lmc_size, device=device) % 300)
                .to(torch.float16)
                .reshape(lmc_shape)
            )
            sgl_k_tensor = (
                (torch.arange(sgl_k_size, device=device) % 300 + 300)
                .to(torch.float16)
                .reshape(sgl_k_shape)
            )
            sgl_v_tensor = (
                (torch.arange(sgl_v_size, device=device) % 300 + 600)
                .to(torch.float16)
                .reshape(sgl_k_shape)
            )

            ops.single_layer_kv_transfer_sgl(
                lmc_tensor,
                sgl_k_tensor,
                sgl_v_tensor,
                slot_mapping_1,
                direction,
                token_major,
            )
            if is_cuda_backend:
                torch.cuda.synchronize()

            result = (
                lmc_tensor.cpu()
                if direction == ops.TransferDirection.D2H
                else sgl_k_tensor.cpu()
            )
            save_result(f"single_layer_kv_transfer_sgl_extra_{tag}", result)

    # On non-CUDA path: pass negative slot [-1] — wrap-around behavior, no crash
    if not is_cuda_backend:
        slot_neg = torch.tensor([-1], device=device, dtype=torch.int64)
        lmc_shape = (1, 2, hidden_size)
        lmc_tensor = torch.zeros(lmc_shape, dtype=torch.float16, device=device)
        sgl_k_tensor = torch.zeros(sgl_k_shape, dtype=torch.float16, device=device)
        sgl_v_tensor = torch.zeros(sgl_k_shape, dtype=torch.float16, device=device)
        # No assertion — just document that it does not crash
        ops.single_layer_kv_transfer_sgl(
            lmc_tensor,
            sgl_k_tensor,
            sgl_v_tensor,
            slot_neg,
            ops.TransferDirection.H2D,
            True,
        )


def scenario_lmcache_memcpy_async_alignments():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)

    # ── Part A: multiple host_buffer_alignment values ──
    valid_alignments = [1, 2, 4, 8, 16, 32, 64, 256, 4096]
    offsets = [0, 1, 7, 15, 16, 255]

    # Use a large enough buffer to accommodate all alignments
    nbytes = 4096
    src_host = torch.randint(1, 255, (nbytes,), dtype=torch.uint8)
    gpu_buffer = torch.zeros(nbytes, dtype=torch.uint8)

    if torch.cuda.is_available():
        dst_host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    else:
        dst_host = torch.zeros(nbytes, dtype=torch.uint8)

    if is_cuda_backend:
        gpu_buffer = gpu_buffer.to(f"cuda:{torch.cuda.current_device()}")

    for alignment in valid_alignments:
        ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            src_host.data_ptr(),
            nbytes,
            ops.TransferDirection.H2D,
            0,
            alignment,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        ops.lmcache_memcpy_async(
            dst_host.data_ptr(),
            gpu_buffer.data_ptr(),
            nbytes,
            ops.TransferDirection.D2H,
            0,
            alignment,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

    # ── Part B: multiple host_buffer_offset values ──
    small_nbytes = 256
    src_small = torch.randint(1, 255, (small_nbytes,), dtype=torch.uint8)
    gpu_small = torch.zeros(small_nbytes, dtype=torch.uint8)
    if is_cuda_backend:
        gpu_small = gpu_small.to(f"cuda:{torch.cuda.current_device()}")

    for offset in offsets:
        ops.lmcache_memcpy_async(
            gpu_small.data_ptr(),
            src_small.data_ptr(),
            small_nbytes,
            ops.TransferDirection.H2D,
            offset,
            16,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

    # ── Part C: non-power-of-two alignments raise ValueError (non-CUDA only) ──
    if not is_cuda_backend:
        for bad_alignment in [3, 5, 7]:
            try:
                ops.lmcache_memcpy_async(
                    gpu_buffer.data_ptr(),
                    src_host.data_ptr(),
                    64,
                    ops.TransferDirection.H2D,
                    0,
                    bad_alignment,
                )
                raise AssertionError(
                    f"Expected ValueError for alignment={bad_alignment}"
                )
            except ValueError:
                pass

    save_result(
        "lmcache_memcpy_async_alignments",
        src_host.cpu(),
    )


def scenario_encode_decode_fast_new_edge_cases():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    configs = [
        (1, 1, 128, 16),   # minimal layers/channels
        (2, 4, 1, 16),     # single token (encode-only)
        (1, 1, 1, 2),      # absolute minimum (encode-only)
        (2, 4, 128, 2),    # minimal alphabet
    ]

    for nlayers, nchannels, ntokens, alphabet_size in configs:
        max_buf_len = max(ntokens * 2, 8)

        step = max(100 // alphabet_size, 1)
        base_cdf = torch.arange(0, 100, step, dtype=torch.int32)
        base_cdf = base_cdf[:alphabet_size]
        cdf_cpu = (
            base_cdf.unsqueeze(0)
            .unsqueeze(0)
            .expand(nlayers, nchannels, -1)
            .contiguous()
        )
        cdf = cdf_cpu.to(dtype=torch.int16, device=device)

        total_syms = nlayers * ntokens * nchannels
        input_cpu = torch.arange(total_syms, dtype=torch.float32)
        input_cpu = (input_cpu % max(alphabet_size - 1, 1)).to(torch.int8)
        input_cpu = input_cpu.reshape(nlayers, ntokens, nchannels)
        input_sym = input_cpu.to(device=device)

        output_buffer = torch.zeros(
            (nlayers, nchannels, max_buf_len), dtype=torch.uint8, device=device,
        )
        output_lengths = torch.zeros(
            (nlayers, nchannels), dtype=torch.int32, device=device,
        )

        ops.encode_fast_new(cdf, input_sym, output_buffer, output_lengths)
        if is_cuda_backend:
            torch.cuda.synchronize()

        tag = f"enc_dec_edge_nl{nlayers}_nc{nchannels}_nt{ntokens}_a{alphabet_size}"

        if ntokens == 1:
            # Encode-only: save output_lengths
            save_result(tag, output_lengths.cpu())
            continue

        # Full roundtrip: decode and verify
        decoded_sym = torch.zeros_like(input_sym, dtype=torch.uint8)
        ops.decode_fast_new(cdf, output_buffer, output_lengths, decoded_sym)
        if is_cuda_backend:
            torch.cuda.synchronize()

        input_uint8 = input_sym.to(torch.uint8)
        mismatch = (input_uint8 != decoded_sym).sum().item()
        assert mismatch == 0, (
            f"Decode mismatch for config nl={nlayers},nc={nchannels},"
            f"nt={ntokens},a={alphabet_size}: {mismatch} errors"
        )

        save_result(tag, decoded_sym[0, :min(20, ntokens), 0].cpu())


def scenario_decode_fast_prefsum_edge_cases():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    configs = [
        (1, 1, 128, 16),
        (2, 4, 1, 16),
        (1, 1, 1, 2),
        (2, 4, 128, 2),
    ]

    for nlayers, nchannels, ntokens, alphabet_size in configs:
        max_buf_len = max(ntokens * 2, 8)
        tag = f"prefsum_edge_nl{nlayers}_nc{nchannels}_nt{ntokens}_a{alphabet_size}"

        cdf = torch.randint(
            1, 100, (nlayers, nchannels, alphabet_size), dtype=torch.int32
        )
        cdf = torch.cumsum(cdf, dim=-1).float()
        cdf = (cdf / cdf[..., -1:] * 65536).to(torch.int32)
        cdf[..., -1] = 65536
        cdf = cdf.to(device).to(torch.int16).contiguous()

        input_sym = torch.randint(
            0,
            max(alphabet_size - 2, 1),
            (nlayers, ntokens, nchannels),
            dtype=torch.int8,
        ).to(device)

        tmp_buf = torch.zeros(
            (nlayers, nchannels, max_buf_len), dtype=torch.uint8, device=device,
        )
        tmp_len = torch.zeros(
            (nlayers, nchannels), dtype=torch.int32, device=device,
        )
        ops.encode_fast_new(cdf, input_sym, tmp_buf, tmp_len)
        if is_cuda_backend:
            torch.cuda.synchronize()

        if ntokens == 1:
            save_result(tag, tmp_len.cpu())
            continue

        lens_flat = tmp_len.cpu().flatten().tolist()
        bufs_flat = tmp_buf.cpu().reshape(-1, max_buf_len).numpy()
        all_bytes = []
        for i, length in enumerate(lens_flat):
            all_bytes.extend(bufs_flat[i, :length].tolist())

        bytestream_1d = torch.tensor(
            all_bytes, dtype=torch.uint8, device=device,
        ).contiguous()

        lengths_prefsum = (
            tmp_len.flatten().cumsum(0).reshape(tmp_len.shape).to(torch.int64).to(device)
        ).contiguous()

        decoded_sym = (
            torch.zeros_like(input_sym, dtype=torch.uint8).to(device).contiguous()
        )

        ops.decode_fast_prefsum(cdf, bytestream_1d, lengths_prefsum, decoded_sym)
        if is_cuda_backend:
            torch.cuda.synchronize()

        input_ref = input_sym.to(torch.uint8)
        mismatch = (input_ref != decoded_sym).sum().item()
        assert mismatch == 0, (
            f"Prefsum mismatch for config nl={nlayers},nc={nchannels},"
            f"nt={ntokens},a={alphabet_size}: {mismatch} errors"
        )

        save_result(tag, decoded_sym[0, :min(20, ntokens), 0].cpu())


def scenario_calculate_cdf_edge_cases():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    # Edge case 1: num_bins=1, all-zero input
    input_1 = torch.zeros((1, 100, 1), dtype=torch.int8)
    if is_cuda_backend:
        input_1 = input_1.to(f"cuda:{torch.cuda.current_device()}")
    raw_1 = ops.calculate_cdf(input_1, 1)
    out_1 = raw_1.flatten().cpu()
    if is_cuda_backend:
        out_int32 = out_1.to(torch.int32)
        out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
        final_1 = out_uint16.float() / 65536.0
    else:
        final_1 = out_1.float()
    save_result("calculate_cdf_edge_allzero_bins1", final_1)

    # Edge case 2: all-same-value input (constant=3, num_bins=8)
    input_2 = torch.full((1, 1000, 1), 3, dtype=torch.int8)
    if is_cuda_backend:
        input_2 = input_2.to(f"cuda:{torch.cuda.current_device()}")
    raw_2 = ops.calculate_cdf(input_2, 8)
    out_2 = raw_2.flatten().cpu()
    if is_cuda_backend:
        out_int32 = out_2.to(torch.int32)
        out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
        final_2 = out_uint16.float() / 65536.0
    else:
        final_2 = out_2.float()
    save_result("calculate_cdf_edge_allsame_bins8", final_2)

    # Edge case 3: uniform distribution (equal counts for 4 bins)
    # Create exactly equal counts: 250 each for bins 0..3
    vals = torch.cat([
        torch.full((250,), i, dtype=torch.int8) for i in range(4)
    ])
    input_3 = vals.reshape(1, 1000, 1)
    if is_cuda_backend:
        input_3 = input_3.to(f"cuda:{torch.cuda.current_device()}")
    raw_3 = ops.calculate_cdf(input_3, 4)
    out_3 = raw_3.flatten().cpu()
    if is_cuda_backend:
        out_int32 = out_3.to(torch.int32)
        out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
        final_3 = out_uint16.float() / 65536.0
    else:
        final_3 = out_3.float()
    save_result("calculate_cdf_edge_uniform_bins4", final_3)


def scenario_rotary_embedding_k_fused_neox_false():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    max_position = 2048
    is_neox = False

    test_configs = [
        (128, 32, 128),   # standard
        (128, 32, 64),    # smaller head
        (1, 8, 128),      # single token
    ]

    for num_tokens, num_kv_heads, head_size in test_configs:
        rotary_dim = head_size

        old_positions = torch.randint(0, 1000, (num_tokens,), dtype=torch.long)
        new_positions = old_positions + 1
        key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32)
        cos_sin_cache = torch.randn(max_position, rotary_dim, dtype=torch.float32)

        if is_cuda_backend:
            target_dev = f"cuda:{torch.cuda.current_device()}"
            old_positions = old_positions.to(target_dev)
            new_positions = new_positions.to(target_dev)
            key = key.to(target_dev)
            cos_sin_cache = cos_sin_cache.to(target_dev)

        ops.rotary_embedding_k_fused(
            old_positions,
            new_positions,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

        tag = f"rotary_neox_false_nt{num_tokens}_nh{num_kv_heads}_hs{head_size}"
        save_result(tag, key.cpu())


def scenario_load_reshape_flash_roundtrip():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    src_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_blocks = 50
    block_size = 16
    num_heads = 4
    head_size = 64
    num_layers = 4
    num_tokens = 64
    dtype = torch.bfloat16

    total_elements = num_blocks * block_size * num_heads * head_size

    kv_cache_cpu = []
    for i in range(num_layers):
        base_tensor = torch.linspace(
            float(i), float(i + 1), total_elements, dtype=torch.float32,
        ).reshape(num_blocks, block_size, num_heads, head_size).to(dtype)
        k = base_tensor.clone()
        v = base_tensor + 0.5
        kv_cache_cpu.append([k, v])

    kv_cache = [
        [layer[0].to(src_device), layer[1].to(src_device)]
        for layer in kv_cache_cpu
    ]

    step = (num_blocks * block_size) // num_tokens
    slot_indices = list(range(0, num_blocks * block_size, step))[:num_tokens]
    slot_mapping = torch.tensor(slot_indices, device=src_device, dtype=torch.int64)

    # ── Extract with load_and_reshape_flash ──
    mem_obj_shape = (2, num_layers, num_tokens, num_heads * head_size)
    mem_obj = torch.zeros(mem_obj_shape, dtype=dtype, device="cpu")
    if is_cuda_backend:
        mem_obj = mem_obj.pin_memory()

    for layer_id in range(num_layers):
        ops.load_and_reshape_flash(
            mem_obj,
            kv_cache[layer_id][0],
            kv_cache[layer_id][1],
            slot_mapping,
            layer_id,
        )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # ── Write back with reshape_and_cache_back_flash ──
    fresh_cache = [
        [
            torch.zeros(num_blocks, block_size, num_heads, head_size,
                        device=src_device, dtype=dtype),
            torch.zeros(num_blocks, block_size, num_heads, head_size,
                        device=src_device, dtype=dtype),
        ]
        for _ in range(num_layers)
    ]

    slot_mapping_dst = torch.tensor(slot_indices, device=src_device, dtype=torch.int64)

    if not mem_obj.is_contiguous():
        mem_obj = mem_obj.contiguous()

    for layer_id in range(num_layers):
        ops.reshape_and_cache_back_flash(
            mem_obj,
            fresh_cache[layer_id][0],
            fresh_cache[layer_id][1],
            slot_mapping_dst,
            layer_id,
        )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # ── Verify: fresh_cache must match original kv_cache at mapped slots ──
    for layer_id in range(num_layers):
        orig_k = kv_cache_cpu[layer_id][0]
        orig_v = kv_cache_cpu[layer_id][1]
        fresh_k = fresh_cache[layer_id][0].cpu()
        fresh_v = fresh_cache[layer_id][1].cpu()

        for tok_idx, slot in enumerate(slot_indices):
            block_idx = slot // block_size
            offset = slot % block_size

            torch.testing.assert_close(
                fresh_k[block_idx, offset],
                orig_k[block_idx, offset],
                rtol=1e-3, atol=1e-3,
                msg=f"K roundtrip mismatch layer={layer_id}, slot={slot}",
            )
            torch.testing.assert_close(
                fresh_v[block_idx, offset],
                orig_v[block_idx, offset],
                rtol=1e-3, atol=1e-3,
                msg=f"V roundtrip mismatch layer={layer_id}, slot={slot}",
            )

    # Also test with layer_idx = num_layers - 1 (last layer)
    last_layer = num_layers - 1
    mem_last = torch.zeros(mem_obj_shape, dtype=dtype, device="cpu")
    if is_cuda_backend:
        mem_last = mem_last.pin_memory()
    ops.load_and_reshape_flash(
        mem_last,
        kv_cache[last_layer][0],
        kv_cache[last_layer][1],
        slot_mapping,
        last_layer,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    fresh_last = [
        torch.zeros(num_blocks, block_size, num_heads, head_size,
                    device=src_device, dtype=dtype),
        torch.zeros(num_blocks, block_size, num_heads, head_size,
                    device=src_device, dtype=dtype),
    ]
    ops.reshape_and_cache_back_flash(
        mem_last,
        fresh_last[0],
        fresh_last[1],
        slot_mapping_dst,
        last_layer,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # ── Save canonical result (layer 0 key, first mapped slot's block) ──
    save_result("load_reshape_flash_roundtrip", fresh_cache[0][0][0].cpu())


def scenario_get_gpu_pci_bus_id_invalid():
    ops, scene_info = get_test_context()

    res = ops.get_gpu_pci_bus_id(999)
    assert not res, f"Expected falsy result for device_id=999, got: {res!r}"

    save_result(
        "get_gpu_pci_bus_id_invalid",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_transfer_direction_enum_values():
    ops, scene_info = get_test_context()

    h2d = ops.TransferDirection.H2D
    d2h = ops.TransferDirection.D2H

    h2d_val = h2d.value if hasattr(h2d, "value") else int(h2d)
    d2h_val = d2h.value if hasattr(d2h, "value") else int(d2h)

    assert h2d_val == 0, f"Expected H2D.value == 0, got {h2d_val}"
    assert d2h_val == 1, f"Expected D2H.value == 1, got {d2h_val}"

    save_result(
        "transfer_direction_enum_values",
        torch.tensor([h2d_val, d2h_val], dtype=torch.int32),
    )


# ==========================================
# 3. Registry
# ==========================================
# cover pybind list in csrc/pybind.cpp
SCENARIO_REGISTRY = {
    "transfer_direction_enum": scenario_transfer_direction_enum,
    "multi_layer_kv_transfer": scenario_multi_layer_kv_transfer,
    "multi_layer_kv_transfer_unilateral": scenario_multi_layer_kv_transfer_unilateral,
    "single_layer_kv_transfer": scenario_single_layer_kv_transfer,
    "single_layer_kv_transfer_sgl": scenario_single_layer_kv_transfer_sgl,
    "load_and_reshape_flash": scenario_load_and_reshape_flash,
    "reshape_and_cache_back_flash": scenario_reshape_and_cache_back_flash,
    "lmcache_memcpy_async": scenario_lmcache_memcpy_async,
    "encode_fast_new": scenario_encode_fast_new,
    "decode_fast_new": scenario_decode_fast_new,
    "decode_fast_prefsum": scenario_decode_fast_prefsum,
    "calculate_cdf": scenario_calculate_cdf,
    "rotary_embedding_k_fused": scenario_rotary_embedding_k_fused,
    "alloc_free_pinned_ptr": scenario_alloc_free_pinned_ptr,
    "alloc_free_pinned_numa_ptr": scenario_alloc_free_pinned_numa_ptr,
    "alloc_free_numa_ptr": scenario_alloc_free_numa_ptr,
    "get_gpu_pci_bus_id": scenario_get_gpu_pci_bus_id,
    "alloc_pinned_ptr_data_readwrite": scenario_alloc_pinned_ptr_data_readwrite,
    "alloc_pinned_ptr_device_id": scenario_alloc_pinned_ptr_device_id,
    "alloc_pinned_numa_ptr_numa_id": scenario_alloc_pinned_numa_ptr_numa_id,
    "free_idempotency": scenario_free_idempotency,
    "multi_layer_kv_transfer_neg_slots": scenario_multi_layer_kv_transfer_neg_slots,
    "multi_layer_kv_transfer_unilateral_neg_slots": (
        scenario_multi_layer_kv_transfer_unilateral_neg_slots
    ),
    "single_layer_kv_transfer_extra": scenario_single_layer_kv_transfer_extra,
    "single_layer_kv_transfer_sgl_extra": scenario_single_layer_kv_transfer_sgl_extra,
    "lmcache_memcpy_async_alignments": scenario_lmcache_memcpy_async_alignments,
    "encode_decode_fast_new_edge_cases": scenario_encode_decode_fast_new_edge_cases,
    "decode_fast_prefsum_edge_cases": scenario_decode_fast_prefsum_edge_cases,
    "calculate_cdf_edge_cases": scenario_calculate_cdf_edge_cases,
    "rotary_embedding_k_fused_neox_false": scenario_rotary_embedding_k_fused_neox_false,
    "load_reshape_flash_roundtrip": scenario_load_reshape_flash_roundtrip,
    "get_gpu_pci_bus_id_invalid": scenario_get_gpu_pci_bus_id_invalid,
    "transfer_direction_enum_values": scenario_transfer_direction_enum_values,
}


# ==========================================
# 4. Subprocess launcher
# ==========================================


def run_scenario(mode, cuda_visible):
    env = os.environ.copy()
    env["LMC_TEST_MODE"] = mode
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    print(
        f"\n>>> Launching Scenario: MODE={mode}, CUDA_VISIBLE_DEVICES='{cuda_visible}'"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-s", "-q"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


# ==========================================
# 5. The test functions pytest sees
# ==========================================

if _is_child:
    # --- Child process: each scenario is its own test case ---

    @pytest.mark.parametrize("name", list(SCENARIO_REGISTRY.keys()))
    def test_scenario(name):
        SCENARIO_REGISTRY[name]()

else:
    # --- Top-level: launch all children once, then compare each function ---

    @pytest.fixture(scope="module")
    def run_all_children():
        """Launch 3 child processes. Runs once for the entire module."""
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)

        for mode, cuda_vis in [("CUDA_OPS", "0"), ("NON_CUDA", "0"), ("NON_CUDA", "")]:
            r = run_scenario(mode, cuda_vis)
            assert r.returncode == 0, (
                f"Scenario {mode}/CUDA_VISIBLE_DEVICES='{cuda_vis}' failed:\n"
                f"{r.stdout}\n{r.stderr}"
            )

    @pytest.mark.parametrize("name", list(SCENARIO_REGISTRY.keys()))
    def test_compare(run_all_children, name):
        """Each scenario function gets its own PASS/FAIL."""
        # Match: exact name or name as prefix (e.g. calculate_cdf → calculate_cdf_bins*)
        exact_files = sorted(RESULTS_DIR.glob(f"{name}@*.pt"))
        prefix_files = sorted(RESULTS_DIR.glob(f"{name}_*@*.pt"))
        all_files = sorted(set(exact_files + prefix_files))

        assert len(all_files) >= 3, (
            f"{name}: expected at least 3 results, found {len(all_files)}"
        )

        # Group by sub-function name
        sub_funcs = sorted(set(f.name.split("@")[0] for f in all_files))

        for sub in sub_funcs:
            sub_files = sorted(RESULTS_DIR.glob(f"{sub}@*.pt"))
            assert len(sub_files) == 3, (
                f"{sub}: expected 3 results, found {len(sub_files)}"
            )

            data = {
                f.name.split("@")[1].replace(".pt", ""): torch.load(
                    f, weights_only=False
                )
                for f in sub_files
            }

            scenes = list(data.keys())
            base_scene = scenes[0]
            base_val = data[base_scene]

            for scene in scenes:
                val = data[scene]

                if isinstance(val, torch.Tensor):
                    v_current = val.detach().cpu().float()
                    v_base = base_val.detach().cpu().float()
                    is_match = torch.allclose(v_current, v_base, rtol=1e-4, atol=1e-4)
                    if not is_match:
                        max_diff = (v_current - v_base).abs().max().item()
                        pytest.fail(
                            f"{sub}: {scene} vs {base_scene} mismatch, "
                            f"max diff = {max_diff:.2e}"
                        )
                else:
                    if val != base_val:
                        pytest.fail(f"{sub}: {scene}={val} != {base_scene}={base_val}")
