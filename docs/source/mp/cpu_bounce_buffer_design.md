# CPU Bounce Buffer Path Design (MP mode, non-CUDA)

## Scope

This document describes the non-CUDA CPU bounce-buffer path added for
LMCache multiprocess mode on the `test` branch, adapted for the branch's
`torch_dev` / `torch_device_type` behavior.

The goal is to support KV transfer for non-CUDA devices (for example CPU/XPU/HPU)
without changing the existing CUDA IPC path.

## Why this path exists

The CUDA path uses IPC wrappers around GPU tensors and existing
`REGISTER_KV_CACHE` / `STORE` / `RETRIEVE` request flow.

For non-CUDA tensors, CUDA IPC is not available. The bounce-buffer path
provides a generic protocol where workers:

1. Gather KV blocks into CPU chunk tensors.
2. Send/store those CPU chunks through MP server storage flow.
3. Retrieve CPU chunks and scatter back into device KV tensors.

## Protocol additions

Three request types are introduced for bounce mode:

- `REGISTER_KV_CACHE_BOUNCE`
- `STORE_CPU_CHUNKS`
- `RETRIEVE_CPU_CHUNKS`

These are registered in MP server dispatch and have corresponding payload/response
contracts in multiprocess protocol definitions.

## Core module

`lmcache/v1/multiprocess/cpu_bounce_context.py` provides:

- `compute_kv_layout`
- `gather_chunks_to_cpu`
- `scatter_cpu_chunks_to_kv`
- `CPUBounceContext`

`CPUBounceContext` stores layout metadata required to interpret chunk payloads:
block size, dtype, number of layers, hidden dimension, and MLA/non-MLA format.

## Tensor/chunk contracts

Chunk formats are unchanged relative to previous behavior:

- non-MLA: `[2, num_layers, chunk_tokens, hidden_dim]`
- MLA: `[num_layers, chunk_tokens, hidden_dim]`

Internal gather/scatter uses block-level indexing to avoid token-level slot
expansion and token-wise select/copy operations.

## Layout handling

Supported KV formats in bounce gather/scatter:

- `NL_X_TWO_NB_BS_NH_HS` (NHD)
- `NL_X_NB_TWO_BS_NH_HS` (NHD flashinfer)
- `NL_X_TWO_NB_NH_BS_HS` (HND)
- `NL_X_NB_TWO_NH_BS_HS` (HND flashinfer)
- `NL_X_NB_BS_HS` (MLA)

### Non-MLA (NHD)

Gather:

- Read K/V blocks directly by `chunk_block_ids`.
- Reshape to token-major `[chunk_tokens, NH*HS]`.

Scatter:

- Reshape chunk payload to `[n_blocks, BS, NH, HS]`.
- Assign directly into destination blocks.

### Non-MLA (HND)

Gather:

- Read K/V blocks as `[n_blocks, NH, BS, HS]`.
- Permute to `[n_blocks, BS, NH, HS]` then flatten to token-major.

Scatter:

- Reshape payload to `[n_blocks, BS, NH, HS]`.
- Permute to `[n_blocks, NH, BS, HS]` and assign by block index.

### MLA

Gather/scatter is direct block reshape between `[NB, BS, HS]` and
`[chunk_tokens, HS]` representation.

## Worker adapter integration

`lmcache/integration/vllm/vllm_multi_process_adapter.py` chooses path by tensor
`device.type`:

- all CUDA -> existing CUDA IPC registration and store/retrieve path
- all non-CUDA -> bounce registration and CPU chunk store/retrieve path

Adapter enforces uniform device type across all layer tensors.

## Server integration

`MPCacheEngine` adds bounce registries and handlers for registration/store/retrieve.

Additional integration points:

- unregister cleanup also removes bounce context metadata
- layout lookup can resolve both classic GPU registration and bounce registration
- status reporting includes bounce context metadata

## Runtime behavior summary

### Store path (non-CUDA)

1. Adapter gathers chunk tensors from KV tensors.
2. Adapter calls `torch_dev.synchronize()` before submit.
3. `STORE_CPU_CHUNKS` sends CPU chunks to server.
4. Server stores chunks into LMCache storage pipeline.

### Retrieve path (non-CUDA)

1. Adapter submits `RETRIEVE_CPU_CHUNKS`.
2. Server returns CPU chunk tensors.
3. Adapter scatters chunks back into KV tensors using block-level writes.

## Validation coverage

`tests/v1/multiprocess/test_cpu_bounce_buffer.py` covers:

- bounce wrapper behavior
- NHD and MLA gather/scatter roundtrip
- HND roundtrip for both HND formats
- `skip_first_n_tokens` behavior
- server-side bounce register/store/retrieve flow

## Non-goals

- No change to existing CUDA IPC path semantics.
- No bounce-specific logic added to shared `gpu_connector/utils.py`.
