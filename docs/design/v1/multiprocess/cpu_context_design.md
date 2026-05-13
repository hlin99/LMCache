# CPU Context Design (MP mode, non-CUDA)

## Scope

This document describes the high-level design of the non-CUDA KV transfer path
for LMCache multiprocess mode.

The purpose of this path is to support KV transfer on non-CUDA devices (for
example CPU, XPU, and HPU) while preserving existing CUDA IPC behavior.

## Why this path exists

CUDA IPC is only available for CUDA tensors. For non-CUDA tensors, workers use
a CPU-context path that:

1. gathers KV blocks into CPU chunks,
2. transfers those chunks to/from the server through `CPUContext`,
3. scatters retrieved chunks back to worker KV tensors.

## Protocol overview

Non-CUDA mode adds three request types:

- `REGISTER_KV_CACHE_CPU_CONTEXT`
- `STORE_CPU_CHUNKS`
- `RETRIEVE_CPU_CHUNKS`

CPU-context registration uses scalar metadata (for example: `instance_id`,
`model_name`, `world_size`, `block_size`, `num_layers`, `hidden_dim_size`,
`dtype_str`, and `use_mla`) so server-side layout can be reconstructed without
transmitting pickled layout objects, reducing serialization coupling and
allowing server-side validation from explicit fields.

## Main components

- `cpu_context.py`
  - defines `CPUContextMetadata`, the `CPUContext` abstraction, and shared
    gather/scatter helpers.
- `cpu_context_pickle.py`
  - current concrete `CPUContext` implementation.
- `transfer_context.py`
  - dispatches between CUDA and CPU transfer paths.
- `vllm_multi_process_adapter.py`
  - owns request lifecycle and future polling.
- `server.py`
  - stores per-instance CPU metadata and handles CPU chunk store/retrieve
    requests.

## Worker-side behavior

`create_transfer_context(kv_caches)` selects transport by device type:

- CUDA tensors -> `CudaTransferContext`
- non-CUDA tensors -> `CPUTransferContext`

The adapter keeps ownership of request completion tracking via
`store_futures` and `retrieve_futures`.

For CPU mode, store/retrieve execution is synchronous inside
`CPUTransferContext` (gather/scatter plus MQ interaction), and the transfer
methods return resolved futures so adapter-side completion flow stays uniform
across CUDA and non-CUDA modes. Here, "resolved futures" means the futures are
already completed when returned (no background async work pending in the CPU
path).

## Server-side behavior

`MPCacheEngine` maintains CPU-context metadata per worker instance and uses that
metadata to resolve layout for CPU chunk writes/reads.

Server handlers:

- register CPU-context metadata,
- store worker-provided CPU chunks into storage,
- retrieve CPU chunks from storage and return them to workers.

Cleanup removes CPU-context state on unregister.

## Format and compatibility notes

- Chunk tensor layout remains consistent with gather/scatter contracts:
  non-MLA chunks are 4D (`[2, num_layers, chunk_tokens, hidden_dim]`) and MLA
  chunks are 3D (`[num_layers, chunk_tokens, hidden_dim]`).
- Existing CUDA IPC semantics are unchanged.
- CPU-context logic remains isolated from shared GPU connector utilities.

## Validation coverage

Tests cover:

- CPU gather/scatter correctness across supported layouts,
- CPU registration and server store/retrieve flow,
- adapter integration with transfer-context submit/get-finished behavior.

## Future extension

The `CPUContext` abstraction is designed to support additional transports
(e.g. shared-memory-based implementations) with minimal adapter/server flow
changes.
