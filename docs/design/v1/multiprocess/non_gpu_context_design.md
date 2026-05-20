# Non-GPU Context Design (MP mode, non-CUDA)

## 1. Motivation

LMCache multiprocess mode relies on **CUDA IPC** to transfer KV cache data
between vLLM worker processes and the LMCache cache server. The existing
path wraps GPU tensors in `CudaIPCWrapper`, exchanges IPC handles via ZMQ
messages, and uses CUDA events for cross-process synchronisation.

This design is fundamentally tied to the CUDA programming model:

| CUDA IPC dependency | Why it blocks non-CUDA devices |
|---|---|
| `CudaIPCWrapper` / `cudaIpcGetMemHandle` | Only works on NVIDIA CUDA tensors |
| `torch.cuda.Event(interprocess=True)` | CUDA-specific IPC event API |
| `cupy.cuda.ExternalStream` | CUDA stream wrapper |
| GPU pointer arithmetic in C++ kernels | Assumes CUDA device pointers |

For non-CUDA accelerators — **CPU, Intel XPU, Habana HPU**, or any future
device — none of these primitives are available.

The **non-GPU context** path introduces a device-agnostic KV transfer mechanism:

1. Workers **gather** paged KV blocks into contiguous CPU chunk tensors.
2. CPU chunks are **transported** to the server through a pluggable
   serialisation layer (pickle today, shared memory in the future).
3. On retrieve, the server returns CPU chunks and workers **scatter** them
   back into device-local paged KV tensors.

The existing CUDA IPC path is **untouched** — the two paths coexist behind a
polymorphic `TransferContext` abstraction.

### Transport comparison

**Store (worker → server storage):**

| Transport | Copies | Data flow |
|---|---|---|
| CUDA IPC | 2 | GPU KV → GPU staging buffer → CPU memory obj |
| Pickle | 4 | GPU KV → CPU chunk → pickle.dumps → pickle.loads → CPU memory obj |
| SHM (TODO) | 1 | GPU KV → CPU memory obj (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| CUDA IPC | 2 | CPU memory obj → GPU staging buffer → GPU KV |
| Pickle | 4 | CPU memory obj → pickle.dumps → pickle.loads → CPU chunk → GPU KV |
| SHM (TODO) | 1 | CPU memory obj (SHM mapped) → GPU KV |

**Applicability:**

| Transport | Platform requirement | Pros | Cons |
|---|---|---|---|
| CUDA IPC | NVIDIA CUDA devices only | Async GPU streams, mature path | CUDA-only |
| Pickle | Any device, no dependencies | Generally available, zero setup | 4 copies + serialisation overhead |
| SHM (TODO) | `/dev/shm` capacity ≥ L1 cache size | Fewest copies (1), no serialisation | Requires sufficient shared memory |

## 2. Architecture Overview

### 2.1 Layered architecture

```
vllm_multi_process_adapter.py    ← Engine adapter, device-agnostic
  └── TransferContext             ← Worker-side transport abstraction (§3)
        ├── CudaTransferContext    ← CUDA IPC + MQ future path
        └── NonCudaTransferContext     ← Synchronous gather/scatter path
              └── NonGpuContext        ← Serialisation abstraction (§4.2)
                    ├── NonGpuContextPickle   ← pickle.dumps/loads (§4.3)
                    └── NonGpuContextShm      ← shared memory (§4.4, TODO)
```

Two layers of abstraction serve different purposes:

- **TransferContext** (§3) — decides **CUDA vs non-CUDA** routing at the
  worker adapter level.
- **NonGpuContext** (§4.2) — decides **how** CPU chunk data is serialised and
  transported (pickle vs SHM). Only used inside `NonCudaTransferContext`.

### 2.2 State machine (worker ↔ server)

```text
                           register_kv_caches()
                                      |
                                      v
                    create_transfer_context(kv_caches)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              [device == cuda]                 [device != cuda]
                     |                                 |
                     v                                 v
      CudaTransferContext.register()     NonCudaTransferContext.register()
      → REGISTER_KV_CACHE               → REGISTER_KV_CACHE_NON_GPU_CONTEXT
        (CUDA IPC handles)                 (scalar metadata fields)
                     |                         + create_non_gpu_context()
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
       transfer_ctx.submit_store()      transfer_ctx.submit_store()
                     |                                 |
                     v                                 v
           STORE (GPU → L1)            _non_gpu_context.prepare_store()
           [async MQ future]           + gather_paged_kv_to_cpu()
                     |                 + _non_gpu_context.commit_store() [sync]
                     v                 return pre-resolved MessagingFuture
                  [READY]                               |
                     +----------------+----------------+
                                      |
                                      v
      transfer_ctx.submit_retrieve()  +  adapter polls future.query()/result()
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
           RETRIEVE (L1 → GPU)     _non_gpu_context.prepare_retrieve() [sync]
           [async MQ future]       + scatter_cpu_to_paged_kv()
                      |            + _non_gpu_context.commit_retrieve()
                      v            return pre-resolved MessagingFuture
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                                      v
                           unregister_kv_cache()
                                      |
                                      v
                                  [TERMINATED]
```

## 3. Worker-side: TransferContext Abstraction

### 3.1 Problem

Before this refactoring, `vllm_multi_process_adapter.py` contained
non-CUDA-specific branching in every method — `register_kv_caches`,
`submit_store_request`, `submit_retrieve_request`, `get_finished`, and the
unhealthy drain path. Adding a third transport would require touching every
branch.

### 3.2 Solution

`transfer_context.py` defines the `TransferContext` ABC with four methods:
`register`, `submit_store`, `submit_retrieve`, and `close`. Both submit methods
return a `MessagingFuture`; the adapter uses `future.query()` /
`future.result()` directly, so polling behavior is implemented in the adapter
rather than in the ABC.

### 3.3 `create_transfer_context()` factory

Inspects device types of all KV cache tensors **exactly once**. CUDA →
`CudaTransferContext`; otherwise → `NonCudaTransferContext`. Mixed device types
are rejected.

### 3.4 `CudaTransferContext`

Wraps the original CUDA IPC path. Sends `REGISTER_KV_CACHE` / `STORE` /
`RETRIEVE` messages with IPC handles and returns async MQ futures. The adapter
queries those futures and handles unhealthy-drain semantics.

### 3.5 `NonCudaTransferContext`

Holds a `NonGpuContext` instance internally. Sends
`REGISTER_KV_CACHE_NON_GPU_CONTEXT` with scalar metadata. Store and retrieve
are **synchronous**: gather/scatter + prepare/commit, and each submit method
returns a pre-resolved `MessagingFuture[bool]`. The adapter still uses the same
`future.query()` / `future.result()` flow as CUDA.

## 4. Server-side: Non-GPU Context Protocol

### 4.1 Why GPU context and non-GPU context need different protocols

| | GPU context | non-GPU context |
|---|---|---|
| Registration | `REGISTER_KV_CACHE` — IPC handles | `REGISTER_KV_CACHE_NON_GPU_CONTEXT` — scalar fields |
| Store | `STORE` — event handle + block IDs, server reads GPU directly | `PREPARE_STORE` + `COMMIT_STORE` — prepare then send serialised CPU tensors |
| Retrieve | `RETRIEVE` — event handle + block IDs, server writes GPU directly | `PREPARE_RETRIEVE` + `COMMIT_RETRIEVE` — fetch CPU tensors then commit |

Registration uses **scalar fields** (`block_size`, `num_layers`,
`hidden_dim_size`, `dtype_str`, `use_mla`) instead of pickled objects
to avoid cross-process pickle security and compatibility concerns. The
server reconstructs `MemoryLayoutDesc` from the scalars internally.

### 4.2 `NonGpuContext` ABC: two-phase prepare/commit

The serialisation layer is abstracted behind `NonGpuContext` so that pickle
and SHM can be swapped without touching `NonCudaTransferContext` or the server.

The ABC defines: `prepare_store`, `commit_store`, `prepare_retrieve`,
`commit_retrieve`, `close`.

Why two phases? SHM needs prepare to allocate a slot, then the worker writes
into mapped memory, then commit tells the server "ready". Pickle keeps the same
shape for protocol consistency: `prepare_store` is an RPC handshake and
`commit_store` performs serialisation + send.

| Phase | Pickle | SHM (TODO) |
|---|---|---|
| `prepare_store` | MQ `PREPARE_STORE` RPC, returns `None` (no pre-allocated buffers) | MQ `PREPARE_STORE` → get SHM offset → `memcpy` into SHM |
| `commit_store` | `pickle.dumps(chunks)` + MQ `COMMIT_STORE`, block for ack | MQ `COMMIT_STORE` → server reads from SHM |
| `prepare_retrieve` | MQ `PREPARE_RETRIEVE` → `pickle.loads` | MQ `PREPARE_RETRIEVE` → server writes to SHM → map tensor views |
| `commit_retrieve` | no-op | MQ `FINISH_READ` → release SHM read lock |

`create_non_gpu_context()` factory currently always returns `NonGpuContextPickle`.
Future: probe `/dev/shm` availability and capacity, fall back to pickle if
insufficient.

## Non-goals

- No change to existing CUDA IPC path semantics.
- No CPU-specific logic added to shared `gpu_connector/utils.py`.
- No wire-protocol incompatibility between CUDA and non-GPU context workers in
  the same cluster.
