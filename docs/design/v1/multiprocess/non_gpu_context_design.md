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
| Pickle | 3 | GPU KV → CPU chunk → pickle.dumps → (MQ send) → pickle.loads → CPU memory obj |
| SHM (TODO) | 1 | GPU KV → CPU memory obj (SHM mapped) |

**Retrieve (server storage → worker):**

| Transport | Copies | Data flow |
|---|---|---|
| CUDA IPC | 2 | CPU memory obj → GPU staging buffer → GPU KV |
| Pickle | 3 | CPU memory obj → (MQ send) → pickle.loads → CPU chunk → GPU KV |
| SHM (TODO) | 1 | CPU memory obj (SHM mapped) → GPU KV |

**Applicability:**

| Transport | Platform requirement | Pros | Cons |
|---|---|---|---|
| CUDA IPC | NVIDIA CUDA devices only | Async GPU streams, mature path | CUDA-only |
| Pickle | Any device, no dependencies | Generally available, zero setup | 3 copies + serialisation overhead, one-way data size = KV size |
| SHM (TODO) | `/dev/shm` capacity ≥ L1 cache size | Fewest copies (1), no serialisation | Requires sufficient shared memory |

## 2. Architecture Overview

### 2.1 Layered architecture

```
vllm_multi_process_adapter.py    ← Engine adapter, device-agnostic
  └── TransferContext             ← Worker-side transport abstraction (§3)
        ├── CudaTransferContext    ← CUDA IPC + MQ future path
        └── NonCudaTransferContext     ← Synchronous gather/scatter path
              └── NonGpuContext        ← Serialisation abstraction (§4)
                    ├── NonGpuContextPickle   ← pickle.dumps/loads (§4.3)
                    └── NonGpuContextShm      ← shared memory (§4.4, TODO)
```

Two layers of abstraction serve different purposes:

- **TransferContext** (§3) — decides **CUDA vs non-CUDA** routing at the
  worker adapter level.
- **NonGpuContext** (§4) — decides **how** CPU chunk data is serialised and
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
           STORE (GPU → L1)            gather_paged_kv_to_cpu()
           [async MQ future]           + prepare_store()  → PREPARE_STORE
                                          + commit_store() → COMMIT_STORE (pickled bytes)
                                          ✓ _store_done[id] = ok
                 [READY]                               |
                     +----------------+----------------+
                                      |
                                      v
      transfer_ctx.submit_retrieve()  + prepare_retrieve() → PREPARE_RETRIEVE
                                      |    (server returns pickled bytes)
                                      v    
          RETRIEVE (L1 → GPU)          + scatter_cpu_to_paged_kv()
          [async MQ future]                 + commit_retrieve() → COMMIT_RETRIEVE
                                          ✓ _retrieve_done[id] = (ok, block_ids)
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

`transfer_context.py` defines the `TransferContext` ABC with six methods:
`register`, `submit_store`, `submit_retrieve`, `poll_finished`, `drain_all`,
and `close`. The adapter holds a single `TransferContext` and delegates —
no `if/else` anywhere.

### 3.3 `create_transfer_context()` factory

Inspects device types of all KV cache tensors **exactly once**. CUDA →
`CudaTransferContext`; otherwise → `NonCudaTransferContext`. Mixed device types
are rejected.

### 3.4 `CudaTransferContext`

Wraps the original CUDA IPC path. Sends `REGISTER_KV_CACHE` / `STORE` /
`RETRIEVE` messages with IPC handles, tracks async MQ futures.
`poll_finished` queries futures; `drain_all` marks all pending as finished
for unhealthy shutdown. Semantics identical to pre-refactoring.

### 3.5 `NonCudaTransferContext`

Holds a `NonGpuContext` instance internally. Sends
`REGISTER_KV_CACHE_NON_GPU_CONTEXT` with scalar metadata. Store and retrieve
are **synchronous**: gather → prepare/commit, then record result in
`_store_done` / `_retrieve_done`. `poll_finished` simply drains these dicts.

## 4. Server-side: Non-GPU Context Protocol

### 4.1 Five new protocol messages

The non-GPU context path introduces five new messages for two-phase store/retrieve:

| Message | Payload | Response | Description |
|---------|---------|----------|-------------|
| `REGISTER_KV_CACHE_NON_GPU_CONTEXT` | `RegisterNonGpuContextPayload` (scalar metadata) | None | Register with scalar fields instead of IPC handles |
| `PREPARE_STORE` | `key, instance_id` | `PrepareStoreResponse(context: dict)` | Allocate resources, returns slot info |
| `COMMIT_STORE` | `key, instance_id, cpu_data: bytes` ` | `bool` | Send serialised data |
| `PREPARE_RETRIEVE` | `key, instance_id` | `PrepareRetrieveResponse(success, data: bytes, context: dict)` | Lookup key, return data |
| `COMMIT_RETRIEVE` | `key, instance_id` | `bool` | Release locks/resources |

### 4.2 Why new messages needed

| GPU context | non-GPU context |
|---|---|
| Uses `REGISTER_KV_CACHE` with IPC handles | Uses `REGISTER_KV_CACHE_NON_GPU_CONTEXT` with scalar fields |
| Single `STORE` message with IPC handle | Two-phase: `PREPARE_STORE` + `COMMIT_STORE` |
| Single `RETRIEVE` message with IPC handle | Two-phase: `PREPARE_RETRIEVE` + `COMMIT_RETRIEVE` |

Why scalar fields for registration? To avoid cross-process pickle security
and compatibility concerns. The server reconstructs `MemoryLayoutDesc` from
the scalars internally.

Two-phase design: Pickle can technically do everything in one step, but
SHM needs prepare to allocate a slot, then worker writes to mapped memory,
then commit tells server "ready". The split accommodates both without forcing
unnecessary round-trips on pickle.

### 4.3 `NonGpuContext` ABC

Abstract base class for serialisation implementations:

```python
class NonGpuContext(ABC):
    def prepare_store(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Send PREPARE_STORE, allocate resources. Returns pre-allocated buffers or None."""

    def commit_store(self, key: Any, instance_id: int, chunks: list[torch.Tensor]) -> bool:
        """Send COMMIT_STORE with serialised data. Returns success."""

    def prepare_retrieve(self, key: Any, instance_id: int) -> list[torch.Tensor] | None:
        """Send PREPARE_RETRIEVE, deserialise response. Returns chunks or None."""

    def commit_retrieve(self, key: Any, instance_id: int) -> bool:
        """Send COMMIT_RETRIEVE for cleanup. Returns success."""

    def allocate_store_buffers(self, size: int) -> list[torch.Tensor] | None:
        """Allocate SHM buffers for store. Default: None (for pickle)."""

    def allocate_retrieve_buffers(self, size: int) -> list[torch.Tensor] | None:
        """Allocate SHM buffers for retrieve. Default: None (for pickle)."""

    def close(self) -> None:
        """Release resources."""
```

### 4.4 Implementation variants

**Pickle (`NonGpuContextPickle`):**
- `prepare_store`: sends `PREPARE_STORE`, returns `None` (no pre-allocated buffers)
- `commit_store`: `pickle.dumps(chunks)` → sends via `COMMIT_STORE`
- `prepare_retrieve`: sends `PREPARE_RETRIEVE`, `pickle.loads(response.data)`
- `commit_retrieve`: sends `COMMIT_RETRIEVE` (no-op for pickle)

**SHM (`NonGpuContextShm`) — TODO:**
- `prepare_store`: allocate SHM slot, return slot info in context
- `commit_store`: confirm write, server reads from SHM
- `prepare_retrieve`: server writes to SHM, return slot info
- `commit_retrieve`: release SHM read lock

`create_non_gpu_context()` factory currently always returns `NonGpuContextPickle`.
Future: probe `/dev/shm` availability and capacity, fall back to pickle if
insufficient.

## 5. Data Path: Gather / Scatter

### 5.1 Chunk format

- **Non-MLA**: `[2, num_layers, chunk_tokens, hidden_dim]` — dim 0 = `(K, V)`.
- **MLA**: `[num_layers, chunk_tokens, hidden_dim]` — single latent vector.

Where `chunk_tokens = blocks_per_chunk × block_size`.

### 5.2 Supported KV layouts

| Format enum | Layout | Shape per layer |
|---|---|---|
| `NL_X_TWO_NB_BS_NH_HS` | NHD | `[2, NB, BS, NH, HS]` |
| `NL_X_NB_TWO_BS_NH_HS` | NHD (flashinfer) | `[NB, 2, BS, NH, HS]` |
| `NL_X_TWO_NB_NH_BS_HS` | HND | `[2, NB, NH, BS, HS]` |
| `NL_X_NB_TWO_NH_BS_HS` | HND (flashinfer) | `[NB, 2, NH, BS, HS]` |
| `NL_X_NB_BS_HS` | MLA | `[NB, BS, HS]` |

### 5.3 Block-level indexing

Gather and scatter operate at **block granularity** (`tensor[block_ids]`)
rather than per-token `index_select` / `index_copy_`. For HND layouts, a
`permute(0, 2, 1, 3)` converts between head-major and token-major order.

### 5.4 Utility functions

- **`compute_kv_layout`** — extracts `(block_size, num_layers, hidden_dim_size, dtype_str, gpu_kv_format)` from live KV tensors.
- **`gather_paged_kv_to_cpu`** — gathers paged blocks into CPU chunk tensors.
- **`scatter_cpu_to_paged_kv`** — scatters CPU chunks back into device paged KV tensors. Respects `skip_first_n_tokens` for partial-prefix retrieval.
- **`gather_paged_kv_to_cpu(out=...)`** — writes directly into pre-allocated buffers (for SHM zero-copy).

## 6. SHM Implementation Details (TODO)

### 6.1 SHM fallback conditions

SHM is the **optimal** transport with only **1 copy**, but requires:

- `/dev/shm` must exist and be writable
- `/dev/shm` capacity ≥ L1 cache size (`worker_l1_cached_chunks * chunk_size_bytes`)

If conditions are not met, automatically fall back to **Pickle** (3 copies).

### 6.2 Zero-copy path with `out=` parameter

The `gather_paged_kv_to_cpu` function accepts an optional `out` parameter:

```python
def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    layout: MemoryLayoutDesc,
    block_ids: list[int],
    out: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if out is not None:
        for chunk, buf in zip(chunks, out, strict=False):
            buf.copy_(chunk, non_blocking=True)
        return out
    return chunks
```

This enables SHM implementations to pre-allocate shared memory buffers and
write directly into them, avoiding an extra copy.

### 6.3 TODO items

- [ ] Implement `NonGpuContextShm` class
- [ ] Implement SHM slot allocation / lifecycle management
- [ ] Add `/dev/shm` capacity check in `create_non_gpu_context()`
- [ ] Implement async SHM read/write with proper locking
- [ ] Add error handling and auto-fallback to Pickle

## Non-goals

- No change to existing CUDA IPC path semantics.
- No CPU-specific logic added to shared `gpu_connector/utils.py`.
- No wire-protocol incompatibility between CUDA and non-GPU context workers in
  the same cluster.