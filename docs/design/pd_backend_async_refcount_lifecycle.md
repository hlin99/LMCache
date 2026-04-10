# PDBackend Async KV Transfer — MemoryObj RefCount Lifecycle

> Commit: `8cc2719464296fa17fb206afb0bf01cdba740c29`
> feat(PDBackend): enable async KV transfer mode

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Sender-Side RefCount State Machine](#2-sender-side-refcount-state-machine)
3. [Receiver-Side RefCount State Machine](#3-receiver-side-refcount-state-machine)
4. [Failed Chunk Release Path](#4-failed-chunk-release-path)
5. [Timeline Summary](#5-timeline-summary)
6. [Key Design Invariants](#6-key-design-invariants)
7. [Inflight Flow-Control Counters](#7-inflight-flow-control-counters)
8. [Before / After: `remove()` Comparison](#8-before--after-remove-comparison)

---

## 1. Architecture Overview

```
┌──────────────────────┐         RDMA / NIXL          ┌──────────────────────┐
│  Sender (Prefiller)  │ ==========================>  │  Receiver (Decoder)  │
│                      │                              │                      │
│  allocate()          │                              │  allocate()          │
│  from_gpu()          │                              │  put(key, obj)       │
│  batched_put()       │                              │  get_blocking()      │
│  async_transfer()    │                              │  remove()            │
└──────────────────────┘                              └──────────────────────┘
         │                                                     │
         │  staging buffer pool                                │  receiver buffer pool
         │  (PagedCpuGpuMemoryAllocator)                       │  (PagedCpuGpuMemoryAllocator)
         └─────────────────────────────────────────────────────┘
```

Each `MemoryObj` maps to one or more physical pages in a paged buffer pool.
When `ref_count` drops to **0**, pages are returned to the pool and become
available for subsequent `allocate()` calls.

---

## 2. Sender-Side RefCount State Machine

```
                     ┌──────────────────────────────────┐
                     │  ① PDBackend.allocate()           │
                     │     rc = 1                        │
                     │     (page allocated from staging   │
                     │      buffer pool)                  │
                     │     _sender_inflight_chunks++      │
                     └───────────────┬──────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │  ② gpu_connector                  │
                     │     .batched_from_gpu()            │
                     │     rc = 1                        │
                     │     (copy GPU KV cache → CPU obj)  │
                     └───────────────┬──────────────────┘
                                     │
                     ╔═══════════════▼═════════════════╗
                     ║  ③ storage_manager               ║
                     ║     .batched_put()               ║
                     ║                                  ║
                     ║  ┌────────────────────────────┐  ║
                     ║  │ 3a. batched_submit_         │  ║
                     ║  │     put_task()              │  ║
                     ║  │                             │  ║
                     ║  │   ref_count_up()            │  ║
                     ║  │   rc: 1 → 2                 │  ║
                     ║  │                             │  ║
                     ║  │   (keep-alive: the async    │  ║
                     ║  │    task still needs the obj) │  ║
                     ║  │   (fire-and-forget return)   │  ║
                     ║  └────────────────────────────┘  ║
                     ║                                  ║
                     ║  ┌────────────────────────────┐  ║
                     ║  │ 3b. end of batched_put()   │  ║
                     ║  │     ref_count_down()        │  ║
                     ║  │     rc: 2 → 1               │  ║
                     ║  │                             │  ║
                     ║  │   (caller releases its own  │  ║
                     ║  │    reference to the obj)    │  ║
                     ║  │   storage_manager.py L432   │  ║
                     ║  └────────────────────────────┘  ║
                     ╚═══════════════╤═════════════════╝
                                     │
                                     │  rc = 1, held only by the async transfer task
                                     ▼
           ┌──────────────────────────────────────────────────────┐
           │  ④ _async_transfer_task()                            │
           │     (runs on the _sender_loop asyncio event loop)    │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ Path A: already_sent (duplicate key)           │  │
           │  │   ref_count_down()                             │  │
           │  │   rc: 1 → 0  ──→  page returned to pool ✓     │  │
           │  └────────────────────────────────────────────────┘  │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ Path B: remote_alloc returned -1               │  │
           │  │         (receiver allocation failure)           │  │
           │  │   ref_count_down()                             │  │
           │  │   rc: 1 → 0  ──→  page returned to pool ✓     │  │
           │  └────────────────────────────────────────────────┘  │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ Path C: normal transfer                        │  │
           │  │   async_batched_write()    (RDMA write)        │  │
           │  │   rc = 1 (in-flight)                           │  │
           │  │         │                                      │  │
           │  │         ▼                                      │  │
           │  │   ref_count_down()                             │  │
           │  │   rc: 1 → 0  ──→  page returned to pool ✓     │  │
           │  └────────────────────────────────────────────────┘  │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ Path D: exception                              │  │
           │  │   for each NOT-YET-released obj:               │  │
           │  │     ref_count_down()                           │  │
           │  │     rc: 1 → 0  ──→  prevents leak ✓           │  │
           │  └────────────────────────────────────────────────┘  │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ finally (executes on ALL paths):               │  │
           │  │   _chunk_semaphore.release()                   │  │
           │  │   _release_sender_staging_chunks()             │  │
           │  │     → _sender_inflight_chunks -= N             │  │
           │  │     → notify_all()                             │  │
           │  │       (unblocks waiting allocate() callers)    │  │
           │  └────────────────────────────────────────────────┘  │
           └──────────────────────────────────────────────────────┘
```

---

## 3. Receiver-Side RefCount State Machine

```
           ┌──────────────────────────────────────────────────────┐
           │  ① _async_allocate_and_put()                         │
           │     (runs on _recv_loop, handling sender's request)   │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ 1a. allocate()                                 │  │
           │  │     rc = 1                                     │  │
           │  │     _inflight_chunks += 1                      │  │
           │  │     (empty page from receiver buffer pool)     │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │  ┌─────────────────▼─────────────────────────────┐  │
           │  │ 1b. put(key, mem_obj)                         │  │
           │  │     self.data[key] = mem_obj                   │  │
           │  │     rc = 1                                     │  │
           │  │     (RDMA target address = this obj's page)    │  │
           │  └────────────────────────────────────────────────┘  │
           └──────────────────────────────────────────────────────┘
                                     │
                                     │  ═══ RDMA write completes ═══
                                     │  sender notifies proxy → proxy notifies decoder vLLM
                                     │
                                     ▼
           ┌──────────────────────────────────────────────────────┐
           │  ② cache_engine.retrieve()                           │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ 2a. storage_manager.batched_get()              │  │
           │  │     → PDBackend.get_blocking(key)              │  │
           │  │     fetches mem_obj from self.data[key]         │  │
           │  │     rc = 1  (no ref_count_up; PD does not pin) │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │  ┌─────────────────▼─────────────────────────────┐  │
           │  │ 2b. gpu_connector.batched_to_gpu()            │  │
           │  │     copy CPU MemoryObj contents back to GPU     │  │
           │  │     rc = 1                                     │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │                    ▼                                 │
           │  ┌───────────────────────────────────────────────┐   │
           │  │ 2c. release phase (cache_engine.py L860-867)  │   │
           │  │                                               │   │
           │  │  if remove_after_retrieve:     ← PD path      │   │
           │  │    storage_manager.remove(key)                │   │
           │  │      └→ PDBackend.remove(key):               │   │
           │  │           with data_lock:                     │   │
           │  │             data.pop(key)                     │   │
           │  │             mem_obj.ref_count_down()          │   │
           │  │               rc: 1 → 0  → page freed ✓      │   │
           │  │             _notify_inflight_freed()          │   │
           │  │               _inflight_chunks -= 1          │   │
           │  │               notify_all()                   │   │
           │  │                                               │   │
           │  │  elif not async_loading:       ← non-PD path  │   │
           │  │    mem_obj.ref_count_down()                   │   │
           │  │      rc: 1 → 0  → page freed ✓               │   │
           │  │                                               │   │
           │  │  ⚠️  if / elif are MUTUALLY EXCLUSIVE         │   │
           │  │     to prevent double-free                    │   │
           │  └───────────────────────────────────────────────┘   │
           └──────────────────────────────────────────────────────┘
```

---

## 4. Failed Chunk Release Path

When `_process_tokens_internal()` discovers that a block retrieval failed,
it must discard all chunks with `end >= last_failed_block_start`.
**These chunks will never reach the normal retrieve → remove path
and must be freed in-place.**

```
  _process_tokens_internal() detects retrieval failure
  last_failed_block_start = S
            │
            ▼
  iterate over reordered_chunks: (key, mem_obj, start, end)
            │
            ├── end < S ?
            │     │
            │     YES → keep; will be released normally via step 2c
            │
            └── end >= S ?
                  │
                  │  this chunk will NOT be used; must free immediately
                  │
                  ├── remove_after_retrieve == True ?
                  │     │
                  │     └→ storage_manager.remove(key)
                  │          └→ PDBackend.remove():
                  │               data.pop(key)
                  │               ref_count_down()     rc: 1 → 0 ✓
                  │               _notify_inflight_freed()
                  │
                  └── remove_after_retrieve == False ?
                        │
                        └→ mem_obj.ref_count_down()    rc: 1 → 0 ✓


  ⚠️  BEFORE this commit (old code):
      A list comprehension silently filtered out these chunks
      WITHOUT freeing them → MEMORY LEAK
      Under high-throughput async workloads the buffer pool
      would be exhausted quickly.
```

---

## 5. Timeline Summary

### Sender Side (store path)

```
time ────────────────────────────────────────────────────────────→

 ① allocate()                rc = 1
                                 │
 ② from_gpu()                    │  rc = 1
                                 │
 ③ ref_count_up()             rc = 2  ← async task keep-alive
                                 │
 ④ batched_put tail            rc = 1  ← caller releases its ref
   ref_count_down()              │
                                 │
   ───── async transfer ─────    │  rc = 1
                                 │
 ⑤ transfer completes            │
   ref_count_down()           rc = 0  ← page returned to pool ✓
                                 │
                            [ FREED ]
```

### Receiver Side (retrieve path)

```
time ────────────────────────────────────────────────────────────→

 ① allocate() + put()         rc = 1     inflight++
                                 │
   ═══ RDMA write ═══            │  rc = 1
                                 │
 ② get_blocking()                │  rc = 1
   batched_to_gpu()              │  rc = 1
                                 │
 ③ PDBackend.remove()            │
   data.pop + ref_count_down  rc = 0     inflight--
   _notify_inflight_freed()      │
                                 │
                            [ FREED ]
```

---

## 6. Key Design Invariants

| # | Rule | Rationale |
|---|------|-----------|
| 1 | `batched_submit_put_task()` calls `ref_count_up()` at entry | Fire-and-forget: the caller will `ref_count_down()` soon, but the obj is still needed by the in-flight RDMA transfer |
| 2 | `storage_manager.batched_put()` calls `ref_count_down()` at the end | The caller (`cache_engine.store`) releases its own reference to the obj |
| 3 | `_async_transfer_task()` calls `ref_count_down()` on completion | Transfer is done; release the async task's hold on the obj so the staging buffer page returns to pool |
| 4 | `PDBackend.remove()` calls `ref_count_down()` internally | Atomicity: `pop + free + inflight decrement` must happen under the same lock to keep counters consistent |
| 5 | `cache_engine.py` uses `elif` instead of `if` | Prevents **double-free**: `remove()` already called `ref_count_down()`, so the `elif` guard skips the second decrement |
| 6 | Dropped chunks are freed explicitly | Discarded chunks never reach the normal `retrieve → remove` path and must be freed in-place to prevent **memory leaks** |
| 7 | Exception path checks `completed_indexes` | Only releases objects that have NOT yet been released, preventing double-free on error |

---

## 7. Inflight Flow-Control Counters

### Sender Side

```
                     ┌───────────────────────────────────┐
                     │  _sender_inflight_chunks           │
                     │  (guarded by threading.Condition)   │
                     └───────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
     allocate()             _release_sender_           allocate()
     succeeds → +1          staging_chunks()           BLOCKS
                            transfer done → -N         when >= max
                            notify_all()
```

### Receiver Side

```
                     ┌───────────────────────────────────┐
                     │  _inflight_chunks                  │
                     │  (guarded by asyncio.Condition)     │
                     └───────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  _async_allocate_            remove()               _async_allocate_
  and_put()                   is called:             and_put()
  alloc succeeds → +1        _notify_inflight_      AWAITS
                              freed()               when >= max
                              -1 + notify_all()
```

### End-to-End Flow-Control Closed Loop

```
sender allocate  →  staging_inflight +1
       │
       ▼
async_transfer   →  remote_allocate  →  receiver allocate  →  inflight +1
       │                                                          │
       ▼                                                          │
RDMA write done  →  staging_inflight -1                           │
                    (notify_all)                                   │
                                                                  ▼
                                    cache_engine.retrieve()  →  to_gpu
                                                                  │
                                                                  ▼
                                    PDBackend.remove()  →  inflight -1
                                                           ref_count_down
                                                           (notify_all)
                                                                  │
                                                                  ▼
                                                       [ page returned to pool ]
                                                       [ blocked allocate() unblocks ]
```

---

## 8. Before / After: `remove()` Comparison

### Before (old sync version)

```python
def remove(self, key, ...):
    # TODO(Jiayi): The logic here is confusing. Ref count down
    # will be done after this function call in cache engine.
    with self.data_lock:
        if mem_obj := self.data.get(key, None):
            if mem_obj.get_ref_count() == 1:
                del self.data[key]
            return True
        return False
```

```python
# cache_engine.py (old):
if self.remove_after_retrieve:
    self.storage_manager.remove(key, ...)   # only deletes dict entry
if not self.async_loading:                  # ← if, NOT elif
    memory_obj.ref_count_down()             # engine frees the page
```

- `remove()` only conditionally deletes the dict entry.
- `ref_count_down()` is the caller's responsibility.
- No inflight counter management.

### After (new async version)

```python
def remove(self, key, ...):
    with self.data_lock:
        mem_obj = self.data.pop(key, None)
        if mem_obj is not None:
            mem_obj.ref_count_down()                    # ① free internally
            if self.pd_config.role == "receiver":
                asyncio.run_coroutine_threadsafe(
                    self._notify_inflight_freed(),      # ② decrement inflight
                    self._recv_loop,
                )
            return True
        return False
```

```python
# cache_engine.py (new):
if self.remove_after_retrieve:
    self.storage_manager.remove(key, ...)   # internally calls ref_count_down
elif not self.async_loading:                # ← elif! prevents double-free
    memory_obj.ref_count_down()
```

**Core change**: `remove()` evolves from "delete dict entry only, leave
freeing to the caller" to "atomically delete + free + notify flow control".
`cache_engine.py` changes `if` to `elif` to match.
