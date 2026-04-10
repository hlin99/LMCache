# PDBackend MemoryObj RefCount State Machine

> Describes the complete lifecycle of `MemoryObj.ref_count` across the
> Sender (Prefiller) and Receiver (Decoder) in async PD transfer mode.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Sender-Side State Machine](#2-sender-side-state-machine)
3. [Receiver-Side State Machine](#3-receiver-side-state-machine)
4. [Failed Chunk Release Path](#4-failed-chunk-release-path)
5. [Timeline Summary](#5-timeline-summary)
6. [Inflight Flow-Control Counters](#6-inflight-flow-control-counters)
7. [Invariants](#7-invariants)

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

Each `MemoryObj` maps to physical pages in a paged buffer pool.
When `ref_count` drops to **0**, the pages are returned to the pool
and become available for subsequent `allocate()` calls.

---

## 2. Sender-Side State Machine

A single `MemoryObj` on the sender goes through the following states:

```
                     ┌──────────────────────────────────┐
                     │  ① PDBackend.allocate()           │
                     │     rc = 1                        │
                     │     _sender_inflight_chunks++     │
                     │     (page claimed from staging    │
                     │      buffer pool)                 │
                     └───────────────┬──────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │  ② gpu_connector                  │
                     │     .batched_from_gpu()            │
                     │     rc = 1                        │
                     │     (GPU KV cache copied into obj) │
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
                     ║  │   keep-alive: the async     │  ║
                     ║  │   task still needs the obj  │  ║
                     ║  │   (fire-and-forget return)  │  ║
                     ║  └────────────────────────────┘  ║
                     ║                                  ║
                     ║  ┌────────────────────────────┐  ║
                     ║  │ 3b. batched_put() tail     │  ║
                     ║  │     ref_count_down()        │  ║
                     ║  │     rc: 2 → 1               │  ║
                     ║  │                             │  ║
                     ║  │   caller releases its ref   │  ║
                     ║  └────────────────────────────┘  ║
                     ╚═══════════════╤═════════════════╝
                                     │
                                     │  rc = 1, held only by the async task
                                     ▼
           ┌──────────────────────────────────────────────────────┐
           │  ④ _async_transfer_task()                            │
           │     (runs on _sender_loop event loop)                │
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
           │  │ finally (ALL paths):                           │  │
           │  │   _chunk_semaphore.release()                   │  │
           │  │   _release_sender_staging_chunks()             │  │
           │  │     → _sender_inflight_chunks -= N             │  │
           │  │     → notify_all()                             │  │
           │  │       (unblocks waiting allocate() callers)    │  │
           │  └────────────────────────────────────────────────┘  │
           └──────────────────────────────────────────────────────┘
```

---

## 3. Receiver-Side State Machine

### 3.1 Allocation & RDMA Receive

```
           ┌──────────────────────────────────────────────────────┐
           │  ① _async_allocate_and_put()                         │
           │     (runs on _recv_loop event loop)                   │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ 1a. allocate()                                 │  │
           │  │     rc = 1                                     │  │
           │  │     _inflight_chunks += 1                      │  │
           │  │     (empty page from receiver buffer pool)     │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │  ┌─────────────────▼──────────────────���──────────┐  │
           │  │ 1b. put(key, mem_obj)                         │  │
           │  │     self.data[key] = mem_obj                   │  │
           │  │     rc = 1                                     │  │
           │  │     (RDMA target address = this obj's page)    │  │
           │  └────────────────────────────────────────────────┘  │
           └──────────────────────────────────────────────────────┘
                                     │
                                     │  ═══ RDMA write completes ═══
                                     ▼
```

### 3.2 Retrieve & Release

```
           ┌──────────────────────────────────────────────────────┐
           │  ② cache_engine.retrieve()                           │
           │                                                      │
           │  ┌────────────────────────────────────────────────┐  │
           │  │ 2a. PDBackend.get_blocking(key)                │  │
           │  │     fetches mem_obj from self.data[key]         │  │
           │  │     rc = 1  (no ref_count_up; PD does not pin) │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │  ┌─────────────────▼─────────────────────────────┐  │
           │  │ 2b. gpu_connector.batched_to_gpu()            │  │
           │  │     CPU MemoryObj → GPU KV buffer              │  │
           │  │     rc = 1                                     │  │
           │  └──────────────────┬─────────────────────────────┘  │
           │                    │                                 │
           │                    ▼                                 │
           │  ┌───────────────────────────────────────────────┐   │
           │  │ 2c. release (cache_engine.py L860-867)        │   │
           │  │                                               │   │
           │  │  if remove_after_retrieve:     ← PD path      │   │
           │  │    storage_manager.remove(key)                │   │
           │  │      └→ PDBackend.remove(key):               │   │
           │  │           data.pop(key)                       │   │
           │  │           mem_obj.ref_count_down()            │   │
           │  │             rc: 1 → 0  → page freed ✓        │   │
           │  │           _notify_inflight_freed()            │   │
           │  │             _inflight_chunks -= 1             │   │
           │  │             notify_all()                      │   │
           │  │                                               │   │
           │  │  elif not async_loading:       ← non-PD path  │   │
           │  │    mem_obj.ref_count_down()                   │   │
           │  │      rc: 1 → 0  → page freed ✓               │   │
           │  │                                               │   │
           │  │  if / elif mutually exclusive                 │   │
           │  │  → prevents double-free                       │   │
           │  └───────────────────────────────────────────────┘   │
           └──────────────────────────────────────────────────────┘
```

---

## 4. Failed Chunk Release Path

When `_process_tokens_internal()` detects a block retrieval failure at
position `S`, all chunks with `end >= S` are discarded.
These chunks never reach the normal retrieve → remove path
and **must be freed in-place**.

```
  _process_tokens_internal()
  last_failed_block_start = S
            │
            ▼
  for (key, mem_obj, start, end) in reordered_chunks:
            │
            ├── end < S ?
            │     │
            │     YES → keep; released later via normal path (§3.2 step 2c)
            │
            └── end >= S ?
                  │
                  │  chunk will NOT be used; free immediately
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
```

---

## 5. Timeline Summary

### Sender (store path)

```
time ────────────────────────────────────────────────────────→

 ① allocate()                rc = 1
                                 │
 ② from_gpu()                    │  rc = 1
                                 │
 ③ ref_count_up()             rc = 2    async task keep-alive
                                 │
 ④ batched_put tail            rc = 1    caller drops its ref
   ref_count_down()              │
                                 │
   ───── async transfer ─────    │  rc = 1
                                 │
 ⑤ transfer done               rc = 0    page returned to pool ✓
   ref_count_down()              │
                            [ FREED ]
```

### Receiver (retrieve path)

```
time ────────────────────────────────────────────────────────→

 ① allocate() + put()        rc = 1     inflight++
                                 │
   ═══ RDMA write ═══            │  rc = 1  (data lands in page)
                                 │
 ② get_blocking()                │  rc = 1
   batched_to_gpu()              │  rc = 1
                                 │
 ③ PDBackend.remove()          rc = 0     inflight--
   data.pop                      │         notify_all()
   ref_count_down()              │
                            [ FREED ]
```

---

## 6. Inflight Flow-Control Counters

### 6.1 Sender

```
                     ┌───────────────────────────────────┐
                     │  _sender_inflight_chunks           │
                     │  (threading.Condition)              │
                     └───────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
     allocate()             _release_sender_           allocate()
     succeeds               staging_chunks()           BLOCKS
       +1                   transfer done              when count
                              -N                       >= max
                            notify_all()
```

### 6.2 Receiver

```
                     ┌───────────────────────────────────┐
                     │  _inflight_chunks                  │
                     │  (asyncio.Condition)                │
                     └───────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
  _async_allocate_            remove()               _async_allocate_
  and_put()                   called                 and_put()
  alloc succeeds              _notify_inflight_      AWAITS
    +1                        freed()                when count
                                -1                   >= max
                              notify_all()
```

### 6.3 End-to-End Closed Loop

```
sender allocate()  →  sender_inflight +1
       │
       ▼
_async_transfer()  →  remote_allocate  →  receiver allocate()  →  inflight +1
       │                                                              │
       ▼                                                              │
RDMA write done    →  sender_inflight -1                              │
                      notify_all()                                    │
                                                                      ▼
                                       cache_engine.retrieve()  →  to_gpu()
                                                                      │
                                                                      ▼
                                       PDBackend.remove()  →  inflight -1
                                                              ref_count_down()
                                                              notify_all()
                                                                      │
                                                                      ▼
                                                           [ page returned to pool ]
                                                           [ blocked allocate() unblocks ]
```

---

## 7. Invariants

| # | Invariant | Enforced By |
|---|-----------|-------------|
| 1 | Every `ref_count_up()` is balanced by exactly one `ref_count_down()` | `_async_transfer_task` owns the `+1` from `batched_submit_put_task` and always calls `-1` (normal, exception, or finally) |
| 2 | `PDBackend.remove()` is the **sole** owner of the receiver-side `ref_count_down()` | `cache_engine.py` uses `elif` to skip its own `ref_count_down()` when `remove()` was called |
| 3 | `_inflight_chunks` is always decremented when a receiver page is freed | `remove()` atomically calls `ref_count_down()` + `_notify_inflight_freed()` under `data_lock` |
| 4 | Dropped chunks (retrieval failure) are always freed | `_process_tokens_internal()` explicitly calls `remove()` or `ref_count_down()` for every discarded chunk |
| 5 | `ref_count` never goes below 0 | Exception path in `_async_transfer_task` tracks `completed_indexes` to skip already-freed objects |
| 6 | `sender_inflight` is always decremented when a sender page is freed | `_async_transfer_task.finally` always calls `_release_sender_staging_chunks(N)` regardless of success/failure |
| 7 | `ref_count == 0` ⟺ page returned to pool | Enforced by `PagedCpuGpuMemoryAllocator`: `ref_count_down()` reaching 0 triggers the free-list return 
