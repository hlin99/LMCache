# PD Async Reservation-Based Admission Control

## Problem

In chunked-prefill mode, large prompts are split into multiple sequential batches. Each batch allocates physical buffer slots on the receiver before the RDMA write. With multiple concurrent requests, this interleaved allocation creates a deadlock: N requests each partially fill the staging buffer, none can complete their full chunk set, and the buffer never drains.

**Example (buffer = 10 chunks):**
```
Req A needs 8 chunks, Req B needs 8 chunks
A allocates 5 → B allocates 5 → buffer full
A needs 3 more → blocked, B needs 3 more → blocked → DEADLOCK
```

## Solution: Reservation-Based Admission

Reserve `total_chunks` upfront per request before any physical allocation. If the buffer can't accommodate the full reservation, the request waits. Once admitted, all subsequent allocations for that request are guaranteed to succeed.

```
Buffer = 10 chunks
A requests admission (8 chunks) → reserved=8, available=2
B requests admission (8 chunks) → 8 > 2 → WAIT
A completes all 8 chunks → RDMA done → release reservation → available=10
B admitted → reserved=8, proceeds
```

## Architecture

### Components

```
┌─────────────────────────────────────────────┐
│                vLLM Worker Thread            │
│  wait_for_save()                            │
│    ├─ try_admit(req_id, total_chunks)  ←─── blocks until buffer available
│    ├─ store() × N batches              ←─── allocate + from_gpu per batch
│    │   └─ batched_submit_put_task()    ←─── submits to sender loop, returns immediately
│    └─ (no release here)                     │
└─────────────────────────────────────────────┘
              │ asyncio.run_coroutine_threadsafe
              ▼
┌─────────────────────────────────────────────┐
│             Sender Event Loop               │
│  _async_transfer_task() (concurrent)        │
│    ├─ _async_remote_allocate()  ←────────── ZMQ REQ/REP to receiver
│    ├─ async_batched_write()     ←────────── RDMA write
│    ├─ ref_count_down()          ←────────── free sender staging buffer
│    ├─ _notify_staging_freed()   ←────────── wake allocate() waiters
│    └─ _check_and_send_proxy_notif()         │
│         ├─ ProxyNotif via ZMQ PUSH          │
│         └─ release_reservation()  ←──────── free reservation AFTER RDMA done
└─────────────────────────────────────────────┘
              │ ZMQ DEALER/ROUTER
              ▼
┌─────────────────────────────────────────────┐
│           Receiver Event Loop               │
│  _handle_alloc_request()                    │
│    ├─ CancelNotif → release keys + reservation
│    └─ AllocRequest → _async_allocate_and_put()
│         ├─ async_try_admit() (first batch)  │
│         ├─ allocate() per chunk             │
│         └─ put() → register KV object       │
└─────────────────────────────────────────────┘
```

### ReservationManager

Shared class used by both sender and receiver with threading and asyncio variants:

- **Sender** (threading): `try_admit()` / `release_reservation()` — called from vLLM worker threads
- **Receiver** (asyncio): `async_try_admit()` / `async_release_reservation()` — called from receiver event loop

Key invariant: `total_reserved <= total_chunks` at all times.

### Lock Inventory

| Path | Lock | Type | Purpose |
|------|------|------|---------|
| Worker thread (sender) | `_reservation_mgr._threading_condition` | threading.Condition | Admission wait |
| Worker thread (sender) | `_reservation_mgr._abort_lock` | threading.Lock | Abort flag check |
| Sender loop | `_async_alloc_locks[receiver_id]` | asyncio.Lock | Serialize ZMQ to same receiver |
| Sender loop | `_staging_condition` | threading.Condition | Wake allocate() waiters |
| Sender loop | `_proxy_send_lock` | asyncio.Lock | Serialize ProxyNotif sends |
| Receiver loop | `_recv_reservation_mgr._async_condition` | asyncio.Condition | Async admission wait |

### ProxyNotif Ordering

Since batches run concurrently, `is_last_prefill` batch may complete RDMA before earlier batches. ProxyNotif fires only when BOTH:
1. `completed_chunks >= total_chunks` (all RDMA done), or `total_chunks == 0` (legacy sender — see below)
2. `req_has_last == True` (is_last_prefill batch completed)

For **legacy senders** (`total_chunks == 0`), no reservation tracking exists so ProxyNotif fires immediately when the `is_last_prefill` batch completes, regardless of how many chunks were transferred. A warning is logged when a legacy sender is detected on the receiver side.

### Abort Flow

```
request_finished(ABORTED)
  → cancel_request(req_id)           # any thread
    → mark_abort(req_id)             # unblocks try_admit if waiting
    → wake _staging_condition        # unblocks allocate if waiting
    → schedule _abort_request()      # on sender loop
      → CancelNotif to receiver      # release remote keys
      → release_reservation()        # free sender buffer
      → clear per-request state
```

### Message Types

| Message | Direction | Purpose |
|---------|-----------|---------|
| AllocRequest | sender → receiver | Allocate remote buffer slots. `total_chunks` field for first-batch reservation |
| AllocResponse | receiver → sender | Remote buffer addresses (-1 on failure) |
| ProxyNotif | sender → proxy | All RDMA done, decoder can start consuming |
| CancelNotif | sender → receiver | Request aborted, release allocated keys |

### Concurrency Model (2P1D example)

```
Buffer = 20 chunks, P1 req A (10), P2 req B (10)

Old (serial admission):
  P1: [=== transfer A ===]
  P2: [     wait          ][=== transfer B ===]
  Total: A + B

New (reservation):
  P1: [=== transfer A ===]
  P2: [=== transfer B ===]    ← concurrent RDMA from different peers
  Total: max(A, B)
```

## Receiver Admission Control

The receiver uses a single `ReservationManager` (`_recv_reservation_mgr`) as the sole source of truth for buffer capacity. On the first batch for each request, the receiver calls `async_try_admit(req_id, total_chunks)` to reserve the full chunk set. Subsequent batches for the same request draw from the existing reservation and allocate immediately.

Legacy senders that do not set `total_chunks` (i.e., `total_chunks == 0`) skip the reservation path. A warning is logged when such senders are detected, as they provide no deadlock protection. Legacy senders are deprecated; all new senders should pass `total_chunks`.

The reservation is released on the receiver side when the last batch of a request is successfully allocated (signaled by `is_last_batch == True` in the `AllocRequest`), or on abort via `CancelNotif`.

## Fail-Fast Detection

If the cumulative number of chunks for a request exceeds the total buffer capacity (`_recv_reservation_mgr._total_chunks`), the receiver raises a `RuntimeError` immediately rather than waiting for an allocation that can never succeed. This prevents indefinite blocking and provides a clear diagnostic message indicating that `pd_buffer_size` should be increased or chunk size reduced.
