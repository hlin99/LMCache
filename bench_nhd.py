#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Device-agnostic perf bench for multi_layer_block_kv_transfer (NHD/HND).
Paged buffer on device, LMCache memory objects on CPU (pinned).

- Random block_ids each iteration to avoid cache hit bias
- Per-iteration sync to measure true single-call latency

Usage:
    python tests/v1/bench_block_kv_transfer_nhd.py
"""

# Standard
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.c_ops as lmc_ops

# ---------- Config ----------
NL = 32
NB = 2048
BS = 16
NH = 8
HS = 128
NUM_OBJS = 4
TOKENS_PER_OBJ = 256
BLOCKS_PER_OBJ = TOKENS_PER_OBJ // BS
TOTAL_BLOCKS = NUM_OBJS * BLOCKS_PER_OBJ  # 64

WARMUP = 10
ITERS = 100

DEVICE = torch.device(torch_device_type, torch_dev.current_device())

NHD_FORMATS = [
    ("NHD_flash_attn", lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS),
    ("NHD_flash_infer", lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS),
    ("HND_flash_attn", lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS),
    ("HND_flash_infer", lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS),
]


def make_paged(fmt_enum):
    """Paged KV buffer on device."""
    if fmt_enum == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        shape = [2, NB, BS, NH, HS]
    elif fmt_enum == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        shape = [NB, 2, BS, NH, HS]
    elif fmt_enum == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        shape = [2, NB, NH, BS, HS]
    elif fmt_enum == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        shape = [NB, 2, NH, BS, HS]
    else:
        raise ValueError(f"Unknown: {fmt_enum}")
    return [torch.randn(shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(NL)]


def make_mem_objects():
    """LMCache memory objects on CPU (pinned)."""
    shape = [2, NL, TOKENS_PER_OBJ, NH * HS]
    return [
        torch.zeros(shape, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_OBJS)
    ]


def random_block_ids():
    """Pick TOTAL_BLOCKS random block indices from [0, NB), no repeat."""
    return torch.randperm(NB, device=DEVICE, dtype=torch.int64)[:TOTAL_BLOCKS]


def run_kernel(paged, mem_objs, block_ids_dev, fmt_enum, dir_enum):
    shape_desc = lmc_ops.PageBufferShapeDesc()
    shape_desc.kv_size = 2
    shape_desc.nl = NL
    shape_desc.nb = NB
    shape_desc.bs = BS
    shape_desc.nh = NH
    shape_desc.hs = HS
    shape_desc.element_size = paged[0].element_size()

    lmc_ops.multi_layer_block_kv_transfer(
        paged,
        mem_objs,
        block_ids_dev,
        DEVICE,
        dir_enum,
        shape_desc,
        TOKENS_PER_OBJ,
        fmt_enum,
        0,
    )


def bench_one(fmt_name, fmt_enum, direction):
    dir_enum = (
        lmc_ops.TransferDirection.H2D
        if direction == "H2D"
        else lmc_ops.TransferDirection.D2H
    )
    paged = make_paged(fmt_enum)
    mem_objs = make_mem_objects()

    # warmup with random blocks
    for _ in range(WARMUP):
        block_ids_dev = random_block_ids()
        run_kernel(paged, mem_objs, block_ids_dev, fmt_enum, dir_enum)
        torch_dev.synchronize()

    # bench: random blocks + per-iteration sync
    times = []
    for _ in range(ITERS):
        block_ids_dev = random_block_ids()
        torch_dev.synchronize()
        t0 = time.perf_counter()
        run_kernel(paged, mem_objs, block_ids_dev, fmt_enum, dir_enum)
        torch_dev.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    elem_size = 2
    bytes_per_call = NUM_OBJS * 2 * NL * TOKENS_PER_OBJ * NH * HS * elem_size
    bw = (bytes_per_call / 1e9) / (avg / 1e3) if avg > 0 else 0
    return fmt_name, direction, avg, mn, p50, p99, mx, bw, bytes_per_call


def main():
    print(f"\nDevice: {torch_device_type}:{torch_dev.current_device()}")
    print(
        f"Paged buffer: {torch_device_type} ({NB} blocks) | "
        f"Memory objects: CPU (pinned)"
    )
    print(f"Config: NL={NL} NB={NB} BS={BS} NH={NH} HS={HS} objs={NUM_OBJS}")
    print(f"Block selection: random {TOTAL_BLOCKS} out of {NB} per iteration")
    print(f"Sync: per-iteration | Warmup={WARMUP} Iters={ITERS}\n")

    hdr = (
        f"{'Format':<20} {'Dir':<12} {'Avg':>7} {'Min':>7} {'P50':>7} "
        f"{'P99':>7} {'Max':>7} {'BW':>8} {'Data':>7}"
    )
    units = (
        f"{'':20} {'':12} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} "
        f"{'(ms)':>7} {'(ms)':>7} {'(GB/s)':>8} {'(MB)':>7}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(units)
    print(sep)

    for fmt_name, fmt_enum in NHD_FORMATS:
        for direction in ("D2H", "H2D"):
            r = bench_one(fmt_name, fmt_enum, direction)
            _, d, avg, mn, p50, p99, mx, bw, bpc = r
            label = (
                f"{torch_device_type}→cpu" if d == "D2H" else f"cpu→{torch_device_type}"
            )
            print(
                f"{fmt_name:<20} {label:<12} {avg:>7.2f} {mn:>7.2f} {p50:>7.2f} "
                f"{p99:>7.2f} {mx:>7.2f} {bw:>8.2f} {bpc / 1e6:>7.2f}"
            )

    print(sep)


if __name__ == "__main__":
    main()
