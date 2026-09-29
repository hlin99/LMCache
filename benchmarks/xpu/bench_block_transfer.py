# SPDX-License-Identifier: Apache-2.0
"""Compare native staged/direct transfers against torch's complete staging path."""

# Standard
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import median
from typing import NotRequired, TypedDict
import argparse
import json
import math
import platform
import time

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format import get_spec_class
from lmcache.v1.platform import torch_ops
import lmcache.lmcache_native as native
import lmcache.xpu_ops as sycl

_FORMATS = (1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 15)
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint8": torch.uint8,
}


class BenchmarkResult(TypedDict):
    """One case's geometry, wall-clock samples, and payload-based throughput."""

    format: str
    format_id: int
    direction: str
    native_path: str
    dtype: str
    layers: int
    block_size: int
    blocks: int
    pool_blocks: int
    heads: int
    head_size: int
    blocks_per_chunk: int
    objects: int
    payload_bytes: int
    native_median_ms: float
    torch_median_ms: float
    native_p95_ms: float
    torch_p95_ms: float
    native_GBps: float
    torch_GBps: float
    speedup: float
    isolated_kernel_median_ms: NotRequired[float]
    isolated_dma_median_ms: NotRequired[float]
    native_samples_ms: list[float]
    torch_samples_ms: list[float]


def _positive(value: str) -> int:
    """Parse a positive CLI integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _elapsed_ms(operation: Callable[[], None], stream: torch.xpu.Stream) -> float:
    """Measure host submission through completion, including the final wait."""
    start = time.perf_counter_ns()
    operation()
    stream.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def _paired_samples(
    operations: tuple[Callable[[], None], Callable[[], None]],
    stream: torch.xpu.Stream,
    warmup: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Alternate execution order to reduce thermal/order bias between paths."""
    samples: tuple[list[float], list[float]] = ([], [])
    for iteration in range(warmup + repeats):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for idx in order:
            elapsed = _elapsed_ms(operations[idx], stream)
            if iteration >= warmup:
                samples[idx].append(elapsed)
    return samples


def _run_case(
    fmt: int,
    layers: int,
    block_size: int,
    blocks: int,
    blocks_per_chunk: int,
    heads: int,
    head_size: int,
    mla_width: int,
    dtype: torch.dtype,
    device: torch.device,
    direction_name: str,
    warmup: int,
    repeats: int,
    seed: int,
    pool_multiplier: int,
    native_path_name: str,
) -> BenchmarkResult:
    """Validate and time a random, non-contiguous per-layer page transfer."""
    if blocks % blocks_per_chunk:
        raise ValueError("--blocks must be divisible by --blocks-per-chunk")
    torch.manual_seed(seed)
    fmt_enum = native.EngineKVFormat(fmt)
    spec = get_spec_class(fmt_enum)
    desc = native.PageBufferShapeDesc()
    desc.nl, desc.bs, desc.nb = layers, block_size, blocks * pool_multiplier
    desc.kv_size = 1 if spec.is_mla or spec.is_fused_packed else 2
    desc.nh = 1 if spec.is_mla else heads
    desc.hs = mla_width if spec.is_mla else head_size
    desc.element_size = dtype.itemsize
    page_shape = spec.paged_layer_shape(desc.nb, desc.bs, desc.nh, desc.hs)
    # Separate layer allocations emulate the engine's pointer list, not one
    # artificial contiguous cross-layer tensor.
    pages = [
        torch.randint(0, 100, page_shape, device=device, dtype=torch.int32).to(dtype)
        for _ in range(layers)
    ]
    ids_host = torch.randperm(desc.nb)[:blocks]
    ids = ids_host.to(device=device, dtype=torch.int64)
    pointers = torch.tensor(
        [page.data_ptr() for page in pages], dtype=torch.uint64, device=device
    )
    chunk_tokens = blocks_per_chunk * block_size
    object_count = blocks // blocks_per_chunk
    shape = (
        (layers, chunk_tokens, desc.nh * desc.hs)
        if desc.kv_size == 1
        else (2, layers, chunk_tokens, desc.nh * desc.hs)
    )
    host_native = [
        torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(object_count)
    ]
    staging = (
        [torch.empty(shape, dtype=dtype, device=device) for _ in range(object_count)]
        if native_path_name == "staged"
        else []
    )
    host_torch = [torch.empty_like(host, pin_memory=True) for host in host_native]
    object_ptrs = [
        obj.data_ptr()
        for obj in (host_native if native_path_name == "direct" else staging)
    ]
    direction = (
        native.TransferDirection.D2H
        if direction_name == "store"
        else native.TransferDirection.H2D
    )
    alignment = 1 << 26
    stream = torch.xpu.Stream(device=device)
    stream.wait_stream(torch.xpu.current_stream(device))

    def gather_or_scatter() -> None:
        sycl.multi_layer_block_kv_transfer(
            pointers,
            object_ptrs,
            ids,
            device,
            direction,
            desc,
            chunk_tokens,
            fmt_enum,
            0,
        )

    def dma() -> None:
        for host, gpu in zip(host_native, staging, strict=True):
            dst, src = (host, gpu) if direction_name == "store" else (gpu, host)
            sycl.lmcache_memcpy_async(
                dst.data_ptr(),
                src.data_ptr(),
                host.nbytes,
                int(direction),
                0,
                alignment,
            )

    def native_path() -> None:
        if native_path_name == "direct":
            gather_or_scatter()
        elif direction_name == "store":
            gather_or_scatter()
            dma()
        else:
            dma()
            gather_or_scatter()

    def torch_path() -> None:
        torch_ops.multi_layer_block_kv_transfer(
            pages, host_torch, ids, device, direction, desc, chunk_tokens, fmt_enum, 0
        )

    with torch.xpu.stream(stream):
        if direction_name == "store":
            native_path()
            torch_path()
            stream.synchronize()
            for actual, expected in zip(host_native, host_torch, strict=True):
                if not torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ):
                    raise RuntimeError("Native and torch store payloads differ")
        else:
            # Initialize identical host payloads outside the timed section.
            torch_ops.multi_layer_block_kv_transfer(
                pages,
                host_torch,
                ids,
                device,
                native.TransferDirection.D2H,
                desc,
                chunk_tokens,
                fmt_enum,
                0,
            )
            stream.synchronize()
            for dst, src in zip(host_native, host_torch, strict=True):
                dst.copy_(src)
            for page in pages:
                page.zero_()
            native_path()
            stream.synchronize()
            expected_pages = [page.cpu() for page in pages]
            for page in pages:
                page.zero_()
            torch_path()
            stream.synchronize()
            for actual, expected in zip(pages, expected_pages, strict=True):
                if not torch.equal(
                    actual.cpu().view(torch.uint8), expected.view(torch.uint8)
                ):
                    raise RuntimeError("Native and torch retrieve pages differ")

        native_ms, torch_ms = _paired_samples(
            (native_path, torch_path), stream, warmup, repeats
        )
        # Diagnostic phases are isolated measurements, not additive components
        # of the full-path samples above (each has its own synchronization).
        phase_results: dict[str, float] = {}
        if native_path_name == "staged":
            kernel_ms, dma_ms = _paired_samples(
                (gather_or_scatter, dma), stream, warmup, repeats
            )
            phase_results = {
                "isolated_kernel_median_ms": median(kernel_ms),
                "isolated_dma_median_ms": median(dma_ms),
            }
    payload = sum(host.nbytes for host in host_native)
    native_median, torch_median = median(native_ms), median(torch_ms)
    p95_index = math.ceil(repeats * 0.95) - 1
    result: BenchmarkResult = {
        "format": fmt_enum.name,
        "format_id": fmt,
        "direction": direction_name,
        "native_path": native_path_name,
        "dtype": str(dtype),
        "layers": layers,
        "block_size": block_size,
        "blocks": blocks,
        "pool_blocks": desc.nb,
        "heads": desc.nh,
        "head_size": desc.hs,
        "blocks_per_chunk": blocks_per_chunk,
        "objects": object_count,
        "payload_bytes": payload,
        "native_median_ms": native_median,
        "torch_median_ms": torch_median,
        "native_p95_ms": sorted(native_ms)[p95_index],
        "torch_p95_ms": sorted(torch_ms)[p95_index],
        "native_GBps": payload / native_median / 1e6,
        "torch_GBps": payload / torch_median / 1e6,
        "speedup": torch_median / native_median,
        "native_samples_ms": native_ms,
        "torch_samples_ms": torch_ms,
    }
    if phase_results:
        result["isolated_kernel_median_ms"] = phase_results["isolated_kernel_median_ms"]
        result["isolated_dma_median_ms"] = phase_results["isolated_dma_median_ms"]
    return result


def main() -> None:
    """Run the XPU transfer sweep and optionally persist metadata and raw samples.

    Arguments are read from the CLI; see ``--help`` for dimensions and defaults.
    Results print as rows and, with ``--output``, a JSON file.

    Raises:
        RuntimeError: If XPU/native kernels are unavailable or payloads differ.
        ValueError: If a transfer does not contain whole chunks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats", type=int, choices=_FORMATS, nargs="+", default=[1, 3, 6]
    )
    parser.add_argument("--layers", type=_positive, nargs="+", default=[8, 32])
    parser.add_argument("--block-sizes", type=_positive, nargs="+", default=[16, 64])
    parser.add_argument("--blocks", type=_positive, nargs="+", default=[16, 64])
    parser.add_argument("--blocks-per-chunk", type=_positive, default=16)
    parser.add_argument("--heads", type=_positive, default=8)
    parser.add_argument("--head-size", type=_positive, default=128)
    parser.add_argument("--mla-width", type=_positive, default=576)
    parser.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    parser.add_argument(
        "--directions", nargs="+", choices=("store", "retrieve"), default=["store"]
    )
    parser.add_argument("--pool-multiplier", type=_positive, default=4)
    parser.add_argument(
        "--native-path",
        choices=("staged", "direct"),
        default="staged",
        help="direct passes pinned CPU object addresses to the kernel (no staging/DMA)",
    )
    parser.add_argument("--warmup", type=_positive, default=10)
    parser.add_argument("--repeats", type=_positive, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.xpu.is_available():
        raise RuntimeError("An available XPU is required")
    if not hasattr(sycl, "multi_layer_block_kv_transfer"):
        raise RuntimeError("Rebuild lmcache.xpu_ops with the block-transfer kernel")
    torch.xpu.set_device(args.device)
    device = torch.device("xpu", args.device)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(torch.xpu.get_device_properties(device)),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "extension": sycl.__file__,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "host submission to stream completion",
        "native_path": args.native_path,
        "native_staging": (
            "preallocated and reused" if args.native_path == "staged" else "none"
        ),
        "host_allocation": "torch pinned CPU (XPU USM host)",
        "torch_staging": "allocated internally by unmodified torch ops",
    }
    results = []
    print(f"Device: {torch.xpu.get_device_name(device)}", flush=True)
    print(f"Native path: {args.native_path}; torch path: internal staging", flush=True)
    print(
        "fmt dir      layers BS blocks MiB     native_ms torch_ms speedup native_GB/s",
        flush=True,
    )
    for fmt, layers, bs, blocks, direction in product(
        args.formats, args.layers, args.block_sizes, args.blocks, args.directions
    ):
        result = _run_case(
            fmt,
            layers,
            bs,
            blocks,
            args.blocks_per_chunk,
            args.heads,
            args.head_size,
            args.mla_width,
            _DTYPES[args.dtype],
            device,
            direction,
            args.warmup,
            args.repeats,
            args.seed,
            args.pool_multiplier,
            args.native_path,
        )
        results.append(result)
        print(
            f"{fmt:3} {direction:8} {layers:3} {bs:4} {blocks:4} "
            f"{result['payload_bytes'] / 2**20:7.1f} "
            f"{result['native_median_ms']:9.3f} {result['torch_median_ms']:8.3f} "
            f"{result['speedup']:6.2f}x {result['native_GBps']:8.2f}",
            flush=True,
        )
    if args.output:
        args.output.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
