# SPDX-License-Identifier: Apache-2.0
"""Execution equivalence of the two multiprocess transfer paths.

Both paths now plan their launches with
:func:`~lmcache.v1.multiprocess.transfer_context.common_exec.plan_copy_batches`
and run them through
:func:`~lmcache.v1.multiprocess.transfer_context.common_exec.execute_copy_batches`;
only the endpoint differs. These tests pin that down where it matters -- the
bytes a store produces and the KV tensors a retrieve produces must be identical
whether the LMCache-driven server or the Engine-driven worker moved them.
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    get_layout_desc,
    plan_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    compute_kv_layout,
    gather_engine_groups,
    scatter_engine_groups,
)
from lmcache.v1.multiprocess.transfer_context.common_copy import (
    DiscoveredGroupLayout,
    GroupTransferPlan,
    build_group_kv_subset,
    build_group_transfer_plans,
    registered_groups_from_engine_infos,
)
from lmcache.v1.platform.cpu.cache_context import CPUCacheContext
import lmcache.c_ops as lmc_ops

# Two layers of full attention plus two layers of sliding-window attention, in
# two engine groups: the hybrid shape the Engine-driven path had to support.
NUM_BLOCKS = 64
BLOCK_SIZE = 4
NUM_HEADS = 2
HEAD_SIZE = 8
TOKENS_PER_CHUNK = 8
SW_SIZE_TOKENS = 4

# Pin the KV layout so both paths detect the same geometry from the same
# tensors; layout detection itself is not what these tests are about.
LAYOUT_HINTS: "LayoutHints" = {"kv_layout": "NHD"}

ENGINE_GROUP_INFOS = [
    EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=BLOCK_SIZE,
        sw_size_tokens=-1,
    ),
    EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(2, 3),
        tokens_per_block=BLOCK_SIZE,
        sw_size_tokens=SW_SIZE_TOKENS,
    ),
]


class _FakeMemoryObj:
    """Host-side storage object double backed by a flat byte tensor.

    Exposes only what the staging helpers use: ``raw_tensor``,
    ``get_size()``, and ``parent()``. ``parent()`` returns ``None`` so the
    helpers take the plain tensor-copy path instead of the lazy-allocator
    pinned-memory path.

    Args:
        num_bytes: Size of the object in bytes.
    """

    def __init__(self, num_bytes: int) -> None:
        self.raw_tensor = torch.zeros(num_bytes, dtype=torch.uint8)

    def get_size(self) -> int:
        """Return the object size in bytes."""
        return int(self.raw_tensor.numel())

    def parent(self) -> None:
        """Return no owning allocator."""
        return None


def _make_kv_caches(seed: int = 0) -> dict[str, torch.Tensor]:
    """Build four NHD KV tensors with deterministic contents.

    Args:
        seed: Torch manual seed used to fill the tensors.

    Returns:
        Mapping from layer name to a ``[2, NB, BS, NH, HS]`` tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    return {
        f"layer_{i}": torch.randn(
            2,
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_HEADS,
            HEAD_SIZE,
            generator=generator,
        )
        for i in range(4)
    }


def _worker_plans(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    *,
    for_retrieve: bool = False,
    skip_first_n_tokens: int = 0,
) -> list[GroupTransferPlan]:
    """Plan a request the way the Engine-driven worker registration does.

    Args:
        kv_caches: Worker KV tensors keyed by layer name.
        block_ids: Per-engine-group block IDs of the request.
        for_retrieve: Apply Sliding Window object-tail selection.
        skip_first_n_tokens: APC prefix guard in logical tokens.

    Returns:
        One plan per engine group, in protocol order.
    """
    layouts = []
    for info in ENGINE_GROUP_INFOS:
        group_kv = build_group_kv_subset(kv_caches, info.layer_indices)
        (
            slots_per_block,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(group_kv, layout_hints=LAYOUT_HINTS)
        layouts.append(
            DiscoveredGroupLayout(
                slots_per_block=slots_per_block,
                num_layers=num_layers,
                hidden_dim_size=hidden_dim_size,
                kv_size=kv_size,
                dtype=getattr(torch, dtype_str),
                engine_kv_format=engine_kv_format,
            )
        )
    return build_group_transfer_plans(
        registered_groups_from_engine_infos(
            ENGINE_GROUP_INFOS, layouts, TOKENS_PER_CHUNK
        ),
        [list(ids) for ids in block_ids],
        for_retrieve=for_retrieve,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def _make_cache_context(kv_caches: dict[str, torch.Tensor]) -> CPUCacheContext:
    """Build a server cache context over worker-shaped KV tensors.

    Args:
        kv_caches: Worker KV tensors keyed by layer name; the context keeps
            references to the very same storage, so copies are observable on
            both sides.

    Returns:
        A CPU cache context registered with the hybrid engine groups.
    """
    wrappers = [cast(Any, _TensorWrapper(tensor)) for tensor in kv_caches.values()]
    return CPUCacheContext(
        wrappers,
        lmcache_tokens_per_chunk=TOKENS_PER_CHUNK,
        layout_hints=LAYOUT_HINTS,
        engine_group_infos=ENGINE_GROUP_INFOS,
    )


class _TensorWrapper:
    """Minimal stand-in for the IPC wrapper the server unwraps.

    Args:
        tensor: KV tensor to expose.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the wrapped KV tensor."""
        return self._tensor


def _object_num_bytes(
    cache_context: CPUCacheContext,
    object_group_id: int,
) -> int:
    """Return the byte size the server reserves for one object of a group.

    The server always reserves a full LMCache chunk, even for a Sliding Window
    group that only fills the windowed head of it, so the object size itself
    differs from the Engine-driven worker's windowed chunk. Only the copied
    payload is comparable across the two paths.

    Args:
        cache_context: Server context.
        object_group_id: Object group whose object size to compute.

    Returns:
        Size of one object in bytes.
    """
    layout = get_layout_desc(cache_context, TOKENS_PER_CHUNK, object_group_id)
    return sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
    )


def _server_object_bytes(
    cache_context: CPUCacheContext,
    block_ids: list[list[int]],
) -> list[list[torch.Tensor]]:
    """Run a full LMCache-driven store and return the produced object bytes.

    Args:
        cache_context: Server context over the source KV tensors.
        block_ids: Per-engine-group block IDs of the request.

    Returns:
        Per object group, the byte tensor of every stored object.
    """
    plans, staged = plan_and_stage_block_ids(
        cache_context, [list(ids) for ids in block_ids]
    )
    manager = cache_context.kv_layer_groups_manager
    per_group: list[list[torch.Tensor]] = []
    for object_group_id in range(manager.num_object_groups):
        plan = next(p for p in plans if p.group.object_group_id == object_group_id)
        num_bytes = _object_num_bytes(cache_context, object_group_id)
        memory_objs = [_FakeMemoryObj(num_bytes) for _ in range(plan.total_objects)]
        transfer_kv_per_object_group(
            cache_context,
            plans,
            staged,
            cast(Any, memory_objs),
            object_group_id,
            cache_context.max_batch_size,
            lmc_ops.TransferDirection.D2H,
        )
        per_group.append([memory_obj.raw_tensor.clone() for memory_obj in memory_objs])
    return per_group


def _worker_object_bytes(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
) -> list[list[torch.Tensor]]:
    """Run an Engine-driven store and return the produced object bytes.

    Args:
        kv_caches: Worker KV tensors keyed by layer name.
        block_ids: Per-engine-group block IDs of the request.

    Returns:
        Per group, the byte view of every gathered chunk.
    """
    plans = _worker_plans(kv_caches, block_ids)
    chunks_per_group = gather_engine_groups(plans, kv_caches, LAYOUT_HINTS)
    return [
        [chunk.contiguous().view(torch.uint8).flatten() for chunk in chunks]
        for chunks in chunks_per_group
    ]


def _assert_payload_equal(
    server_bytes: list[list[torch.Tensor]],
    worker_bytes: list[list[torch.Tensor]],
) -> None:
    """Assert both paths wrote the same payload into every object.

    The server reserves every object at full-chunk size, so a Sliding Window
    group only fills the head of it and leaves a tail of stale staging-buffer
    bytes that no reader ever looks at. The copied payload is therefore the
    comparable part.

    Args:
        server_bytes: Objects produced by the LMCache-driven path.
        worker_bytes: Chunks produced by the Engine-driven path.

    Raises:
        AssertionError: If any payload differs, or the server wrote past it.
    """
    assert len(server_bytes) == len(worker_bytes)
    for group_index, (server_group, worker_group) in enumerate(
        zip(server_bytes, worker_bytes, strict=True)
    ):
        assert len(server_group) == len(worker_group), group_index
        for object_index, (server_obj, worker_obj) in enumerate(
            zip(server_group, worker_group, strict=True)
        ):
            where = (group_index, object_index)
            payload = worker_obj.numel()
            assert server_obj.numel() >= payload, where
            assert torch.equal(server_obj[:payload], worker_obj), where


def test_store_produces_identical_object_bytes_on_both_paths() -> None:
    """A store moves the same bytes whether the server or the worker copies."""
    kv_caches = _make_kv_caches()
    block_ids = [list(range(8)), list(range(10, 18))]

    worker_bytes = _worker_object_bytes(kv_caches, block_ids)
    cache_context = _make_cache_context(kv_caches)
    try:
        server_bytes = _server_object_bytes(cache_context, block_ids)
    finally:
        cache_context.close()

    _assert_payload_equal(server_bytes, worker_bytes)

    # The Sliding Window group must actually be windowed, or the test would
    # compare two full-attention paths and prove nothing about trimming.
    assert worker_bytes[1][0].numel() < worker_bytes[0][0].numel()


def test_store_batching_boundary_does_not_change_object_bytes() -> None:
    """More objects than one batch holds still yields byte-identical results."""
    kv_caches = _make_kv_caches(seed=3)
    # 12 objects per group: three batches at the server's batch size of four.
    block_ids = [list(range(24)), list(range(30, 54))]

    worker_bytes = _worker_object_bytes(kv_caches, block_ids)
    cache_context = _make_cache_context(kv_caches)
    try:
        assert cache_context.max_batch_size < 12
        server_bytes = _server_object_bytes(cache_context, block_ids)
    finally:
        cache_context.close()

    _assert_payload_equal(server_bytes, worker_bytes)


@pytest.mark.parametrize("skip_first_n_tokens", [0, TOKENS_PER_CHUNK * 5 + 2])
def test_retrieve_writes_identical_kv_on_both_paths(skip_first_n_tokens: int) -> None:
    """A retrieve writes the same KV whether the server or the worker copies.

    The non-zero skip crosses a batch boundary (object five of a four-object
    batch), which is exactly where a per-batch skip must not be re-applied.

    Args:
        skip_first_n_tokens: APC prefix guard in logical tokens.
    """
    source = _make_kv_caches(seed=7)
    block_ids = [list(range(24)), list(range(30, 54))]
    stored = _worker_object_bytes(source, block_ids)

    plans = _worker_plans(
        source, block_ids, for_retrieve=True, skip_first_n_tokens=skip_first_n_tokens
    )
    # Rebuild the typed chunks a grouped retrieve hands back to the worker.
    chunks_per_group = [
        [
            stored[group_index][plan.first_object + i]
            .clone()
            .view(plan.group.dtype)
            .view(plan.group.shape)
            for i in range(plan.transfer_objects)
        ]
        for group_index, plan in enumerate(plans)
    ]

    worker_dest = {name: torch.zeros_like(t) for name, t in source.items()}
    scatter_engine_groups(plans, worker_dest, chunks_per_group, LAYOUT_HINTS)

    server_dest = {name: torch.zeros_like(t) for name, t in source.items()}
    cache_context = _make_cache_context(server_dest)
    try:
        server_plans, staged = plan_and_stage_block_ids(
            cache_context,
            [list(ids) for ids in block_ids],
            for_retrieve=True,
            skip_first_n_tokens=skip_first_n_tokens,
        )
        for object_group_id, plan in enumerate(server_plans):
            num_bytes = _object_num_bytes(cache_context, object_group_id)
            memory_objs: list[_FakeMemoryObj | None] = [
                None for _ in range(plan.first_object)
            ]
            for chunk in chunks_per_group[object_group_id]:
                memory_obj = _FakeMemoryObj(num_bytes)
                payload = chunk.contiguous().view(torch.uint8).flatten()
                memory_obj.raw_tensor[: payload.numel()].copy_(payload)
                memory_objs.append(memory_obj)
            transfer_kv_per_object_group(
                cache_context,
                server_plans,
                staged,
                cast(Any, memory_objs),
                object_group_id,
                cache_context.max_batch_size,
                lmc_ops.TransferDirection.H2D,
            )
    finally:
        cache_context.close()

    for name in source:
        assert torch.equal(server_dest[name], worker_dest[name]), name

    first_block = block_ids[0][0]
    if skip_first_n_tokens:
        # The guarded prefix must stay untouched on both paths.
        assert not torch.any(worker_dest["layer_0"][:, first_block])
    else:
        assert torch.equal(
            worker_dest["layer_0"][:, first_block],
            source["layer_0"][:, first_block],
        )


class _FakeLazyMemoryObj(_FakeMemoryObj):
    """Storage object double that also satisfies the native staging descriptor.

    Args:
        num_bytes: Size of the object in bytes.
    """

    def __init__(self, num_bytes: int) -> None:
        super().__init__(num_bytes)
        self.data_ptr = self.raw_tensor.data_ptr()
        self.meta = SimpleNamespace(address=0)


def test_native_endpoint_records_the_stream_endpoint_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batched native plan carries exactly the immediate launch geometry.

    ``execute_object_group_transfer`` is a native-only entry point, so this
    records what the endpoint would hand it and compares that against the
    launches the stream endpoint actually issues.

    Args:
        monkeypatch: Fixture used to install the native symbols.
    """
    kv_caches = _make_kv_caches(seed=11)
    block_ids = [list(range(24)), list(range(30, 54))]

    stream_launches: list[tuple[int, list[int], int]] = []
    original_transfer = lmc_ops.multi_layer_block_kv_transfer

    def _recording_transfer(
        kv_pointers: torch.Tensor,
        object_ptrs: list,
        group_block_ids: torch.Tensor,
        *args: Any,
    ) -> None:
        stream_launches.append(
            (len(object_ptrs), group_block_ids.tolist(), int(args[-1]))
        )
        original_transfer(kv_pointers, object_ptrs, group_block_ids, *args)

    cache_context = _make_cache_context(kv_caches)
    try:
        monkeypatch.setattr(
            lmc_ops, "multi_layer_block_kv_transfer", _recording_transfer
        )
        _server_object_bytes(cache_context, block_ids)
        monkeypatch.undo()

        recorded: list[tuple] = []
        staged_block_ids: dict[int, list[int]] = {}

        def _execute_object_group_transfer(
            direction: Any,
            device: Any,
            pin_chunk_size: int,
            kernel_group_specs: list,
            batch_steps: list,
        ) -> None:
            for step in batch_steps:
                for launch in step[1]:
                    spec = kernel_group_specs[launch[0]]
                    recorded.append(
                        (
                            launch[3],
                            spec[1][launch[1] : launch[1] + launch[2]],
                            launch[4],
                        )
                    )

        monkeypatch.setattr(
            lmcache_driven_transfer, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", True
        )
        monkeypatch.setattr(
            lmc_ops,
            "KernelGroupSpec",
            lambda kv_ptr, buffers, shape, sw, fmt, ids_ptr, ids_len: (
                kv_ptr,
                staged_block_ids[ids_ptr],
            ),
            raising=False,
        )
        monkeypatch.setattr(
            lmc_ops,
            "LaunchVar",
            lambda *fields: fields,
            raising=False,
        )
        monkeypatch.setattr(
            lmc_ops,
            "BatchStep",
            lambda staging, launches: (staging, list(launches)),
            raising=False,
        )
        monkeypatch.setattr(
            lmc_ops, "StagingCopy", lambda *fields: fields, raising=False
        )
        monkeypatch.setattr(
            lmc_ops,
            "execute_object_group_transfer",
            _execute_object_group_transfer,
            raising=False,
        )

        plans, staged = plan_and_stage_block_ids(
            cache_context, [list(ids) for ids in block_ids]
        )
        staged_block_ids.update(
            {tensor.data_ptr(): tensor.tolist() for tensor in staged}
        )
        for object_group_id in range(
            cache_context.kv_layer_groups_manager.num_object_groups
        ):
            plan = next(p for p in plans if p.group.object_group_id == object_group_id)
            num_bytes = _object_num_bytes(cache_context, object_group_id)
            transfer_kv_per_object_group(
                cache_context,
                plans,
                staged,
                cast(
                    Any,
                    [_FakeLazyMemoryObj(num_bytes) for _ in range(plan.total_objects)],
                ),
                object_group_id,
                cache_context.max_batch_size,
                lmc_ops.TransferDirection.D2H,
            )
    finally:
        cache_context.close()

    assert recorded == stream_launches
    assert recorded
