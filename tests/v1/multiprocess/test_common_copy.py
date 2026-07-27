# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.common_copy import (
    DiscoveredGroupLayout,
    RegisteredGroup,
    build_group_transfer_plans,
    flatten_chunks_group_major,
    gather_engine_groups,
    registered_groups_from_engine_infos,
    scatter_engine_groups,
    sliding_window_first_object,
    unflatten_chunks_group_major,
    validate_group_block_ids,
    validate_registered_groups,
)


def _group(
    kernel_group_id: int,
    engine_group_id: int,
    layer_indices: tuple[int, ...],
    *,
    object_group_id: int | None = None,
    tokens_per_block: int = 4,
    slots_per_block: int = 4,
    blocks_per_chunk: int = 2,
    copy_blocks_per_chunk: int = 2,
    sw_size_tokens: int = -1,
    copy_format: int = 10,
) -> RegisteredGroup:
    """Build compact transfer metadata for planner contract tests.

    Args:
        kernel_group_id: Registration order of the group.
        engine_group_id: Engine block-ID address space.
        layer_indices: Registered layer indices assigned to the group.
        object_group_id: Storage namespace; defaults to ``kernel_group_id``.
        tokens_per_block: Logical tokens represented by one block.
        slots_per_block: Physical slots stored by one block.
        blocks_per_chunk: Full engine blocks per LMCache chunk.
        copy_blocks_per_chunk: Trailing blocks copied after window trimming.
        sw_size_tokens: Sliding Window size, or ``-1`` for full attention.
        copy_format: Native copy-format test identifier.

    Returns:
        A transfer group suitable for planner contract tests.
    """
    return RegisteredGroup(
        kernel_group_id=kernel_group_id,
        object_group_id=(
            kernel_group_id if object_group_id is None else object_group_id
        ),
        engine_group_id=engine_group_id,
        layer_indices=layer_indices,
        tokens_per_block=tokens_per_block,
        slots_per_block=slots_per_block,
        blocks_per_chunk=blocks_per_chunk,
        copy_blocks_per_chunk=copy_blocks_per_chunk,
        chunk_tokens=8,
        shape=torch.Size(
            [2, len(layer_indices), copy_blocks_per_chunk * slots_per_block, 1]
        ),
        dtype=torch.float32,
        engine_kv_format=copy_format,
        sw_size_tokens=sw_size_tokens,
    )


def test_validate_group_block_ids_requires_exact_equal_coverage() -> None:
    """Every group must exactly cover the same logical chunk count."""
    assert validate_group_block_ids([[0, 1, 2, 3], [4, 5]], [2, 1]) == 2

    with pytest.raises(ValueError, match="exact multiple"):
        validate_group_block_ids([[0, 1, 2]], [2])
    with pytest.raises(ValueError, match="different logical chunk counts"):
        validate_group_block_ids([[0, 1, 2, 3], [4]], [2, 1])
    with pytest.raises(ValueError, match="must be positive"):
        validate_group_block_ids([[]], [0])


def test_registered_groups_validate_ids_layers_and_geometry() -> None:
    """Group registration rejects invalid IDs and incomplete layer maps."""
    validate_registered_groups([_group(0, 0, (0,)), _group(1, 0, (1,))], 2)

    # Several kernel groups may share one object group, in contiguous order.
    validate_registered_groups(
        [_group(0, 0, (0,), object_group_id=0), _group(1, 0, (1,), object_group_id=0)],
        2,
    )

    with pytest.raises(ValueError, match="kernel group IDs"):
        validate_registered_groups([_group(1, 0, (0,))], 1)
    with pytest.raises(ValueError, match="object group IDs must be dense"):
        validate_registered_groups([_group(0, 0, (0,), object_group_id=1)], 1)
    with pytest.raises(ValueError, match="must not interleave"):
        validate_registered_groups(
            [
                _group(0, 0, (0,), object_group_id=1),
                _group(1, 0, (1,), object_group_id=0),
                _group(2, 0, (2,), object_group_id=1),
            ],
            3,
        )
    with pytest.raises(ValueError, match="dense"):
        validate_registered_groups([_group(0, 1, (0,))], 1)
    with pytest.raises(ValueError, match="duplicated"):
        validate_registered_groups([_group(0, 0, (0,)), _group(1, 0, (0,))], 1)
    with pytest.raises(ValueError, match="missing"):
        validate_registered_groups([_group(0, 0, (0,))], 2)
    with pytest.raises(ValueError, match="outside"):
        validate_registered_groups([_group(0, 0, (2,))], 2)


def test_validate_registered_groups_with_explicit_cross_layer_aliases() -> None:
    """Only explicitly excluded cross-layer aliases may lack an owner group."""
    groups = [_group(0, 0, (0,)), _group(1, 1, (1,))]

    validate_registered_groups(groups, 3, excluded_layer_indices={2})

    with pytest.raises(ValueError, match="missing"):
        validate_registered_groups(groups, 3)
    with pytest.raises(ValueError, match="excluded layer indices.*outside"):
        validate_registered_groups(groups, 3, excluded_layer_indices={3})
    with pytest.raises(ValueError, match="must not be assigned"):
        validate_registered_groups(groups, 3, excluded_layer_indices={1})


def test_plan_preserves_shared_engine_group_and_group_geometry() -> None:
    """Kernel groups may share an engine group while retaining copy identity."""
    groups = [
        _group(0, 0, (0,), copy_format=11),
        _group(
            1,
            0,
            (1,),
            tokens_per_block=8,
            slots_per_block=2,
            blocks_per_chunk=1,
            copy_blocks_per_chunk=1,
            copy_format=22,
        ),
    ]

    plans = build_group_transfer_plans(groups, [[0, 1, 2, 3], [10, 11]])

    assert [plan.group.engine_group_id for plan in plans] == [0, 0]
    assert [plan.group.copy_blocks_per_chunk for plan in plans] == [2, 1]
    assert [plan.group.engine_kv_format for plan in plans] == [11, 22]
    assert [plan.total_objects for plan in plans] == [2, 2]


def test_plan_selects_sliding_window_tail_blocks_and_objects() -> None:
    """Retrieve retains only window-valid objects and trailing subchunk blocks."""
    groups = [
        _group(0, 0, (0,)),
        _group(1, 1, (1,), copy_blocks_per_chunk=1, sw_size_tokens=8),
    ]
    block_ids = [list(range(8)), list(range(10, 18))]

    store_plans = build_group_transfer_plans(groups, block_ids)
    retrieve_plans = build_group_transfer_plans(groups, block_ids, for_retrieve=True)

    assert store_plans[1].selected_block_ids == (11, 13, 15, 17)
    assert store_plans[1].first_object == 0
    assert store_plans[1].transfer_objects == 4
    assert retrieve_plans[1].first_object == 3
    assert retrieve_plans[1].transfer_objects == 1
    assert retrieve_plans[1].selected_block_ids == (17,)
    assert retrieve_plans[0].first_object == 0
    assert retrieve_plans[0].selected_block_ids == tuple(range(8))


def test_sliding_window_first_object_matches_window_size() -> None:
    """Object-tail selection keeps exactly the window's trailing objects."""
    full = _group(0, 0, (0,))
    windowed = _group(1, 1, (1,), sw_size_tokens=24)

    assert sliding_window_first_object(full, 8) == 0
    assert sliding_window_first_object(windowed, 8) == 5
    assert sliding_window_first_object(windowed, 2) == 0
    with pytest.raises(ValueError, match="non-negative"):
        sliding_window_first_object(windowed, -1)


def test_logical_skip_converts_through_trimmed_geometry() -> None:
    """Skip conversion accounts for compression and per-chunk trimming."""
    compressed = _group(
        0,
        0,
        (0,),
        tokens_per_block=8,
        slots_per_block=2,
        blocks_per_chunk=1,
        copy_blocks_per_chunk=1,
    )
    assert compressed.blocks_to_skip(8) == 1
    assert compressed.physical_skip(8) == 2
    # Sub-block skips round down to the nearest whole block.
    assert compressed.physical_skip(1) == 0

    trimmed = _group(
        0,
        0,
        (0,),
        tokens_per_block=2,
        slots_per_block=2,
        blocks_per_chunk=4,
        copy_blocks_per_chunk=2,
    )
    # The first two blocks of every chunk are never selected for copy.
    assert trimmed.blocks_to_skip(4) == 0
    assert trimmed.blocks_to_skip(6) == 1
    assert trimmed.blocks_to_skip(8) == 2
    assert trimmed.physical_skip(8) == 4
    with pytest.raises(ValueError, match="non-negative"):
        trimmed.blocks_to_skip(-1)


def test_plan_skip_is_relative_to_the_first_transferred_object() -> None:
    """The window tail and the APC skip are resolved together, exactly once."""
    groups = [_group(0, 0, (0,), sw_size_tokens=8)]

    plans = build_group_transfer_plans(
        groups, [list(range(8))], for_retrieve=True, skip_first_n_tokens=28
    )

    # Three objects fall outside the window; only the remaining four tokens of
    # the fourth object are protected by the APC skip.
    assert plans[0].first_object == 3
    assert plans[0].physical_skip == 4
    with pytest.raises(ValueError, match="non-negative"):
        build_group_transfer_plans(groups, [list(range(8))], skip_first_n_tokens=-1)


def test_worker_scatter_consumes_server_trimmed_sliding_window_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker aligns server-trimmed objects without slicing them a second time."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import common_copy

    groups = [
        _group(0, 0, (0,)),
        _group(1, 1, (1,), copy_blocks_per_chunk=1, sw_size_tokens=8),
    ]
    plans = build_group_transfer_plans(
        groups,
        [list(range(8)), list(range(10, 18))],
        for_retrieve=True,
    )
    kv_caches = {"full": torch.empty(2, 8, 4, 1), "window": torch.empty(2, 8, 4, 1)}
    seen: list[tuple[list[int], list[int]]] = []

    def fake_scatter(
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        chunks: list[torch.Tensor],
        _blocks_per_chunk: int,
        **_kwargs: object,
    ) -> None:
        seen.append((block_ids, [int(chunk.item()) for chunk in chunks]))

    monkeypatch.setattr(common_copy, "scatter_cpu_to_paged_kv", fake_scatter)
    scatter_engine_groups(
        plans,
        kv_caches,
        [[torch.tensor(i) for i in range(4)], [torch.tensor(3)]],
    )

    assert seen == [(list(range(8)), [0, 1, 2, 3]), ([17], [3])]


def test_scatter_rejects_object_counts_that_do_not_match_the_plan() -> None:
    """A grouped retrieve must return exactly the planned object count."""
    plans = build_group_transfer_plans([_group(0, 0, (0,))], [[0, 1, 2, 3]])
    kv_caches = {"full": torch.empty(2, 4, 4, 1)}

    with pytest.raises(ValueError, match="expected 2"):
        scatter_engine_groups(plans, kv_caches, [[torch.tensor(0)]])
    with pytest.raises(ValueError, match="chunks_per_group has"):
        scatter_engine_groups(plans, kv_caches, [])


def test_group_major_wire_helpers_are_deterministic_and_strict() -> None:
    """Flatten/unflatten preserve deterministic ownership and reject bad counts."""
    chunks = [[torch.tensor([0]), torch.tensor([1])], [torch.tensor([2])]]
    flat = flatten_chunks_group_major(chunks)
    assert [chunk.item() for chunk in flat] == [0, 1, 2]
    assert unflatten_chunks_group_major(flat, [2, 1]) == chunks
    with pytest.raises(ValueError, match="negative"):
        unflatten_chunks_group_major(flat, [4, -1])
    with pytest.raises(ValueError, match="sum"):
        unflatten_chunks_group_major(flat, [1, 1])


def test_gather_uses_each_groups_copy_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution forwards each group's discovered native copy format."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import common_copy

    groups = [_group(0, 0, (0,), copy_format=11), _group(1, 1, (1,), copy_format=22)]
    kv_caches = {"a": torch.empty(1), "b": torch.empty(1)}
    plans = build_group_transfer_plans(groups, [[0, 1], [2, 3]])
    seen_formats: list[int] = []

    def fake_gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_per_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        seen_formats.append(cast(int, kwargs["engine_kv_format"]))
        return [torch.empty(1)]

    monkeypatch.setattr(common_copy, "gather_paged_kv_to_cpu", fake_gather)

    gather_engine_groups(plans, kv_caches)

    assert seen_formats == [11, 22]


def test_empty_transfer_is_valid_for_every_group() -> None:
    """Zero chunks is a valid no-op when every group is empty."""
    plans = build_group_transfer_plans(
        [_group(0, 0, (0,)), _group(1, 1, (1,))], [[], []]
    )
    assert [plan.total_objects for plan in plans] == [0, 0]
    assert [plan.selected_block_ids for plan in plans] == [(), ()]


def test_engine_registration_uses_authoritative_lmcache_chunk_size() -> None:
    """Engine-driven registration derives geometry from the LMCache chunk size."""
    infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0,), tokens_per_block=32),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=16,
            sw_size_tokens=64,
        ),
    ]
    layouts = [
        DiscoveredGroupLayout(32, 1, 8, 2, torch.float16, 3),
        DiscoveredGroupLayout(16, 1, 8, 1, torch.float16, 4),
    ]

    groups = registered_groups_from_engine_infos(infos, layouts, 256)

    assert [group.blocks_per_chunk for group in groups] == [8, 16]
    assert [group.copy_blocks_per_chunk for group in groups] == [8, 4]
    assert [group.kernel_group_id for group in groups] == [0, 1]
    assert [group.object_group_id for group in groups] == [0, 1]
    assert [group.chunk_tokens for group in groups] == [256, 256]
    assert groups[0].shape == torch.Size([2, 1, 256, 8])
    assert groups[1].shape == torch.Size([1, 64, 8])
    validate_registered_groups(groups, 2)

    with pytest.raises(ValueError, match="not divisible"):
        registered_groups_from_engine_infos(infos, layouts, 24)
    with pytest.raises(ValueError, match="layouts"):
        registered_groups_from_engine_infos(infos, layouts[:1], 256)


def test_both_paths_register_and_plan_the_same_transfer() -> None:
    """The server and worker adapters describe the same hybrid transfer.

    The LMCache-driven server derives its groups from the KV layer groups
    manager, while the Engine-driven worker derives them from the registration
    payload. Both must produce the same copy geometry and the same request
    plan, otherwise the two paths would disagree on what is stored.
    """
    # First Party
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
    from lmcache.v1.multiprocess.transfer_context.common_copy import (
        registered_groups_from_kv_layer_groups,
    )

    kv_format = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    # [kv, num_blocks, block_size, num_heads, head_size] per layer.
    tensors: list[DiscoverableKVCache] = [
        torch.empty(2, 64, 16, 8, 64, dtype=torch.float16) for _ in range(2)
    ]
    infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0,), tokens_per_block=16),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=16,
            sw_size_tokens=64,
        ),
    ]
    manager = KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[kv_format] * 2,
        engine_group_infos=infos,
    )

    server_groups = registered_groups_from_kv_layer_groups(manager, 256)
    worker_groups = registered_groups_from_engine_infos(
        infos,
        [DiscoveredGroupLayout(16, 1, 512, 2, torch.float16, kv_format)] * 2,
        256,
    )

    for server, worker in zip(server_groups, worker_groups, strict=True):
        assert server.kernel_group_id == worker.kernel_group_id
        assert server.object_group_id == worker.object_group_id
        assert server.engine_group_id == worker.engine_group_id
        assert server.layer_indices == worker.layer_indices
        assert server.tokens_per_block == worker.tokens_per_block
        assert server.slots_per_block == worker.slots_per_block
        assert server.blocks_per_chunk == worker.blocks_per_chunk
        assert server.copy_blocks_per_chunk == worker.copy_blocks_per_chunk
        assert server.chunk_tokens == worker.chunk_tokens
        assert server.shape == worker.shape
        assert server.dtype == worker.dtype
        assert server.engine_kv_format == worker.engine_kv_format
        # The manager rounds the window up to whole chunks; both descriptions
        # must still cover the same number of objects.
        assert server.objects_in_window == worker.objects_in_window

    block_ids = [list(range(48)), list(range(100, 148))]
    server_plans = build_group_transfer_plans(
        server_groups, block_ids, for_retrieve=True, skip_first_n_tokens=32
    )
    worker_plans = build_group_transfer_plans(
        worker_groups, block_ids, for_retrieve=True, skip_first_n_tokens=32
    )

    for server_plan, worker_plan in zip(server_plans, worker_plans, strict=True):
        assert server_plan.selected_block_ids == worker_plan.selected_block_ids
        assert server_plan.total_objects == worker_plan.total_objects
        assert server_plan.first_object == worker_plan.first_object
        assert server_plan.transfer_objects == worker_plan.transfer_objects
        assert server_plan.physical_skip == worker_plan.physical_skip

    # The windowed group keeps only the last object and only its last four
    # blocks per chunk; the full-attention group keeps everything.
    assert server_plans[0].transfer_objects == 3
    assert server_plans[1].transfer_objects == 1
    assert server_plans[1].selected_block_ids == (144, 145, 146, 147)
    assert server_plans[1].physical_skip == 0
