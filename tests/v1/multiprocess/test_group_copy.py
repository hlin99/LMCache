# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context.group_copy import (
    RegisteredGroup,
    flatten_chunks_group_major,
    gather_engine_groups,
    plan_group_copy,
    unflatten_chunks_group_major,
    validate_group_block_ids,
    validate_registered_groups,
)


def _group(
    object_group_id: int,
    engine_group_id: int,
    layer_indices: tuple[int, ...],
    *,
    tokens_per_block: int = 4,
    slots_per_block: int = 4,
    blocks_per_chunk: int = 2,
    copy_blocks_per_chunk: int = 2,
    sw_size_tokens: int = -1,
    copy_format: int = 10,
) -> RegisteredGroup:
    """Build compact transfer metadata for planner contract tests."""
    return RegisteredGroup(
        object_group_id=object_group_id,
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
    valid = [_group(0, 0, (0,)), _group(1, 0, (1,))]
    validate_registered_groups(valid, 2)

    with pytest.raises(ValueError, match="object group IDs"):
        validate_registered_groups([_group(1, 0, (0,))], 1)
    with pytest.raises(ValueError, match="dense"):
        validate_registered_groups([_group(0, 1, (0,))], 1)
    with pytest.raises(ValueError, match="duplicated"):
        validate_registered_groups([_group(0, 0, (0,)), _group(1, 0, (0,))], 1)
    with pytest.raises(ValueError, match="missing"):
        validate_registered_groups([_group(0, 0, (0,))], 2)
    with pytest.raises(ValueError, match="outside"):
        validate_registered_groups([_group(0, 0, (2,))], 2)


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
    kv_caches = {
        "layer_0": torch.empty(2, 4, 4, 1),
        "layer_1": torch.empty(2, 2, 2, 1),
    }

    plans = plan_group_copy(
        kv_caches,
        [[0, 1, 2, 3], [10, 11]],
        groups,
    )

    assert [plan.engine_group_id for plan in plans] == [0, 0]
    assert [plan.blocks_in_chunk for plan in plans] == [2, 1]
    assert [plan.group.engine_kv_format for plan in plans] == [11, 22]
    assert [list(plan.kv_subset) for plan in plans] == [["layer_0"], ["layer_1"]]


def test_plan_selects_sliding_window_tail_blocks_and_objects() -> None:
    """Retrieve retains only window-valid objects and trailing subchunk blocks."""
    groups = [
        _group(0, 0, (0,)),
        _group(
            1,
            1,
            (1,),
            copy_blocks_per_chunk=1,
            sw_size_tokens=8,
        ),
    ]
    kv_caches = {
        "full": torch.empty(2, 8, 4, 1),
        "window": torch.empty(2, 8, 4, 1),
    }
    block_ids = [
        list(range(8)),
        [10, 11, 12, 13, 14, 15, 16, 17],
    ]

    store_plans = plan_group_copy(kv_caches, block_ids, groups)
    retrieve_plans = plan_group_copy(kv_caches, block_ids, groups, for_retrieve=True)

    assert store_plans[1].flat_block_ids == [11, 13, 15, 17]
    assert retrieve_plans[1].first_object == 3
    assert retrieve_plans[1].flat_block_ids == [17]
    assert retrieve_plans[0].first_object == 0


def test_compressed_group_converts_logical_skip_to_physical_slots() -> None:
    """Compressed logical-token geometry converts APC skips exactly."""
    group = _group(
        0,
        0,
        (0,),
        tokens_per_block=8,
        slots_per_block=2,
        blocks_per_chunk=1,
        copy_blocks_per_chunk=1,
    )
    assert group.physical_skip(8) == 2
    with pytest.raises(ValueError, match="not representable"):
        group.physical_skip(1)


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
    from lmcache.v1.multiprocess.transfer_context import group_copy

    groups = [
        _group(0, 0, (0,), copy_format=11),
        _group(1, 1, (1,), copy_format=22),
    ]
    kv_caches = {"a": torch.empty(1), "b": torch.empty(1)}
    plans = plan_group_copy(kv_caches, [[0, 1], [2, 3]], groups)
    seen_formats: list[int] = []

    def fake_gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_per_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        seen_formats.append(cast(int, kwargs["engine_kv_format"]))
        return [torch.empty(1)]

    monkeypatch.setattr(group_copy, "gather_paged_kv_to_cpu", fake_gather)

    gather_engine_groups(plans)

    assert seen_formats == [11, 22]


def test_empty_transfer_is_valid_for_every_group() -> None:
    """Zero chunks is a valid no-op when every group is empty."""
    groups = [_group(0, 0, (0,)), _group(1, 1, (1,))]
    plans = plan_group_copy(
        {"a": torch.empty(1), "b": torch.empty(1)},
        [[], []],
        groups,
    )
    assert [plan.num_chunks for plan in plans] == [0, 0]
