# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.wrapper.contiguous import ContiguousTransferWrapper
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

KEY = IPCCacheServerKey(
    model_name="model",
    world_size=1,
    worker_id=0,
    token_ids=(1, 2, 3, 4),
    start=0,
    end=4,
    request_id="req",
)

RetrieveResult = list[torch.Tensor] | tuple[list[torch.Tensor], list[int]] | None


class _FakeContext:
    """Minimal engine-driven transport recording retrieve lifecycle calls."""

    def __init__(self, retrieve_result: RetrieveResult) -> None:
        self.retrieve_result = retrieve_result
        self.committed = 0
        self.aborted = 0

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> RetrieveResult:
        """Return the canned response for this fake transport."""
        return self.retrieve_result

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> None:
        """Record a successful retrieve completion."""
        self.committed += 1

    def abort_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> None:
        """Record an unsuccessful retrieve completion."""
        self.aborted += 1


def _wrapper(result: RetrieveResult) -> tuple[ContiguousTransferWrapper, _FakeContext]:
    """Build a contiguous wrapper over a fake context returning ``result``."""
    context = _FakeContext(result)
    return ContiguousTransferWrapper(context, 4), context  # type: ignore[arg-type]


def test_retrieve_accepts_legacy_and_structured_single_group_responses() -> None:
    """Both response shapes assemble one contiguous tensor and commit once."""
    chunks = [torch.zeros(2, 1, 4, 8), torch.ones(2, 1, 4, 8)]

    for result in (list(chunks), (list(chunks), [2])):
        wrapper, context = _wrapper(result)

        kv = wrapper.retrieve(KEY, 0)

        assert kv is not None
        assert kv.shape == torch.Size([2, 1, 8, 8])
        assert (context.committed, context.aborted) == (1, 0)


def test_retrieve_aborts_on_miss_for_every_empty_response() -> None:
    """A miss releases the retrieve without committing it."""
    empty_results: list[RetrieveResult] = [None, [], ([], [])]

    for result in empty_results:
        wrapper, context = _wrapper(result)

        assert wrapper.retrieve(KEY, 0) is None
        assert (context.committed, context.aborted) == (0, 1)


def test_retrieve_rejects_multi_group_responses_and_aborts() -> None:
    """A single contiguous tensor cannot represent several transfer groups."""
    wrapper, context = _wrapper(([torch.zeros(2, 1, 4, 8)] * 2, [1, 1]))

    with pytest.raises(ValueError, match="single transfer group"):
        wrapper.retrieve(KEY, 0)

    assert (context.committed, context.aborted) == (0, 1)


def test_retrieve_aborts_when_chunks_cannot_be_concatenated() -> None:
    """Malformed chunks abort the retrieve and surface the original error."""
    wrapper, context = _wrapper(([torch.zeros(2, 1, 4, 8), torch.zeros(3)], [2]))

    with pytest.raises(RuntimeError):
        wrapper.retrieve(KEY, 0)

    assert (context.committed, context.aborted) == (0, 1)
