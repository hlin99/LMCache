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

    def __init__(
        self,
        retrieve_result: RetrieveResult,
        commit_result: bool = True,
        commit_error: Exception | None = None,
        abort_error: Exception | None = None,
    ) -> None:
        self.retrieve_result = retrieve_result
        self.commit_result = commit_result
        self.commit_error = commit_error
        self.abort_error = abort_error
        self.committed = 0
        self.aborted = 0

    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> RetrieveResult:
        """Return the canned response for this fake transport."""
        return self.retrieve_result

    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Record a retrieve completion and report the canned outcome."""
        self.committed += 1
        if self.commit_error is not None:
            raise self.commit_error
        return self.commit_result

    def abort_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Record an unsuccessful retrieve completion."""
        self.aborted += 1
        if self.abort_error is not None:
            raise self.abort_error
        return True


def _wrapper(
    result: RetrieveResult,
    commit_result: bool = True,
    commit_error: Exception | None = None,
    abort_error: Exception | None = None,
) -> tuple[ContiguousTransferWrapper, _FakeContext]:
    """Build a contiguous wrapper over a fake context returning ``result``."""
    context = _FakeContext(result, commit_result, commit_error, abort_error)
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


def test_retrieve_returns_none_when_commit_fails() -> None:
    """An unsuccessful commit aborts once and reports a miss."""
    wrapper, context = _wrapper(([torch.zeros(2, 1, 4, 8)], [1]), commit_result=False)

    assert wrapper.retrieve(KEY, 0) is None
    assert (context.committed, context.aborted) == (1, 1)


def test_retrieve_aborts_and_reraises_when_commit_raises() -> None:
    """A commit error aborts the retrieve and surfaces the original error."""
    wrapper, context = _wrapper(
        ([torch.zeros(2, 1, 4, 8)], [1]), commit_error=RuntimeError("commit failed")
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        wrapper.retrieve(KEY, 0)

    assert (context.committed, context.aborted) == (1, 1)


def test_retrieve_keeps_commit_error_when_abort_also_raises() -> None:
    """A failing abort must not replace the commit error seen by the caller."""
    wrapper, context = _wrapper(
        ([torch.zeros(2, 1, 4, 8)], [1]),
        commit_error=RuntimeError("commit failed"),
        abort_error=RuntimeError("abort failed"),
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        wrapper.retrieve(KEY, 0)

    assert (context.committed, context.aborted) == (1, 1)
