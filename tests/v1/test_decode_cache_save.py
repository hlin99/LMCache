# SPDX-License-Identifier: Apache-2.0

"""Tests for decode cache save behavior in from_request_tracker.

Verifies that when save_decode_cache=True, the chunk_boundary skip logic
does not prevent saves during the decode phase.
"""

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta, RequestTracker


def _make_tracker(
    prompt_len: int,
    token_ids: list[int],
    num_saved_tokens: int = 0,
    is_decode_phase: bool = False,
    block_ids: list[int] | None = None,
) -> RequestTracker:
    """Create a RequestTracker for testing."""
    if block_ids is None:
        # Each block holds block_size tokens; allocate enough blocks
        block_ids = list(range((len(token_ids) + 15) // 16))
    tracker = RequestTracker(
        req_id="test-req",
        prompt_len=prompt_len,
        token_ids=token_ids,
        allocated_block_ids=block_ids,
        num_saved_tokens=num_saved_tokens,
    )
    tracker.is_decode_phase = is_decode_phase
    return tracker


class TestDecodeCacheSave:
    """Tests for save_decode_cache behavior in from_request_tracker."""

    def test_decode_save_not_skipped_when_enabled(self) -> None:
        """When save_decode_cache=True, decode-phase saves should not be
        skipped by the chunk_boundary check."""
        prompt_len = 10
        # Simulate: prompt fully saved, now in decode with one extra token
        token_ids = list(range(prompt_len + 1))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            load_spec=None,
            discard_partial_chunks=False,
            save_decode_cache=True,
        )

        # Should NOT return None; a ReqMeta should be created
        assert result is not None
        assert result.save_spec is not None
        assert result.save_spec.can_save is True

    def test_decode_save_skipped_when_disabled(self) -> None:
        """When save_decode_cache=False (default), decode-phase saves
        should be skipped."""
        prompt_len = 10
        token_ids = list(range(prompt_len + 1))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            load_spec=None,
            discard_partial_chunks=False,
            save_decode_cache=False,
        )

        # Should return None since decode saves are disabled
        assert result is None

    def test_decode_save_updates_num_saved_tokens(self) -> None:
        """During decode with save_decode_cache=True and
        discard_partial_chunks=False, num_saved_tokens should
        be updated to include decode tokens."""
        prompt_len = 10
        token_ids = list(range(prompt_len + 5))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=256,
            load_spec=None,
            discard_partial_chunks=False,
            save_decode_cache=True,
        )

        assert result is not None
        # With discard_partial_chunks=False, all tokens should be saved
        assert tracker.num_saved_tokens == prompt_len + 5
        assert len(result.token_ids) == prompt_len + 5

    def test_decode_save_respects_chunk_boundary_with_discard(self) -> None:
        """During decode with save_decode_cache=True and
        discard_partial_chunks=True, saves should still happen but
        tokens are rounded to chunk boundaries."""
        prompt_len = 256
        chunk_size = 256
        # 1 decode token beyond prompt
        token_ids = list(range(prompt_len + 1))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=chunk_size,
            load_spec=None,
            discard_partial_chunks=True,
            save_decode_cache=True,
        )

        # Should create a ReqMeta (not skipped)
        assert result is not None
        # But tokens saved should be rounded down to chunk boundary
        assert len(result.token_ids) == 256

    def test_decode_save_at_chunk_boundary(self) -> None:
        """When decode tokens reach a chunk boundary with
        discard_partial_chunks=True and save_decode_cache=True,
        new tokens should be saved."""
        prompt_len = 256
        chunk_size = 256
        # Exactly at next chunk boundary
        token_ids = list(range(prompt_len + chunk_size))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=chunk_size,
            load_spec=None,
            discard_partial_chunks=True,
            save_decode_cache=True,
        )

        assert result is not None
        assert result.save_spec is not None
        assert result.save_spec.can_save is True
        # Should save up to 512 tokens (2 chunks)
        assert len(result.token_ids) == 512
        assert tracker.num_saved_tokens == 512

    def test_prefill_chunk_boundary_still_applies(self) -> None:
        """During prefill (not decode), the chunk_boundary skip should
        still apply regardless of save_decode_cache setting."""
        prompt_len = 512
        chunk_size = 256
        # First chunk of prefill: 100 tokens
        token_ids = list(range(100))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=0,
            is_decode_phase=False,
        )

        # First call: prefill with 100 tokens (not at chunk boundary)
        result = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            lmcache_chunk_size=chunk_size,
            load_spec=None,
            discard_partial_chunks=True,
            save_decode_cache=True,
        )

        # Should still create ReqMeta (num_saved_tokens == 0)
        assert result is not None
        # num_tokens_to_save = 100 // 256 * 256 = 0
        assert tracker.num_saved_tokens == 0

    def test_decode_save_incremental(self) -> None:
        """Simulate multiple decode steps with save_decode_cache=True
        and discard_partial_chunks=False, verifying incremental saves."""
        prompt_len = 10
        chunk_size = 256

        # Start with prompt fully saved
        token_ids = list(range(prompt_len))
        tracker = _make_tracker(
            prompt_len=prompt_len,
            token_ids=token_ids,
            num_saved_tokens=prompt_len,
            is_decode_phase=True,
        )

        # Simulate 3 decode steps
        for i in range(3):
            # Add one decode token
            tracker.token_ids.append(prompt_len + i)
            prev_saved = tracker.num_saved_tokens

            result = ReqMeta.from_request_tracker(
                tracker,
                block_size=16,
                lmcache_chunk_size=chunk_size,
                load_spec=None,
                discard_partial_chunks=False,
                save_decode_cache=True,
            )

            assert result is not None, (
                f"Decode step {i}: from_request_tracker should not return None"
            )
            assert result.save_spec is not None
            assert result.save_spec.can_save is True
            # num_saved_tokens should increase by 1 each step
            assert tracker.num_saved_tokens == prev_saved + 1
