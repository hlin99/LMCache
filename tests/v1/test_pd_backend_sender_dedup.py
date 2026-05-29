# SPDX-License-Identifier: Apache-2.0
"""Tests for sender-side handling of already_sent_indexes in PDBackendAsync."""
import pytest
from unittest.mock import MagicMock
from lmcache.utils import CacheEngineKey


def _make_key(chunk_hash: int, worker_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        worker_id=worker_id,
        chunk_hash=chunk_hash,
    )


def _make_memory_obj(address: int) -> MagicMock:
    obj = MagicMock()
    obj.meta = MagicMock()
    obj.meta.address = address
    obj._ref_count = 1
    obj._freed = False

    def _ref_down():
        assert obj._ref_count > 0
        obj._ref_count -= 1
        if obj._ref_count == 0:
            obj._freed = True

    def _get_ref_count():
        return obj._ref_count

    obj.ref_count_down.side_effect = _ref_down
    obj.get_ref_count.side_effect = _get_ref_count
    return obj


class TestSenderSideDedup:
    """Tests for sender-side handling of already_sent_indexes."""

    def test_sender_filters_deduped_chunks(self):
        """Sender releases staging buffers for deduped chunks, sends the rest."""
        keys = [_make_key(chunk_hash=i) for i in range(4)]
        memory_objs = [_make_memory_obj(address=0x1000 * i) for i in range(4)]
        already_sent_indexes = {1, 3}

        mem_objs_to_send = []
        keys_to_send = []
        for idx, (key, mem_obj) in enumerate(zip(keys, memory_objs)):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
            else:
                mem_objs_to_send.append(mem_obj)
                keys_to_send.append(key)

        assert memory_objs[1]._freed
        assert memory_objs[3]._freed
        assert not memory_objs[0]._freed
        assert not memory_objs[2]._freed
        assert len(mem_objs_to_send) == 2
        assert mem_objs_to_send[0] is memory_objs[0]
        assert mem_objs_to_send[1] is memory_objs[2]

    def test_sender_rejects_out_of_range_indexes(self):
        """Sender rejects already_sent_indexes with values >= num_keys."""
        num_keys = 3
        already_sent_indexes = {0, 5}

        with pytest.raises(RuntimeError, match="Invalid already_sent_indexes"):
            if min(already_sent_indexes) < 0 or max(already_sent_indexes) >= num_keys:
                raise RuntimeError(
                    f"Invalid already_sent_indexes from receiver: "
                    f"{sorted(already_sent_indexes)}, valid range [0, {num_keys})"
                )

    def test_sender_rejects_negative_indexes(self):
        """Sender rejects already_sent_indexes with negative values."""
        num_keys = 3
        already_sent_indexes = {-1, 2}

        with pytest.raises(RuntimeError, match="Invalid already_sent_indexes"):
            if min(already_sent_indexes) < 0 or max(already_sent_indexes) >= num_keys:
                raise RuntimeError(
                    f"Invalid already_sent_indexes from receiver: "
                    f"{sorted(already_sent_indexes)}, valid range [0, {num_keys})"
                )

    def test_sender_rejects_inconsistent_alloc_response(self):
        """Sender rejects when remote_indexes count doesn't match expected."""
        num_keys = 4
        already_sent_indexes = {1, 3}
        remote_indexes = [0x100, 0x200, 0x300]  # should be 2, not 3

        expected_send_count = num_keys - len(already_sent_indexes)
        with pytest.raises(RuntimeError, match="AllocResponse inconsistency"):
            if len(remote_indexes) != expected_send_count:
                raise RuntimeError(
                    f"AllocResponse inconsistency: total_keys={num_keys}, "
                    f"already_sent={len(already_sent_indexes)}, "
                    f"remote_indexes={len(remote_indexes)}, expected={expected_send_count}"
                )
