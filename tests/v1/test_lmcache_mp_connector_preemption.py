# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPConnectorMetadata,
)


def _new_connector_without_init() -> LMCacheMPConnector:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    return connector


def test_build_connector_meta_sets_need_flush_from_preemption_signal() -> None:
    connector = _new_connector_without_init()
    connector._process_retrieve_requests = lambda metadata: None
    connector._process_new_requests = lambda scheduler_output, metadata: None
    connector._process_cached_requests = lambda scheduler_output, metadata: None
    connector._report_block_allocation_deltas = lambda scheduler_output: None

    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=["req-1"])
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(metadata, LMCacheMPConnectorMetadata)
    assert metadata.need_flush is True


def test_build_connector_meta_keeps_need_flush_false_without_signal() -> None:
    connector = _new_connector_without_init()
    connector._process_retrieve_requests = lambda metadata: None
    connector._process_new_requests = lambda scheduler_output, metadata: None
    connector._process_cached_requests = lambda scheduler_output, metadata: None
    connector._report_block_allocation_deltas = lambda scheduler_output: None

    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=[],
            resumed_from_preemption=[],
        ),
        preempted_req_ids=[],
        evicted_req_ids=[],
    )
    metadata = connector.build_connector_meta(scheduler_output)
    assert isinstance(metadata, LMCacheMPConnectorMetadata)
    assert metadata.need_flush is False


@pytest.mark.parametrize(
    "scheduler_output",
    [
        SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(
                resumed_req_ids=[],
                resumed_from_preemption=[False, True],
            )
        ),
        SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(
                resumed_req_ids=[],
                resumed_from_preemption=[],
            ),
            preempted_req_ids=["req-1"],
        ),
        SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(
                resumed_req_ids=[],
                resumed_from_preemption=[],
            ),
            preempted_req_ids=[],
            evicted_req_ids=["req-2"],
        ),
    ],
)
def test_scheduler_step_needs_flush_for_all_supported_signals(
    scheduler_output: SimpleNamespace,
) -> None:
    connector = _new_connector_without_init()
    assert connector._scheduler_step_needs_flush(scheduler_output) is True


def test_handle_preemptions_forwards_flush_hint_to_worker_adapter() -> None:
    connector = _new_connector_without_init()
    connector._role = KVConnectorRole.WORKER
    connector.worker_adapter = MagicMock()
    metadata = LMCacheMPConnectorMetadata()

    connector.handle_preemptions(metadata)
    connector.worker_adapter.handle_preemptions.assert_called_once_with(False)

    connector.worker_adapter.reset_mock()
    metadata.need_flush = True
    connector.handle_preemptions(metadata)
    connector.worker_adapter.handle_preemptions.assert_called_once_with(True)


def test_scheduler_step_needs_flush_conservative_on_unknown_schema() -> None:
    connector = _new_connector_without_init()
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(unexpected_field=["req-1"])
    )
    assert connector._scheduler_step_needs_flush(scheduler_output) is True


def test_scheduler_step_needs_flush_false_for_recognized_no_preemption() -> None:
    connector = _new_connector_without_init()
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=[],
            resumed_from_preemption=[],
        )
    )
    assert connector._scheduler_step_needs_flush(scheduler_output) is False
