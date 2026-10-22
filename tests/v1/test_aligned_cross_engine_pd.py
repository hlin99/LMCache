# SPDX-License-Identifier: Apache-2.0
"""Server-free checks for the aligned cross-engine PD example."""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch
import importlib.util

# Third Party
import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples/disagg_prefill_mp/vllm_to_sglang.py"
)
_spec = importlib.util.spec_from_file_location("aligned_pd_example", _EXAMPLE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load PD example from {_EXAMPLE}")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)
IPCCacheServerKey = pd.IPCCacheServerKey


def cache_client() -> Mock:
    client = Mock()
    client.get_chunk_size.return_value.result.return_value = 4
    client.wait_prefetch_status.return_value.result.return_value = 2
    return client


def prefill_response() -> dict:
    return {
        "choices": [{"token_ids": [99], "finish_reason": "length"}],
        "usage": {"prompt_tokens": 8},
    }


def decode_response(storage: int = 8) -> dict:
    return {
        "output_ids": [100, 101],
        "meta_info": {
            "prompt_tokens": 9,
            "cached_tokens": storage,
            "cached_tokens_details": {"storage": storage, "device": 0, "host": 0},
        },
    }


def run_handoff(prefill: Mock, decode: Mock, **kwargs) -> dict:
    return pd.handoff(
        model="model",
        input_ids=kwargs.pop("input_ids", list(range(8))),
        chunk_size=4,
        max_tokens=kwargs.pop("max_tokens", 3),
        prefill_url="http://prefill",
        decode_url="http://decode",
        prefill_cache=prefill,
        decode_cache=decode,
        **kwargs,
    )


def test_handoff_preserves_token_ids_salt_budget_and_prefix_locks():
    prefill, decode = cache_client(), cache_client()

    def respond(url, payload, timeout):
        if url.endswith("/v1/completions"):
            assert payload["max_tokens"] == 1
            assert payload["return_token_ids"] is True
            return prefill_response()
        assert url == "http://decode/generate"
        assert payload["input_ids"] == list(range(8)) + [99]
        assert payload["sampling_params"]["max_new_tokens"] == 2
        for client in (prefill, decode):
            key = client.lookup.call_args.args[0]
            assert key.end == 8 and key.token_ids == tuple(range(8))
            assert key.world_size == 1
            assert key.num_kv_readers == 1
            assert client.lookup.call_args.args[1] == 1
            assert key.cache_salt == payload["cache_salt"]
            assert client.wait_prefetch_status.called
            client.free_lookup_locks.assert_not_called()
        return decode_response()

    with patch.object(pd, "request_json", side_effect=respond):
        result = run_handoff(prefill, decode)
    assert result["output_ids"] == [99, 100, 101]
    for client in (prefill, decode):
        client.free_lookup_locks.assert_called_once()
        assert client.free_lookup_locks.call_args.args[1] == 1
        client.end_session.assert_called_once()


@pytest.mark.parametrize("input_ids", [[], [1, 2, 3]])
def test_unaligned_prompt_rejected_before_requests(input_ids):
    prefill, decode = cache_client(), cache_client()
    with patch.object(pd, "request_json") as request:
        with pytest.raises(ValueError, match="aligned"):
            run_handoff(prefill, decode, input_ids=input_ids)
    request.assert_not_called()
    prefill.lookup.assert_not_called()


def test_server_chunk_size_mismatch_rejected():
    prefill, decode = cache_client(), cache_client()
    decode.get_chunk_size.return_value.result.return_value = 8
    with patch.object(pd, "request_json") as request:
        with pytest.raises(ValueError, match="Server chunk size"):
            run_handoff(prefill, decode)
    request.assert_not_called()


@pytest.mark.parametrize("storage", [0, 4])
def test_partial_restore_is_not_reported_as_pd_success(storage):
    with patch.object(
        pd, "request_json", side_effect=[prefill_response(), decode_response(storage)]
    ):
        with pytest.raises(RuntimeError, match="entire prefix"):
            run_handoff(cache_client(), cache_client())


def test_http_failure_releases_both_prefixes():
    prefill, decode = cache_client(), cache_client()
    with patch.object(
        pd,
        "request_json",
        side_effect=[prefill_response(), RuntimeError("HTTP failed")],
    ):
        with pytest.raises(RuntimeError, match="HTTP failed"):
            run_handoff(prefill, decode)
    for client in (prefill, decode):
        client.free_lookup_locks.assert_called_once()
        client.end_session.assert_called_once()


@pytest.mark.parametrize("terminal", [True, False])
def test_terminal_first_token_or_single_token_budget_skips_decode(terminal):
    prefill, decode = cache_client(), cache_client()
    response = prefill_response()
    if terminal:
        response["choices"][0]["finish_reason"] = "stop"
    with patch.object(pd, "request_json", return_value=response) as request:
        result = run_handoff(prefill, decode, max_tokens=3 if terminal else 1)
    assert result["output_ids"] == [99]
    assert request.call_count == 1
    prefill.lookup.assert_not_called()
    decode.lookup.assert_not_called()


def test_missing_first_token_fails_without_retokenizing_text():
    response = prefill_response()
    response["choices"][0]["token_ids"] = []
    with patch.object(pd, "request_json", return_value=response):
        with pytest.raises(RuntimeError, match="nonterminal first token"):
            run_handoff(cache_client(), cache_client())


def test_prefix_retry_releases_partial_hit_before_next_lookup():
    client = cache_client()
    client.wait_prefetch_status.return_value.result.side_effect = [1, 2]
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=None,
        num_kv_readers=1,
        token_ids=tuple(range(8)),
        start=0,
        end=8,
        request_id="test",
    )
    with patch.object(pd.time, "sleep"):
        with pd.hold_prefix(client, key, 4, 10):
            assert client.lookup.call_count == 2
            assert client.free_lookup_locks.call_count == 1
    assert client.free_lookup_locks.call_count == 2
    assert client.end_session.call_count == 2
    assert (
        client.lookup.call_args_list[0].args[0].request_id
        != client.lookup.call_args_list[1].args[0].request_id
    )


def test_prefix_timeout_releases_session():
    client = cache_client()
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=None,
        num_kv_readers=1,
        token_ids=(1, 2, 3, 4),
        start=0,
        end=4,
        request_id="test",
    )
    with patch.object(pd.time, "monotonic", side_effect=[0, 0, 2]):
        with pytest.raises(TimeoutError, match="Timed out"):
            with pd.hold_prefix(client, key, 4, 1):
                pytest.fail("An unfinished prefix must not be yielded")
    client.free_lookup_locks.assert_called_once()
    client.end_session.assert_called_once()
