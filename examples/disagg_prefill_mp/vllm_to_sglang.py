# SPDX-License-Identifier: Apache-2.0
"""Experimental greedy, chunk-aligned TP1 handoff through LMCache MP P2P."""

# Standard
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
import argparse
import json
import time
import urllib.request
import uuid

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def hold_prefix(
    client: RequestClient,
    key: IPCCacheServerKey,
    chunk_size: int,
    timeout: float,
) -> Iterator[None]:
    """Wait for a complete prefix and retain its lookup locks during handoff."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        attempt = replace(key, request_id=f"aligned-pd-{uuid.uuid4().hex}")
        try:
            client.lookup(attempt, 1).result(timeout=timeout)
            hit_chunks = None
            while hit_chunks is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for the aligned prefix")
                wait = min(remaining, 1.0)
                hit_chunks = client.wait_prefetch_status(
                    attempt.request_id, wait
                ).result(timeout=wait + 5)
            if hit_chunks * chunk_size == key.end:
                yield
                return
        finally:
            try:
                client.free_lookup_locks(attempt, 1).result(timeout=timeout)
            finally:
                client.end_session(attempt.request_id).result(timeout=timeout)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"Prefix [0, {key.end}) was not available within {timeout}s")


def handoff(
    *,
    model: str,
    input_ids: list[int],
    chunk_size: int,
    max_tokens: int,
    prefill_url: str,
    decode_url: str,
    prefill_cache: RequestClient,
    decode_cache: RequestClient,
    timeout: float = 120,
) -> dict[str, Any]:
    """Prefill once, transfer the aligned prefix, and continue from its first token."""
    if chunk_size <= 0 or not input_ids or len(input_ids) % chunk_size:
        raise ValueError("The nonempty prompt must be aligned to a positive chunk_size")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    for client in (prefill_cache, decode_cache):
        actual = client.get_chunk_size().result(timeout=timeout)
        if actual != chunk_size:
            raise ValueError(f"Server chunk size {actual} differs from {chunk_size}")
    salt = f"aligned-pd-{uuid.uuid4().hex}"
    prefill = request_json(
        f"{prefill_url.rstrip('/')}/v1/completions",
        {
            "model": model,
            "prompt": input_ids,
            "max_tokens": 1,
            "temperature": 0,
            "return_token_ids": True,
            "cache_salt": salt,
        },
        timeout,
    )
    if prefill["usage"]["prompt_tokens"] != len(input_ids):
        raise RuntimeError("Prefill did not process the exact input token sequence")
    choice = prefill["choices"][0]
    first_ids = choice["token_ids"]
    if len(first_ids) > 1 or any(type(token) is not int for token in first_ids):
        raise RuntimeError("Prefill must return at most one exact output token ID")
    result = {
        "input_ids": input_ids,
        "cache_salt": salt,
        "first_token_ids": first_ids,
        "prefill": prefill,
    }
    if choice["finish_reason"] == "stop":
        result["output_ids"] = first_ids
        return result
    if len(first_ids) != 1 or choice["finish_reason"] != "length":
        raise RuntimeError("Prefill did not return a nonterminal first token")
    if max_tokens == 1:
        result["output_ids"] = first_ids
        return result
    key = IPCCacheServerKey(
        model_name=model,
        world_size=1,
        worker_id=None,
        num_kv_readers=1,
        token_ids=tuple(input_ids),
        start=0,
        end=len(input_ids),
        request_id="",
        cache_salt=salt,
    )
    with (
        hold_prefix(prefill_cache, key, chunk_size, timeout),
        hold_prefix(decode_cache, key, chunk_size, timeout),
    ):
        continuation = request_json(
            f"{decode_url.rstrip('/')}/generate",
            {
                "input_ids": input_ids + first_ids,
                "cache_salt": salt,
                "sampling_params": {
                    "max_new_tokens": max_tokens - 1,
                    "temperature": 0,
                },
            },
            timeout,
        )
    meta = continuation["meta_info"]
    cached = meta["cached_tokens_details"]
    if (
        meta["prompt_tokens"] != len(input_ids) + 1
        or meta["cached_tokens"] != len(input_ids)
        or cached["storage"] != len(input_ids)
        or cached["device"] != 0
        or cached["host"] != 0
    ):
        raise RuntimeError(
            "Decode did not restore the entire prefix from LMCache: "
            f"{meta['cached_tokens_details']}"
        )
    result["decode"] = continuation
    result["output_ids"] = first_ids + continuation["output_ids"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:18000")
    parser.add_argument("--decode-url", default="http://127.0.0.1:18002")
    parser.add_argument("--prefill-cache", default="tcp://127.0.0.1:15655")
    parser.add_argument("--decode-cache", default="tcp://127.0.0.1:15656")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    input_ids = json.loads(args.input_ids.read_text())
    if not isinstance(input_ids, list) or any(
        type(token) is not int or token < 0 for token in input_ids
    ):
        parser.error("--input-ids must contain a JSON array of nonnegative token IDs")
    prefill_cache = RequestClientFactory.create(args.prefill_cache)
    try:
        decode_cache = RequestClientFactory.create(args.decode_cache)
        try:
            result = handoff(
                model=args.model,
                input_ids=input_ids,
                chunk_size=args.chunk_size,
                max_tokens=args.max_tokens,
                prefill_url=args.prefill_url,
                decode_url=args.decode_url,
                prefill_cache=prefill_cache,
                decode_cache=decode_cache,
                timeout=args.timeout,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"output_ids": result["output_ids"]}))
        finally:
            decode_cache.close()
    finally:
        prefill_cache.close()


if __name__ == "__main__":
    main()
