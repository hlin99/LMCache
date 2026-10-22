
## PD disaggregation in LMCache multiprocess mode

![Architecture](architecture.png)

The figure above shows the PD architecture. The workflow goes as:

- The router sends one request to prefill instance with `max_token=1` and then wait for an event.
- The prefill instance finishes storing all the KV caches.
- The `request_telemetry` class reports the finished store event back to the router
- The router then send the request to decode instance.


**NOTE:** the current code only support 1p1d for now.
**NOTE:** the current code is not optimized for perf.

## How to run

Launch the following in 4 different terminal windows

```bash
# LMCache multi-process server
lmcache server --l1-size-gb 100 --eviction-policy LRU
# Prefill instance, enforce_eager for faster startup
LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi LMCACHE_REQUEST_TELEMETRY_ENDPOINT=http://localhost:5768/api/v1/telemetry vllm serve Qwen/Qwen3-14B --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}' --gpu-memory-utilization 0.7 --no-enable-prefix-caching --enforce-eager --port 8100
# Decode instance, enforce_eager for faster startup
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-14B --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}' --gpu-memory-utilization 0.7 --no-enable-prefix-caching --enforce-eager --port 8200
# proxy server
python disagg_proxy_server
```

And then you can test the implementation with example curl request

```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "Qwen/Qwen3-14B",
    "messages": [
      {
        "role": "user",
        "content": "1Hello, how are you? I am fine, thank you. But I am not sure about you. Can you tell me more about yourself? I am a human."
      }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Or run some benchmark

```
vllm bench serve \
    --dataset-name random \
    --random-input-len 5000 \
    --random-output-len 100 \
    --request-rate 0.25 \
    --num-prompts 10 \
    --random-range-ratio 0.0 \
    --ignore-eos \
    --backend vllm \
    --model Qwen/Qwen3-14B \
    --port 8000
```

## Experimental aligned vLLM-to-SGLang handoff

`vllm_to_sglang.py` is a separate, non-streaming matching-TP example.
It uses two LMCache MP servers and the existing NIXL P2P CPU transfer:

For a one-command Qwen3.5-0.8B functional check, run:

```bash
MODEL=/absolute/path/to/Qwen3.5-0.8B \
P_PYTHON=/path/to/vllm-env/bin/python \
D_PYTHON=/path/to/sglang-env/bin/python \
TP=2 bash verify_vllm_to_sglang.sh
```

With the existing `/workspace` experiment layout, simply run
`bash examples/disagg_prefill_mp/verify_vllm_to_sglang.sh` from the repository
root. Defaults are the sibling `kv-interop/.venv/bin/python` for prefill,
`kv-interop/sglang/.venv/bin/python` for decode, and the sibling
`models/Qwen3.5-0.8B` if present (otherwise `Qwen/Qwen3.5-0.8B`).
The environment variables above override these defaults.

The script prepares a two-chunk needle prompt, starts the coordinator, both CPU
daemons and engines, and checks the complete storage restore and NIXL P2P
counter, with no remaining read/write locks or active sessions after the
handoff. It then stops the engines and starts a fresh cache-disabled SGLang
control, comparing continuation token IDs with the same first-token input.
It prints `PASS` only if all checks succeed, keeps logs and JSON results in
the printed temporary directory, and cleans up its own process groups on exit.
Service stdout and stderr are shown live in the terminal and saved both in
per-service logs and the combined `run.log`. Python output is unbuffered.
It refuses to run if any of its ports are already in use.

The default is TP2 on GPUs 0/1 and 2/3; `TP=1` uses GPUs 0 and 1. Override
`PREFILL_GPUS` and `DECODE_GPUS` for different disjoint GPU sets.
`DISABLE_GPU_P2P=1` is the default functional-test workaround for local engine
collective initialization; set it to `0` on hosts that do not need it.
It disables NCCL P2P/CUMEM and custom all-reduce in both engines, not NIXL CPU
transfers.
Set `LMCACHE_MP_TRANSFER_MODE=engine_driven` to force worker-side GPU/CPU
copies on both engines; CPU-to-CPU transfer still uses NIXL. The default is
`auto` (LMCache-driven on the tested CUDA setup).
`CHUNK_SIZE=1088` and `MAMBA_BLOCK_SIZE=544` are the validated TP1/TP2 defaults.
For attention-only models such as Llama, set `MAMBA_BLOCK_SIZE=0` to disable
GDN packing; for example, use `CHUNK_SIZE=256`.
For other TP sizes, set these to match the actual writer block size and both
engines' checkpoint grids. Both Python environments must load this repository's
LMCache, with compatible engine dependencies and NIXL/UCX in the daemon
environment. This script is not an installer or a performance benchmark.

```text
vLLM GPU -> prefill LMCache CPU L1
         -> NIXL READ -> decode LMCache CPU L1 -> SGLang GPU
```

The prompt must contain an exact multiple of the LMCache chunk size. The
example does not implement partial chunks or change either engine's scheduler.
It requests exactly one output token from vLLM, takes its **token ID** without
detokenizing/re-tokenizing, and submits `prompt_ids + [first_token_id]` to
SGLang. SGLang can then restore the entire aligned prompt and compute just the
new input token. For hybrid models this restores `state(N)` and advances it
with that token; it never attempts to rewind a recurrent state.

Before submitting to SGLang, the example looks up and locks the full prefix on
the prefill daemon, waiting for STORE completion, then on the decode daemon,
waiting for P2P prefetch completion. It releases both sets of locks and sessions
on completion or failure. A fresh salt isolates each handoff from earlier runs.
The result is rejected unless SGLang reports the entire original prompt as an
LMCache storage hit. This is a functional prototype, not a production PD proxy:
it supports greedy generation only, no streaming, custom stop sequences,
penalties, prompt logprobs, heterogeneous TP, or multimodal inputs. Requests
must complete within the daemons' read-lock TTL.

`--tp-size` accepts any positive common TP size supported by the model and
both engines, with PP1. Both engines must use compatible per-rank head
sharding/replication and wire layouts; matching the TP count alone does not
establish that compatibility. Real-engine validation currently covers TP1
and TP2 only.

### Start the services

Use separate compatible engine environments, both loading this LMCache source.
Set `P_PYTHON` to the vLLM/LMCache Python and `D_PYTHON` to the SGLang Python.
The daemon environment also needs the NIXL Python package for its CUDA version
and a working UCX backend. Keep the engines' existing Torch versions when
installing optional dependencies.

The following configuration is for Qwen3.5-0.8B, BF16, DS convolution state,
FP32 recurrent state, and a resolved vLLM Mamba block size of 544. Both engines
must use identical weights, model name, token IDs, and hash configuration.
See the [adapter interoperability documentation](../../docs/design/integration/sglang/lmcache-unified-radix-cache.md)
for the experimental wire-format restrictions. The shared chunk size is
`1088 = lcm(544, 64)`.

Run each long-lived command in a separate terminal, exporting the same
environment variables:

```bash
export MODEL=/absolute/path/to/Qwen3.5-0.8B
export P_PYTHON=/path/to/vllm-env/bin/python
export D_PYTHON=/path/to/sglang-env/bin/python

# Coordinator.
"$P_PYTHON" -m lmcache.cli.main coordinator --host 127.0.0.1 --port 19300

# Prefill-side CPU cache.
"$P_PYTHON" -m lmcache.cli.main server \
  --host 127.0.0.1 --port 15655 --http-port 19000 \
  --chunk-size 1088 --separate-object-groups \
  --l1-size-gb 1 --l1-align-bytes 65536 --eviction-policy LRU \
  --instance-id aligned-prefill \
  --coordinator-url http://127.0.0.1:19300 \
  --coordinator-advertise-ip 127.0.0.1 \
  --p2p-advertise-url 127.0.0.1:18555

# Decode-side CPU cache: a distinct L1 allocation and distinct ports.
"$P_PYTHON" -m lmcache.cli.main server \
  --host 127.0.0.1 --port 15656 --http-port 19001 \
  --chunk-size 1088 --separate-object-groups \
  --l1-size-gb 1 --l1-align-bytes 65536 --eviction-policy LRU \
  --instance-id aligned-decode \
  --coordinator-url http://127.0.0.1:19300 \
  --coordinator-advertise-ip 127.0.0.1 \
  --p2p-advertise-url 127.0.0.1:18556

# Prefill engine.
CUDA_VISIBLE_DEVICES=0 VLLM_SSM_CONV_STATE_LAYOUT=DS \
"$P_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port 18000 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 2 \
  --max-num-batched-tokens 544 --gpu-memory-utilization 0.25 \
  --enforce-eager --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":15655}}'

# Decode engine; run from this example directory for the YAML path.
CUDA_VISIBLE_DEVICES=1 "$D_PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port 18002 \
  --tp-size 1 --dtype bfloat16 --context-length 4096 \
  --mem-fraction-static 0.25 --max-running-requests 2 \
  --disable-cuda-graph --attention-backend triton \
  --mamba-ssm-dtype float32 \
  --enable-lmcache --lmcache-config-file vllm_to_sglang.yaml
```

Wait for both engines' `/health` endpoints and verify both daemons report a
connected P2P peer at `/status`. For multiple hosts, replace the advertised
addresses, engine URLs, and cache URLs with routable addresses; the SGLang YAML
must point to its own local daemon.

### Submit an aligned prompt

Create `input_ids.json` as a JSON array of exactly 2176 prompt token IDs,
including the question at its end. Use the model's tokenizer, not dummy padding
tokens added just to align an existing prompt. Then run:

```bash
"$P_PYTHON" vllm_to_sglang.py \
  --model "$MODEL" --input-ids input_ids.json \
  --chunk-size 1088 --max-tokens 32 --output handoff.json
```

`handoff.json` contains the exact combined `output_ids`, the first token ID,
both engine responses, and the request salt. Decode the combined IDs with the
model tokenizer for display; the SGLang response text alone omits the first
token, which was produced by vLLM. The total output budget includes that first
token. A terminal first token or a one-token budget skips decode altogether.

For transport evidence, inspect the decode daemon's `/metrics`:
`lmcache_mp_l2_load_completed_requests_total{l2_name="p2p"}` must increase, and
its only L2 adapter should be the prefill peer. The transport logs must identify
the NIXL UCX backend. Localhost exercises NIXL CPU-to-CPU functionality, not
cross-node RDMA performance.

The functional validation used a 2176-token Qwen3.5-0.8B prompt and a 32-token
output budget. SGLang reported 2176 LMCache storage-hit tokens and computed one
new input token. P2P loaded three objects: two Full KV chunks and the final GDN
checkpoint. All 31 continuation token IDs matched a separate cold SGLang run
given the identical `prompt_ids + [vLLM_first_token_id]`. This does not assert
equality to an independently sampled, uninterrupted vLLM completion.

### Matching TP2 on both sides

Keep both CPU daemons and the hybrid YAML unchanged. Start vLLM with
`CUDA_VISIBLE_DEVICES=0,1 --tensor-parallel-size 2` (the environment assignment
precedes the Python command), and SGLang with `CUDA_VISIBLE_DEVICES=2,3` and
`--tp-size 2`. Pass `--tp-size 2` to `vllm_to_sglang.py` as well:

```bash
"$P_PYTHON" vllm_to_sglang.py \
  --model "$MODEL" --input-ids input_ids.json \
  --chunk-size 1088 --tp-size 2 --max-tokens 32 --output handoff-tp2.json
```

The readiness lookups cover both KV ranks. Every rank has its own Full KV
and GDN objects, with one lookup reader per object, not two readers of a
replicated object. For Qwen3.5-0.8B each rank owns one Full KV head and eight
GDN heads. Rank 0 transfers to rank 0, and rank 1 to rank 1; this does not
implement TP2-to-TP1 or TP4-to-TP2 resharding. The vLLM block-size setting must
still match the actual writer configuration.

The TP2-to-TP2 functional run used GPUs 0/1 for vLLM and GPUs 2/3 for
SGLang. NIXL loaded six objects (two Full KV chunks and one final GDN
checkpoint per rank). SGLang restored all 2176 prompt tokens and computed
one new input token. All 31 continuation IDs matched an independent cold
TP2 SGLang run with the same first-token input.

On the validation host, vLLM's local GPU-P2P collective initialization stalled.
The functional run used `NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0` in the vLLM
environment and added `--disable-custom-all-reduce` to its launch command.
These are host-specific collective workarounds, not requirements of the
LMCache NIXL CPU transfer, and the run makes no performance claim.