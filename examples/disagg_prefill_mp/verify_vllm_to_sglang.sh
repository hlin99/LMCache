#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
DEFAULT_MODEL=Qwen/Qwen3.5-0.8B
if [[ -d "${WORKSPACE_DIR}/models/Qwen3.5-0.8B" ]]; then
    DEFAULT_MODEL="${WORKSPACE_DIR}/models/Qwen3.5-0.8B"
fi
MODEL="${MODEL:-${DEFAULT_MODEL}}"
P_PYTHON="${P_PYTHON:-${WORKSPACE_DIR}/kv-interop/.venv/bin/python}"
D_PYTHON="${D_PYTHON:-${WORKSPACE_DIR}/kv-interop/sglang/.venv/bin/python}"
TP="${TP:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-1088}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-544}"
DISABLE_GPU_P2P="${DISABLE_GPU_P2P:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
export LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-auto}"

case "${LMCACHE_MP_TRANSFER_MODE}" in
    auto|lmcache_driven|engine_driven) ;;
    *)
        echo "LMCACHE_MP_TRANSFER_MODE must be auto, lmcache_driven or engine_driven" >&2
        exit 1
        ;;
esac

for value in "${TP}" "${CHUNK_SIZE}" "${STARTUP_TIMEOUT}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TP, CHUNK_SIZE and STARTUP_TIMEOUT must be positive integers" >&2
        exit 1
    fi
done
if [[ ! "${MAMBA_BLOCK_SIZE}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "MAMBA_BLOCK_SIZE must be a nonnegative integer (0 for attention-only models)" >&2
    exit 1
fi
PREFILL_BATCH_SIZE="${MAMBA_BLOCK_SIZE}"
if [[ "${MAMBA_BLOCK_SIZE}" == 0 ]]; then
    PREFILL_BATCH_SIZE="${CHUNK_SIZE}"
fi
if [[ "${DISABLE_GPU_P2P}" != 0 && "${DISABLE_GPU_P2P}" != 1 ]]; then
    echo "DISABLE_GPU_P2P must be 0 or 1" >&2
    exit 1
fi
for executable in "${P_PYTHON}" "${D_PYTHON}" setsid curl tee; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "Missing executable: ${executable}" >&2
        echo "Set P_PYTHON and D_PYTHON to your installed engine environments." >&2
        exit 1
    fi
done

p_gpus=""
d_gpus=""
for ((i = 0; i < TP; i++)); do
    p_gpus="${p_gpus:+${p_gpus},}${i}"
    d_gpus="${d_gpus:+${d_gpus},}$((TP + i))"
done
PREFILL_GPUS="${PREFILL_GPUS:-${p_gpus}}"
DECODE_GPUS="${DECODE_GPUS:-${d_gpus}}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LMCACHE_TRACK_USAGE=false VLLM_NO_USAGE_STATS=1 PYTHONUNBUFFERED=1
WORK_DIR="$(mktemp -d -t lmcache-vllm-sglang-pd-XXXXXX)"
declare -A PIDS=()
exec {TERMINAL_OUT}>&1 {TERMINAL_ERR}>&2
exec > >(tee "${WORK_DIR}/run.log") 2>&1
LOG_PID=$!

stop_service() {
    local name="$1" pid="${PIDS[$1]:-}" i
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for ((i = 0; i < 150; i++)); do
        if ! kill -0 -- "-${pid}" 2>/dev/null && ! kill -0 "${pid}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    PIDS["${name}"]=""
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    for name in cold decode prefill decode-cache prefill-cache coordinator; do
        stop_service "${name}"
    done
    echo "Logs and results: ${WORK_DIR}"
    exec 1>&"${TERMINAL_OUT}" 2>&"${TERMINAL_ERR}"
    wait "${LOG_PID}" || status=$?
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_service() {
    local name="$1"
    shift
    setsid bash -c 'set -o pipefail; log="$1"; shift; "$@" 2>&1 | tee "${log}"' \
        bash "${WORK_DIR}/${name}.log" "$@" &
    PIDS["${name}"]=$!
}

wait_for_health() {
    local name="$1" url="$2"
    for ((attempt = 0; attempt < STARTUP_TIMEOUT; attempt++)); do
        if ! kill -0 "${PIDS[$name]}" 2>/dev/null; then
            echo "${name} exited during startup" >&2
            tail -n 80 "${WORK_DIR}/${name}.log" >&2
            return 1
        fi
        if curl --connect-timeout 1 --max-time 2 -fsS "${url}" >/dev/null 2>&1; then
            echo "${name} ready"
            return
        fi
        sleep 1
    done
    echo "Timed out waiting for ${name}" >&2
    tail -n 80 "${WORK_DIR}/${name}.log" >&2
    return 1
}

echo "TP=${TP}; prefill GPUs=${PREFILL_GPUS}; decode GPUs=${DECODE_GPUS}"
echo "MODEL=${MODEL}"
echo "P_PYTHON=${P_PYTHON}"
echo "D_PYTHON=${D_PYTHON}"
echo "LMCACHE_MP_TRANSFER_MODE=${LMCACHE_MP_TRANSFER_MODE}"
echo "Logs and results: ${WORK_DIR}"
cd "${EXAMPLE_DIR}"
"${P_PYTHON}" - "${MODEL}" "${TP}" "${CHUNK_SIZE}" "${MAMBA_BLOCK_SIZE}" \
    "${PREFILL_GPUS}" "${DECODE_GPUS}" "${WORK_DIR}" <<'PY'
import json
import socket
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import _load_nixl

model, tp, chunk, block, p_gpus, d_gpus, work_dir = sys.argv[1:]
tp, chunk, block = int(tp), int(chunk), int(block)
if (block and chunk % block) or chunk % 64:
    raise SystemExit("CHUNK_SIZE must align to MAMBA_BLOCK_SIZE and SGLang's 64-token grid")
if 2 * chunk + 32 > 4096:
    raise SystemExit("Two prompt chunks plus output must fit the 4096-token context")
p_ids, d_ids = p_gpus.split(","), d_gpus.split(",")
if (
    len(p_ids) != tp or len(d_ids) != tp
    or len(set(p_ids + d_ids)) != 2 * tp
):
    raise SystemExit("Specify TP distinct GPUs per engine, with no overlap")
available = {str(i) for i in range(torch.cuda.device_count())}
if not set(p_ids + d_ids) <= available:
    raise SystemExit(f"GPU IDs must be among {sorted(available)}")
for port in (19300, 15655, 15656, 19000, 19001, 18555, 18556, 18000, 18002):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
_load_nixl()
tokenizer = AutoTokenizer.from_pretrained(model)
passage = (
    "The harbor has three piers and the keeper is named Ada. "
    "Every evening the keeper checks the lens and records the weather. "
    "The daily log lists the tide, wind speed, and arriving ships. "
)
prefix = tokenizer.encode(
    "Archive fact: the secret access code is ZEPHYR-73921.\n" + passage * chunk,
    add_special_tokens=False,
)
question = tokenizer.encode(
    "\nQuestion: What is the secret access code stated at the beginning? "
    "Return only the exact code.\nAnswer:",
    add_special_tokens=False,
)
count = 2 * chunk
if len(question) >= count or len(prefix) < count - len(question):
    raise SystemExit("Cannot construct an aligned prompt with this tokenizer")
ids = prefix[:count - len(question)] + question
if "ZEPHYR-73921" not in tokenizer.decode(ids):
    raise SystemExit("Aligned prompt lost the needle")
Path(work_dir, "input_ids.json").write_text(json.dumps(ids))
print(f"Prepared {len(ids)} prompt tokens")
PY
"${D_PYTHON}" -c 'import torch; import sglang'

cat >"${WORK_DIR}/lmcache.yaml" <<YAML
mp_host: 127.0.0.1
mp_port: 15656
extra_config:
  sglang.vllm_fused_kv: true
  sglang.vllm_mamba_block_size: ${MAMBA_BLOCK_SIZE}
YAML

start_service coordinator "${P_PYTHON}" -m lmcache.cli.main coordinator \
    --host 127.0.0.1 --port 19300
wait_for_health coordinator http://127.0.0.1:19300/healthz

for side in prefill decode; do
    if [[ "${side}" == prefill ]]; then
        port=15655; http_port=19000; transfer_port=18555
    else
        port=15656; http_port=19001; transfer_port=18556
    fi
    start_service "${side}-cache" env -u CUDA_VISIBLE_DEVICES \
        "${P_PYTHON}" -m lmcache.cli.main server \
        --host 127.0.0.1 --port "${port}" --http-port "${http_port}" \
        --chunk-size "${CHUNK_SIZE}" --separate-object-groups \
        --supported-transfer-mode "${LMCACHE_MP_TRANSFER_MODE}" \
        --l1-size-gb 1 --l1-align-bytes 65536 --eviction-policy LRU \
        --instance-id "aligned-${side}" \
        --coordinator-url http://127.0.0.1:19300 \
        --coordinator-advertise-ip 127.0.0.1 \
        --p2p-transfer-engine nixl --p2p-advertise-url "127.0.0.1:${transfer_port}"
    wait_for_health "${side}-cache" "http://127.0.0.1:${http_port}/healthcheck"
done

collective_env=()
collective_args=()
if [[ "${DISABLE_GPU_P2P}" == 1 ]]; then
    collective_env+=(NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0)
    collective_args+=(--disable-custom-all-reduce)
fi
start_service prefill env "CUDA_VISIBLE_DEVICES=${PREFILL_GPUS}" \
    VLLM_SSM_CONV_STATE_LAYOUT=DS "${collective_env[@]}" \
    "${P_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --host 127.0.0.1 --port 18000 \
    --tensor-parallel-size "${TP}" --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 2 \
    --max-num-batched-tokens "${PREFILL_BATCH_SIZE}" --gpu-memory-utilization 0.25 \
    --enforce-eager --enable-prefix-caching "${collective_args[@]}" \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":15655}}'

decode_command=(
    env "CUDA_VISIBLE_DEVICES=${DECODE_GPUS}" "${collective_env[@]}"
    "${D_PYTHON}" -m sglang.launch_server
    --model-path "${MODEL}" --host 127.0.0.1 --port 18002
    --tp-size "${TP}" --dtype bfloat16 --context-length 4096
    --mem-fraction-static 0.25 --max-running-requests 2
    --disable-cuda-graph --attention-backend triton --mamba-ssm-dtype float32
    "${collective_args[@]}"
)
start_service decode "${decode_command[@]}" \
    --enable-lmcache --lmcache-config-file "${WORK_DIR}/lmcache.yaml"
wait_for_health prefill http://127.0.0.1:18000/health
wait_for_health decode http://127.0.0.1:18002/health

for side in prefill decode; do
    if [[ "${side}" == prefill ]]; then port=19000; else port=19001; fi
    curl -fsS --max-time 30 "http://127.0.0.1:${port}/status" >"${WORK_DIR}/${side}-status-before.json"
    curl -fsS --max-time 30 "http://127.0.0.1:${port}/metrics" >"${WORK_DIR}/${side}-metrics-before.txt"
done

"${P_PYTHON}" vllm_to_sglang.py \
    --model "${MODEL}" --input-ids "${WORK_DIR}/input_ids.json" \
    --chunk-size "${CHUNK_SIZE}" --tp-size "${TP}" --max-tokens 32 \
    --timeout 120 --output "${WORK_DIR}/handoff.json"

for side in prefill decode; do
    if [[ "${side}" == prefill ]]; then port=19000; else port=19001; fi
    curl -fsS --max-time 30 "http://127.0.0.1:${port}/status" >"${WORK_DIR}/${side}-status-after.json"
    curl -fsS --max-time 30 "http://127.0.0.1:${port}/metrics" >"${WORK_DIR}/${side}-metrics-after.txt"
done

stop_service decode
stop_service prefill
start_service cold "${decode_command[@]}"
wait_for_health cold http://127.0.0.1:18002/health

"${P_PYTHON}" - "${WORK_DIR}" "${MODEL}" "${TP}" "${LMCACHE_MP_TRANSFER_MODE}" <<'PY'
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm_to_sglang import request_json

root, model, tp = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
transfer_mode = sys.argv[4]
result = json.loads((root / "handoff.json").read_text())
if "decode" not in result:
    raise SystemExit("FAIL: prefill terminated before a PD handoff")
cold = request_json(
    "http://127.0.0.1:18002/generate",
    {
        "input_ids": result["input_ids"] + result["first_token_ids"],
        "sampling_params": {"max_new_tokens": 31, "temperature": 0},
    },
    120,
)
(root / "cold.json").write_text(json.dumps(cold, indent=2))
if cold["meta_info"]["cached_tokens"] != 0:
    raise SystemExit("FAIL: cold control unexpectedly hit a cache")
if cold["output_ids"] != result["decode"]["output_ids"]:
    raise SystemExit("FAIL: PD and cold continuation token IDs differ")

def load_count(stage):
    metrics = (root / f"decode-metrics-{stage}.txt").read_text()
    values = re.findall(
        r'^lmcache_mp_l2_load_completed_requests_total\{l2_name="p2p"\} ([0-9.eE+-]+)$',
        metrics, re.MULTILINE,
    )
    if len(values) > 1:
        raise SystemExit("FAIL: ambiguous P2P counter")
    return float(values[0]) if values else 0

load_delta = load_count("after") - load_count("before")
if load_delta <= 0:
    raise SystemExit("FAIL: no NIXL P2P loads completed")
for side in ("prefill", "decode"):
    before = json.loads((root / f"{side}-status-before.json").read_text())
    after = json.loads((root / f"{side}-status-after.json").read_text())
    registration_field = (
        "registered_non_cuda_instance_ids"
        if transfer_mode == "engine_driven"
        else "registered_gpu_ids"
    )
    if len(before[registration_field]) != tp:
        raise SystemExit(f"FAIL: {side} did not register all TP ranks")
    if transfer_mode == "engine_driven" and before.get("registered_gpu_ids", []):
        raise SystemExit(f"FAIL: {side} unexpectedly registered CUDA IPC contexts")
    adapters = before["storage_manager"]["l2_adapters"]
    if len(adapters) != 1 or adapters[0]["type"] != "P2PL2Adapter":
        raise SystemExit(f"FAIL: {side} must have only its peer as L2")
    if before["p2p_peer_count"] != 1:
        raise SystemExit(f"FAIL: {side} peer not connected")
    l1 = after["storage_manager"]["l1_manager"]
    if l1["read_locked_count"] or l1["write_locked_count"] or after["active_sessions"]:
        raise SystemExit(f"FAIL: {side} leaked locks or sessions")
decode_before = json.loads((root / "decode-status-before.json").read_text())
if decode_before["storage_manager"]["l1_manager"]["total_object_count"] != 0:
    raise SystemExit("FAIL: decode CPU cache was not initially empty")
tokenizer = AutoTokenizer.from_pretrained(model)
text = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
if "ZEPHYR-73921" not in text:
    raise SystemExit("FAIL: output lost the cached-prefix needle")
validation = {
    "tp_size": tp,
    "mp_transfer_mode": (
        "engine_driven" if transfer_mode == "engine_driven" else "lmcache_driven"
    ),
    "prompt_tokens": len(result["input_ids"]),
    "restored_tokens": result["decode"]["meta_info"]["cached_tokens"],
    "computed_input_tokens": (
        result["decode"]["meta_info"]["prompt_tokens"]
        - result["decode"]["meta_info"]["cached_tokens"]
    ),
    "p2p_completed_loads_delta": load_delta,
    "cold_continuation_ids_equal": True,
    "output_text": text,
}
(root / "validation.json").write_text(json.dumps(validation, indent=2))
print("PASS")
print(json.dumps(validation, indent=2))
PY
