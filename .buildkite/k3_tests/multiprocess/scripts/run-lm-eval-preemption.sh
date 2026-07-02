#!/usr/bin/env bash
# Preemption correctness test for LMCache.
#
# Runs lm_eval (gsm8k) twice against a vLLM+LMCache server that has been
# started with a reduced gpu-memory-utilization (GPU_MEMORY_UTILIZATION=0.5
# in the pipeline) to force a small KV cache, triggering vLLM preemption
# under high concurrency.
#
# Verifications (any failure exits non-zero):
#   1. Both lm_eval runs complete without crash (propagated via set -e).
#   2. Preemption actually occurred during each run (grep for "<preempted>"
#      in the vLLM log; count must increase -- non-vacuous check).
#   3. gsm8k exact_match score drift between runs is <= SCORE_TOLERANCE.
#   4. Both scores exceed SCORE_FLOOR (sanity check the model is not broken).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration (all overridable via environment)
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-300}"
BUILD_ID="${BUILD_ID:-local_$$}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.01}"
SCORE_FLOOR="${SCORE_FLOOR:-0.3}"
VLLM_LOG="${VLLM_LOG:-/tmp/build_${BUILD_ID}_vllm.log}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

PREEMPTION_DIR="$RESULTS_DIR/lm_eval_preemption"
FIRST_RUN_DIR="$PREEMPTION_DIR/first_run"
SECOND_RUN_DIR="$PREEMPTION_DIR/second_run"

echo "=== LM-Eval Preemption Test ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Score tolerance: $SCORE_TOLERANCE"
echo "Score floor: $SCORE_FLOOR"
echo "vLLM log: $VLLM_LOG"
echo "Results dir: $PREEMPTION_DIR"
echo ""

mkdir -p "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

# Run one lm_eval gsm8k pass against the vLLM server.
#
# Arguments:
#   $1 run_name   - human-readable label used only in progress log lines.
#   $2 output_dir - directory lm_eval writes results_*.json / samples_*.jsonl to.
# Returns:
#   lm_eval's exit status (non-zero if the evaluation run fails).
run_lm_eval() {
    local run_name="$1"
    local output_dir="$2"

    echo "=== Running lm_eval ($run_name) ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    echo "$run_name completed"
    echo ""
}

# Count occurrences of the preemption marker in the vLLM log.
#
# vLLM logs "<preempted>" in its scheduler output lines whenever a sequence is
# preempted.  Counting these lines gives a monotonically increasing watermark.
# Globals (read):
#   VLLM_LOG - path to the vLLM process log file.
# Outputs:
#   The integer count of lines containing "<preempted>" to stdout (0 if the
#   log file does not exist or contains no preemption lines).
count_preemptions() {
    [ -f "$VLLM_LOG" ] || { echo 0; return; }
    grep -c "<preempted>" "$VLLM_LOG" 2>/dev/null || echo 0
}

# ── Run 1 ────────────────────────────────────────────────────
preemptions_before_run1=$(count_preemptions)
echo "Preemption count before run 1: ${preemptions_before_run1}"

run_lm_eval "first_run" "$FIRST_RUN_DIR"

preemptions_after_run1=$(count_preemptions)
echo "Preemption count after run 1: ${preemptions_after_run1}"

if [ "$preemptions_after_run1" -le "$preemptions_before_run1" ]; then
    echo "FAIL: No preemption occurred during run 1 (before=${preemptions_before_run1}, after=${preemptions_after_run1})"
    echo "Try lowering GPU_MEMORY_UTILIZATION or raising NUM_CONCURRENT to trigger preemption."
    exit 1
fi
echo "Preemption confirmed during run 1 (+$((preemptions_after_run1 - preemptions_before_run1)) events)"
echo ""

# ── Run 2 ────────────────────────────────────────────────────
preemptions_before_run2=$preemptions_after_run1

run_lm_eval "second_run" "$SECOND_RUN_DIR"

preemptions_after_run2=$(count_preemptions)
echo "Preemption count after run 2: ${preemptions_after_run2}"

if [ "$preemptions_after_run2" -le "$preemptions_before_run2" ]; then
    echo "FAIL: No preemption occurred during run 2 (before=${preemptions_before_run2}, after=${preemptions_after_run2})"
    echo "Try lowering GPU_MEMORY_UTILIZATION or raising NUM_CONCURRENT to trigger preemption."
    exit 1
fi
echo "Preemption confirmed during run 2 (+$((preemptions_after_run2 - preemptions_before_run2)) events)"
echo ""

# ── Score comparison ─────────────────────────────────────────
echo "============================================"
echo "=== Verifying score consistency ==="
echo "============================================"

python3 - "$FIRST_RUN_DIR" "$SECOND_RUN_DIR" \
    "$SCORE_TOLERANCE" "$SCORE_FLOOR" <<'PYEOF'
import glob
import json
import os
import sys

first_run_dir, second_run_dir, tol_s, floor_s = sys.argv[1:5]
tol = float(tol_s)
floor = float(floor_s)


def gsm8k_score_and_stderr(results_dir: str) -> tuple[float, float]:
    """Return the gsm8k (exact_match, stderr) from an lm_eval results directory.

    Searches recursively for the newest results_*.json produced by lm_eval and
    extracts the exact_match score, preferring the strict-match variant.

    Assumes the results JSON contains a "results" key with a "gsm8k" sub-key
    (i.e., only a single gsm8k task variant is expected in the results).

    Args:
        results_dir: Directory passed to ``lm_eval --output_path``.

    Returns:
        ``(score, stderr)``: the gsm8k ``exact_match`` accuracy in
        ``[0.0, 1.0]`` and its reported sampling stderr (0.0 if absent).

    Raises:
        SystemExit: If no ``results_*.json`` exists, the JSON lacks the
            expected ``results.gsm8k`` structure, or no ``exact_match``
            metric is found.
    """
    files = glob.glob(
        os.path.join(results_dir, "**", "results_*.json"), recursive=True
    )
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    try:
        metrics = data["results"]["gsm8k"]
    except KeyError as exc:
        raise SystemExit(
            f"results_*.json is missing expected gsm8k results structure in {latest}: {exc}"
        ) from exc
    preferred = "exact_match,strict-match"
    if preferred in metrics:
        stderr = float(metrics.get("exact_match_stderr,strict-match", 0.0))
        return float(metrics[preferred]), stderr
    for key, value in metrics.items():
        if key.startswith("exact_match,") and "stderr" not in key:
            variant = key.split(",", 1)[1]
            stderr = float(metrics.get(f"exact_match_stderr,{variant}", 0.0))
            return float(value), stderr
    raise SystemExit(f"No exact_match metric in {latest}: {sorted(metrics)}")


s1, e1 = gsm8k_score_and_stderr(first_run_dir)
s2, e2 = gsm8k_score_and_stderr(second_run_dir)

print(f"  Run 1 gsm8k exact_match = {s1:.4f} +/- {e1:.4f}")
print(f"  Run 2 gsm8k exact_match = {s2:.4f} +/- {e2:.4f}")
print(f"  tolerance = {tol}, floor = {floor}")

failures = []
if abs(s1 - s2) > tol:
    failures.append(
        f"score drift between runs: |{s1:.4f} - {s2:.4f}| = "
        f"{abs(s1 - s2):.4f} > {tol}"
    )
if s1 <= floor:
    failures.append(f"run 1 score {s1:.4f} is below floor {floor}")
if s2 <= floor:
    failures.append(f"run 2 score {s2:.4f} is below floor {floor}")

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: scores match within tolerance ({tol}) and both exceed floor ({floor})."
)
PYEOF

echo ""
echo "============================================"
echo "=== LM-Eval Preemption test passed ==="
echo "============================================"
