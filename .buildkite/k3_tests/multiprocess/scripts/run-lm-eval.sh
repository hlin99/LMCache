#!/usr/bin/env bash
# Run lm_eval workload test against vLLM server.
# Sends the same requests twice to test LMCache caching behavior.
# Adapted from the old Docker-based run-lm-eval.sh -- no venv setup needed
# (setup-env.sh + extras already installed by run.sh).
#
# Verification mode is selected via LM_EVAL_VERIFY_MODE:
#   samples     (default) -- sort and diff the two runs' samples_gsm8k_*.jsonl
#                            files to confirm bit-exact cache correctness.
#   preemption  -- parse gsm8k results_*.json scores, check score drift <=
#                  SCORE_TOLERANCE, both scores >= SCORE_MIN, and that
#                  preemption events were observed in the vLLM log.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Common configuration ─────────────────────────────────────────────────────
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-300}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

# ── Mode selection ───────────────────────────────────────────────────────────
# samples     -- bit-exact diff of per-sample JSONL outputs (default)
# preemption  -- score drift + preemption-count checks
LM_EVAL_VERIFY_MODE="${LM_EVAL_VERIFY_MODE:-samples}"

# ── Preemption-mode configuration (ignored in samples mode) ─────────────────
# Max absolute difference allowed between the two runs' gsm8k scores.
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.05}"
# Minimum acceptable gsm8k score for either run (correctness floor).
SCORE_MIN="${SCORE_MIN:-0.80}"
# vLLM server log, scanned to confirm preemption actually occurred.
VLLM_LOG="${VLLM_LOG:-/tmp/build_${BUILD_ID}_vllm.log}"

# ── Output directories ───────────────────────────────────────────────────────
# Use separate subdirectories so results from the two modes don't collide.
if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    LM_EVAL_DIR="$RESULTS_DIR/lm_eval_preemption"
else
    LM_EVAL_DIR="$RESULTS_DIR/lm_eval"
fi
FIRST_RUN_DIR="$LM_EVAL_DIR/first_run"
SECOND_RUN_DIR="$LM_EVAL_DIR/second_run"

echo "=== LM-Eval Workload Test (mode: ${LM_EVAL_VERIFY_MODE}) ==="
echo "Model: $MODEL"
echo "vLLM Port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Results dir: $LM_EVAL_DIR"
if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    echo "Score tolerance: $SCORE_TOLERANCE"
    echo "Score minimum: $SCORE_MIN"
fi
echo ""

mkdir -p "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"

# Run one lm_eval gsm8k pass against a vLLM OpenAI-compatible server.
#
# Arguments:
#   $1 run_name   - human-readable label used only in progress log lines.
#   $2 output_dir - directory lm_eval writes results_*.json / samples_*.jsonl to.
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

# ── Verification helpers ─────────────────────────────────────────────────────

verify_samples_match() {
    local first_dir="$1"
    local second_dir="$2"

    echo "=== Verifying samples files match ==="

    first_samples=$(find "$first_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)
    second_samples=$(find "$second_dir" -name "samples_gsm8k_*.jsonl" -type f 2>/dev/null | head -1)

    if [ -z "$first_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in first run directory: $first_dir"
        find "$first_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    if [ -z "$second_samples" ]; then
        echo "Could not find samples_gsm8k_*.jsonl in second run directory: $second_dir"
        find "$second_dir" -type f -name "*.jsonl" || true
        return 1
    fi

    echo "First run samples: $first_samples"
    echo "Second run samples: $second_samples"

    first_sorted=$(mktemp)
    second_sorted=$(mktemp)

    sort "$first_samples" > "$first_sorted"
    sort "$second_samples" > "$second_sorted"

    if diff -q "$first_sorted" "$second_sorted" > /dev/null 2>&1; then
        echo "Samples files are identical!"
        rm -f "$first_sorted" "$second_sorted"
        return 0
    else
        echo "Samples files differ!"
        echo ""
        echo "=== Diff (first 50 lines) ==="
        diff "$first_sorted" "$second_sorted" | head -50 || true
        rm -f "$first_sorted" "$second_sorted"
        return 1
    fi
}

# Count how many times the vLLM log mentions preemption events so far.
#
# Globals (read):
#   VLLM_LOG - path to the vLLM server log file.
count_preemptions() {
    [ -f "$VLLM_LOG" ] || { echo 0; return; }
    local count
    count=$(grep -c "<preempted>" "$VLLM_LOG" 2>/dev/null || true)
    echo "${count:-0}"
}

verify_scores_and_preemptions() {
    local first_dir="$1"
    local second_dir="$2"
    local preemptions_before="$3"
    local preemptions_after="$4"

    echo "vLLM preemptions logged: before=${preemptions_before}, after=${preemptions_after}"

    python3 - "$first_dir" "$second_dir" \
        "$SCORE_TOLERANCE" "$SCORE_MIN" "$preemptions_before" "$preemptions_after" <<'PYEOF'
import glob
import json
import os
import sys

first_dir, second_dir, tol_s, score_min_s, before_s, after_s = sys.argv[1:7]
tol = float(tol_s)
score_min = float(score_min_s)
preemptions_before = int(before_s)
preemptions_after = int(after_s)


def gsm8k_score_and_stderr(results_dir: str) -> tuple[float, float]:
    """Return the gsm8k (exact_match, stderr) from an lm_eval results directory.

    Prefers the strict-match variant; falls back to any non-stderr
    ``exact_match`` metric key (paired with its ``exact_match_stderr`` twin).

    Args:
        results_dir: Directory passed to ``lm_eval --output_path``. Searched
            recursively for the newest ``results_*.json`` (lm_eval nests it
            under a per-model subdirectory and stamps the filename with a
            timestamp).

    Returns:
        ``(score, stderr)``: the gsm8k ``exact_match`` accuracy in
        ``[0.0, 1.0]`` and its reported sampling stderr (0.0 if absent).

    Raises:
        SystemExit: If no ``results_*.json`` exists under ``results_dir`` or the
            newest one contains no ``exact_match`` metric for the gsm8k task.
    """
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    metrics = data["results"]["gsm8k"]
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


s_first, e_first = gsm8k_score_and_stderr(first_dir)
s_second, e_second = gsm8k_score_and_stderr(second_dir)

print(f"  First run  gsm8k exact_match = {s_first:.4f} +/- {e_first:.4f}")
print(f"  Second run gsm8k exact_match = {s_second:.4f} +/- {e_second:.4f}")
print(f"  tolerance = {tol}")
print(f"  score_min = {score_min}")

failures = []
# Score drift: a broken LMCache preemption-resume path would corrupt KV and
# skew results between runs.
if abs(s_first - s_second) > tol:
    failures.append(
        f"score drift between runs: |{s_first:.4f} - {s_second:.4f}| = "
        f"{abs(s_first - s_second):.4f} > {tol}"
    )
# Score floor: catastrophic regression check.
for label, score in [("first_run", s_first), ("second_run", s_second)]:
    if score < score_min:
        failures.append(
            f"{label} score {score:.4f} < score_min {score_min}"
        )
# Non-vacuous: preemption must have actually occurred during the test runs.
if preemptions_after <= preemptions_before:
    failures.append(
        "vLLM logged no preemption events during the test runs "
        f"(before={preemptions_before}, after={preemptions_after}); "
        "check GPU_MEMORY_UTILIZATION is set low enough to trigger preemption"
    )

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: score drift {abs(s_first - s_second):.4f} <= {tol}; "
    f"both scores >= {score_min}; "
    f"preemptions observed: {preemptions_after - preemptions_before}."
)
PYEOF
}

# ── Run lm_eval twice ────────────────────────────────────────────────────────

# First run -- populates cache
echo "============================================"
echo "=== First lm_eval run (cache population) ==="
echo "============================================"
if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    preemptions_before=$(count_preemptions)
fi
run_lm_eval "first_run" "$FIRST_RUN_DIR"

# Second run -- should use cached results
echo "============================================"
echo "=== Second lm_eval run (cache hit) ==="
echo "============================================"
run_lm_eval "second_run" "$SECOND_RUN_DIR"
if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    preemptions_after=$(count_preemptions)
fi

# ── Verify correctness ───────────────────────────────────────────────────────
echo "============================================"
echo "=== Verifying output consistency ==="
echo "============================================"

if [ "$LM_EVAL_VERIFY_MODE" = "preemption" ]; then
    if ! verify_scores_and_preemptions \
            "$FIRST_RUN_DIR" "$SECOND_RUN_DIR" \
            "$preemptions_before" "$preemptions_after"; then
        echo "Preemption verification failed"
        exit 1
    fi
else
    if ! verify_samples_match "$FIRST_RUN_DIR" "$SECOND_RUN_DIR"; then
        echo "Verification failed: samples files do not match"
        exit 1
    fi
fi

echo "============================================"
echo "=== LM-Eval workload test completed ==="
echo "============================================"
