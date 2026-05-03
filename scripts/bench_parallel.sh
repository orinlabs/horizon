#!/usr/bin/env bash
# Run every harness in parallel as separate `harbor run` processes.
# Workaround for Harbor's in-process multi-trial orchestration hanging
# on Modal — independent processes run concurrently fine.
#
# Usage:
#   source .env && export OPENROUTER_API_KEY
#   scripts/bench_parallel.sh evals/01-example-catering-vendor
set -euo pipefail

EVAL_PATH="${1:-evals/01-example-catering-vendor}"
EVAL_SLUG=$(basename "$EVAL_PATH")
JOB_PREFIX="${JOB_PREFIX:-bench-$(date +%H%M%S)}"
ENV_PATH="${ENV_PATH:-horizon_environment:HorizonModalEnvironment}"

# (agent_name, import_path, model)
AGENTS=(
  "trace_window|trace_window.agent:TraceWindowAgent|google/gemini-2.5-flash"
  "trace_summary|trace_summary.agent:TraceSummaryAgent|google/gemini-2.5-flash"
  "trace_keyword|trace_keyword.agent:TraceKeywordAgent|google/gemini-2.5-flash"
  "trace_rag|trace_rag.agent:TraceRagAgent|google/gemini-2.5-flash"
  "trace_mem0|trace_mem0.agent:TraceMem0Agent|google/gemini-2.5-flash"
)

mkdir -p logs/bench
pids=()
for spec in "${AGENTS[@]}"; do
  IFS='|' read -r name imp model <<<"$spec"
  job="${JOB_PREFIX}-${name}"
  out="logs/bench/${job}.out"
  echo "starting: $name -> $job"
  PYTHONPATH=agents harbor run \
    --environment-import-path "$ENV_PATH" \
    -p "$EVAL_PATH" \
    --agent-import-path "$imp" \
    -m "$model" \
    --job-name "$job" \
    -y \
    >"$out" 2>&1 &
  pids+=($!)
done

echo
echo "launched ${#pids[@]} runs:"
for p in "${pids[@]}"; do echo "  pid=$p"; done
echo
echo "tail logs/bench/${JOB_PREFIX}-<agent>.out for live progress"
echo "waiting for all to finish..."

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    fail=$((fail + 1))
  fi
done

echo
echo "=== results ==="
for spec in "${AGENTS[@]}"; do
  IFS='|' read -r name _ _ <<<"$spec"
  job="${JOB_PREFIX}-${name}"
  rj="jobs/${job}/${EVAL_SLUG}*/result.json"
  reward=$(python -c "import glob, json; ps=glob.glob('$rj'); print(json.load(open(ps[0])).get('reward', 'NA') if ps else 'NO_RESULT')" 2>/dev/null || echo "?")
  printf "  %-15s reward=%s\n" "$name" "$reward"
done
echo
echo "$fail process(es) exited non-zero"
exit "$fail"
