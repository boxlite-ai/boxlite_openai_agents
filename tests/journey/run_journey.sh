#!/usr/bin/env bash
# Run each journey stage separately with a hard per-stage timeout.
# Emits one line per stage: "<stage>\t<status>\t<elapsed_s>"
set -u

STAGES=(
  "test_stage_00_quickstart"
  "test_stage_01_sandbox_basics"
  "test_stage_02_manifest_full"
  "test_stage_03_capabilities"
  "test_stage_04_resume_across_days"
  "test_stage_05_custom_llm"
  "test_stage_06_orchestration"
  "test_stage_07_function_tool"
  "test_stage_08_guardrails"
  "test_stage_09_tracing"
  "test_stage_10_airgapped_default_deny"
)

PER_STAGE_TIMEOUT=${PER_STAGE_TIMEOUT:-120}
RESULT_FILE=${RESULT_FILE:-/tmp/journey-results.tsv}
LOG_DIR=${LOG_DIR:-/tmp/journey-logs}
mkdir -p "$LOG_DIR"
: > "$RESULT_FILE"

for stage in "${STAGES[@]}"; do
  log="$LOG_DIR/${stage}.log"
  echo ">>> Running $stage (timeout ${PER_STAGE_TIMEOUT}s)"
  start_epoch=$(date +%s)
  status="UNKNOWN"
  output=""
  # Note: omit `-s` (no-capture). The rich console used by openai-agents
  # detects non-tty stdout under -s and blocks on a layout call, hanging
  # the test in non-interactive runs. Default capture is fine; we capture
  # the per-stage stdout/stderr to the log file via the redirection below.
  if BOXLITE_RUN_LIVE_LLM=1 PYTHONUNBUFFERED=1 timeout "$PER_STAGE_TIMEOUT" \
       .venv/bin/python -m pytest "tests/journey/test_journey.py::$stage" \
       -v --tb=short --no-header > "$log" 2>&1; then
    status="PASS"
  else
    rc=$?
    if [[ $rc -eq 124 ]]; then
      status="TIMEOUT"
    else
      status="FAIL"
    fi
  fi
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  output=$(grep -E "^=== Stage|^Stage |stage |passed|failed|FAIL|PASS" "$log" 2>/dev/null | tail -3 | tr '\n' '|' | sed 's/|$//')
  printf "%s\t%s\t%ds\t%s\n" "$stage" "$status" "$elapsed" "${output:-(no output)}" | tee -a "$RESULT_FILE"
done

echo ""
echo "=== Summary ==="
column -ts $'\t' "$RESULT_FILE" || cat "$RESULT_FILE"
