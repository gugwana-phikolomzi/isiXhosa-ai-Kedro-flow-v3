#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all.sh                # run all in order
#   bash run_all.sh model_A3       # start from model_A3 and continue
#   START_FROM=model_B0 bash run_all.sh   # same as above (env var option)

PIPELINES=(
  model_A0
  model_A1
  model_A2
  model_A3
  model_A4
  model_A5
  model_A6
  model_A7
  model_B0
  model_B1
  model_B2
  model_C0
  model_C1
)

mkdir -p logs
START_FROM="${1:-${START_FROM:-}}"

# Find index to start from (if provided)
START_IDX=0
if [[ -n "${START_FROM}" ]]; then
  for i in "${!PIPELINES[@]}"; do
    if [[ "${PIPELINES[$i]}" == "${START_FROM}" ]]; then
      START_IDX="$i"
      break
    fi
  done
fi

for (( i=START_IDX; i<${#PIPELINES[@]}; i++ )); do
  p="${PIPELINES[$i]}"
  ts="$(date +'%Y%m%d-%H%M%S')"
  log="logs/${p}_${ts}.log"

  echo "============================"
  echo "Running pipeline: ${p}"
  echo "Log: ${log}"
  echo "============================"

  # Run and tee output to a timestamped log
  /usr/bin/time -f "Elapsed: %E  CPU: %P  MaxRSS: %M KB" \
    kedro run --pipeline "${p}" 2>&1 | tee -a "${log}"

  echo "✅ Completed: ${p}"
done

echo "🎉 All requested pipelines finished."
