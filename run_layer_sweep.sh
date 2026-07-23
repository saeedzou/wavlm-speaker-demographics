#!/usr/bin/env bash
# Sweep gender/age/accent classification + speaker verification over WavLM
# hidden_states layers (stride of 2) and collect results in
# results/<task_name>/metrics.csv
#
# Usage:
#   ./run_layer_sweep.sh
#   LAYER_START=0 LAYER_END=12 LAYER_STRIDE=2 MODEL=microsoft/wavlm-base-plus ./run_layer_sweep.sh

set -euo pipefail

MODEL="${MODEL:-microsoft/wavlm-base-plus}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-12}"       # inclusive; 12 for wavlm-base-plus (13 hidden_states: 0..12)
LAYER_STRIDE="${LAYER_STRIDE:-1}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EXTRA_ARGS=(--model "$MODEL")      # add e.g. --max_train_samples 2000 here while testing

# task_name -> script file
declare -A TASK_SCRIPTS=(
  [gender_classification]="gender_classification.py"
  [age_classification]="age_classification.py"
  [accent_classification]="accent_classification.py"
  [speaker_verification]="speaker_verification.py"
)

layers=($(seq "$LAYER_START" "$LAYER_STRIDE" "$LAYER_END"))
echo "Layers to sweep: ${layers[*]}"

for task in "${!TASK_SCRIPTS[@]}"; do
  script="${TASK_SCRIPTS[$task]}"
  task_dir="${RESULTS_DIR}/${task}"
  raw_dir="${task_dir}/_by_layer"
  mkdir -p "$raw_dir"

  echo "=== ${task} (${script}) ==="
  for layer in "${layers[@]}"; do
    layer_csv="${raw_dir}/layer_${layer}.csv"
    echo "--- layer ${layer} ---"
    python "$script" \
      "${EXTRA_ARGS[@]}" \
      --layer "$layer" \
      --metrics_csv "$layer_csv"
  done

  # Concatenate per-layer CSVs into one file, keeping the header only once.
  final_csv="${task_dir}/metrics.csv"
  first=1
  : > "$final_csv"
  for layer in "${layers[@]}"; do
    layer_csv="${raw_dir}/layer_${layer}.csv"
    if [[ $first -eq 1 ]]; then
      cat "$layer_csv" >> "$final_csv"
      first=0
    else
      tail -n +2 "$layer_csv" >> "$final_csv"
    fi
  done
  echo "Wrote ${final_csv}"
done

echo "Done. Per-layer intermediates kept under ${RESULTS_DIR}/<task>/_by_layer/ for debugging."