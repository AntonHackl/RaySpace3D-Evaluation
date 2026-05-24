#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

JOB_SCRIPTS=(
  "$SCRIPT_DIR/slurm_intersection_breakdown.sh"
  # "$SCRIPT_DIR/slurm_intersection_mesh_complexity.sh"
  "$SCRIPT_DIR/slurm_overlap_cube_scalability.sh"
  "$SCRIPT_DIR/slurm_overlap_mesh_complexity.sh"
  "$SCRIPT_DIR/slurm_overlap_microns.sh"
  "$SCRIPT_DIR/slurm_overlap_large_nu_v_scalability.sh"
  "$SCRIPT_DIR/slurm_overlap_large_nu_nn_scalability.sh"
  "$SCRIPT_DIR/slurm_dataset_table_benchmark.sh"
  "$SCRIPT_DIR/slurm_overlap_nn_scalability.sh"
  "$SCRIPT_DIR/slurm_overlap_nu_scalability.sh"
  "$SCRIPT_DIR/slurm_overlap_selectivity.sh"
  "$SCRIPT_DIR/slurm_query_comparison_microns.sh"
  "$SCRIPT_DIR/slurm_query_comparison_nu_v.sh"
  "$SCRIPT_DIR/slurm_query_comparison_nu_nn.sh"
)

if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer, got: $MAX_CONCURRENT" >&2
  exit 1
fi

echo "Submitting all benchmarks with max concurrent jobs: $MAX_CONCURRENT"
echo "Using script directory: $SCRIPT_DIR"

declare -a SLOT_LAST_JOB_IDS=()

submit_job() {
  local script="$1"
  local slot="$2"
  local job_id
  local display_script="${script#$REPO_ROOT/}"

  if [[ ! -f "$script" ]]; then
    echo "Skipping missing script: $display_script" >&2
    return 0
  fi

  if [[ -n "${SLOT_LAST_JOB_IDS[$slot]:-}" ]]; then
    job_id="$(sbatch --parsable --dependency=afterany:${SLOT_LAST_JOB_IDS[$slot]} "$script")"
    echo "Submitted $display_script as job $job_id (slot $slot, afterany:${SLOT_LAST_JOB_IDS[$slot]})"
  else
    job_id="$(sbatch --parsable "$script")"
    echo "Submitted $display_script as job $job_id (slot $slot)"
  fi

  SLOT_LAST_JOB_IDS[$slot]="$job_id"
}

for idx in "${!JOB_SCRIPTS[@]}"; do
  slot=$(( idx % MAX_CONCURRENT ))
  submit_job "${JOB_SCRIPTS[$idx]}" "$slot"
done

echo "All benchmarks submitted."
