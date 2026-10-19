#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

config_path="configs/last_token.json"
run_id="${RUN_ID:-}"
start_at="prepare"
dry_run=false
current_stage="setup"

usage() {
  cat <<'EOF'
Run the complete last-token affect-geometry experiment.

Usage:
  experiments/01_run_last_token.sh [options]

Options:
  --config PATH       Experiment config relative to the project root
                      (default: configs/last_token.json)
  --run-id ID         Shared run ID (default: RUN_ID or a new UTC timestamp)
  --start-at STAGE    Start at prepare, extract, rdm, embed, analyze, or plot
                      (default: prepare)
  --dry-run           Validate and print the commands without running them
  -h, --help          Show this help

To resume a completed earlier stage, reuse its run ID, for example:
  experiments/01_run_last_token.sh --run-id 20261019T153045Z --start-at rdm
EOF
}

while (($#)); do
  case "$1" in
    --config)
      config_path="${2:?--config requires a path}"
      shift 2
      ;;
    --run-id)
      run_id="${2:?--run-id requires an ID}"
      shift 2
      ;;
    --start-at)
      start_at="${2:?--start-at requires a stage}"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [[ "${config_path}" != /* ]]; then
  config_path="${PROJECT_ROOT}/${config_path}"
fi
if [[ ! -f "${config_path}" ]]; then
  echo "Configuration not found: ${config_path}" >&2
  exit 2
fi
if [[ -z "${run_id}" ]]; then
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
fi

model_name="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"]["name"])' "${config_path}")"
poolings="$(python -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["activations"]["poolings"]))' "${config_path}")"
if [[ "${poolings}" != "last_token" ]]; then
  echo "Refusing to run: activations.poolings must be exactly [\"last_token\"]." >&2
  echo "Configured poolings: ${poolings}" >&2
  exit 2
fi

output_dir="$(python -c 'import sys; from affect_geometry.config import load_config; print(load_config(sys.argv[1], run_id=sys.argv[2]).run_dir.resolve())' "${config_path}" "${run_id}")"
log_dir="$(python -c 'import sys; from affect_geometry.config import load_config; print(load_config(sys.argv[1], run_id=sys.argv[2]).log_dir.resolve())' "${config_path}" "${run_id}")"

stages=(prepare extract rdm embed analyze plot)
start_found=false
for stage in "${stages[@]}"; do
  if [[ "${stage}" == "${start_at}" ]]; then
    start_found=true
    break
  fi
done
if [[ "${start_found}" != true ]]; then
  echo "Invalid --start-at stage: ${start_at}" >&2
  echo "Expected one of: ${stages[*]}" >&2
  exit 2
fi

trap 'status=$?; echo; echo "Experiment failed during ${current_stage} (exit ${status})." >&2; echo "Resume with: experiments/01_run_last_token.sh --run-id ${run_id} --start-at ${current_stage}" >&2; exit "${status}"' ERR

echo "Last-token affect-geometry experiment"
echo "  model:      ${model_name}"
echo "  run ID:     ${run_id}"
echo "  config:     ${config_path}"
echo "  output:     ${output_dir}"
echo "  logs:       ${log_dir}"
echo "  start at:   ${start_at}"
echo "  poolings:   ${poolings}"

should_run=false
for stage in "${stages[@]}"; do
  if [[ "${stage}" == "${start_at}" ]]; then
    should_run=true
  fi
  if [[ "${should_run}" != true ]]; then
    continue
  fi

  current_stage="${stage}"
  echo
  echo "==> ${stage}"
  if [[ "${dry_run}" == true ]]; then
    printf 'python -m %q --config %q --run-id %q\n' "${stage}" "${config_path}" "${run_id}"
  else
    python -m "${stage}" --config "${config_path}" --run-id "${run_id}"
  fi
done

echo
if [[ "${dry_run}" == true ]]; then
  echo "Dry run complete; no experiment stages were executed."
else
  echo "Experiment complete."
  echo "Outputs: ${output_dir}"
  echo "Logs:    ${log_dir}"
fi
