#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python run.py preflight --config configs/pilot.json "$@"
python run.py pipeline --config configs/pilot.json --stage baselines "$@"
