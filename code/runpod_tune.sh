#!/usr/bin/env bash
# Run after features.parquet is in code/data/processed/.
# Usage: bash runpod_tune.sh
set -euo pipefail
cd "$(dirname "$0")"
export N_TRIALS="${N_TRIALS:-24}"
if [[ ! -f data/processed/features.parquet ]]; then
  echo "missing data/processed/features.parquet — gdown it first" >&2
  exit 1
fi
echo "N_TRIALS=$N_TRIALS (DL only; trees skipped; SQLite resume at outputs/optuna_study.sqlite)"
nohup python run_all.py --stage tune --arch dl > /workspace/tune.log 2>&1 &
echo "pid $!"
echo "tail -f /workspace/tune.log"
