#!/usr/bin/env bash
# Hour-1 measurement only. Do NOT start the 216-run search from this script.
# Requires data/processed/features.parquet on the persistent volume.
set -euo pipefail
cd "$(dirname "$0")"
export N_TRIALS="${N_TRIALS:-24}"
if [[ ! -f data/processed/features.parquet ]]; then
  echo "missing data/processed/features.parquet — upload it first" >&2
  exit 1
fi
echo "CLOCK TRIAL N_TRIALS=$N_TRIALS (budget unchanged; this run is 1 trial)"
python run_all.py --stage tune --clock-trial
echo "---- clock_trial.json ----"
cat ../outputs/clock_trial.json 2>/dev/null || cat outputs/clock_trial.json
