#!/usr/bin/env bash
# One pod session: honest milestone bootstrap (client T5.R1). Run from /workspace/ML-Model/code.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export REGEN_TRAIN_CAP=0
export REGEN_USE_JOBLIB=1
export REGEN_NO_REFIT=1
export REGEN_HONEST_BOOTSTRAP=1
export REGEN_WRITE_HEADLINE_TABLES=0
export REGEN_STOP_ON_PARITY_MISMATCH=0
export REGEN_TABLE_ASSERT="${REGEN_TABLE_ASSERT:-$(pwd)/outputs/tables/main_h136}"
export TABLE_RESULTS_DIR="${TABLE_RESULTS_DIR:-$(pwd)/outputs/tables/main_h136}"

test -f data/processed/features.parquet || { echo "missing features.parquet"; exit 1; }
test -f outputs/tables/production_scaler.json || { echo "missing production_scaler.json"; exit 1; }
test -d outputs/tables/main_h136 || { echo "missing outputs/tables/main_h136"; exit 1; }

python write_run_env.py
python backfill_tree_rounds.py
echo "=== refit_trees_pod (single process — do not start a second) ==="
python -u refit_trees_pod.py 2>&1 | tee outputs/tables/tree_refit_report.txt
python pod_tree_preflight.py

bash run_milestone_bootstrap.sh
ls -lh ../deliverables/T5_milestone_compute.zip
echo "Download zip; send client milestone_acceptance_report.txt from zip."
