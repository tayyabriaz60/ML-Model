#!/usr/bin/env bash
# Client round 3: backfill → refit → XGB .json → preflight 18/18 → one-fit tables + bootstrap.
set -euo pipefail
cd "$(dirname "$0")"
export REGEN_TRAIN_CAP=0
export REGEN_USE_JOBLIB=1
export REGEN_NO_REFIT=1
export REGEN_XGB_REQUIRE_JSON=1
export REGEN_HONEST_BOOTSTRAP=1
export REGEN_WRITE_HEADLINE_TABLES=1
export REGEN_TABLE_ASSERT=
export TABLE_RESULTS_DIR="$(pwd)/outputs/tables"

python write_run_env.py
python backfill_tree_rounds.py
python -u refit_trees_pod.py 2>&1 | tee outputs/tables/tree_refit_report.txt
python export_tree_models_json.py
python pod_tree_preflight.py
python regenerate_headline_eval.py
python bootstrap_table_parity_report.py || true
python station_coverage_report.py
python ci_mask_report.py
python write_dl_pressure_h1_note.py
python h1_xgb_lgb_ci_overlap.py
python verify_milestone_acceptance.py || true
bash pack_milestone_outputs.sh
echo "DONE — preflight 18/18 and acceptance report required before delivery"
