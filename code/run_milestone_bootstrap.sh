#!/usr/bin/env bash
# Re-evaluate bootstrap + per-station from backfilled joblibs; tables unchanged.
set -euo pipefail
cd "$(dirname "$0")"
export REGEN_TRAIN_CAP=0
export REGEN_USE_JOBLIB=1
export REGEN_NO_REFIT=1
export REGEN_HONEST_BOOTSTRAP="${REGEN_HONEST_BOOTSTRAP:-1}"
export REGEN_WRITE_HEADLINE_TABLES=0
# Use ${VAR-default} not ${VAR:-default} so REGEN_TABLE_ASSERT= disables fatal parity assert.
export REGEN_TABLE_ASSERT="${REGEN_TABLE_ASSERT-$(pwd)/outputs/tables/main_h136}"
export TABLE_RESULTS_DIR="${TABLE_RESULTS_DIR:-$(pwd)/outputs/tables/main_h136}"
# Do not abort regenerate on RMSE≠table; client traces mask/n from parity report.
export REGEN_STOP_ON_PARITY_MISMATCH="${REGEN_STOP_ON_PARITY_MISMATCH:-0}"

python regenerate_headline_eval.py
python bootstrap_table_parity_report.py || true
python station_coverage_report.py
python ci_mask_report.py
python write_dl_pressure_h1_note.py
python h1_xgb_lgb_ci_overlap.py
python verify_milestone_acceptance.py || true
bash pack_milestone_outputs.sh
echo "DONE — check milestone_acceptance_report.txt (63/63 parity must arise naturally, not forced)"
