#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export REGEN_TRAIN_CAP="${REGEN_TRAIN_CAP:-0}"
export REGEN_USE_JOBLIB="${REGEN_USE_JOBLIB:-1}"
export REGEN_WRITE_HEADLINE_TABLES="${REGEN_WRITE_HEADLINE_TABLES:-0}"

python write_run_env.py
python pod_tree_preflight.py
if [[ "${SKIP_JSON_EXPORT:-0}" != "1" ]]; then
  python export_tree_models_json.py || true
fi
python regenerate_headline_eval.py
python station_coverage_report.py
python ci_mask_report.py
python h1_xgb_lgb_ci_overlap.py
bash pack_milestone_outputs.sh
echo "DONE — download ../deliverables/T5_milestone_compute.zip"
