#!/usr/bin/env bash
set -euo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$CODE")"
OUT="$PROJ/deliverables/T5_milestone_compute.zip"
STAGING="${TMPDIR:-/tmp}/t5_milestone_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING" "$PROJ/deliverables"
cp "$CODE/outputs/tables/bootstrap_ci.csv" "$STAGING/"
cp "$CODE/outputs/tables/per_station_metrics.csv" "$STAGING/"
cp "$CODE/outputs/tables/table09_regime_shap.csv" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/ci_mask_report.txt" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/station_coverage_note.txt" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/station_coverage_main_h136.csv" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/tree_preflight_report.txt" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/h1_xgb_lgb_ci_overlap.txt" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/tables/run_env.txt" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/figures/F8_regime_shap.png" "$STAGING/" 2>/dev/null || true
cp "$CODE/outputs/figures/F8_regime_shap.pdf" "$STAGING/" 2>/dev/null || true
rm -f "$OUT"
(cd "$STAGING" && zip -r "$OUT" .)
echo "wrote $OUT"
