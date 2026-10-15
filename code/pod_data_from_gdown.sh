#!/usr/bin/env bash
# After gdown offbox bundle: copy only safe paths into /workspace/ML-Model/code.
# Do NOT rsync nested outputs/models/*.joblib over a fresh refit.
set -euo pipefail
BUNDLE="${1:-/workspace/offbox.zip}"
ROOT="/workspace/ML-Model/code"
TMP="${TMPDIR:-/tmp}/offbox_extract_$$"
mkdir -p "$TMP"
unzip -q "$BUNDLE" -d "$TMP"
SRC="$TMP"
if [[ -d "$TMP/ML-Model/ML-Model/code" ]]; then
  SRC="$TMP/ML-Model/ML-Model/code"
elif [[ -d "$TMP/ML-Model/code" ]]; then
  SRC="$TMP/ML-Model/code"
elif [[ -d "$TMP/code" ]]; then
  SRC="$TMP/code"
fi
echo "Using bundle root: $SRC"
mkdir -p "$ROOT/data/processed" "$ROOT/outputs/tables/main_h136" \
  "$ROOT/outputs/models" "$ROOT/outputs/figures"
cp -a "$SRC/data/processed/features.parquet" "$ROOT/data/processed/"
cp -a "$SRC/outputs/tables/"*.json "$ROOT/outputs/tables/" 2>/dev/null || true
cp -a "$SRC/outputs/tables/main_h136/." "$ROOT/outputs/tables/main_h136/" 2>/dev/null || true
cp -a "$SRC/outputs/tables/table09_regime_shap.csv" "$ROOT/outputs/tables/" 2>/dev/null || true
cp -a "$SRC/outputs/figures/F8_regime_shap."* "$ROOT/outputs/figures/" 2>/dev/null || true
for ext in pt joblib; do
  if compgen -G "$SRC/outputs/models/*.$ext" >/dev/null; then
    echo "Copying *.$ext checkpoints (trees will be refit — joblibs are optional seed only)"
    cp -a "$SRC/outputs/models/*.$ext" "$ROOT/outputs/models/" 2>/dev/null || true
  fi
done
rm -rf "$TMP"
echo "DATA_OK features=$(du -h "$ROOT/data/processed/features.parquet" | cut -f1)"
