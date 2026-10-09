# Pod eval pre-flight checklist (send before instance start)

## A. Instance
- 16 vCPU, 64 GB RAM, 100 GB disk (CPU only unless DL retrain needed)
- Ubuntu 22.04 or same OS as prior successful pod run

## B. Upload to pod (paths under `/workspace/ML-Model/` or equivalent)

| Path | Purpose |
|------|---------|
| `data/processed/features.parquet` | Full study grid (~402 MB) |
| `outputs/models/xgb_{target}_h{h}.joblib` | 9 files (3 targets × h=1,3,6) |
| `outputs/models/lgb_{target}_h{h}.joblib` | 9 files if saved from prior pod |
| `outputs/models/{lstm,gru,transformer}_{target}_h{h}.pt` | DL checkpoints (main horizons) |
| `outputs/tables/best_hyperparameters.json` | 24-trial winners for refit fallback |
| `outputs/tables/production_scaler.json` | Scaler params |
| `outputs/tables/split_meta.json` | Chronological cuts |
| `outputs/tables/table08_summary_all.csv` | **Authoritative RMSE/n for exact CI point match** |
| `code/` | This repo `T5_revision/code` |

## C. Python stack (record in deliverable `run_env.txt`)
Match prior training pod when possible. Minimum to test:
- Python 3.10 or 3.11 (avoid 3.13 if prior pod was 3.10)
- `xgboost`, `lightgbm`, `scikit-learn`, `torch`, `pandas`, `pyarrow`, `joblib`, `numpy`

After install, run:
```bash
python -c "import sys,xgboost,lgb,sklearn; print(sys.version); print('xgboost', xgboost.__version__); print('lightgbm', lgb.__version__); print('sklearn', sklearn.__version__)"
```

## D. Pre-flight script (before bootstrap / per-station)

```bash
cd code
python pod_tree_preflight.py   # to be run first on pod
```

For each of 9 tree joblibs (xgb/lgb × 3 targets × h=1,3,6):
1. Load joblib
2. Predict validation slice (chronological val split, same features/scaler as train)
3. RMSE vs native `y`; **PASS** if RMSE ≤ 2×σ_target and not ~|mean(y)|

If **all PASS**: export native format immediately:
```bash
python export_tree_models_json.py   # xgb/lgb → .json under outputs/models/
```

If **any FAIL**: refit that model on **full** train (no row cap) using `best_hyperparameters.json`, then save joblib + `.json`.

## E. Main eval job (after pre-flight)

```bash
python regenerate_headline_eval.py   # pod mode: full train, joblib or refit, target-aware masks
python regenerate_dl_headline.py     # LSTM/Transformer pressure h=1 vs reference
python station_coverage_report.py
python ci_mask_report.py             # exact equality bootstrap point vs table08
python pack_milestone_outputs.ps1    # or pack_milestone_outputs.sh
```

## F. Deliverables (milestone zip)
1. `bootstrap_ci.csv` (observed-target mask; point RMSE **identical** to Table 8)
2. `per_station_metrics.csv` + `station_coverage_*.txt/csv` (target-aware feature mask)
3. DL pressure h=1 LSTM + Transformer rows matching reference (or written note if checkpoint bad)
4. `table09_regime_shap.csv`, `F8_regime_shap.png/pdf` (unchanged, bundled)
5. `run_env.txt` (Python, XGBoost, LightGBM versions)
6. `tree_preflight_report.txt` + `h1_xgb_lgb_ci_overlap.txt` (overlap yes/no)

## G. Blockers (report same day)
- Missing joblib/pt files
- Pre-flight fail on all stacks tried
- DL checkpoint load failure on CPU
- Table 8 row definition mismatch (observed vs feature-complete)
