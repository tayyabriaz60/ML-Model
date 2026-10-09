"""
Recompute bootstrap CIs (observed-target mask), results_full.json, and per-station
metrics for main horizons h=1,3,6 without re-tuning.

Uses split-filtered parquet reads (same pattern as dm_trees_vs_dl.py) so the
~2.6M-row feature grid fits in RAM on a laptop.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dm_trees_vs_dl import (
    _load_dl,
    _load_part,
    _split_cuts,
    _train_val_mats,
)
from run_all import FEAT, _load_best, _tree_params, log
from src import config as C
from src import models, evaluate

DL_KINDS = ("lstm", "gru", "transformer")
HORIZONS = list(C.HORIZONS_MAIN)
SCALER_JSON = os.path.join(C.OUT_TABLES, "production_scaler.json")
def _train_cap() -> int | None:
    raw = os.environ.get("REGEN_TRAIN_CAP", "120000")
    if raw in ("", "0", "none", "None"):
        return None
    return int(raw)


def _production_feature_cols() -> list[str]:
    if not os.path.exists(SCALER_JSON):
        raise SystemExit(f"missing {SCALER_JSON}")
    with open(SCALER_JSON) as f:
        return list(json.load(f)["params"].keys())


def _fit_linear_headline(Xtr, ytr, Xva, yva, max_n: int = 150_000):
    if len(ytr) > max_n:
        rng = np.random.RandomState(C.SEED)
        idx = rng.choice(len(ytr), max_n, replace=False)
        if hasattr(Xtr, "iloc"):
            Xtr = Xtr.iloc[idx]
        else:
            Xtr = Xtr[idx]
        ytr = ytr[idx]
        log(f"  LinearRegression train subsample n={max_n}")
    return models.fit_linear(Xtr, ytr, Xva, yva)


def _joblib_plausible(m, Xva, yva) -> bool:
    """Pod joblib + local Python/XGBoost often loads but predicts ~0 (RMSE ~ |y|)."""
    pred_va = np.asarray(m.predict(Xva), float)
    rmse_va = float(np.sqrt(np.mean((yva - pred_va) ** 2)))
    y_scale = float(np.nanstd(yva)) or 1.0
    if rmse_va > max(10.0, 2.5 * y_scale):
        return False
    if abs(float(np.nanmean(yva))) > 5.0 and abs(float(np.nanmean(pred_va))) < 0.05 * abs(
            float(np.nanmean(yva))):
        return False
    return True


def _load_tree_model(name: str, tag: str, target: str, h: int, trn, val, best: dict):
    fit_fn = models.fit_xgboost if tag == "xgb" else models.fit_lightgbm
    tp = _tree_params(best, tag, target)
    path = os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.joblib")
    Xtr, ytr = trn[0], trn[1]
    Xva, yva = val[0], val[1]
    use_joblib = os.environ.get("REGEN_USE_JOBLIB", "1") != "0"
    if use_joblib and os.path.exists(path):
        try:
            m = joblib.load(path)
            if _joblib_plausible(m, Xva, yva):
                log(f"  loaded {name} {target} h={h} from joblib")
                return m
            log(f"  {name} joblib failed plausibility on val — refit")
        except Exception as e:
            log(f"  joblib load failed {name} {target} h={h}: {e}; refit")
    else:
        if not os.path.exists(path):
            log(f"  refit {name} {target} h={h} (no joblib)")
    log(f"  refit {name} {target} h={h} n={len(ytr)}")
    return fit_fn(Xtr, ytr, Xva, yva, tp)


def _test_block(sc, cols, target: str, h: int, cut_test: pd.Timestamp):
    ycol = f"y_{target}_h{h}"
    obs = f"{ycol}__observed"
    persist = f"persist_{target}_h{h}"
    te = _load_part(cols, [ycol, obs, persist], cut_lo=cut_test)
    m = te[ycol].notna() & te[cols].notna().all(axis=1)
    te_ok = te.loc[m]
    Xte = sc.transform(te_ok[cols])
    yte = te_ok[ycol].to_numpy(np.float32)
    persist_v = te_ok[persist].to_numpy(np.float32)
    obs_m = te_ok[obs].values.astype(bool) if obs in te_ok.columns else None
    stn = te_ok[C.COL_STATION].values
    del te
    gc.collect()
    return Xte, yte, persist_v, obs_m, stn, te_ok


def main():
    if not os.path.exists(FEAT):
        raise SystemExit(
            f"missing {FEAT}. Extract from offbox_exports "
            "(core_tables_study_*.tar.gz → data/processed/features.parquet).")
    cols = _production_feature_cols()
    sc = models.Scaler.load(SCALER_JSON)
    cut_train, cut_test = _split_cuts()
    best = _load_best()
    tcol_map = C.TARGETS
    results, ci_rows = {}, []

    for target in C.CONFIG.targets:
        tcol = tcol_map[target]
        for h in HORIZONS:
            log(f"=== {target} h={h} ===")
            trn, val = _train_val_mats(
                sc, cols, target, h, cut_train, cut_test, cut_test,
                max_train_rows=_train_cap())
            Xte, yte, persist, obs_m, stn, te_raw = _test_block(
                sc, cols, target, h, cut_test)

            preds = {"Persistence": persist}
            preds["LinearRegression"] = _fit_linear_headline(
                trn[0], trn[1], val[0], val[1]).predict(Xte)
            for name, tag in (("XGBoost", "xgb"), ("LightGBM", "lgb")):
                m = _load_tree_model(name, tag, target, h, trn, val, best)
                preds[name] = m.predict(Xte)

            for kind in DL_KINDS:
                te_aln, y_dl, yhat = _load_dl(kind, target, h, cols, te_raw, best)
                if te_aln is None:
                    continue
                persist_dl = te_aln[f"persist_{target}_h{h}"].values
                obs_col = f"y_{target}_h{h}__observed"
                obs_dl_m = (te_aln[obs_col].values.astype(bool)
                            if obs_col in te_aln.columns else None)
                label = kind.upper() if kind != "transformer" else "Transformer"
                results[(target, h, label)] = evaluate.evaluate_predictions(
                    y_dl, yhat, persist_dl, observed_mask=obs_dl_m,
                    stations=te_aln[C.COL_STATION].values)
                ci = evaluate.block_bootstrap_ci(y_dl, yhat, mask=obs_dl_m)
                ci_rows.append({
                    "target": target, "horizon_h": h, "model": label,
                    "rmse": ci["point"], "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                })

            for name, yhat in preds.items():
                results[(target, h, name)] = evaluate.evaluate_predictions(
                    yte, yhat, persist, observed_mask=obs_m, stations=stn,
                    scaler=sc, target_col=tcol)
                ci = evaluate.block_bootstrap_ci(yte, yhat, mask=obs_m)
                ci_rows.append({
                    "target": target, "horizon_h": h, "model": name,
                    "rmse": ci["point"], "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                })
            log(f"  {target} h={h} evaluated")
            del te_raw, Xte, trn, val
            gc.collect()

    if os.environ.get("REGEN_WRITE_HEADLINE_TABLES", "0") == "1":
        evaluate.results_to_tables(results)
    pd.DataFrame(ci_rows).to_csv(os.path.join(C.OUT_TABLES, "bootstrap_ci.csv"),
                                 index=False)
    with open(os.path.join(C.OUT_TABLES, "results_full.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()},
                  f, indent=2, default=float)
    import export_per_station
    export_per_station.main()
    log("done — rerun: python run_all.py --stage figures && python run_all.py --stage numbers")


if __name__ == "__main__":
    main()
