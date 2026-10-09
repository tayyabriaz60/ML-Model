"""
Build Table 9 (regime-conditioned SHAP) and Fig. 8 from saved XGBoost + features.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import joblib
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dm_trees_vs_dl import _load_part, _split_cuts
from run_all import FEAT, log
from src import config as C
from src import extremes, explain, models, figures

SCALER_JSON = os.path.join(C.OUT_TABLES, "production_scaler.json")


def _production_feature_cols() -> list[str]:
    with open(SCALER_JSON) as f:
        return list(json.load(f)["params"].keys())


def _regime_series(clean: pd.DataFrame) -> pd.Series:
    rp = os.path.join(C.DATA_PROC, "regime_labels.parquet")
    if os.path.exists(rp):
        return pd.read_parquet(rp)["regime"]
    fog = extremes.fog_events(clean)
    thr = extremes.heat_threshold(clean)
    heat = extremes.heat_events(clean, thr)
    lab = extremes.regime_labels(clean, fog, heat)
    pd.DataFrame({"regime": lab}).to_parquet(rp)
    return lab


def _load_regime_frame() -> pd.DataFrame:
    names = set(pq.read_schema(FEAT).names)
    want = [C.COL_STATION, C.COL_TIME, "VISIBILITY_DISTANCE", "AIR_TEMPERATURE"]
    for extra in ("VISIBILITY_DISTANCE__observed", "AIR_TEMPERATURE__observed"):
        if extra in names:
            want.append(extra)
    use = [c for c in want if c in names]
    return pd.read_parquet(FEAT, columns=use)


def _shap_by_regime(model, X: pd.DataFrame, lab: pd.Series, min_n: int = 200):
    lab = lab.reindex(X.index).fillna("normal")
    out = {}
    for regime in lab.unique():
        idx = lab.index[lab == regime]
        if len(idx) < min_n:
            log(f"  skip regime {regime}: n={len(idx)} < {min_n}")
            continue
        Xi = X.loc[idx]
        _, sv = explain.shap_values_tree(model, Xi)
        out[str(regime)] = explain.global_importance(sv, Xi.columns)
    return out


def main():
    if not os.path.exists(FEAT):
        raise SystemExit(f"missing {FEAT}")
    slim = _load_regime_frame()
    lab_full = _regime_series(slim)
    del slim
    gc.collect()

    cols = _production_feature_cols()
    sc = models.Scaler.load(SCALER_JSON)
    _, cut_test = _split_cuts()
    h = 1
    tables = {}
    for target in C.CONFIG.targets:
        path = os.path.join(C.OUT_MODELS, f"xgb_{target}_h{h}.joblib")
        if not os.path.exists(path):
            log(f"  skip {target}: no xgb h1")
            continue
        ycol = f"y_{target}_h{h}"
        obs = f"{ycol}__observed"
        te = _load_part(cols, [ycol, obs], cut_lo=cut_test)
        m = te[ycol].notna() & te[cols].notna().all(axis=1)
        te_ok = te.loc[m]
        Xte = sc.transform(te_ok[cols])
        m = joblib.load(path)
        lab = lab_full.reindex(te_ok.index).fillna("normal")
        by_r = _shap_by_regime(m, Xte, lab, min_n=150)
        if by_r:
            tables[target] = extremes.regime_shap_table(by_r)
        del te, te_ok, Xte
        gc.collect()

    if not tables:
        raise SystemExit("no regime SHAP tables produced")
    primary = (tables["temperature"] if "temperature" in tables
               else next(iter(tables.values())))
    primary.to_csv(os.path.join(C.OUT_TABLES, "table09_regime_shap.csv"), index=False)
    regimes = [c for c in primary.columns
               if c != "feature" and not str(c).endswith("_vs_normal_pct")]
    if regimes:
        figures.fig_regime_shap(primary, regimes)
        log("  wrote F8_regime_shap")
    for tgt, tbl in tables.items():
        if tgt != "temperature":
            tbl.to_csv(os.path.join(C.OUT_TABLES, f"table09_regime_shap_{tgt}.csv"),
                       index=False)
    log("done build_regime_shap")


if __name__ == "__main__":
    main()
