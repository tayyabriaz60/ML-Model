"""
Memory-safe ablations for local CPU (no full features.parquet in RAM).

Requires production_scaler.json from dm_trees_vs_dl.py for the production
scaling path. Visibility lag rebuilds from hourly_clean.parquet once.
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dm_trees_vs_dl import (
    FEAT, _feature_columns, _split_cuts, _load_part, _train_val_mats,
)
from run_all import CLEAN, log
from src import ablations, config as C, features, models

TARGET = "temperature"
H = 1
ABL_TRAIN_CAP = 350_000  # local CPU RAM; same split, capped train rows for XGB fit


def _cap_train(trn, val):
    Xtr, ytr = trn
    if len(ytr) <= ABL_TRAIN_CAP:
        return trn, val
    idx = np.random.default_rng(C.SEED).choice(len(ytr), ABL_TRAIN_CAP, replace=False)
    return (Xtr.iloc[idx], ytr[idx]), val


def _temp_mats(sc, cols, cut_train, cut_val):
    return _train_val_mats(
        sc, cols, TARGET, H, cut_train, cut_val, cut_val, max_train_rows=ABL_TRAIN_CAP)


def _load_test(sc, cols, cut_val, target, h):
    ycol = f"y_{target}_h{h}"
    te = _load_part(cols, [ycol], cut_lo=cut_val)
    m = te[ycol].notna() & te[cols].notna().all(axis=1)
    te = te.loc[m]
    return sc.transform(te[cols]), te[ycol].to_numpy(np.float32)


def feature_count_ablation(sc, cols, cut_train, cut_val):
    shap_p = os.path.join(C.OUT_XAI, f"{TARGET}_h{H}", "shap_global.csv")
    imp = pd.read_csv(shap_p)
    trn, val = _temp_mats(sc, cols, cut_train, cut_val)
    Xte, yte = _load_test(sc, cols, cut_val, TARGET, H)
    ablations.run_feature_count_ablation(
        models.fit_xgboost, trn[0], trn[1], val[0], val[1], Xte, yte, imp)
    log("  feature-count ablation written")


def scaling_ablation(cols, cut_train, cut_val):
    rows = []
    from src.evaluate import _metrics
    for strategy in C.SCALING_STRATEGIES:
        sc = models.Scaler(strategy=strategy)
        for c in cols:
            tbl = __import__("pyarrow.parquet", fromlist=["pq"]).read_table(
                FEAT, columns=[c],
                filters=[("DATE", "<", cut_train.to_pydatetime())])
            v = tbl.column(0).to_numpy(zero_copy_only=False).astype(np.float32)
            if len(v) > 250_000:
                v = v[np.random.default_rng(C.SEED).choice(len(v), 250_000, replace=False)]
            skewed = set(C.SKEWED_VARS)
            log1p = strategy == "log1p_standard" and sc._needs_log(c, skewed)
            if log1p:
                v = np.log1p(np.clip(v, 0, None))
            if strategy == "robust":
                centre = float(np.nanmedian(v))
                q75, q25 = np.nanpercentile(v, [75, 25])
                scale = float(q75 - q25) or 1.0
            else:
                centre = float(np.nanmean(v))
                scale = float(np.nanstd(v)) or 1.0
            sc.params[c] = {"log": log1p, "centre": centre, "scale": scale}
            del tbl, v
            gc.collect()
        trn, val = _temp_mats(sc, cols, cut_train, cut_val)
        Xte, yte = _load_test(sc, cols, cut_val, TARGET, H)
        m = models.fit_xgboost(trn[0], trn[1], val[0], val[1])
        r = _metrics(yte, m.predict(Xte))
        r.update({"strategy": strategy, "target": TARGET, "horizon_h": H, "units": "native"})
        rows.append(r)
        del trn, val, Xte, yte, m, sc
        gc.collect()
    pd.DataFrame(rows).to_csv(
        os.path.join(C.OUT_TABLES, f"ablation_scaling_{TARGET}_h{H}.csv"), index=False)
    log("  scaling sensitivity written")


def visibility_lag_ablation(cut_train, cut_val, cols):
    del cols  # full vs short feature sets differ; rebuilt from clean below
    log("  visibility lag: rebuild full vs short features from clean…")
    df = pd.read_parquet(CLEAN)
    if len(df) > 600_000:
        df = (df.sample(600_000, random_state=C.SEED)
              .sort_values([C.COL_STATION, C.COL_TIME])
              .reset_index(drop=True))
        log("  visibility lag: 600k-row clean subsample (local RAM limit)")
    F_full = features.build_features(df, vis_short_only=False)
    F_short = features.build_features(df, vis_short_only=True)
    del df
    gc.collect()
    frames = {"full": {}, "short_only": {}}
    _mat = __import__("run_all", fromlist=["_matrices"])._matrices
    for label, frame in (("full", F_full), ("short_only", F_short)):
        c = features.feature_columns(frame)
        for hh in (1, 3, 6):
            (a, b, c_te), _ = _mat(frame, "visibility", hh, c)
            trn, val = _cap_train((a[0], a[1]), (b[0], b[1]))
            frames[label][hh] = (trn[0], trn[1], val[0], val[1], c_te[0], c_te[1])
        del frame
        gc.collect()
    ablations.run_visibility_lag_ablation(models.fit_xgboost, frames)
    log("  visibility-lag ablation written")


def subsample_ablation(sc, cols, cut_train, cut_val):
    trn, val = _temp_mats(sc, cols, cut_train, cut_val)
    Xte, yte = _load_test(sc, cols, cut_val, TARGET, H)

    def train_fn(X, y, Xv, yv):
        m = models.fit_xgboost(X, y, Xv, yv)
        return {"predict": m.predict}

    pack = (np.asarray(trn[0]), np.asarray(trn[1]), np.asarray(val[0]),
            np.asarray(val[1]), np.asarray(Xte), np.asarray(yte))
    ablations.run_subsample_ablation(train_fn, pack)
    log("  subsample ablation written")


def main():
    cut_train, cut_val = _split_cuts()
    cols = _feature_columns()
    sc = models.Scaler.load(os.path.join(C.OUT_TABLES, "production_scaler.json"))
    feature_count_ablation(sc, cols, cut_train, cut_val)
    scaling_ablation(cols, cut_train, cut_val)
    visibility_lag_ablation(cut_train, cut_val, cols)
    subsample_ablation(sc, cols, cut_train, cut_val)


if __name__ == "__main__":
    main()
