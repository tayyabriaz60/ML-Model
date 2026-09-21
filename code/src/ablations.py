"""
Step 9: the four ablations the reviewers asked for.

  A. Feature-count ablation (Reviewer 1, exp. design 1)
     Does the full 122-feature set earn its complexity, or does the top-20 by
     mean|SHAP| retain the skill? Justifies both the feature set and the
     decision to display only the top features in the SHAP figures.

  B. Visibility long-lag ablation (Reviewer 1, exp. design 2)
     Are 12 h and 24 h visibility lags appropriate for a quantity that varies
     far faster than T or p? Compare the full set against short-lags-only.

  C. Scaling sensitivity (Reviewer 1, exp. design 4)
     StandardScaler vs RobustScaler vs log1p+standard for the skewed variables.

  D. Subsampling ablation (Reviewer 2, bullet 1)
     Quantifies exactly how much the original 30 % neural subsample cost, so
     the response letter can state the effect rather than assert it was small.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from . import config as C


def run_feature_count_ablation(fit_fn, Xtr, ytr, Xva, yva, Xte, yte,
                               importance: pd.DataFrame,
                               ks=(10, 20, 30, 50, None)) -> pd.DataFrame:
    from .evaluate import _metrics
    rows = []
    for k in ks:
        cols = (list(importance.feature.head(k)) if k else list(Xtr.columns))
        m = fit_fn(Xtr[cols], ytr, Xva[cols], yva)
        r = _metrics(yte, m.predict(Xte[cols]))
        r["n_features"] = len(cols)
        r["k"] = k if k else "all"
        rows.append(r)
    t = pd.DataFrame(rows)
    base = t[t.k == "all"].iloc[0]
    t["r2_retained_pct"] = (100 * t.r2 / base.r2).round(2)
    t["rmse_penalty_pct"] = (100 * (t.rmse - base.rmse) / base.rmse).round(2)
    t.to_csv(os.path.join(C.OUT_TABLES, "ablation_feature_count.csv"), index=False)
    return t


def run_visibility_lag_ablation(fit_fn, frames: dict, horizons=(1, 3, 6)) -> pd.DataFrame:
    """`frames` maps 'full'/'short_only' -> (Xtr,ytr,Xva,yva,Xte,yte) per horizon."""
    from .evaluate import _metrics
    rows = []
    for variant, byh in frames.items():
        for h in horizons:
            Xtr, ytr, Xva, yva, Xte, yte = byh[h]
            m = fit_fn(Xtr, ytr, Xva, yva)
            r = _metrics(yte, m.predict(Xte))
            r.update({"variant": variant, "horizon_h": h,
                      "n_features": Xtr.shape[1]})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(C.OUT_TABLES, "ablation_visibility_lags.csv"), index=False)
    return t


def run_scaling_sensitivity(build_fn, fit_fn, df, target, horizon,
                            strategies=C.SCALING_STRATEGIES) -> pd.DataFrame:
    """build_fn(df, strategy) -> (Xtr,ytr,Xva,yva,Xte,yte,scaler,target_col)."""
    from .evaluate import _metrics
    rows = []
    for s in strategies:
        Xtr, ytr, Xva, yva, Xte, yte, scaler, tcol = build_fn(df, s)
        m = fit_fn(Xtr, ytr, Xva, yva)
        yhat = scaler.inverse_target(m.predict(Xte), tcol)
        r = _metrics(scaler.inverse_target(yte, tcol), yhat)
        r.update({"strategy": s, "target": target, "horizon_h": horizon,
                  "units": "native"})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(C.OUT_TABLES,
                          f"ablation_scaling_{target}_h{horizon}.csv"), index=False)
    return t


def run_subsample_ablation(train_fn, data, fractions=(0.3, 0.5, 1.0),
                           seed=C.SEED) -> pd.DataFrame:
    from .evaluate import _metrics
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    rows = []
    for f in fractions:
        n = int(f * len(Xtr))
        idx = rng.choice(len(Xtr), n, replace=False) if f < 1 else np.arange(len(Xtr))
        model = train_fn(Xtr[idx], ytr[idx], Xva, yva)
        r = _metrics(yte, model["predict"](Xte))
        r.update({"train_fraction": f, "n_train": n})
        rows.append(r)
    t = pd.DataFrame(rows)
    base = t[t.train_fraction == 1.0]
    if len(base):
        t["r2_gap_vs_full"] = (t.r2 - float(base.r2.iloc[0])).round(4)
    t.to_csv(os.path.join(C.OUT_TABLES, "ablation_dl_subsample.csv"), index=False)
    return t


def sequence_vs_engineered_ablation(results: dict) -> pd.DataFrame:
    """Reviewer 1, exp. design 3, second half: LSTM/GRU already learn temporal
    structure, so feeding them engineered lag features may be redundant.
    Compare three input regimes: raw sequences only, engineered features only,
    and both. `results` maps regime -> metric dict."""
    t = pd.DataFrame([{"input_regime": k, **v} for k, v in results.items()])
    t.to_csv(os.path.join(C.OUT_TABLES, "ablation_sequence_inputs.csv"), index=False)
    return t
