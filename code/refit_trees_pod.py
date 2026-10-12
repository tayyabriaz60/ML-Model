#!/usr/bin/env python
"""Refit nine tree joblibs using fixed boosting rounds from best_hyperparameters.json."""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all import FEAT, _load_best, _matrices, _tree_params, log
from src import config as C
from src import features, models

HORIZONS = list(C.HORIZONS_MAIN)
SPECS = (("xgb", models.fit_xgboost), ("lgb", models.fit_lightgbm))


def main() -> int:
    if not os.path.exists(FEAT):
        raise SystemExit(f"missing {FEAT}")
    best = _load_best()
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    rows = []
    for target in C.CONFIG.targets:
        for h in HORIZONS:
            (trn, val, _), _ = _matrices(F, target, h, cols)
            for tag, fit_fn in SPECS:
                key = f"{tag}_{target}"
                hp = best.get(key) or {}
                if not hp.get("fixed_boost_rounds") and not hp.get("n_estimators"):
                    log(f"WARN {key} h={h}: no fixed_boost_rounds in JSON — run backfill_tree_rounds.py")
                tp = _tree_params(best, tag, target)
                m = fit_fn(trn[0], trn[1], val[0], val[1], tp)
                pred_va = m.predict(val[0])
                rmse_va = float(np.sqrt(np.mean((val[1] - pred_va) ** 2)))
                path = os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.joblib")
                joblib.dump(m, path)
                rows.append({
                    "target": target, "h": h, "family": tag,
                    "n_estimators": tp.n_estimators,
                    "early_stopping": tp.early_stopping_rounds,
                    "val_rmse": rmse_va,
                    "path": path,
                })
                log(f"  saved {path} val_rmse={rmse_va:.4f} n={tp.n_estimators}")
    rep = __import__("pandas").DataFrame(rows)
    out = os.path.join(C.OUT_TABLES, "tree_refit_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(rep.to_csv(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
