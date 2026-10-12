#!/usr/bin/env python
"""Capture best_iteration from tuning winners into best_hyperparameters.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all import BEST_HP, FEAT, _load_best, _matrices, _save_best, _tree_params, log
from src import config as C
from src import features, models
from src.tree_rounds import annotate_fixed_rounds, best_n_estimators


def _val_rmse(m, Xva, yva) -> float:
    pred = m.predict(Xva)
    return float(np.sqrt(np.mean((yva - pred) ** 2)))


SCALER_PATH = os.path.join(C.OUT_TABLES, "production_scaler.json")


def main() -> int:
    if not os.path.exists(FEAT):
        raise SystemExit(f"missing {FEAT}")
    if not os.path.exists(SCALER_PATH):
        raise SystemExit(f"missing {SCALER_PATH}")
    sc_prod = models.Scaler.load(SCALER_PATH)
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    best = _load_best()
    lines = []
    for family in ("xgb", "lgb"):
        fit = models.fit_xgboost if family == "xgb" else models.fit_lightgbm
        for target in C.CONFIG.targets:
            key = f"{family}_{target}"
            row = dict(best.get(key) or {})
            if not row:
                log(f"skip {key}: no hyperparameters")
                continue
            (trn, val, _), _ = _matrices(F, target, 1, cols, scaler=sc_prod)
            tp_tune = C.TreeParams(**{k: v for k, v in row.items()
                                      if k in C.TreeParams.__annotations__})
            tp_tune = C.TreeParams(
                max_depth=int(tp_tune.max_depth),
                learning_rate=float(tp_tune.learning_rate),
                min_child_weight=float(tp_tune.min_child_weight),
                subsample=float(tp_tune.subsample),
                colsample_bytree=float(tp_tune.colsample_bytree),
                reg_lambda=float(tp_tune.reg_lambda),
                n_estimators=C.TreeParams().n_estimators,
                early_stopping_rounds=C.TreeParams().early_stopping_rounds,
            )
            m = fit(trn[0], trn[1], val[0], val[1], tp_tune)
            rmse_es = _val_rmse(m, val[0], val[1])
            updated = annotate_fixed_rounds(row, m, family)
            n_fix = int(updated.get("n_estimators", 0))
            tp_fix = _tree_params({key: updated}, family, target)
            m_fix = fit(trn[0], trn[1], val[0], val[1], tp_fix)
            rmse_fix = _val_rmse(m_fix, val[0], val[1])
            ref_rmse = row.get("rmse")
            lines.append(
                f"{key}: tune_csv_val_rmse={ref_rmse} es_refit_val_rmse={rmse_es:.4f} "
                f"n_estimators={n_fix} fixed_val_rmse={rmse_fix:.4f}")
            best[key] = updated
            log(lines[-1])
    _save_best(best)
    out = os.path.join(C.OUT_TABLES, "tree_rounds_backfill.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
