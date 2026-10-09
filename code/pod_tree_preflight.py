"""Load-test nine tree joblibs on val split before milestone eval."""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dm_trees_vs_dl import _split_cuts, _train_val_mats
from regenerate_headline_eval import _joblib_plausible, _production_feature_cols
from src import config as C
from src import models

OUT = os.path.join(C.OUT_TABLES, "tree_preflight_report.txt")
HORIZONS = list(C.HORIZONS_MAIN)
SPECS = (("XGBoost", "xgb"), ("LightGBM", "lgb"))


def main():
    cols = _production_feature_cols()
    sc = models.Scaler.load(os.path.join(C.OUT_TABLES, "production_scaler.json"))
    c1, c2 = _split_cuts()
    cap = os.environ.get("REGEN_TRAIN_CAP")
    max_tr = int(cap) if cap and int(cap) > 0 else None
    rows = []
    for target in C.CONFIG.targets:
        for h in HORIZONS:
            trn, val = _train_val_mats(sc, cols, target, h, c1, c2, c2, max_train_rows=max_tr)
            for name, tag in SPECS:
                path = os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.joblib")
                status, rmse = "missing", None
                if os.path.exists(path):
                    try:
                        m = joblib.load(path)
                        pred = m.predict(val[0])
                        rmse = float(np.sqrt(np.mean((val[1] - pred) ** 2)))
                        status = "PASS" if _joblib_plausible(m, val[0], val[1]) else "FAIL"
                    except Exception as e:
                        status = f"LOAD_ERR: {e}"
                rows.append({
                    "target": target, "h": h, "model": name, "status": status, "val_rmse": rmse,
                    "path": path,
                })
    rep = pd.DataFrame(rows)
    n_pass = int((rep.status == "PASS").sum())
    summary = f"tree preflight: {n_pass}/18 PASS (9 cells x2 families)\n"
    print(summary + rep.to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write(rep.to_csv(index=False))
    return 0 if n_pass == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
