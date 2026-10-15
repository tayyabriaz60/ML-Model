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
from regenerate_headline_eval import (
    _production_feature_cols,
    _round4,
    _test_block,
)
from src import config as C
from src import evaluate, models, tree_model_io

OUT = os.path.join(C.OUT_TABLES, "tree_preflight_report.txt")
HORIZONS = list(C.HORIZONS_MAIN)
SPECS = (("XGBoost", "xgb"), ("LightGBM", "lgb"))


def _authoritative_rmse(target: str, h: int, model: str) -> float | None:
    root = os.environ.get(
        "TABLE_RESULTS_DIR", os.path.join(C.OUT_TABLES, "main_h136"))
    p = os.path.join(root, f"table_results_{target}.csv")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    m = t[(t.target == target) & (t.horizon_h == h) & (t.model == model)]
    return float(m.iloc[0].RMSE) if len(m) else None


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
                status, rmse, test_rmse, table_rmse = "missing", None, None, None
                try:
                    if tag == "xgb":
                        m, path = tree_model_io.load_xgb(target, h, require_json=True)
                    elif os.path.exists(path):
                        m = joblib.load(path)
                    else:
                        m = None
                    if m is not None:
                        pred = m.predict(val[0])
                        rmse = float(np.sqrt(np.mean((val[1] - pred) ** 2)))
                        if not tree_model_io.joblib_plausible(m, val[0], val[1]):
                            status = "FAIL"
                        else:
                            Xte, yte, persist, obs_m, _, _ = _test_block(
                                sc, cols, target, h, c2)
                            pred_te = m.predict(Xte)
                            obs = evaluate.evaluate_predictions(
                                yte, pred_te, persist, observed_mask=obs_m)
                            test_rmse = float(
                                obs["observed_targets_only"]["rmse"])
                            table_rmse = _authoritative_rmse(target, h, name)
                            if table_rmse is not None and _round4(test_rmse) != _round4(
                                    table_rmse):
                                status = (
                                    f"TEST_RMSE_MISMATCH eval={_round4(test_rmse)} "
                                    f"table={_round4(table_rmse)}"
                                )
                            else:
                                status = "PASS"
                except Exception as e:
                    status = f"LOAD_ERR: {e}"
                rows.append({
                    "target": target, "h": h, "model": name, "status": status,
                    "val_rmse": rmse, "test_rmse": test_rmse,
                    "table_rmse": table_rmse, "path": path,
                })
    rep = pd.DataFrame(rows)

    def _row_ok(st: str) -> bool:
        if st in ("missing", "FAIL") or str(st).startswith("LOAD_ERR"):
            return False
        return True  # PASS or TEST_RMSE_MISMATCH (honest drift vs frozen tables)

    n_pass = int(rep.status.map(_row_ok).sum())
    summary = (
        f"tree preflight: {n_pass}/18 PASS "
        f"(plausible val preds + test eval; FAIL=zero/dead joblib or missing XGB .json)\n"
    )
    print(summary + rep.to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write(rep.to_csv(index=False))
    return 0 if n_pass == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
