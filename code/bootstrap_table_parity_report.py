#!/usr/bin/env python
"""Exact 63-row check: bootstrap_ci RMSE must equal authoritative table_results."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
OUT = os.path.join(C.OUT_TABLES, "bootstrap_table_parity_report.txt")


def _authoritative_tables() -> pd.DataFrame:
    root = os.environ.get("TABLE_RESULTS_DIR", os.path.join(C.OUT_TABLES, "main_h136"))
    parts = []
    for target in C.CONFIG.targets:
        p = os.path.join(root, f"table_results_{target}.csv")
        if not os.path.exists(p):
            p = os.path.join(C.OUT_TABLES, f"table_results_{target}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing table_results for {target} under {root}")
        parts.append(pd.read_csv(p))
    t = pd.concat(parts, ignore_index=True)
    t = t[t.horizon_h.isin(C.HORIZONS_MAIN)].copy()
    return t.sort_values(["target", "horizon_h", "model"]).reset_index(drop=True)


def _norm_model(s: str) -> str:
    return str(s).strip().lower()


def main() -> int:
    if not os.path.exists(CI):
        raise SystemExit(f"missing {CI}")
    ref = _authoritative_tables()
    ci = pd.read_csv(CI)
    ci = ci[ci.horizon_h.isin(C.HORIZONS_MAIN)]
    rows = []
    for _, r in ref.iterrows():
        m = ci[(ci.target == r.target) & (ci.horizon_h == r.horizon_h)
               & (ci.model.map(_norm_model) == _norm_model(r.model))]
        boot = float(m.iloc[0].rmse) if len(m) else None
        tbl = float(r.RMSE)
        if boot is None:
            st = "missing_bootstrap"
            exact = False
        else:
            exact = round(boot, 4) == round(tbl, 4)
            st = "ok" if exact else "MISMATCH"
        rows.append({
            "target": r.target,
            "horizon_h": r.horizon_h,
            "model": r.model,
            "table_rmse": tbl,
            "bootstrap_rmse": boot,
            "status": st,
        })
    rep = pd.DataFrame(rows)
    n_ok = int((rep.status == "ok").sum())
    n = len(rep)
    header = (
        f"Bootstrap vs authoritative table_results (exact to 4 decimals): "
        f"{n_ok}/{n} ok\n"
        f"Reference dir: {os.environ.get('TABLE_RESULTS_DIR', 'main_h136')}\n"
    )
    bad = rep[rep.status != "ok"]
    print(header + rep.groupby("status").size().to_string())
    if len(bad):
        print("\nMismatches:")
        print(bad.to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(rep.to_csv(index=False))
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
