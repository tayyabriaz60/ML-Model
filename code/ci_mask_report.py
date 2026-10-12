"""Compare bootstrap_ci to authoritative table_results (all 63 rows, exact RMSE)."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
OUT = os.path.join(C.OUT_TABLES, "ci_mask_report.txt")


def _authoritative_tables() -> pd.DataFrame:
    root = os.environ.get(
        "TABLE_RESULTS_DIR", os.path.join(C.OUT_TABLES, "main_h136"))
    parts = []
    for target in C.CONFIG.targets:
        p = os.path.join(root, f"table_results_{target}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        parts.append(pd.read_csv(p))
    t = pd.concat(parts, ignore_index=True)
    return t[t.horizon_h.isin(C.HORIZONS_MAIN)]


def main():
    if not os.path.exists(CI):
        raise SystemExit(f"missing {CI}")
    ref = _authoritative_tables()
    ci = pd.read_csv(CI)
    ci = ci[ci.horizon_h.isin(C.HORIZONS_MAIN)]
    rows = []
    for _, r in ref.iterrows():
        m = ci[(ci.target == r.target) & (ci.horizon_h == r.horizon_h)
               & (ci.model.str.lower() == str(r.model).lower())]
        if len(m) == 0:
            rows.append((r.target, r.horizon_h, r.model, r.RMSE, None, "missing_ci"))
            continue
        boot = float(m.iloc[0].rmse)
        tbl = float(r.RMSE)
        st = "ok" if round(boot, 4) == round(tbl, 4) else "MISMATCH"
        rows.append((r.target, r.horizon_h, r.model, tbl, boot, st))
    rep = pd.DataFrame(rows, columns=[
        "target", "h", "model", "table_rmse", "bootstrap_rmse", "status"])
    n_ok = int((rep.status == "ok").sum())
    note = (
        f"Bootstrap vs authoritative table_results ({len(ref)} rows, all targets): "
        f"{n_ok}/{len(rep)} exact RMSE match (4 dp). "
        f"Reference: {os.environ.get('TABLE_RESULTS_DIR', 'main_h136')}."
    )
    print(note)
    print(rep.groupby("status").size().to_string())
    bad = rep[rep.status != "ok"]
    if len(bad):
        print(bad.head(20).to_string(index=False))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(note + "\n")
        f.write(rep.to_csv(index=False))
    return 0 if n_ok == len(rep) else 1


if __name__ == "__main__":
    raise SystemExit(main())
