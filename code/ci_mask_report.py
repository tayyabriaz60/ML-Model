"""Compare bootstrap_ci to authoritative Table 8 + plausibility bounds."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CODE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(
    CODE, "workspace", "backup_tables_sep30", "main_h136", "table08_summary_all.csv")
T08 = os.environ.get("TABLE08_REFERENCE", DEFAULT_REF)
CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
SCALER = os.path.join(C.OUT_TABLES, "production_scaler.json")


def _target_std_native() -> dict:
    """Use production scaler scale as native-unit σ prior for plausibility."""
    with open(SCALER) as f:
        params = json.load(f)["params"]
    out = {}
    for t, col in C.TARGETS.items():
        p = params.get(col, {})
        out[t] = float(p.get("scale", np.nan))
    return out


def main():
    if not os.path.exists(T08):
        raise SystemExit(f"missing reference table {T08}")
    t = pd.read_csv(T08)
    ci = pd.read_csv(CI)
    ci = ci[ci.horizon_h.isin(C.HORIZONS_MAIN)]
    sig = _target_std_native()
    rows = []
    for _, r in t.iterrows():
        m = ci[(ci.target == r.target) & (ci.horizon_h == r.horizon_h)
               & (ci.model.str.lower() == str(r.model).lower())]
        if len(m) == 0:
            rows.append((r.target, r.horizon_h, r.model, r.RMSE, None, r.n,
                         "missing_ci", np.nan))
            continue
        c = m.iloc[0]
        ref_ok = abs(c.rmse - r.RMSE) < max(0.05, 0.01 * r.RMSE)
        plaus = c.rmse <= 2.0 * sig.get(r.target, np.inf)
        if not plaus:
            st = "implausible_rmse"
        elif ref_ok:
            st = "ok"
        else:
            st = "MISMATCH_REF"
        rows.append((r.target, r.horizon_h, r.model, r.RMSE, c.rmse, r.n, st,
                     sig.get(r.target, np.nan)))
    rep = pd.DataFrame(rows, columns=[
        "target", "h", "model", "ref_table_rmse", "bootstrap_rmse", "ref_n",
        "status", "target_sigma_prior"])
    bad = rep[~rep.status.isin(("ok",))]
    print(rep.groupby("status").size().to_string())
    if len(bad):
        print("\nIssues (first 20):")
        print(bad.head(20).to_string(index=False))
    n_ok = int((rep.status == "ok").sum())
    note = (
        f"Mask check vs frozen reference Table 8 ({os.path.basename(T08)}): "
        f"{n_ok}/{len(rep)} rows match within tolerance AND pass RMSE <= 2x target sigma "
        f"(sigma from production_scaler.json). "
        f"Comparing bootstrap to a co-generated table08 cannot detect broken models."
    )
    print("\n" + note)
    out = os.path.join(C.OUT_TABLES, "ci_mask_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(note + "\n")
        f.write(rep.to_csv(index=False))
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
