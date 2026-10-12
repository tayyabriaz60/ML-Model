#!/usr/bin/env python
"""Document DL pressure h=1 bootstrap vs frozen Table 8 (priority-4 flag)."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CODE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get(
    "TABLE08_REFERENCE",
    os.path.join(CODE, "outputs", "tables", "main_h136", "table_results_pressure.csv"),
)
CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
OUT = os.path.join(C.OUT_TABLES, "dl_pressure_h1_note.txt")


def main() -> int:
    if not os.path.exists(CI):
        raise SystemExit(f"missing {CI}")
    ci = pd.read_csv(CI)
    ref = pd.read_csv(REF) if os.path.exists(REF) else None
    lines = [
        "DL pressure h=1 reconciliation (milestone flag)",
        f"Reference: {REF}",
        "",
    ]
    for model in ("LSTM", "Transformer"):
        c = ci[(ci.target == "pressure") & (ci.horizon_h == 1)
               & (ci.model == model)]
        boot = float(c.iloc[0].rmse) if len(c) else None
        ref_rmse = None
        if ref is not None:
            r = ref[(ref.horizon_h == 1) & (ref.model == model)]
            if len(r):
                ref_rmse = float(r.iloc[0].RMSE)
        status = "unknown"
        if boot is not None and ref_rmse is not None:
            status = "match" if abs(boot - ref_rmse) < max(0.05, 0.01 * ref_rmse) else "MISMATCH"
        lines.append(f"{model}: ref_table_RMSE={ref_rmse} bootstrap_RMSE={boot} status={status}")
    lines.extend([
        "",
        "These cells use frozen .pt weights from the original search; bootstrap RMSE "
        "can differ from Table 8 if eval mask, stack, or sequence alignment changed. "
        "No DL retrain in T5.R1 unless explicitly scoped.",
    ])
    text = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
