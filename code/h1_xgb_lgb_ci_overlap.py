"""Write one-line h=1 XGB vs LGB bootstrap CI overlap answer for milestone zip."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
OUT = os.path.join(C.OUT_TABLES, "h1_xgb_lgb_ci_overlap.txt")


def _intervals_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return not (hi_a < lo_b or hi_b < lo_a)


def main() -> int:
    if not os.path.exists(CI):
        raise SystemExit(f"missing {CI} — run regenerate_headline_eval.py first")
    ci = pd.read_csv(CI)
    ci = ci[ci.horizon_h == 1]
    parts = []
    any_overlap = False
    for target in C.CONFIG.targets:
        sub = ci[ci.target == target]
        x = sub[sub.model.str.lower() == "xgboost"]
        l = sub[sub.model.str.lower() == "lightgbm"]
        if len(x) != 1 or len(l) != 1:
            parts.append(f"{target}=UNKNOWN")
            continue
        x, l = x.iloc[0], l.iloc[0]
        ov = _intervals_overlap(x.ci_low, x.ci_high, l.ci_low, l.ci_high)
        any_overlap = any_overlap or ov
        parts.append(f"{target}={'yes' if ov else 'no'}")
    line = (
        "h=1 XGBoost vs LightGBM: 95% bootstrap CI overlap by target — "
        + "; ".join(parts)
        + f"; any_overlap={'yes' if any_overlap else 'no'}."
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
