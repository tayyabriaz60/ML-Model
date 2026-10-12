#!/usr/bin/env python
"""Write tree_refit_audit.txt for milestone zip (answers client tree-count question)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all import BEST_HP
from src import config as C

OUT = os.path.join(C.OUT_TABLES, "tree_refit_audit.txt")
DEFAULTS = C.TreeParams()


def main() -> int:
    best = json.load(open(BEST_HP)) if os.path.exists(BEST_HP) else {}
    lines = [
        "Tree hyperparameter audit (best_hyperparameters.json vs production defaults)",
        f"Default n_estimators={DEFAULTS.n_estimators}, "
        f"early_stopping_rounds={DEFAULTS.early_stopping_rounds}",
        "",
    ]
    for family in ("xgb", "lgb"):
        for target in C.CONFIG.targets:
            key = f"{family}_{target}"
            row = best.get(key) or {}
            n = row.get("n_estimators")
            fixed = row.get("fixed_boost_rounds") or row.get("use_early_stopping") is False
            lines.append(
                f"{key}: n_estimators={n!r} fixed_boost_rounds={fixed} "
                f"tuning_val_rmse={row.get('rmse')}")
    lines.append("")
    lines.append(
        "If n_estimators is missing, pod refit used defaults (2000 + early stopping) "
        "and can diverge from the 24-trial search effective tree count.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
