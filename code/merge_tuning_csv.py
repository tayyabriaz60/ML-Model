#!/usr/bin/env python
"""Build best_hyperparameters.json from tuning_{xgb|lgb}_{target}.csv files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Directory with tuning_*.csv")
    ap.add_argument("-o", "--out", default="outputs/tables/best_hyperparameters.json")
    args = ap.parse_args()
    folder = Path(args.folder)
    best = {}
    for p in sorted(folder.glob("tuning_*.csv")):
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        family, target = parts[1], "_".join(parts[2:])
        t = pd.read_csv(p).sort_values("rmse")
        best[f"{family}_{target}"] = t.iloc[0].to_dict()
        print(f"  {family}_{target}: val RMSE {t.rmse.iloc[0]:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({len(best)} keys)")


if __name__ == "__main__":
    main()
