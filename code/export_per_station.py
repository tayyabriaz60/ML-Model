"""Flatten per-station metrics from results_full.json to CSV."""
from __future__ import annotations

import json
import os

import pandas as pd

from src import config as C

RESULTS = os.path.join(C.OUT_TABLES, "results_full.json")
OUT = os.path.join(C.OUT_TABLES, "per_station_metrics.csv")


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit(f"missing {RESULTS} — run regenerate_headline_eval.py")
    data = json.load(open(RESULTS))
    rows = []
    for key, blob in data.items():
        ps = blob.get("per_station") or {}
        tbl = ps.get("table") or []
        if not tbl:
            continue
        target, h, model = key.split("|")
        for r in tbl:
            rows.append({
                "target": target,
                "horizon_h": int(h),
                "model": model,
                "station": r.get("station"),
                "n": r.get("n"),
                "rmse": r.get("rmse"),
                "mae": r.get("mae"),
                "r2": r.get("r2"),
                "skill_vs_persistence": r.get("skill_vs_persistence"),
            })
    if not rows:
        raise SystemExit("no per_station tables in results_full.json")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
