"""Rebuild table08_summary_all.csv from per-target table_results_*.csv."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from src import config as C

ROOT = Path(C.OUT_TABLES)


def main():
    parts = []
    for target in C.CONFIG.targets:
        p = ROOT / f"table_results_{target}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        raise SystemExit("no table_results_*.csv")
    t = pd.concat(parts, ignore_index=True).sort_values(["target", "horizon_h", "model"])
    t.to_csv(ROOT / "table08_summary_all.csv", index=False)
    print(f"table08 rows: {len(t)}")


if __name__ == "__main__":
    main()
