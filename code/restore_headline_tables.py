"""Restore h=1,3,6 headline tables from main_h136 backup (after supp train overwrote root)."""
from __future__ import annotations

import shutil
from pathlib import Path

from src import config as C

MAIN = Path(C.OUT_TABLES) / "main_h136"
ROOT = Path(C.OUT_TABLES)
NAMES = (
    "bootstrap_ci.csv",
    "table08_summary_all.csv",
    "results_full.json",
    "table_results_visibility.csv",
    "table_results_temperature.csv",
    "table_results_pressure.csv",
)


def main():
    if not MAIN.is_dir():
        raise SystemExit(f"missing {MAIN}")
    for name in NAMES:
        src = MAIN / name
        if src.exists():
            shutil.copy2(src, ROOT / name)
            print(f"copied {name}")


if __name__ == "__main__":
    main()
