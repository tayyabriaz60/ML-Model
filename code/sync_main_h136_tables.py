"""Copy headline main-h136 tables from main_h136/ to outputs/tables/."""
from __future__ import annotations

import shutil
from pathlib import Path

from src import config as C

SRC = Path(C.OUT_TABLES) / "main_h136"
DST = Path(C.OUT_TABLES)


def main():
    if not SRC.is_dir():
        raise SystemExit(f"missing {SRC}")
    for name in (
        "table_results_temperature.csv",
        "table_results_visibility.csv",
        "table_results_pressure.csv",
        "table08_summary_all.csv",
        "bootstrap_ci.csv",
        "diebold_mariano.csv",
    ):
        sp = SRC / name
        if sp.exists():
            shutil.copy2(sp, DST / name)
            print(f"copied {name}")
    import merge_table08
    merge_table08.main()


if __name__ == "__main__":
    main()
