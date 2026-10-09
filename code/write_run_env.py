"""Write Python / XGB / LGB versions for milestone zip."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

OUT = os.path.join(C.OUT_TABLES, "run_env.txt")


def main():
    import sklearn
    import xgboost
    import lightgbm
    line = (
        f"python={sys.version.split()[0]}\n"
        f"xgboost={xgboost.__version__}\n"
        f"lightgbm={lightgbm.__version__}\n"
        f"sklearn={sklearn.__version__}\n"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(line)
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
