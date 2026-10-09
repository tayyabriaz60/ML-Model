"""Export tree joblibs to native JSON (run after preflight PASS)."""
from __future__ import annotations

import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

HORIZONS = (1, 3, 6)
TARGETS = ("temperature", "visibility", "pressure")
TAGS = ("xgb", "lgb")


def _save_xgb(m, path: str):
    m.save_model(path)


def _save_lgb(m, path: str):
    m.booster_.save_model(path)


def main():
    n = 0
    for target in TARGETS:
        for h in HORIZONS:
            for tag in TAGS:
                jp = os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.joblib")
                if not os.path.exists(jp):
                    print(f"skip missing {jp}")
                    continue
                m = joblib.load(jp)
                out = os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.json")
                if tag == "xgb":
                    _save_xgb(m, out)
                else:
                    _save_lgb(m, out)
                print(f"wrote {out}")
                n += 1
    print(f"exported {n} json models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
