#!/usr/bin/env python
"""Equal-budget DL search (LSTM / GRU / transformer x 3 targets).

Each finished trial is committed to SQLite before the next starts.
Trees are not run. Set N_TRIALS in the environment (paper default 24).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as C
from src import features, models, evaluate
from src.trial_store import TrialStore

FEAT = os.path.join(C.DATA_PROC, "features.parquet")
STUDY_DB = os.path.join(C.OUT, "optuna_study.sqlite")
BEST_HP = os.path.join(C.OUT_TABLES, "best_hyperparameters.json")
KINDS = ("lstm", "gru", "transformer")


def _save_best(best: dict) -> None:
    os.makedirs(C.OUT_TABLES, exist_ok=True)
    existing = {}
    if os.path.exists(BEST_HP):
        existing = json.load(open(BEST_HP))
    existing.update(best)
    with open(BEST_HP, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(C.TARGETS), default=None)
    ap.add_argument("--arch", choices=list(KINDS), default=None)
    args = ap.parse_args()
    targets = [args.target] if args.target else list(C.CONFIG.targets)
    kinds = [args.arch] if args.arch else list(KINDS)
    if not os.path.exists(FEAT):
        raise SystemExit(f"missing {FEAT}")
    print(
        f"N_TRIALS={models.N_TRIALS} targets={targets} kinds={kinds} "
        f"loading {FEAT}",
        flush=True,
    )
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    tr, va, _, _ = models.chronological_split(F)
    store = TrialStore(STUDY_DB)
    cache = {}
    best = {}

    def ds(frame, target, seq_len):
        key = (id(frame), target, seq_len)
        if key not in cache:
            cache[key] = models.window_dataset(frame, cols, f"y_{target}_h1", seq_len)
        return cache[key]

    for target in targets:
        for kind in kinds:
            study = f"{kind}_{target}"
            done = store.rows(study)
            print(f"=== {study} ({len(done)} stored) ===", flush=True)
            for i, trial in enumerate(models.random_search(models.DL_SEARCH_SPACE)):
                if store.has(study, i):
                    print(f"  skip {study} trial {i+1}/{models.N_TRIALS}", flush=True)
                    continue
                p = models.dl_params_from_search(kind, trial)
                dtr, dva = ds(tr, target, p.seq_len), ds(va, target, p.seq_len)
                if len(dtr) < p.batch_size or len(dva) == 0:
                    print(f"  skip {study} trial {i}: empty", flush=True)
                    continue
                t0 = time.time()
                net = models.build_sequence_model(kind, len(cols), p)
                net, hist = models.train_sequence_model(net, dtr, dva, p)
                yhat = models.predict_sequence(net, dva)
                ytrue = np.array([dva[j][1] for j in range(len(dva))], dtype=float)
                r = evaluate._metrics(ytrue, yhat)
                wall = time.time() - t0
                row = {**trial, **r, "trial": i,
                       "epochs_run": hist["epochs_run"], "wall_s": wall}
                store.put(study, i, row)
                rows = store.rows(study)
                pd.DataFrame(rows).sort_values("rmse").to_csv(
                    os.path.join(C.OUT_TABLES, f"tuning_{kind}_{target}.csv"),
                    index=False,
                )
                print(
                    f"  {study} trial {i+1}/{models.N_TRIALS} "
                    f"RMSE {r['rmse']:.4f} {wall/60.0:.1f} min "
                    f"{hist['epochs_run']} ep",
                    flush=True,
                )
            rows = store.rows(study)
            if rows:
                t = pd.DataFrame(rows).sort_values("rmse")
                best[study] = t.iloc[0].to_dict()
                _save_best(best)
                print(f"  {study} BEST RMSE {t.rmse.iloc[0]:.4f}", flush=True)
    print("DL SEARCH DONE", flush=True)
    print(json.dumps({k: v.get("rmse") for k, v in best.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
