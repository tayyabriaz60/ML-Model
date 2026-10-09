#!/usr/bin/env python
"""Hour-1 clock trial: one LSTM / temperature training run.

Does not change N_TRIALS. Writes outputs/clock_trial.json and commits
trial 0 of study lstm_temperature so the full search can resume.
"""
from __future__ import annotations

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
CLOCK_JSON = os.path.join(C.OUT, "clock_trial.json")
N_JOBS = 9


def main():
    if not os.path.exists(FEAT):
        raise SystemExit(f"missing {FEAT} — upload features.parquet first")
    print(f"loading {FEAT}", flush=True)
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    tr, va, _, _ = models.chronological_split(F)
    trial = next(models.random_search(models.DL_SEARCH_SPACE))
    p = models.dl_params_from_search("lstm", trial)
    print(f"params {trial} seq={p.seq_len} batch={p.batch_size}", flush=True)
    dtr = models.window_dataset(tr, cols, "y_temperature_h1", p.seq_len)
    dva = models.window_dataset(va, cols, "y_temperature_h1", p.seq_len)
    if len(dtr) < p.batch_size or len(dva) == 0:
        raise SystemExit("not enough sequence rows for this trial")
    t0 = time.time()
    net = models.build_sequence_model("lstm", len(cols), p)
    net, hist = models.train_sequence_model(net, dtr, dva, p)
    yhat = models.predict_sequence(net, dva)
    ytrue = np.array([dva[j][1] for j in range(len(dva))], dtype=float)
    r = evaluate._metrics(ytrue, yhat)
    wall = time.time() - t0
    row = {**trial, **r, "trial": 0, "epochs_run": hist["epochs_run"], "wall_s": wall}
    store = TrialStore(STUDY_DB)
    store.put("lstm_temperature", 0, row)
    minutes = wall / 60.0
    search_h = (N_JOBS * models.N_TRIALS) * minutes / 60.0
    report = {
        "arch": "lstm",
        "target": "temperature",
        "wall_clock_seconds": round(wall, 1),
        "wall_clock_minutes": round(minutes, 2),
        "epochs_run": hist["epochs_run"],
        "val_rmse": r["rmse"],
        "n_trials": models.N_TRIALS,
        "n_search_runs": N_JOBS * models.N_TRIALS,
        "search_hours_est": round(search_h, 1),
        "post_search_hours_lo": 12,
        "post_search_hours_hi": 20,
        "total_hours_lo": round(search_h + 12, 1),
        "total_hours_hi": round(search_h + 20, 1),
        "params": trial,
    }
    with open(CLOCK_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"CLOCK TRIAL {minutes:.1f} min  →  search ~{search_h:.1f} h  +  "
        f"post-search 12–20 h",
        flush=True,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
