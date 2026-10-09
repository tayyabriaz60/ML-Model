"""
Diebold-Mariano: tree models vs deep learning on aligned test sequences.

Loads features.parquet in split-filtered chunks to stay within RAM on CPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all import FEAT, _load_best, _tree_params, log
from src import config as C
from src import features, models, evaluate

DL_KINDS = ("lstm", "gru", "transformer")
HORIZONS = list(C.HORIZONS_MAIN)
TREE_NAMES = tuple(
    x.strip() for x in os.environ.get("DM_TREE_NAMES", "XGBoost,LightGBM").split(",") if x.strip())
TRAIN_CAP = int(os.environ.get("DM_TRAIN_CAP", "120000"))


def _split_cuts() -> tuple[pd.Timestamp, pd.Timestamp]:
    p = os.path.join(C.OUT_TABLES, "split_meta.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p}")
    m = json.load(open(p))
    c1 = pd.Timestamp(m["cut_train_val"])
    c2 = pd.Timestamp(m["cut_val_test"])
    return c1, c2


def _fit_scaler(cols: list[str], cut_train: pd.Timestamp) -> models.Scaler:
    sc = models.Scaler(strategy=C.SCALING_PRODUCTION)
    sc.params = {}
    skewed = set(C.SKEWED_VARS)
    for c in cols:
        tbl = pq.read_table(
            FEAT, columns=[c], filters=[("DATE", "<", cut_train.to_pydatetime())])
        v = tbl.column(0).to_numpy(zero_copy_only=False).astype(np.float32)
        if len(v) > 250_000:
            v = v[np.random.default_rng(C.SEED).choice(len(v), 250_000, replace=False)]
        log1p = sc.strategy == "log1p_standard" and sc._needs_log(c, skewed)
        if log1p:
            v = np.log1p(np.clip(v, 0, None))
        del tbl
        gc.collect()
        if sc.strategy == "robust":
            centre = float(np.nanmedian(v))
            q75, q25 = np.nanpercentile(v, [75, 25])
            scale = float(q75 - q25) or 1.0
        else:
            centre = float(np.nanmean(v))
            scale = float(np.nanstd(v)) or 1.0
        sc.params[c] = {"log": log1p, "centre": centre, "scale": scale}
    return sc


def _load_part(cols, extra, cut_lo: pd.Timestamp | None = None,
               cut_hi: pd.Timestamp | None = None,
               max_rows: int | None = None) -> pd.DataFrame:
    use = list(dict.fromkeys(cols + extra + [C.COL_TIME, C.COL_STATION]))
    filters = []
    if cut_lo is not None:
        filters.append(("DATE", ">=", cut_lo.to_pydatetime()))
    if cut_hi is not None:
        filters.append(("DATE", "<", cut_hi.to_pydatetime()))
    if filters:
        df = pd.read_parquet(FEAT, columns=use, filters=filters)
    else:
        df = pd.read_parquet(FEAT, columns=use)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=C.SEED)
    return df


def _tree_models(target: str, h: int, trn, val, best: dict) -> dict:
    """Load or refit tree families listed in DM_TREE_NAMES (once per target, h)."""
    Xtr, ytr = trn[0], trn[1]
    Xva, yva = val[0], val[1]
    use_joblib = not os.environ.get("DM_REFIT_TREES") and os.environ.get(
        "DM_USE_JOBLIB", "1") != "0"
    specs = {
        "XGBoost": ("xgb", models.fit_xgboost, f"xgb_{target}_h{h}.joblib"),
        "LightGBM": ("lgb", models.fit_lightgbm, f"lgb_{target}_h{h}.joblib"),
    }
    out = {}
    for name in TREE_NAMES:
        if name not in specs:
            log(f"  skip unknown tree name {name!r}")
            continue
        tag, fit_fn, fname = specs[name]
        path = os.path.join(C.OUT_MODELS, fname)
        if use_joblib and os.path.exists(path):
            try:
                cand = joblib.load(path)
                pred_va = cand.predict(Xva)
                rmse_va = float(np.sqrt(np.mean((yva - pred_va) ** 2)))
                # Pod joblib + local parquet often disagree; refit if val RMSE is absurd.
                y_scale = float(np.nanstd(yva)) or 1.0
                if rmse_va > max(10.0, 2.5 * y_scale):
                    log(f"  {name} joblib val RMSE={rmse_va:.2f} (y_std={y_scale:.2f}) "
                        f"— refitting for DM")
                else:
                    out[name] = cand
                    log(f"  loaded {name} {target} h={h} from joblib (val RMSE={rmse_va:.3f})")
                    continue
            except Exception as e:
                log(f"  {name} joblib load failed ({target} h={h}): {e}; refitting")
        log(f"  refit {name} {target} h={h} n={len(ytr)}")
        out[name] = fit_fn(
            Xtr, ytr, Xva, yva, _tree_params(best, tag, target))
    return out


def _load_dl(kind: str, target: str, h: int, cols, te_df, best: dict):
    pt = os.path.join(C.OUT_MODELS, f"{kind}_{target}_h{h}.pt")
    if not os.path.exists(pt):
        return None, None, None
    try:
        blob = torch.load(pt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(pt, map_location="cpu")
    trial = blob.get("params") or best.get(f"{kind}_{target}", {})
    p = models.dl_params_from_search(kind, trial) if trial else (
        C.TransformerParams() if kind == "transformer" else C.DLParams())
    ycol = f"y_{target}_h{h}"
    dte = models.window_dataset(te_df, cols, ycol, p.seq_len)
    if len(dte) == 0:
        return None, None, None
    net = models.build_sequence_model(kind, blob.get("n_features", len(cols)), p)
    net.load_state_dict(blob["state_dict"])
    yhat = models.predict_sequence(net, dte)
    te_aln = te_df.loc[dte.row_index]
    y = np.array([dte[j][1] for j in range(len(dte))], dtype=float)
    return te_aln, y, yhat


def _obs_mask(te_aln, target: str, h: int):
    col = f"y_{target}_h{h}__observed"
    if col not in te_aln.columns:
        return None
    return te_aln[col].values.astype(bool)


def _train_val_mats(sc, cols, target, h, cut_train, cut_val, cut_test=None,
                    max_train_rows: int | None = None):
    ycol = f"y_{target}_h{h}"
    cap = max_train_rows if max_train_rows else None
    tr = _load_part(cols, [ycol], cut_hi=cut_train, max_rows=cap)
    va = _load_part(cols, [ycol], cut_lo=cut_train, cut_hi=cut_val)
    mtr = tr[ycol].notna()
    for c in cols:
        mtr &= tr[c].notna()
    mva = va[ycol].notna()
    for c in cols:
        mva &= va[c].notna()
    tr_ok = tr.loc[mtr]
    Xtr = sc.transform(tr_ok[cols])
    Xva = sc.transform(va.loc[mva, cols])
    ytr = tr_ok[ycol].to_numpy(np.float32)
    yva = va.loc[mva, ycol].to_numpy(np.float32)
    del tr, va
    gc.collect()
    return (Xtr, ytr), (Xva, yva)


def _feature_columns() -> list[str]:
    bad = (C.COL_STATION, C.COL_TIME)
    names = pq.read_schema(FEAT).names
    return [c for c in names
            if c not in bad
            and not c.startswith(("y_", "persist_"))
            and not c.endswith("__observed")]


def run() -> pd.DataFrame:
    cols = _feature_columns()
    cut_train, cut_val = _split_cuts()
    best = _load_best()
    log("  fitting scaler (column-wise on train split)...")
    sc = _fit_scaler(cols, cut_train)
    sc_path = os.path.join(C.OUT_TABLES, "production_scaler.json")
    sc.save(sc_path)
    log(f"  saved {sc_path}")
    rows = []
    for target in C.CONFIG.targets:
        for h in HORIZONS:
            ycol = f"y_{target}_h{h}"
            obs = f"{ycol}__observed"
            persist = f"persist_{target}_h{h}"
            extra = [ycol, obs, persist]
            te = _load_part(cols, extra, cut_lo=cut_val)
            mte = te[ycol].notna() & te[cols].notna().all(axis=1)
            te = te.loc[mte]
            trn, val = _train_val_mats(
                sc, cols, target, h, cut_train, cut_val, cut_val, max_train_rows=TRAIN_CAP)
            trees = _tree_models(target, h, trn, val, best)
            for kind in DL_KINDS:
                te_aln, y_dl, pred_dl = _load_dl(kind, target, h, cols, te, best)
                if te_aln is None:
                    log(f"  skip DM {target} h={h} {kind}: missing model or sequences")
                    continue
                X_aln = sc.transform(te_aln[cols])
                for tree in TREE_NAMES:
                    pred_tree = trees[tree].predict(X_aln)
                    y, a, b = y_dl, pred_tree, pred_dl
                    mask = _obs_mask(te_aln, target, h)
                    if mask is not None:
                        y, a, b = y[mask], a[mask], b[mask]
                    dm = evaluate.diebold_mariano(y, a, b, h=h)
                    rows.append({
                        "target": target,
                        "horizon_h": h,
                        "model_a": tree,
                        "model_b": kind.upper() if kind != "transformer" else "Transformer",
                        "comparison": "trees_vs_dl",
                        **dm,
                    })
                    log(f"  DM {tree} vs {kind} {target} h={h}: "
                        f"dm={dm['dm_stat']:.3f} p={dm['p_value']:.4g} better={dm['better']}")
            del te
            gc.collect()
    out = pd.DataFrame(rows)
    dm_path = os.path.join(C.OUT_TABLES, "dm_trees_vs_dl.csv")
    out.to_csv(dm_path, index=False)
    main_dm = os.path.join(C.OUT_TABLES, "main_h136", "diebold_mariano.csv")
    legacy = os.path.join(C.OUT_TABLES, "diebold_mariano.csv")
    if os.path.exists(main_dm):
        base = pd.read_csv(main_dm)
        if "comparison" not in base.columns:
            base["comparison"] = "trees_vs_persistence"
    elif os.path.exists(legacy):
        base = pd.read_csv(legacy)
        if "comparison" not in base.columns:
            base["comparison"] = "trees_vs_persistence"
        base = base[base.get("comparison", "trees_vs_persistence") == "trees_vs_persistence"]
    else:
        base = pd.DataFrame()
    combined = pd.concat([base, out], ignore_index=True)
    combined.to_csv(legacy, index=False)
    log(f"  wrote {len(out)} DM trees-vs-DL rows -> {dm_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run()
