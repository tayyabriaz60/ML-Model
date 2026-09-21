#!/usr/bin/env python
"""
Orchestrator for the manuscript revision.

Run the stages IN ORDER. Stage 1 is a gate: if the station-grouping audit
confirms cross-station contamination, everything downstream must be rerun and
the manuscript numbers regenerated from scratch. Do not skip it and do not
start writing until it has returned a verdict.

    python run_all.py --stage audit
    python run_all.py --stage features
    python run_all.py --stage tune
    python run_all.py --stage train
    python run_all.py --stage explain
    python run_all.py --stage extremes
    python run_all.py --stage ablations
    python run_all.py --stage figures
    python run_all.py --stage numbers
    python run_all.py --stage all
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
from src import data, features, models, evaluate, explain, extremes, ablations, figures
from src import export_numbers

CLEAN = os.path.join(C.DATA_PROC, "hourly_clean.parquet")
FEAT = os.path.join(C.DATA_PROC, "features.parquet")
BEST_HP = os.path.join(C.OUT_TABLES, "best_hyperparameters.json")
RUN_META = os.path.join(C.OUT, "run_meta.json")
DL_KINDS = ("lstm", "gru", "transformer")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_run_meta(**extra):
    meta = {
        "n_trials": models.N_TRIALS,
        "seed": C.SEED,
        "scaling": C.SCALING_PRODUCTION,
        "dl_subsample": C.DL_SUBSAMPLE,
        "horizons": list(C.CONFIG.horizons),
    }
    meta.update({k: v for k, v in extra.items() if v is not None})
    with open(RUN_META, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _load_best():
    if os.path.exists(BEST_HP):
        return json.load(open(BEST_HP))
    return {}


def _save_best(best):
    with open(BEST_HP, "w") as f:
        json.dump(best, f, indent=2, default=str)


# --------------------------------------------------------------------------
def stage_audit(args):
    log("STAGE 1: load, clean, missingness report, STATION-GROUPING AUDIT")
    df = data.build(args.raw)
    a = json.load(open(os.path.join(C.OUT_AUDIT, "audit_all_targets.json")))
    print(json.dumps({k: v["verdict"] for k, v in a.items()}, indent=2))
    log("Read outputs/audit/*.json before proceeding. This is a decision gate.")
    return df


def stage_features(args):
    log("STAGE 2: station-grouped feature construction")
    df = pd.read_parquet(CLEAN)
    F = features.build_features(df)
    features.assert_no_cross_station_leakage(F)      # hard gate
    cols = features.feature_columns(F)
    log(f"  {len(cols)} features built; leakage test passed")
    features.correlation_report(F, cols)
    features.vif_report(F, cols[:60])
    features.autocorrelation_profile(df)
    F.to_parquet(FEAT)
    return F


def _matrices(F, target, horizon, cols, scaler=None, strategy=None):
    tr, va, te, _ = models.chronological_split(F)
    y = f"y_{target}_h{horizon}"
    obs = f"{y}__observed"
    sc = scaler or models.Scaler(strategy or C.SCALING_PRODUCTION).fit(tr[cols])
    parts = []
    for part in (tr, va, te):
        m = part[y].notna() & part[cols].notna().all(axis=1)
        parts.append((sc.transform(part.loc[m, cols]), part.loc[m, y].values,
                      part.loc[m], m))
    return parts, sc


def _tune_trees(F, cols, targets, family):
    fit = models.fit_xgboost if family == "xgb" else models.fit_lightgbm
    best_rows = {}
    for target in targets:
        (trn, val, tst), sc = _matrices(F, target, 1, cols)
        rows = []
        for i, p in enumerate(models.random_search(models.TREE_SEARCH_SPACE)):
            tp = C.TreeParams(**{k: v for k, v in p.items()
                                 if k in C.TreeParams.__annotations__})
            m = fit(trn[0], trn[1], val[0], val[1], tp)
            r = evaluate._metrics(val[1], m.predict(val[0]))
            rows.append({**p, **r, "trial": i})
        t = pd.DataFrame(rows).sort_values("rmse")
        t.to_csv(os.path.join(C.OUT_TABLES, f"tuning_{family}_{target}.csv"), index=False)
        best_rows[f"{family}_{target}"] = t.iloc[0].to_dict()
        log(f"  {target}: {family} search done, best val RMSE {t.rmse.iloc[0]:.4f}")
    return best_rows


def _tune_dl(F, cols, targets, kinds):
    tr, va, te, _ = models.chronological_split(F)
    best_rows = {}
    cache = {}

    def ds(frame, target, seq_len):
        key = (id(frame), target, seq_len)
        if key not in cache:
            cache[key] = models.window_dataset(frame, cols, f"y_{target}_h1", seq_len)
        return cache[key]

    for target in targets:
        for kind in kinds:
            rows = []
            for i, trial in enumerate(models.random_search(models.DL_SEARCH_SPACE)):
                p = models.dl_params_from_search(kind, trial)
                dtr, dva = ds(tr, target, p.seq_len), ds(va, target, p.seq_len)
                if len(dtr) < p.batch_size or len(dva) == 0:
                    log(f"  skip {kind}/{target} trial {i}: not enough sequence rows")
                    continue
                net = models.build_sequence_model(kind, len(cols), p)
                net, hist = models.train_sequence_model(net, dtr, dva, p)
                yhat = models.predict_sequence(net, dva)
                ytrue = np.array([dva[j][1] for j in range(len(dva))], dtype=float)
                r = evaluate._metrics(ytrue, yhat)
                rows.append({**trial, **r, "trial": i, "epochs_run": hist["epochs_run"]})
                log(f"  {kind} {target} trial {i+1}/{models.N_TRIALS} val RMSE {r['rmse']:.4f}")
            if not rows:
                continue
            t = pd.DataFrame(rows).sort_values("rmse")
            t.to_csv(os.path.join(C.OUT_TABLES, f"tuning_{kind}_{target}.csv"), index=False)
            best_rows[f"{kind}_{target}"] = t.iloc[0].to_dict()
            log(f"  {target}: {kind} search done, best val RMSE {t.rmse.iloc[0]:.4f}")
    return best_rows


def stage_tune(args):
    log("STAGE 3: hyperparameter search, EQUAL BUDGET for trees and DL")
    write_run_meta(parallel=getattr(args, "parallel", 1))
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    targets = [args.target] if getattr(args, "target", None) else list(C.CONFIG.targets)
    arch = getattr(args, "arch", None)
    best = _load_best()
    do_trees = arch in (None, "tree", "xgb", "lgb")
    do_dl = arch in (None,) or arch in DL_KINDS
    if getattr(args, "parallel", 1) and args.parallel > 1:
        log("  --parallel is for splitting pods: use --target and --arch on each GPU")
    if do_trees and arch in (None, "tree", "xgb"):
        best.update(_tune_trees(F, cols, targets, "xgb"))
    if do_trees and arch in (None, "tree", "lgb"):
        best.update(_tune_trees(F, cols, targets, "lgb"))
    if do_dl:
        kinds = (arch,) if arch in DL_KINDS else DL_KINDS
        best.update(_tune_dl(F, cols, targets, kinds))
    _save_best(best)
    write_run_meta(n_best=len(best))
    return best


def stage_train(args):
    log("STAGE 4: train all families on the FULL training set, evaluate")
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    results, ci_rows, dm_rows = {}, [], []
    for target in C.CONFIG.targets:
        tcol = C.TARGETS[target]
        for h in C.CONFIG.horizons:
            (trn, val, tst), sc = _matrices(F, target, h, cols)
            Xtr, ytr = trn[0], trn[1]
            Xva, yva = val[0], val[1]
            Xte, yte, te_df = tst[0], tst[1], tst[2]
            persist = te_df[f"persist_{target}_h{h}"].values
            obs = te_df.get(f"y_{target}_h{h}__observed")
            stn = te_df[C.COL_STATION].values

            preds = {"Persistence": persist}
            for name, fn in (("LinearRegression", models.fit_linear),
                             ("XGBoost", models.fit_xgboost),
                             ("LightGBM", models.fit_lightgbm)):
                m = fn(Xtr, ytr, Xva, yva)
                preds[name] = m.predict(Xte)
                if name == "XGBoost":
                    import joblib
                    joblib.dump(m, os.path.join(
                        C.OUT_MODELS, f"xgb_{target}_h{h}.joblib"))
            tr_df, va_df, te_df2, _ = models.chronological_split(F)
            ycol = f"y_{target}_h{h}"
            best = _load_best()
            for kind in DL_KINDS:
                trial = best.get(f"{kind}_{target}", {})
                p = models.dl_params_from_search(kind, trial) if trial else (
                    C.TransformerParams() if kind == "transformer" else C.DLParams())
                dtr = models.window_dataset(tr_df, cols, ycol, p.seq_len)
                dva = models.window_dataset(va_df, cols, ycol, p.seq_len)
                dte = models.window_dataset(te_df2, cols, ycol, p.seq_len)
                if len(dtr) == 0 or len(dte) == 0:
                    log(f"  skip {kind} {target} h={h}: empty sequences")
                    continue
                net = models.build_sequence_model(kind, len(cols), p)
                net, _ = models.train_sequence_model(net, dtr, dva, p)
                yhat = models.predict_sequence(net, dte)
                # align persistence / observed to sequence row index
                te_aln = te_df2.loc[dte.row_index]
                preds_dl = yhat
                persist_dl = te_aln[f"persist_{target}_h{h}"].values
                obs_dl = te_aln.get(f"y_{target}_h{h}__observed")
                stn_dl = te_aln[C.COL_STATION].values
                yte_dl = np.array([dte[j][1] for j in range(len(dte))], dtype=float)
                results[(target, h, kind.upper() if kind != "transformer" else "Transformer")] = (
                    evaluate.evaluate_predictions(
                        yte_dl, preds_dl, persist_dl,
                        observed_mask=(obs_dl.values if obs_dl is not None else None),
                        stations=stn_dl))
                import torch
                torch.save(
                    {"kind": kind, "params": trial, "n_features": len(cols),
                     "state_dict": net.state_dict()},
                    os.path.join(C.OUT_MODELS, f"{kind}_{target}_h{h}.pt"))
                ci = evaluate.block_bootstrap_ci(yte_dl, preds_dl)
                ci_rows.append({"target": target, "horizon_h": h,
                                "model": kind, "rmse": ci["point"],
                                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
            for name, yhat in preds.items():
                results[(target, h, name)] = evaluate.evaluate_predictions(
                    yte, yhat, persist,
                    observed_mask=(obs.values if obs is not None else None),
                    stations=stn, scaler=sc, target_col=tcol)
                ci = evaluate.block_bootstrap_ci(yte, yhat)
                ci_rows.append({"target": target, "horizon_h": h, "model": name,
                                "rmse": ci["point"], "ci_low": ci["ci_low"],
                                "ci_high": ci["ci_high"]})
            for a in ("XGBoost", "LightGBM"):
                dm = evaluate.diebold_mariano(yte, preds[a], persist, h=h)
                dm_rows.append({"target": target, "horizon_h": h,
                                "model_a": a, "model_b": "Persistence", **dm})
            log(f"  {target} h={h} done")
    evaluate.results_to_tables(results)
    pd.DataFrame(ci_rows).to_csv(os.path.join(C.OUT_TABLES, "bootstrap_ci.csv"), index=False)
    pd.DataFrame(dm_rows).to_csv(os.path.join(C.OUT_TABLES, "diebold_mariano.csv"), index=False)
    return results


def stage_explain(args):
    log("STAGE 5: SHAP / LIME / attention + QUANTIFIED cross-method agreement")
    import joblib
    F = pd.read_parquet(FEAT)
    cols = features.feature_columns(F)
    agreement_all = {}
    for target in C.CONFIG.targets:
        for h in C.CONFIG.horizons:
            p = os.path.join(C.OUT_MODELS, f"xgb_{target}_h{h}.joblib")
            if not os.path.exists(p):
                continue
            m = joblib.load(p)
            (trn, val, tst), sc = _matrices(F, target, h, cols)
            Xs, sv = explain.shap_values_tree(m, tst[0])
            imp = explain.global_importance(sv, Xs.columns)
            signed = explain.signed_shap_summary(sv, Xs)
            cov = explain.topk_coverage(imp)
            lime_t = explain.lime_importance(m, trn[0], tst[0])
            rankings = {"SHAP": list(imp.feature),
                        "LIME": list(lime_t.feature)}
            pt = os.path.join(C.OUT_MODELS, f"transformer_{target}_h{h}.pt")
            lstm_pt = os.path.join(C.OUT_MODELS, f"lstm_{target}_h{h}.pt")
            seq_path = pt if os.path.exists(pt) else lstm_pt
            if os.path.exists(seq_path):
                import torch
                try:
                    blob = torch.load(seq_path, map_location="cpu", weights_only=False)
                except TypeError:
                    blob = torch.load(seq_path, map_location="cpu")
                kind = blob.get("kind", "transformer")
                p = models.dl_params_from_search(kind, blob.get("params") or {})
                tr_df, va_df, te_df2, _ = models.chronological_split(F)
                dte = models.window_dataset(te_df2, cols, f"y_{target}_h{h}", p.seq_len)
                net = models.build_sequence_model(kind, blob.get("n_features", len(cols)), p)
                net.load_state_dict(blob["state_dict"])
                n = min(2000, len(dte))
                Xs = np.stack([dte[i][0] for i in range(n)])
                ys = np.array([dte[i][1] for i in range(n)], dtype=float)
                pert = explain.perturbation_importance(
                    lambda X: models.predict_sequence(net, X), Xs, ys, cols)
                rankings["perturbation"] = list(pert.feature)
                explain.save_explainability(target, h, {"perturbation": pert})
                if kind == "transformer" and len(dte):
                    att = explain.attention_weights(net, Xs)
                    att.to_csv(os.path.join(C.OUT_XAI, f"{target}_h{h}", "attention.csv"),
                               index=False)
            ag = explain.cross_method_agreement(rankings)
            agreement_all[(target, h)] = ag
            explain.save_explainability(target, h, {
                "shap_global": imp, "shap_signed": signed,
                "shap_topk_coverage": cov, "lime_global": lime_t,
                "agreement": ag,
                "agreement_divergence": explain.agreement_divergence_table(rankings),
            })
            log(f"  {target} h={h}: median Jaccard@10 = {ag['median_jaccard@10']:.3f}")
    return agreement_all


def stage_extremes(args):
    log("STAGE 6: WMO-compliant fog and heat definitions, regime analysis")
    df = pd.read_parquet(CLEAN)
    fog = extremes.fog_events(df)
    thr = extremes.heat_threshold(df)
    heat = extremes.heat_events(df, thr)
    lab = extremes.regime_labels(df, fog, heat)
    comp = extremes.normal_class_composition(df, lab)
    log(f"  fog episodes: {len(fog)}; heat episodes: {len(heat)} "
        f"(heatwaves: {(heat.label=='heatwave').sum() if len(heat) else 0})")
    log(f"  normal class = {comp['pct_of_all']}% of hours")
    pd.DataFrame({"regime": lab}).to_parquet(
        os.path.join(C.DATA_PROC, "regime_labels.parquet"))
    return fog, heat, lab


def stage_ablations(args):
    log("STAGE 7: ablations (feature count, visibility lags, scaling, subsample)")
    F = pd.read_parquet(FEAT)
    df = pd.read_parquet(CLEAN)
    cols = features.feature_columns(F)
    target = "temperature"
    h = 1
    (trn, val, tst), sc = _matrices(F, target, h, cols)
    Xtr, ytr, Xva, yva, Xte, yte = trn[0], trn[1], val[0], val[1], tst[0], tst[1]
    shap_p = os.path.join(C.OUT_XAI, f"{target}_h{h}", "shap_global.csv")
    if os.path.exists(shap_p):
        imp = pd.read_csv(shap_p)
        ablations.run_feature_count_ablation(
            models.fit_xgboost, Xtr, ytr, Xva, yva, Xte, yte, imp)
        log("  feature-count ablation written")
    frames = {"full": {}, "short_only": {}}
    F_short = features.build_features(df, vis_short_only=True)
    for label, frame in (("full", F), ("short_only", F_short)):
        c = features.feature_columns(frame)
        for hh in (1, 3, 6):
            (a, b, c_te), _ = _matrices(frame, "visibility", hh, c)
            frames[label][hh] = (a[0], a[1], b[0], b[1], c_te[0], c_te[1])
    ablations.run_visibility_lag_ablation(models.fit_xgboost, frames)
    log("  visibility-lag ablation written")

    def build_fn(raw_df, strategy):
        # raw_df unused; features already built. Re-scale the temperature fold.
        (a, b, c_te), scaler = _matrices(F, target, h, cols, strategy=strategy)
        return a[0], a[1], b[0], b[1], c_te[0], c_te[1], scaler, C.TARGETS[target]

    ablations.run_scaling_sensitivity(build_fn, models.fit_xgboost, df, target, h)
    log("  scaling sensitivity written")

    def train_fn(X, y, Xv, yv):
        m = models.fit_xgboost(X, y, Xv, yv)
        return {"predict": m.predict}

    pack = (np.asarray(Xtr), np.asarray(ytr), np.asarray(Xva),
            np.asarray(yva), np.asarray(Xte), np.asarray(yte))
    ablations.run_subsample_ablation(train_fn, pack)
    log("  subsample ablation written")
    tree_m = evaluate._metrics(yte, models.fit_xgboost(Xtr, ytr, Xva, yva).predict(Xte))
    ablations.sequence_vs_engineered_ablation({
        "engineered_features_xgb": tree_m,
    })
    log("  sequence-input ablation stub written (full DL grid is Stage 4)")


def stage_figures(args):
    log("STAGE 8: consolidated figures (24 -> 13)")
    prof = pd.read_csv(os.path.join(C.OUT_TABLES, "autocorrelation_profile.csv"))
    figures.fig_autocorrelation(prof)
    p = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
    if os.path.exists(p):
        figures.fig_performance_with_ci(pd.read_csv(p))
    log("  figures written to outputs/figures/")


def stage_numbers(args):
    log("STAGE 9: export manuscript numbers")
    n = export_numbers.collect()
    log(f"  {len(n)} values written to {export_numbers.NUMBERS_PATH}")
    if args.manuscript:
        export_numbers.check_placeholders(args.manuscript)


STAGES = {
    "audit": stage_audit, "features": stage_features, "tune": stage_tune,
    "train": stage_train, "explain": stage_explain, "extremes": stage_extremes,
    "ablations": stage_ablations, "figures": stage_figures, "numbers": stage_numbers,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--manuscript", default=None)
    ap.add_argument("--parallel", type=int, default=1,
                    help="Hint only: split jobs across GPUs with --target/--arch")
    ap.add_argument("--target", default=None, choices=list(C.TARGETS))
    ap.add_argument("--arch", default=None,
                    choices=["tree", "xgb", "lgb"] + list(DL_KINDS))
    args = ap.parse_args()
    todo = list(STAGES) if args.stage == "all" else [args.stage]
    for s in todo:
        STAGES[s](args)
    log("done")
