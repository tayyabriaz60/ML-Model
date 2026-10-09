"""
Step 11: the bridge between the pipeline and the manuscript.

Every number that appears in the revised manuscript is written here into
`outputs/manuscript_numbers.json` under a stable key. The manuscript source
contains placeholders of the form [[KEY]] and `build_manuscript.py`
substitutes them. Nothing is retyped by hand, so the text, the tables and the
figures cannot drift apart - which is precisely how the submitted version ended
up with three different values for the fog-regime temperature importance
(Table 9 = 0.35, Fig. 16 ~ 0.85, Fig. 22 ~ 0.37).

Run `python -m src.export_numbers --check` after a build to list any
placeholder in the manuscript that has no value yet.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import numpy as np
import pandas as pd

from . import config as C

NUMBERS_PATH = os.path.join(C.OUT, "manuscript_numbers.json")


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def collect() -> dict:
    n: dict = {}

    # ---- run metadata ----------------------------------------------------
    # The tuning budget is reported in the manuscript and the letter as
    # [[N_TRIALS]]. It is read back from the run rather than hard-coded, so a
    # budget reduced to fit a compressed schedule is stated accurately instead
    # of the text continuing to claim a number nobody spent.
    p = os.path.join(C.OUT, "run_meta.json")
    if os.path.exists(p):
        meta = json.load(open(p))
        if "n_trials" in meta:
            n["N_TRIALS"] = str(meta["n_trials"])
        if "scaling" in meta:
            n["SCALING_PRODUCTION"] = str(meta["scaling"])
    else:
        from .models import N_TRIALS as _NT
        n["N_TRIALS"] = str(_NT)
    n.setdefault("SCALING_PRODUCTION", C.SCALING_PRODUCTION)

    # ---- audit -----------------------------------------------------------
    p = os.path.join(C.OUT_AUDIT, "audit_all_targets.json")
    if os.path.exists(p):
        a = json.load(open(p))
        for t, v in a.items():
            n[f"AUDIT_{t.upper()}_PERSIST_R2_GROUPED"] = _fmt(v["grouped"]["r2"])
            n[f"AUDIT_{t.upper()}_PERSIST_R2_UNGROUPED"] = _fmt(v["ungrouped"]["r2"])
            n[f"AUDIT_{t.upper()}_PERSIST_RMSE_GROUPED"] = _fmt(v["grouped"]["rmse"])
            n[f"AUDIT_{t.upper()}_PERSIST_RMSE_UNGROUPED"] = _fmt(v["ungrouped"]["rmse"])
            n[f"AUDIT_{t.upper()}_PERSIST_MAE_GROUPED"] = _fmt(v["grouped"]["mae"])
            n[f"AUDIT_{t.upper()}_AUTOCORR_1H"] = _fmt(v["grouped"]["lag1_autocorr"])
            n[f"AUDIT_{t.upper()}_R2_FROM_LAG1"] = _fmt(
                2.0 * v["grouped"]["lag1_autocorr"] - 1.0)
            rmse_g = v["grouped"]["rmse"]
            mae_g = v["grouped"]["mae"]
            n[f"AUDIT_{t.upper()}_RMSE_OVER_MAE_GROUPED"] = _fmt(
                rmse_g / mae_g if mae_g else float("nan"), 2)
            n[f"AUDIT_{t.upper()}_SD"] = _fmt(v["grouped"]["target_sd"])
            n[f"AUDIT_{t.upper()}_VERDICT"] = v["verdict"]

    # ---- missingness -----------------------------------------------------
    p = os.path.join(C.OUT_AUDIT, "missingness_report.json")
    if os.path.exists(p):
        m = json.load(open(p))
        for col, v in m["per_variable"].items():
            n[f"MISS_{col}_PCT"] = _fmt(v["pct_missing"], 2)
        for col, v in m.get("gap_distribution", {}).items():
            n[f"GAP_{col}_MEDIAN_H"] = _fmt(v["median_gap_h"], 1)
            n[f"GAP_{col}_MAX_H"] = _fmt(v["max_gap_h"], 0)
            n[f"GAP_{col}_PCT_LE6H"] = _fmt(v["pct_rows_in_gaps_le_6h"], 2)
        if "n_rows_full_hourly_grid" in m:
            n["N_ROWS_HOURLY"] = f"{int(m['n_rows_full_hourly_grid']):,}"
        if m.get("per_station"):
            n["N_STATIONS"] = str(len(m["per_station"]))
    n.setdefault("N_STATIONS", "29")
    n["STUDY_START"] = C.STUDY_START
    n["STUDY_END"] = C.STUDY_END
    n["LIME_N"] = str(getattr(C, "LIME_N_INSTANCES", 200))

    # ---- split -----------------------------------------------------------
    p = os.path.join(C.OUT_TABLES, "split_meta.json")
    if os.path.exists(p):
        s = json.load(open(p))
        n["SPLIT_CUT_TRAIN_VAL"] = s["cut_train_val"][:10]
        n["SPLIT_CUT_VAL_TEST"] = s["cut_val_test"][:10]
        for k in ("n_train", "n_val", "n_test"):
            n[f"SPLIT_{k.upper()}"] = f"{s[k]:,}"

    # ---- main results ----------------------------------------------------
    p = os.path.join(C.OUT_TABLES, "table08_summary_all.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for _, r in t.iterrows():
            k = f"{r.target.upper()}_H{int(r.horizon_h)}_{str(r.model).upper()}"
            n[f"{k}_R2"] = _fmt(r.R2)
            n[f"{k}_RMSE"] = _fmt(r.RMSE)
            n[f"{k}_MAE"] = _fmt(r.MAE)
            n[f"{k}_SKILL"] = _fmt(r.skill_vs_persistence, 3)
            if str(r.model).lower() == "persistence":
                n[f"{r.target.upper()}_H{int(r.horizon_h)}_PERSISTENCE_R2"] = _fmt(r.R2)
                n[f"{r.target.upper()}_H{int(r.horizon_h)}_PERSISTENCE_RMSE"] = _fmt(r.RMSE)
        for tgt, g in t.groupby("target"):
            for h, gh in g.groupby("horizon_h"):
                ml = gh[~gh.model.str.lower().str.contains("persist")]
                if len(ml):
                    best = ml.loc[ml.R2.idxmax()]
                    n[f"BEST_{tgt.upper()}_H{int(h)}_MODEL"] = str(best.model)
                    n[f"BEST_{tgt.upper()}_H{int(h)}_R2"] = _fmt(best.R2)
                    n[f"BEST_{tgt.upper()}_H{int(h)}_RMSE"] = _fmt(best.RMSE)

    # ---- Diebold-Mariano (trees vs persistence + trees vs DL) ------------
    p = os.path.join(C.OUT_TABLES, "dm_trees_vs_dl.csv")
    if os.path.exists(p):
        dm = pd.read_csv(p)
        for _, r in dm.iterrows():
            a = str(r.model_a).upper().replace(" ", "")
            b = str(r.model_b).upper().replace(" ", "")
            k = f"DM_{r.target.upper()}_H{int(r.horizon_h)}_{a}_VS_{b}"
            n[f"{k}_STAT"] = _fmt(r.dm_stat, 3)
            n[f"{k}_P"] = _fmt(r.p_value, 4)
            n[f"{k}_BETTER"] = str(r.better)

    # ---- agreement -------------------------------------------------------
    for f in glob.glob(os.path.join(C.OUT_XAI, "*", "agreement.json")):
        key = os.path.basename(os.path.dirname(f)).upper()
        a = json.load(open(f))
        n[f"AGREE_{key}_MEDIAN_J10"] = _fmt(a.get("median_jaccard@10"), 3)
        n[f"AGREE_{key}_MIN_J10"] = _fmt(a.get("min_jaccard@10"), 3)
        for pair, v in a.get("pairs", {}).items():
            pk = pair.replace("__vs__", "_")
            n[f"AGREE_{key}_{pk.upper()}_J10"] = _fmt(v.get("jaccard@10"), 3)
            if "spearman_shared" in v:
                n[f"AGREE_{key}_{pk.upper()}_RHO"] = _fmt(v["spearman_shared"], 3)

    # ---- extremes --------------------------------------------------------
    for name, key in (("fog_events.csv", "FOG"), ("heat_events.csv", "HEAT")):
        p = os.path.join(C.OUT_TABLES, name)
        if os.path.exists(p):
            t = pd.read_csv(p)
            n[f"{key}_N_EVENTS"] = f"{len(t):,}"
            if key == "FOG" and len(t):
                n["FOG_MEDIAN_DURATION_H"] = _fmt(t.duration_h.median(), 1)
                n["FOG_N_DENSE"] = f"{int(t.dense.sum()):,}"
                n["FOG_MIN_VIS_MEDIAN_M"] = _fmt(t.min_visibility_m.median(), 0)
            if key == "HEAT" and len(t):
                hw = t[t.label == "heatwave"]
                n["HEAT_N_HEATWAVE"] = f"{int(len(hw)):,}"
                n["HEAT_N_SHORT"] = f"{int((t.label != 'heatwave').sum()):,}"
                if len(hw):
                    n["HEAT_MEDIAN_DURATION_D"] = _fmt(hw.duration_days.median(), 1)

    p = os.path.join(C.OUT_TABLES, "normal_class_composition.json")
    if os.path.exists(p):
        c = json.load(open(p))
        n["NORMAL_PCT_OF_ALL"] = _fmt(c["pct_of_all"], 2)
        n["NORMAL_PCT_VIS_BELOW_5KM"] = _fmt(c["pct_visibility_below_5km"], 2)
        n["NORMAL_MEDIAN_VIS_M"] = _fmt(c["visibility_m"]["median"], 0)

    # ---- ablations -------------------------------------------------------
    p = os.path.join(C.OUT_TABLES, "ablation_feature_count.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        r = t[t.k.astype(str) == "20"]
        if len(r):
            n["ABL_TOP20_R2_RETAINED_PCT"] = _fmt(float(r.r2_retained_pct.iloc[0]), 1)
            n["ABL_TOP20_RMSE_PENALTY_PCT"] = _fmt(float(r.rmse_penalty_pct.iloc[0]), 2)

    p = os.path.join(C.OUT_TABLES, "ablation_dl_subsample.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        r30 = t[np.isclose(t.train_fraction, 0.30)]
        if len(r30):
            n["ABL_SUBSAMPLE30_R2_GAP"] = _fmt(float(r30.r2_gap_vs_full.iloc[0]), 4)

    p = os.path.join(C.OUT_TABLES, "ablation_visibility_lags.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for _, r in t.iterrows():
            n[f"ABL_VIS_{str(r.variant).upper()}_H{int(r.horizon_h)}_R2"] = _fmt(r.r2)

    # ---- feature diagnostics --------------------------------------------
    p = os.path.join(C.OUT_TABLES, "collinearity_report.json")
    if os.path.exists(p):
        c = json.load(open(p))
        n["COLLIN_N_PAIRS_ABOVE_095"] = str(c["n_pairs_above_thresh"])
        n["COLLIN_MEAN_ABS_R"] = _fmt(c["mean_abs_corr"], 3)
        n["N_FEATURES_TOTAL"] = str(c.get("n_features", 122))
    elif os.path.exists(os.path.join(C.OUT_TABLES, "vif_report.csv")):
        n["N_FEATURES_TOTAL"] = str(len(pd.read_csv(
            os.path.join(C.OUT_TABLES, "vif_report.csv"))))
    else:
        n["N_FEATURES_TOTAL"] = "122"

    with open(NUMBERS_PATH, "w") as f:
        json.dump(n, f, indent=2, sort_keys=True)
    return n


def check_placeholders(manuscript_path: str) -> dict:
    """List placeholders in the manuscript that have no value yet."""
    txt = open(manuscript_path, encoding="utf-8").read()
    used = set(re.findall(r"\[\[([A-Z0-9_@]+)\]\]", txt))
    have = set(json.load(open(NUMBERS_PATH))) if os.path.exists(NUMBERS_PATH) else set()
    rep = {"n_placeholders": len(used), "n_resolved": len(used & have),
           "missing": sorted(used - have), "unused_values": sorted(have - used)}
    print(json.dumps({k: v for k, v in rep.items() if k != "unused_values"}, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="MANUSCRIPT", default=None)
    args = ap.parse_args()
    collect()
    print(f"wrote {NUMBERS_PATH}")
    if args.check:
        check_placeholders(args.check)
