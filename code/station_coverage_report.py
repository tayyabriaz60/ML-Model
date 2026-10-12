"""
Document which stations appear in per-station headline eval (observed-target mask).
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

from dm_trees_vs_dl import _load_part, _split_cuts
from run_all import FEAT
from src import config as C
from src.eval_masks import eval_row_mask

SCALER_JSON = os.path.join(C.OUT_TABLES, "production_scaler.json")
OUT_CSV = os.path.join(C.OUT_TABLES, "station_coverage_main_h136.csv")
OUT_TXT = os.path.join(C.OUT_TABLES, "station_coverage_note.txt")
PRESSURE_COL = C.TARGETS["pressure"]


def _feat_cols():
    with open(SCALER_JSON) as f:
        return list(json.load(f)["params"].keys())


def _study_and_test_stations(cut_test: pd.Timestamp) -> tuple[list[str], list[str]]:
    study = sorted(
        pq.read_table(FEAT, columns=[C.COL_STATION]).column(0).unique().to_pylist())
    te_st = _load_part([], [PRESSURE_COL], cut_lo=cut_test)
    test = sorted(te_st[C.COL_STATION].unique())
    return study, test


def _pressure_null_stations_on_test(cut_test: pd.Timestamp) -> list[str]:
    te = _load_part([], [PRESSURE_COL], cut_lo=cut_test)
    frac = te.groupby(C.COL_STATION)[PRESSURE_COL].apply(lambda s: float(s.isna().mean()))
    return sorted(frac[frac >= 1.0 - 1e-12].index.astype(str))


def main():
    cols = _feat_cols()
    _, cut_test = _split_cuts()
    study_stations, test_stations = _study_and_test_stations(cut_test)
    pressure_null_stns = _pressure_null_stations_on_test(cut_test)
    absent_from_test = sorted(set(study_stations) - set(test_stations))
    all_stations = set()
    rows = []
    for target in C.CONFIG.targets:
        for h in C.HORIZONS_MAIN:
            ycol = f"y_{target}_h{h}"
            obs = f"{ycol}__observed"
            persist = f"persist_{target}_h{h}"
            te = _load_part(cols, [ycol, obs, persist], cut_lo=cut_test)
            all_stations.update(te[C.COL_STATION].unique())
            base = te[ycol].notna() & te[cols].notna().all(axis=1)
            tree_m = eval_row_mask(te, cols, target, "LightGBM", ycol, obs)
            obs_ok = te[obs] if obs in te.columns else True
            for stn, g in te.groupby(C.COL_STATION):
                m_eval = base.loc[g.index] & obs_ok.loc[g.index]
                m_tree = tree_m.loc[g.index]
                rows.append({
                    "target": target,
                    "horizon_h": h,
                    "station": stn,
                    "test_rows_total": len(g),
                    "rows_feature_complete": int(base.loc[g.index].sum()),
                    "rows_observed_target_eval": int(m_eval.sum()),
                    "rows_tree_relaxed_eval": int(m_tree.sum()),
                    "included_in_per_station_trees": bool(m_tree.sum() > 0),
                    "included_in_per_station_strict": bool(m_eval.sum() > 0),
                })
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT_CSV, index=False)
    # Stations with zero eval rows in every target-h (dropped everywhere)
    pivot_strict = rep.groupby("station")["rows_observed_target_eval"].sum()
    pivot_trees = rep.groupby("station")["rows_tree_relaxed_eval"].sum()
    study = sorted(all_stations)
    active_strict = sorted(pivot_strict[pivot_strict > 0].index)
    active_trees = sorted(pivot_trees[pivot_trees > 0].index)
    dropped_strict = sorted(set(study) - set(active_strict))
    pressure_null_set = set(pressure_null_stns)
    tree_indicative = sorted(pressure_null_set & set(active_trees))
    lines = [
        f"Study grid stations (features.parquet): {len(study_stations)}",
        f"Stations with >=1 hourly row on or after val/test cut: {len(test_stations)}",
        f"Stations in study grid but absent from test-period rows: "
        f"{len(absent_from_test)}"
        + (f" ({', '.join(absent_from_test)})" if absent_from_test else ""),
        "",
    ]
    kharj = next((s for s in absent_from_test if "KHARJ" in s.upper()), None)
    if kharj:
        lines.append(
            f"{kharj}: retained in the {len(study_stations)}-station study grid but "
            "contributes no test-period hours after the chronological val/test cut, "
            f"so headline test metrics cover {len(test_stations)} stations.")
        lines.append("")
    elif absent_from_test:
        lines.append(
            "Stations in the study grid with no test-period hours after the val/test cut: "
            + ", ".join(absent_from_test)
            + f" (headline test metrics cover {len(test_stations)} stations).")
        lines.append("")
    lines.extend([
        f"Stations with 100% missing {PRESSURE_COL} on the test split: "
        f"{len(pressure_null_stns)}"
        + (f" ({', '.join(pressure_null_stns)})" if pressure_null_stns else ""),
        "Per-station tree metrics (XGBoost/LightGBM) use a relaxed mask for temperature "
        "and visibility: pressure-block features may be NaN while other inputs are present. "
        "Headline Table 8 / bootstrap remain on the strict feature-complete mask.",
        "",
        f"Stations with >=1 tree relaxed eval row (any target-h): {len(active_trees)}",
        f"Stations with >=1 strict feature-complete eval row (any target-h): {len(active_strict)}",
        f"Stations with zero strict eval rows on all main horizons: {len(dropped_strict)}",
        "",
    ])
    if tree_indicative:
        lines.append(
            "Pressure-all-null sites with tree relaxed coverage (indicative only; "
            "pressure block NaN passed through at predict time):")
        for stn in tree_indicative:
            sub = rep[rep.station == stn]
            lines.append(
                f"  - {stn}: max tree_relaxed={int(sub.rows_tree_relaxed_eval.max())}, "
                f"max strict={int(sub.rows_observed_target_eval.max())}")
        lines.append("")
    if dropped_strict:
        lines.append(
            "Strict-mask dropped stations (zero feature-complete observed-target rows "
            "on h=1,3,6 for all targets):")
        for stn in dropped_strict:
            if stn in tree_indicative:
                continue
            sub = rep[rep.station == stn]
            lines.append(
                f"  - {stn}: max test_rows_total={int(sub.test_rows_total.max())}, "
                f"max feature_complete={int(sub.rows_feature_complete.max())}, "
                f"max strict_eval={int(sub.rows_observed_target_eval.max())}")
    else:
        lines.append(
            "No station is entirely absent under the strict mask; per-station CSV row "
            "counts still differ by model (sequence-aligned DL vs tree relaxed grid).")
    note = "\n".join(lines)
    print(note)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(note + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
