#!/usr/bin/env python
"""Client hypothesis (2026-09-23): Table 5 is persistence on a mean-imputed
target frame, including imputed rows — not z-scored residuals.

Scaling cannot change R² or RMSE/MAE. This script reconstructs the hourly
temperature series from observed flags, applies several submitted-like
imputes (including mean-fill of TARGETS), and scores 1 h within-station
persistence on ALL finite rows.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from src import config as C

SUB = {"rmse": 0.8864, "mae": 0.0568, "r2": -0.5144}
COL = "AIR_TEMPERATURE"
OBS = f"{COL}__observed"
OUT = os.path.join(C.OUT_AUDIT, "audit_imputed_target_persistence.json")


def _metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    err = yhat - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {
        "n": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse_over_mae": rmse / mae if mae > 0 else float("nan"),
        "target_sd": float(y.std()),
        "target_mean": float(y.mean()),
    }


def persist(d: pd.DataFrame, col: str) -> dict:
    d = d.sort_values([C.COL_STATION, C.COL_TIME]).copy()
    lag = d.groupby(C.COL_STATION, sort=False)[col].shift(1)
    return _metrics(d[col], lag)


def persist_slices(d: pd.DataFrame, col: str, observed: pd.Series) -> dict:
    d = d.sort_values([C.COL_STATION, C.COL_TIME]).copy()
    obs = observed.reindex(d.index)
    lag = d.groupby(C.COL_STATION, sort=False)[col].shift(1)
    both_obs = obs & obs.groupby(d[C.COL_STATION]).shift(1).fillna(False)
    either_imp = ~both_obs
    out = {"all_rows": _metrics(d[col], lag)}
    out["both_ends_observed"] = _metrics(d.loc[both_obs, col], lag.loc[both_obs])
    out["at_least_one_imputed"] = _metrics(d.loc[either_imp, col], lag.loc[either_imp])
    return out


def fill_ffill_interp(s: pd.Series, idx, max_h=6) -> pd.Series:
    s = s.copy()
    s.index = idx
    s = s.ffill(limit=max_h).bfill(limit=max_h)
    s = s.interpolate(method="time", limit=max_h, limit_area="inside")
    return s.reset_index(drop=True)


def apply_station_month_mean(df: pd.DataFrame, col: str) -> pd.Series:
    month = pd.to_datetime(df[C.COL_TIME], utc=True).dt.month
    return df.groupby([df[C.COL_STATION], month])[col].transform(lambda s: s.fillna(s.mean()))


def apply_global_mean(s: pd.Series) -> pd.Series:
    return s.fillna(s.mean())


def match(m: dict) -> dict:
    return {
        "rmse_abs_err": abs(m["rmse"] - SUB["rmse"]),
        "mae_abs_err": abs(m["mae"] - SUB["mae"]),
        "r2_abs_err": abs(m["r2"] - SUB["r2"]),
        "ratio_abs_err": abs(m["rmse_over_mae"] - SUB["rmse"] / SUB["mae"]),
        "close": (
            abs(m["rmse"] - SUB["rmse"]) < 0.15
            and abs(m["mae"] - SUB["mae"]) < 0.05
            and abs(m["r2"] - SUB["r2"]) < 0.15
        ),
    }


def main():
    path = os.path.join(C.DATA_PROC, "hourly_clean.parquet")
    df = pd.read_parquet(path, columns=[C.COL_STATION, C.COL_TIME, COL, OBS])
    df[C.COL_TIME] = pd.to_datetime(df[C.COL_TIME], utc=True)
    observed = df[OBS].astype(bool)
    raw = df[COL].where(observed)

    miss = {
        "n_hourly_grid": int(len(df)),
        "n_temperature_missing": int((~observed).sum()),
        "temperature_missing_pct": round(100 * (~observed).mean(), 2),
        "pressure_missing_pct_from_audit": 78.18,
        "note": "Temperature missingness is on the complete hourly grid "
                "before any fill, reconstructed from AIR_TEMPERATURE__observed.",
    }

    variants = {}

    # 0. Observed-only (no imputed targets in the score)
    d0 = df[[C.COL_STATION, C.COL_TIME]].copy()
    d0[COL] = raw
    variants["observed_only_dropna"] = persist(d0, COL)

    # 1. Revision impute already in parquet (ffill/interp; leftover NaN; no target mean)
    variants["revision_ffill_interp_no_target_mean"] = persist(df, COL)

    # 2. ffill/interp then STATION-MONTH mean on the TARGET (client hypothesis)
    filled = df[[C.COL_STATION, C.COL_TIME]].copy()
    pieces = []
    for stn, g in df.groupby(C.COL_STATION, sort=False):
        s = fill_ffill_interp(raw.loc[g.index], g[C.COL_TIME])
        p = g[[C.COL_STATION, C.COL_TIME]].copy()
        p[COL] = s.values
        pieces.append(p)
    d2 = pd.concat(pieces, ignore_index=True)
    d2[COL] = apply_station_month_mean(d2, COL)
    variants["ffill_interp_then_station_month_mean_on_target"] = persist_slices(
        d2, COL, observed.reset_index(drop=True) if False else observed
    )
    # observed index may not align after concat — rebuild flag from original order
    # groupby sort=False + concat preserves row order if groups are in file order.
    # Safer: work on a copy of df in place.
    df2 = df[[C.COL_STATION, C.COL_TIME]].copy()
    df2[COL] = raw
    acc = []
    for stn, g in df2.groupby(C.COL_STATION, sort=False):
        acc.append(fill_ffill_interp(g[COL], g[C.COL_TIME]))
    df2[COL] = pd.concat(acc).to_numpy()
    df2[COL] = apply_station_month_mean(df2, COL)
    variants["ffill_interp_then_station_month_mean_on_target"] = persist_slices(
        df2, COL, observed)

    # 3. Station-month mean ONLY (no ffill) — pure mean-impute of targets
    df3 = df[[C.COL_STATION, C.COL_TIME]].copy()
    df3[COL] = raw
    df3[COL] = apply_station_month_mean(df3, COL)
    variants["station_month_mean_only"] = persist_slices(df3, COL, observed)

    # 4. Global mean only
    df4 = df[[C.COL_STATION, C.COL_TIME]].copy()
    df4[COL] = apply_global_mean(raw)
    variants["global_mean_only"] = persist_slices(df4, COL, observed)

    # 5. Station mean only (not by month)
    df5 = df[[C.COL_STATION, C.COL_TIME]].copy()
    df5[COL] = raw
    df5[COL] = df5.groupby(C.COL_STATION)[COL].transform(lambda s: s.fillna(s.mean()))
    variants["station_mean_only"] = persist_slices(df5, COL, observed)

    # 6. Negative control: z-score after correct observed persist (R² and ratio must hold)
    m_obs = variants["observed_only_dropna"]
    variants["zscore_negative_control"] = {
        "rmse_if_divide_by_sd": m_obs["rmse"] / m_obs["target_sd"],
        "mae_if_divide_by_sd": m_obs["mae"] / m_obs["target_sd"],
        "r2_unchanged": m_obs["r2"],
        "ratio_unchanged": m_obs["rmse_over_mae"],
        "note": "Affine rescaling cannot move R2 or RMSE/MAE. "
                "z-scoring observed persistence gives RMSE≈0.16, not 0.8864.",
    }

    scored = {}
    any_close = False
    for name, blob in variants.items():
        core = blob["all_rows"] if isinstance(blob, dict) and "all_rows" in blob else blob
        if "rmse" in core:
            core = dict(core)
            core["match"] = match(core)
            any_close = any_close or core["match"]["close"]
            if isinstance(blob, dict) and "all_rows" in blob:
                blob = dict(blob)
                blob["all_rows"] = core
                variants[name] = blob
            else:
                variants[name] = core
        scored[name] = core.get("match") if isinstance(core, dict) else None

    report = {
        "submitted_table5_1h_persistence": SUB,
        "submitted_rmse_over_mae": SUB["rmse"] / SUB["mae"],
        "missingness": miss,
        "any_variant_reproduces_table5": any_close,
        "verdict": (
            "REPRODUCED: mean-imputed-target persistence matches Table 5 "
            "within tolerance."
            if any_close else
            "NOT REPRODUCED: none of the imputed-target persistence variants "
            "match Table 5 (RMSE 0.8864 / MAE 0.0568 / R2 -0.5144). "
            "The submitted computation cannot be reconstructed from this "
            "extract. Report corrected observed-target values; do not assert "
            "a mechanism."
        ),
        "variants": variants,
    }
    os.makedirs(C.OUT_AUDIT, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({
        "temperature_missing_pct": miss["temperature_missing_pct"],
        "n_missing": miss["n_temperature_missing"],
        "n_grid": miss["n_hourly_grid"],
        "any_close": any_close,
        "verdict": report["verdict"],
        "all_row_scores": {
            k: {kk: variants[k]["all_rows"][kk] if "all_rows" in variants[k]
                else variants[k].get(kk)
                for kk in ("n", "rmse", "mae", "r2", "rmse_over_mae")}
            if isinstance(variants[k], dict) else variants[k]
            for k in variants
            if k != "zscore_negative_control"
        },
        "zscore_control": variants["zscore_negative_control"],
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
