"""
Step 1-2 of the revision plan: load the raw KAPSARC hourly export, clean it,
document the missingness, and run the STATION-GROUPING AUDIT.

The audit in `audit_station_grouping()` is the single most important function
in this package. It tests the hypothesis that the originally submitted feature
pipeline computed lag / rolling / persistence features on a frame sorted by
timestamp with all 29 stations interleaved, i.e. WITHOUT a groupby(station).
If that happened, `lag_1` is the previous ROW (a different station in the same
hour), not the previous HOUR at the same station.

Symptom in the submitted manuscript (Table 5, 1 h air temperature):
    persistence  RMSE = 0.8864, MAE = 0.0568, R2 = -0.5144
    XGBoost      RMSE = 0.6485, R2 = 0.1893
A RMSE/MAE ratio of ~15 and an implied hourly autocorrelation of ~0.24 are not
physically possible for station air temperature. Cross-station contamination
explains it, and it also explains why pressure (spatially coherent) and
visibility (saturated at 10 km) were barely affected while temperature
collapsed.

Run this BEFORE touching anything else.
"""
from __future__ import annotations

import json
import os
import zipfile
import numpy as np
import pandas as pd

from . import config as C


def _study_mask(dates: pd.Series) -> pd.Series:
    start = pd.Timestamp(C.STUDY_START)
    end = pd.Timestamp(C.STUDY_END) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (dates >= start) & (dates <= end)


def _read_csv_chunks(source, **kwargs) -> pd.DataFrame:
    """Chunked read, keep the manuscript date window, rename headers."""
    frames = []
    n_in = 0
    for chunk in pd.read_csv(source, usecols=list(C.RAW_USECOLS),
                             chunksize=250_000, low_memory=False, **kwargs):
        n_in += len(chunk)
        if C.COLUMN_MAP:
            chunk = chunk.rename(columns=C.COLUMN_MAP)
        chunk[C.COL_TIME] = pd.to_datetime(chunk[C.COL_TIME], errors="coerce")
        chunk = chunk.loc[_study_mask(chunk[C.COL_TIME])]
        if len(chunk):
            frames.append(chunk)
    if not frames:
        raise ValueError(
            f"No rows in {C.STUDY_START}–{C.STUDY_END} after reading {n_in:,} rows."
        )
    df = pd.concat(frames, ignore_index=True)
    print(f"  load_raw: {n_in:,} rows scanned, {len(df):,} kept in study window",
          flush=True)
    return df


def _historical_member(zf: zipfile.ZipFile) -> str:
    """Pick the raw KAPSARC export inside a zip; never the processed feature table."""
    hits = [n for n in zf.namelist()
            if n.replace("\\", "/").endswith("saudi-hourly-weather-data_Historical.csv")]
    if not hits:
        raise FileNotFoundError("saudi-hourly-weather-data_Historical.csv not in zip")
    return hits[0]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the raw hourly export (CSV, directory of CSVs, or the client zip)."""
    path = path or C.DATA_RAW
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.endswith((".csv", ".gz"))
                 and "subset" not in f.lower()
                 and "sample" not in f.lower()
                 and "processed" not in f.lower()]
        if not files:
            raise FileNotFoundError(f"No raw KAPSARC CSV under {path}")
        return pd.concat([_read_csv_chunks(f) for f in files], ignore_index=True)
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            member = _historical_member(zf)
            print(f"  load_raw: zip member {member}", flush=True)
            with zf.open(member) as fh:
                return _read_csv_chunks(fh)
    return _read_csv_chunks(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel removal, range screening, hourly index, chronological sort.

    IMPORTANT: sorting is by (station, time). Every downstream operation
    assumes this ordering.
    """
    df = df.copy()
    df[C.COL_TIME] = pd.to_datetime(df[C.COL_TIME], errors="coerce", utc=True)
    df = df.dropna(subset=[C.COL_TIME, C.COL_STATION])
    start = pd.Timestamp(C.STUDY_START, tz="UTC")
    end = pd.Timestamp(C.STUDY_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[(df[C.COL_TIME] >= start) & (df[C.COL_TIME] <= end)]
    counts = df.groupby(C.COL_STATION).size()
    keep = counts[counts >= C.MIN_STATION_RECORDS].index
    print(f"  clean: {len(df):,} rows, {counts.size} stations in window; "
          f"keeping {len(keep)} with >={C.MIN_STATION_RECORDS} records", flush=True)
    df = df[df[C.COL_STATION].isin(keep)].copy()

    for col in C.BASE_VARS:
        if col not in df.columns:
            raise KeyError(f"Expected column {col!r} not found. Columns: {list(df.columns)[:40]}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(C.SENTINELS), col] = np.nan
        lo, hi = C.PLAUSIBLE_RANGE[col]
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Collapse to a strict hourly grid per station (mean of sub-hourly reports)
    df["_hour"] = df[C.COL_TIME].dt.floor("h")
    df = (df.groupby([C.COL_STATION, "_hour"], as_index=False)[C.BASE_VARS]
            .mean())
    df = df.rename(columns={"_hour": C.COL_TIME})
    df = df.sort_values([C.COL_STATION, C.COL_TIME]).reset_index(drop=True)
    return df


def reindex_full_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Insert explicit NaN rows for missing hours so gap statistics are honest.

    Without this step a 30-hour outage looks like a 1-row gap and
    `ffill(limit=6)` silently bridges it. Reviewer 2 asked for exactly this
    level of detail on the staged imputation.
    """
    out = []
    for stn, g in df.groupby(C.COL_STATION, sort=True):
        idx = pd.date_range(g[C.COL_TIME].min(), g[C.COL_TIME].max(),
                            freq="h", tz=g[C.COL_TIME].dt.tz)
        g = g.set_index(C.COL_TIME).reindex(idx)
        g[C.COL_STATION] = stn
        g.index.name = C.COL_TIME
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True).sort_values(
        [C.COL_STATION, C.COL_TIME]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Missingness reporting  (Reviewer 2, basic reporting, bullet 3)
# --------------------------------------------------------------------------
def gap_runs(series: pd.Series) -> pd.Series:
    """Length of each consecutive run of NaNs."""
    isna = series.isna().values
    if not isna.any():
        return pd.Series(dtype=int)
    idx = np.flatnonzero(np.diff(np.r_[0, isna.view(np.int8), 0]))
    return pd.Series(idx[1::2] - idx[0::2])


def missingness_report(df: pd.DataFrame) -> dict:
    """Per-variable and per-station missingness + gap-length distribution."""
    rep = {"per_variable": {}, "per_station": {}, "gap_distribution": {}}
    n = len(df)
    for col in C.BASE_VARS:
        rep["per_variable"][col] = {
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": round(100 * df[col].isna().mean(), 2),
        }
        runs = pd.concat([gap_runs(g[col]) for _, g in df.groupby(C.COL_STATION)])
        if len(runs):
            rep["gap_distribution"][col] = {
                "n_gaps": int(len(runs)),
                "median_gap_h": float(runs.median()),
                "p90_gap_h": float(runs.quantile(0.90)),
                "max_gap_h": int(runs.max()),
                "pct_rows_in_gaps_le_6h": round(
                    100 * runs[runs <= 6].sum() / max(n, 1), 2),
                "pct_rows_in_gaps_gt_24h": round(
                    100 * runs[runs > 24].sum() / max(n, 1), 2),
            }
    per_stn = df.groupby(C.COL_STATION)[C.BASE_VARS].apply(
        lambda g: g.isna().mean() * 100).round(2)
    rep["per_station"] = json.loads(per_stn.to_json(orient="index"))
    rep["n_rows_full_hourly_grid"] = int(n)
    return rep


def staged_impute(df: pd.DataFrame, max_ffill: int = 6,
                  max_interp: int = 6) -> pd.DataFrame:
    """Staged imputation, applied STRICTLY WITHIN STATION, with an audit flag.

    Two changes versus the submitted version:
      1. groupby(station) is explicit and non-optional;
      2. every imputed cell is flagged so that the test-set evaluation can be
         restricted to OBSERVED targets only (Reviewer 2, bullet 3). Reporting
         R2 = 0.98 for pressure when 46 % of pressure was imputed is not a
         defensible claim.
    Mean imputation is now applied per station-month rather than globally, and
    only to PREDICTORS. Targets are never mean-imputed; rows whose target is
    still missing are dropped from that target's supervised set.
    """
    df = df.sort_values([C.COL_STATION, C.COL_TIME]).copy()
    for col in C.BASE_VARS:
        df[f"{col}__observed"] = df[col].notna()

    pieces = []
    for stn, g in df.groupby(C.COL_STATION, sort=True):
        g = g.sort_values(C.COL_TIME).set_index(C.COL_TIME)
        for col in C.BASE_VARS:
            s = g[col].ffill(limit=max_ffill).bfill(limit=max_ffill)
            s = s.interpolate(method="time", limit=max_interp, limit_area="inside")
            g[col] = s
        g = g.reset_index()
        g[C.COL_STATION] = stn
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    # Predictor-only fallback: station x month climatological mean
    df["_month"] = df[C.COL_TIME].dt.month
    for col in C.PREDICTORS:
        df[col] = df.groupby([C.COL_STATION, "_month"])[col].transform(
            lambda s: s.fillna(s.mean()))
    df = df.drop(columns=["_month"])
    return df


# --------------------------------------------------------------------------
# THE AUDIT  (Reviewer 2, basic reporting, bullet 4 / Red Flag 1)
# --------------------------------------------------------------------------
def audit_station_grouping(df: pd.DataFrame, target: str = "AIR_TEMPERATURE",
                           save: bool = True) -> dict:
    """Compare correct (grouped) vs contaminated (ungrouped) persistence.

    Returns a dict that the response letter quotes directly. If
    `ungrouped_matches_submission` is True, the submitted Table 5 was produced
    by the ungrouped pipeline and every result in the paper must be regenerated.
    """
    d = df.sort_values([C.COL_STATION, C.COL_TIME]).copy()

    # (a) CORRECT: previous hour at the same station
    d["lag1_grouped"] = d.groupby(C.COL_STATION)[target].shift(1)

    # (b) CONTAMINATED: previous row after sorting by time only
    d2 = df.sort_values([C.COL_TIME, C.COL_STATION]).copy()
    d2["lag1_ungrouped"] = d2[target].shift(1)
    d = d.merge(d2[[C.COL_STATION, C.COL_TIME, "lag1_ungrouped"]],
                on=[C.COL_STATION, C.COL_TIME], how="left")

    res = {}
    for name, lagcol in (("grouped", "lag1_grouped"),
                         ("ungrouped", "lag1_ungrouped")):
        m = d[[target, lagcol]].dropna()
        err = m[target].values - m[lagcol].values
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((m[target].values - m[target].values.mean()) ** 2))
        res[name] = {
            "n": int(len(m)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - ss_res / ss_tot),
            "lag1_autocorr": float(np.corrcoef(m[target], m[lagcol])[0, 1]),
            "target_sd": float(m[target].std()),
        }

    sub = {"rmse": 0.8864, "mae": 0.0568, "r2": -0.5144}  # submitted Table 5, 1 h
    res["submitted_table5_1h_persistence"] = sub
    res["ungrouped_matches_submission"] = bool(
        abs(res["ungrouped"]["r2"] - sub["r2"]) < 0.15)
    res["verdict"] = (
        "CONFIRMED: submitted results came from an ungrouped (cross-station) "
        "feature pipeline. Full regeneration required."
        if res["ungrouped_matches_submission"] else
        "NOT CONFIRMED by this test: investigate scaling, target leakage and "
        "the RMSE/MAE inconsistency separately before regenerating."
    )
    # Secondary diagnostic: RMSE >> MAE means a few huge errors dominate.
    res["rmse_over_mae_grouped"] = res["grouped"]["rmse"] / max(res["grouped"]["mae"], 1e-9)
    res["rmse_over_mae_submitted"] = sub["rmse"] / sub["mae"]

    if save:
        with open(os.path.join(C.OUT_AUDIT, f"audit_station_grouping_{target}.json"),
                  "w") as f:
            json.dump(res, f, indent=2)
    return res


def audit_all(df: pd.DataFrame) -> dict:
    out = {t: audit_station_grouping(df, col)
           for t, col in C.TARGETS.items()}
    with open(os.path.join(C.OUT_AUDIT, "audit_all_targets.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def audit_scaling_of_reported_units(df: pd.DataFrame) -> dict:
    """Second half of Red Flag 1: is Table 5 really in degrees Celsius?

    Submitted 1 h persistence RMSE = 0.8864 'degC'. If the true hourly
    persistence RMSE is ~1.3-1.6 degC and the true SD is ~9-10 degC, then the
    reported numbers are z-scores that were mislabelled as degC. Both failure
    modes must be excluded before the manuscript is rewritten.
    """
    out = {}
    for t, col in C.TARGETS.items():
        g = df.sort_values([C.COL_STATION, C.COL_TIME]).groupby(C.COL_STATION)[col]
        diff = g.diff().dropna()
        out[t] = {
            "sd_native_units": float(df[col].std()),
            "persistence_rmse_native_units": float(np.sqrt((diff ** 2).mean())),
            "persistence_mae_native_units": float(diff.abs().mean()),
        }
    with open(os.path.join(C.OUT_AUDIT, "audit_units.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def build(path: str | None = None) -> pd.DataFrame:
    """Full Step 1-2 pipeline. Writes the cleaned parquet and the audit files."""
    raw = load_raw(path)
    df = clean(raw)
    df = reindex_full_hourly(df)
    rep = missingness_report(df)
    with open(os.path.join(C.OUT_AUDIT, "missingness_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    audit_all(df)
    audit_scaling_of_reported_units(df)
    df = staged_impute(df)
    df.to_parquet(os.path.join(C.DATA_PROC, "hourly_clean.parquet"))
    return df
