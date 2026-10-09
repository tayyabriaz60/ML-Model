"""
Step 6: evaluation with the statistical rigour Reviewer 2 asked for
(bullets 7 and 8), plus native-unit reporting for visibility (Reviewer 1,
basic reporting 8 and 9).

Key points that go into the revised manuscript:

* All headline metrics are reported in NATIVE UNITS (degC, m, hPa). Scaled
  values are relegated to a supplementary table.
* R2 is invariant to any affine rescaling, so the cross-variable ordering
  "pressure > visibility > temperature" is NOT an artefact of scaling. That is
  stated explicitly in the revised Section 6.2.1 in answer to Reviewer 1's
  basic-reporting point 9. What IS misleading is comparing RMSE across
  variables with different units, so a unit-free skill score is added.
* Skill score vs persistence, SS = 1 - MSE_model / MSE_persistence, is the
  primary cross-variable comparator. It directly addresses Reviewer 2's
  bullet 8: pressure R2 = 0.989 looks spectacular only until the persistence
  baseline of 0.981 is taken into account.
* Test-set metrics are computed twice: on all rows, and on rows whose TARGET
  was actually observed rather than imputed.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
def _metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("n", "rmse", "mae", "bias", "r2")}
    err = yhat - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def skill_score(y, yhat, y_ref) -> float:
    """SS = 1 - MSE(model) / MSE(reference). Unit-free, comparable across
    variables, and the metric the revised Section 6.2 leads with."""
    y, yhat, y_ref = map(lambda a: np.asarray(a, float), (y, yhat, y_ref))
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(y_ref)
    mse_m = np.mean((yhat[ok] - y[ok]) ** 2)
    mse_r = np.mean((y_ref[ok] - y[ok]) ** 2)
    return float(1 - mse_m / mse_r) if mse_r > 0 else np.nan


def evaluate_predictions(y_true, y_pred, y_persist, observed_mask=None,
                         stations=None, scaler=None, target_col=None,
                         y_was_scaled: bool = False) -> dict:
    """Full metric bundle for one model x target x horizon.

    Targets in this pipeline are trained in native units (°C, m, hPa).
    Only invert the scaler when `y_was_scaled=True`. Calling inverse on
    already-native y was turning headline RMSE/MAE into garbage.
    """
    out = {}
    if y_was_scaled and scaler is not None and target_col is not None:
        y_true = scaler.inverse_target(np.asarray(y_true, float), target_col)
        y_pred = scaler.inverse_target(np.asarray(y_pred, float), target_col)
        y_persist = scaler.inverse_target(np.asarray(y_persist, float), target_col)
        out["units"] = "native"
    else:
        out["units"] = "native"
    out["all_rows"] = _metrics(y_true, y_pred)
    out["all_rows"]["skill_vs_persistence"] = skill_score(y_true, y_pred, y_persist)
    out["persistence"] = _metrics(y_true, y_persist)

    if observed_mask is not None:
        m = np.asarray(observed_mask, bool)
        out["observed_targets_only"] = _metrics(y_true[m], y_pred[m])
        out["observed_targets_only"]["skill_vs_persistence"] = skill_score(
            y_true[m], y_pred[m], y_persist[m])
        out["observed_targets_only"]["pct_of_test"] = round(100 * m.mean(), 2)

    if stations is not None:
        out["per_station"] = per_station_metrics(y_true, y_pred, y_persist, stations)
    return out


def per_station_metrics(y_true, y_pred, y_persist, stations) -> dict:
    """Reviewer 2, bullet 7: variance across the 29 stations."""
    d = pd.DataFrame({"s": np.asarray(stations), "y": y_true,
                      "p": y_pred, "b": y_persist})
    rows = []
    for s, g in d.groupby("s"):
        m = _metrics(g.y, g.p)
        m["station"] = s
        m["skill_vs_persistence"] = skill_score(g.y, g.p, g.b)
        rows.append(m)
    t = pd.DataFrame(rows)
    return {
        "n_stations": int(len(t)),
        "rmse_mean": float(t.rmse.mean()), "rmse_sd": float(t.rmse.std()),
        "rmse_min": float(t.rmse.min()), "rmse_max": float(t.rmse.max()),
        "r2_mean": float(t.r2.mean()), "r2_sd": float(t.r2.std()),
        "r2_p10": float(t.r2.quantile(0.10)), "r2_p90": float(t.r2.quantile(0.90)),
        "skill_mean": float(t.skill_vs_persistence.mean()),
        "skill_sd": float(t.skill_vs_persistence.std()),
        "table": t.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Moving-block bootstrap  (Reviewer 2, bullet 7)
# --------------------------------------------------------------------------
def block_bootstrap_ci(y_true, y_pred, stat="rmse", n_boot=C.BOOTSTRAP_N,
                       block=C.BOOTSTRAP_BLOCK_H, level=C.CI_LEVEL,
                       seed=C.SEED, mask=None) -> dict:
    """Moving-block bootstrap CI. Plain i.i.d. resampling would be wrong here
    because hourly forecast errors are strongly autocorrelated.

    When ``mask`` is set (observed-target rows), CIs match headline Table 8.
    """
    y, yhat = np.asarray(y_true, float), np.asarray(y_pred, float)
    if mask is not None:
        m = np.asarray(mask, bool)
        y, yhat = y[m], yhat[m]
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    n = len(y)
    if n < 2 * block:
        block = max(2, n // 10)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, max(1, n - block))
    vals = []
    for _ in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = idx[idx < n]
        m = _metrics(y[idx], yhat[idx])
        vals.append(m[stat])
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"stat": stat, "point": _metrics(y, yhat)[stat],
            "ci_low": float(lo), "ci_high": float(hi),
            "level": level, "n_boot": n_boot, "block_h": block}


# --------------------------------------------------------------------------
# Diebold-Mariano  (Reviewer 2, bullets 7 and 8)
# --------------------------------------------------------------------------
def diebold_mariano(y_true, pred_a, pred_b, h: int = 1, loss: str = C.DM_LOSS) -> dict:
    """Is model A significantly better than model B?

    Needed above all for the pressure comparison: XGBoost R2 = 0.9892 versus
    persistence 0.9813 sounds decisive, but the increment is small and the
    manuscript must say whether it is statistically distinguishable.
    Newey-West HAC variance with a (h-1) lag truncation, Harvey-Leybourne-
    Newbold small-sample correction applied.
    """
    y = np.asarray(y_true, float)
    a, b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    ea, eb = y - a, y - b
    d = (ea ** 2 - eb ** 2) if loss == "squared" else (np.abs(ea) - np.abs(eb))
    n = len(d)
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    gsum = 0.0
    for k in range(1, h):
        ck = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        gsum += 2 * ck
    var = (gamma0 + gsum) / n
    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n}
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return {"dm_stat": float(dm_hln), "p_value": float(p), "n": int(n),
            "mean_loss_diff": float(dbar),
            "better": "A" if dbar < 0 else "B",
            "significant_at_5pct": bool(p < 0.05)}


# --------------------------------------------------------------------------
def results_to_tables(results: dict, out_dir: str = C.OUT_TABLES) -> dict:
    """Flatten the nested results dict into the manuscript tables 5-8."""
    rows = []
    for (target, horizon, model), r in results.items():
        base = r.get("observed_targets_only", r["all_rows"])
        rows.append({
            "target": target, "horizon_h": horizon, "model": model,
            "R2": round(base["r2"], 4),
            "RMSE": round(base["rmse"], 4),
            "MAE": round(base["mae"], 4),
            "bias": round(base["bias"], 4),
            "skill_vs_persistence": round(base.get("skill_vs_persistence", np.nan), 4),
            "n": base["n"],
        })
    t = pd.DataFrame(rows).sort_values(["target", "horizon_h", "model"])
    for target, g in t.groupby("target"):
        g.to_csv(os.path.join(out_dir, f"table_results_{target}.csv"), index=False)
    t.to_csv(os.path.join(out_dir, "table08_summary_all.csv"), index=False)
    with open(os.path.join(out_dir, "results_full.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()},
                  f, indent=2, default=float)
    return {"summary_table": t}
