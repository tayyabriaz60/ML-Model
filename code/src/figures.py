"""
Step 10: consolidated figures.

Both reviewers asked for fewer, denser figures (Reviewer 1 basic reporting 7;
Reviewer 2 bullet 9). The submitted paper has 24 figures, several of which show
the same information as a table. The revised set is 11:

  F1  SHAP global importance, 2x3 panel  (targets x {1h, 6h})   <- old Figs 1-6
  F2  SHAP direction / dependence, 3 panels                      <- NEW
  F3  LIME aggregated importance, 3 panels                       <- old Figs 7-9
  F4  Transformer attention over lag positions, 3 panels         <- old Figs 10-12 (corrected)
  F5  Perturbation importance, 3 panels                          <- old Figs 10-12 (renamed)
  F6  Autocorrelation vs lag, 3 panels                           <- NEW, answers R1 validity 2
  F7  Cross-method agreement heatmap                             <- NEW, answers R2 bullet 5
  F8  Regime-conditioned SHAP, 3 panels                          <- old Figs 16-18
  F9  Model performance + bootstrap CIs, 3 panels                <- old Figs 19-21
  F10 Per-station skill distribution (boxplots)                  <- NEW, answers R2 bullet 7
  F11 Fog and extreme-heat case studies, 2x2 panel               <- old Figs 22-23

WITHDRAWN in this revision, with the manuscript sections they belonged to:

  old Figs 13-15  Temporal saliency curves   -> Section 5.4 removed. The lag
                  information they carried is already in the SHAP rankings
                  (F1), and the memory-window sanity check is now served by the
                  autocorrelation figure F6, which compares attention against
                  the statistical memory of the series rather than against a
                  second attribution measure.
  old Fig  24     XAI dashboard mockup       -> Section 7.6 removed. Never
                  deployed, instrumented or user-tested, so the figure has no
                  evidential content.

Do not re-add these. If a reader asks for them, the answer is Section 9.

Old Figs 19-21 are retained only because they carry bootstrap CIs that the
tables cannot show compactly; the raw numbers move to Tables 5-7.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "figure.constrained_layout.use": True,
})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.OUT_FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_shap_panel(imp_by_key: dict, topk=15, name="F1_shap_global"):
    """imp_by_key: (target, horizon) -> DataFrame(feature, mean_abs_shap)."""
    keys = sorted(imp_by_key)
    ncol = len(set(k[1] for k in keys))
    nrow = len(set(k[0] for k in keys))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow))
    axes = np.atleast_2d(axes)
    for ax, k in zip(axes.ravel(), keys):
        t = imp_by_key[k].head(topk).iloc[::-1]
        ax.barh(t.feature, t.mean_abs_shap, color=CB[0])
        ax.set_title(f"{k[0]} - {k[1]} h")
        ax.set_xlabel("mean |SHAP|")
        ax.tick_params(axis="y", labelsize=5)
    _save(fig, name)


def fig_autocorrelation(profile: pd.DataFrame, name="F7_autocorrelation"):
    """The figure that answers Reviewer 1's validity point 2."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for ax, (t, g) in zip(axes, profile.groupby("target")):
        ax.plot(g.lag_h, g.autocorr, color=CB[0], marker="o", ms=2)
        ax.axhline(0, color="k", lw=0.5)
        for h in (6, 12, 24):
            ax.axvline(h, color=CB[1], ls=":", lw=0.8)
        ax.set_title(t); ax.set_xlabel("lag (h)")
    axes[0].set_ylabel("autocorrelation")
    _save(fig, name)


def fig_agreement_heatmap(agreement_by_key: dict, name="F8_agreement"):
    keys = sorted(agreement_by_key)
    pairs = sorted(agreement_by_key[keys[0]]["pairs"])
    M = np.array([[agreement_by_key[k]["pairs"][p].get("jaccard@10", np.nan)
                   for p in pairs] for k in keys])
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(pairs), 0.5 + 0.35 * len(keys)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("__vs__", "\nvs\n") for p in pairs], fontsize=6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{a} {b}h" for a, b in keys], fontsize=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="Jaccard@10")
    ax.set_title("Cross-method agreement (top-10 feature overlap)")
    _save(fig, name)


def fig_performance_with_ci(ci_table: pd.DataFrame, name="F10_performance"):
    """ci_table: target, model, rmse, ci_low, ci_high."""
    targets = sorted(ci_table.target.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 2.8))
    for ax, t in zip(np.atleast_1d(axes), targets):
        g = ci_table[ci_table.target == t].sort_values("rmse")
        y = np.arange(len(g))
        ax.errorbar(g.rmse, y,
                    xerr=[g.rmse - g.ci_low, g.ci_high - g.rmse],
                    fmt="o", color=CB[0], capsize=2, ms=3)
        ax.set_yticks(y); ax.set_yticklabels(g.model, fontsize=6)
        ax.set_title(t); ax.set_xlabel("RMSE (native units)")
    _save(fig, name)


def fig_per_station_box(per_station: dict, name="F11_per_station"):
    """per_station: target -> DataFrame with a per-station r2 column."""
    fig, axes = plt.subplots(1, len(per_station), figsize=(3.2 * len(per_station), 2.8))
    for ax, (t, df) in zip(np.atleast_1d(axes), per_station.items()):
        data = [g.r2.values for _, g in df.groupby("model")]
        ax.boxplot(data, labels=sorted(df.model.unique()), showfliers=True)
        ax.set_title(f"{t}: R2 across 29 stations")
        ax.tick_params(axis="x", rotation=60, labelsize=6)
    _save(fig, name)


def fig_regime_shap(table: pd.DataFrame, regimes, name="F9_regime_shap"):
    t = table.head(12).iloc[::-1]
    y = np.arange(len(t)); w = 0.8 / len(regimes)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, r in enumerate(regimes):
        ax.barh(y + i * w, t[r], height=w, label=r, color=CB[i % len(CB)])
    ax.set_yticks(y + 0.4 - w / 2); ax.set_yticklabels(t.feature, fontsize=6)
    ax.set_xlabel("mean |SHAP|"); ax.legend()
    ax.set_title("Attribution by weather regime (single source: Table 9)")
    _save(fig, name)
