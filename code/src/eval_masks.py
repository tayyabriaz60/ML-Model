"""Target- and model-aware evaluation row masks (per-station vs headline)."""
from __future__ import annotations

import pandas as pd

from . import config as C

PRESSURE_COL = C.TARGETS["pressure"]


def pressure_block_columns(cols: list[str]) -> list[str]:
    """Feature columns derived from sea-level pressure (may be all-NaN at some sites)."""
    out = []
    for c in cols:
        if PRESSURE_COL in c or c == "pressure_tendency_3h":
            out.append(c)
    return out


def required_feature_columns(cols: list[str], target: str, model: str) -> list[str]:
    """Columns that must be non-null for a model to score a test row."""
    if model in ("XGBoost", "LightGBM") and target in ("temperature", "visibility"):
        skip = set(pressure_block_columns(cols))
        return [c for c in cols if c not in skip]
    return list(cols)


def eval_row_mask(
    frame: pd.DataFrame,
    cols: list[str],
    target: str,
    model: str,
    ycol: str,
    obs_col: str | None,
) -> pd.Series:
    """Boolean Series aligned to ``frame.index``."""
    m = frame[ycol].notna()
    if obs_col and obs_col in frame.columns:
        m &= frame[obs_col].fillna(False).astype(bool)
    need = required_feature_columns(cols, target, model)
    if need:
        m &= frame[need].notna().all(axis=1)
    return m
