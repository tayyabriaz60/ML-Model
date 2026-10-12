"""Persist and apply tuned tree counts (early-stopping best iteration)."""
from __future__ import annotations

from . import config as C


def best_n_estimators(model, family: str) -> int | None:
    """Number of boosting rounds to use for a fixed-round production refit."""
    if family == "xgb":
        bi = getattr(model, "best_iteration", None)
        if bi is None:
            try:
                bi = model.get_booster().best_iteration
            except Exception:
                bi = None
    else:
        bi = getattr(model, "best_iteration_", None)
    if bi is None:
        return None
    return int(bi) + 1


def annotate_fixed_rounds(row: dict, model, family: str) -> dict:
    """Merge tuning winner row with captured ``n_estimators`` (no early stopping)."""
    n = best_n_estimators(model, family)
    if n is None:
        return row
    out = dict(row)
    out["n_estimators"] = n
    out["use_early_stopping"] = False
    out["fixed_boost_rounds"] = True
    return out


def apply_fixed_rounds(tp: C.TreeParams) -> C.TreeParams:
    """When JSON stores a fixed round count, disable early stopping on refit."""
    if tp.early_stopping_rounds <= 0:
        return tp
    return C.TreeParams(
        max_depth=tp.max_depth,
        learning_rate=tp.learning_rate,
        n_estimators=tp.n_estimators,
        early_stopping_rounds=0,
        subsample=tp.subsample,
        colsample_bytree=tp.colsample_bytree,
        min_child_weight=tp.min_child_weight,
        reg_lambda=tp.reg_lambda,
        random_state=tp.random_state,
    )
