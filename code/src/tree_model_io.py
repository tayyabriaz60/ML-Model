"""Load tree checkpoints for eval; XGBoost prefers native JSON (joblib pickle breaks across XGB versions)."""
from __future__ import annotations

import os

import joblib
import numpy as np

from src import config as C


def _json_path(tag: str, target: str, h: int) -> str:
    return os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.json")


def _joblib_path(tag: str, target: str, h: int) -> str:
    return os.path.join(C.OUT_MODELS, f"{tag}_{target}_h{h}.joblib")


def joblib_plausible(m, Xva, yva) -> bool:
    """Pod joblib + local Python/XGBoost often loads but predicts ~0 (RMSE ~ |y|)."""
    pred_va = np.asarray(m.predict(Xva), float)
    rmse_va = float(np.sqrt(np.mean((yva - pred_va) ** 2)))
    y_scale = float(np.nanstd(yva)) or 1.0
    if rmse_va > max(10.0, 2.5 * y_scale):
        return False
    if abs(float(np.nanmean(yva))) > 5.0 and abs(float(np.nanmean(pred_va))) < 0.05 * abs(
            float(np.nanmean(yva))):
        return False
    return True


def load_xgb(target: str, h: int, *, require_json: bool | None = None):
    """Load XGBRegressor from native JSON when present (client milestone requirement)."""
    import xgboost as xgb

    if require_json is None:
        require_json = os.environ.get("REGEN_XGB_REQUIRE_JSON", "1") == "1"
    jp, mp = _json_path("xgb", target, h), _joblib_path("xgb", target, h)
    if os.path.exists(jp):
        m = xgb.XGBRegressor()
        m.load_model(jp)
        return m, jp
    if require_json:
        raise FileNotFoundError(f"missing XGB JSON {jp} (export after refit: export_tree_models_json.py)")
    if os.path.exists(mp):
        m = joblib.load(mp)
        return m, mp
    raise FileNotFoundError(f"missing XGB model for {target} h={h}")


def load_lgb(target: str, h: int):
    mp = _joblib_path("lgb", target, h)
    if not os.path.exists(mp):
        raise FileNotFoundError(mp)
    return joblib.load(mp), mp


def load_tree(tag: str, target: str, h: int):
    if tag == "xgb":
        return load_xgb(target, h)
    if tag == "lgb":
        return load_lgb(target, h)
    raise ValueError(tag)
