"""
Central configuration for the manuscript revision pipeline.

EVERY path, threshold and hyperparameter that the reviewers asked us to justify
lives here so that it can be cited by name in the manuscript and in the
response letter. Do not hard-code these values anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.environ.get("KAPSARC_RAW", os.path.join(ROOT, "data", "raw"))
DATA_INTERIM = os.path.join(ROOT, "data", "interim")
DATA_PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs")
OUT_TABLES = os.path.join(OUT, "tables")
OUT_FIGS = os.path.join(OUT, "figures")
OUT_MODELS = os.path.join(OUT, "models")
OUT_XAI = os.path.join(OUT, "xai")
OUT_AUDIT = os.path.join(OUT, "audit")

for _p in (DATA_INTERIM, DATA_PROC, OUT, OUT_TABLES, OUT_FIGS, OUT_MODELS,
           OUT_XAI, OUT_AUDIT):
    os.makedirs(_p, exist_ok=True)

SEED = 20260317

# Manuscript window (Sections 3.x). Raw KAPSARC file starts 1946; do not
# reindex the full archive or the hourly grid will not fit in memory.
STUDY_START = "2009-01-01"
STUDY_END = "2019-05-24"
# Same floor the submitted preprocessor used (select_stations min_records=30000).
# Applied AFTER the 2009–2019 cut so sparse 1940s-only sites never enter the grid.
MIN_STATION_RECORDS = 30000

# --------------------------------------------------------------------------
# Column names as they appear in the KAPSARC hourly export
# --------------------------------------------------------------------------
COL_STATION = "STATION"
COL_TIME = "DATE"
# Headers in saudi-hourly-weather-data_Historical.csv
COLUMN_MAP: Dict[str, str] = {
    "STATION_NAME": COL_STATION,
    "OBSERVATION_DATE": COL_TIME,
}
# Read only modelling columns from the 5 GB export.
RAW_USECOLS: Tuple[str, ...] = (
    "STATION_NAME",
    "OBSERVATION_DATE",
    "AIR_TEMPERATURE",
    "AIR_TEMPERATURE_DEW_POINT",
    "VISIBILITY_DISTANCE",
    "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
    "WIND_SPEED_RATE",
    "WIND_DIRECTION_ANGLE",
    "SKY_CEILING_HEIGHT",
)
TARGETS: Dict[str, str] = {
    "temperature": "AIR_TEMPERATURE",
    "visibility": "VISIBILITY_DISTANCE",
    "pressure": "ATMOSPHERIC_SEA_LEVEL_PRESSURE",
}
PREDICTORS: List[str] = [
    "AIR_TEMPERATURE_DEW_POINT",
    "WIND_SPEED_RATE",
    "WIND_DIRECTION_ANGLE",
    "SKY_CEILING_HEIGHT",
]
BASE_VARS: List[str] = list(TARGETS.values()) + PREDICTORS

# Sentinel codes documented in Section 3.3 of the submitted manuscript
SENTINELS: Tuple[float, ...] = (9999.9, 99999.0, 999999.0, 999.0, 999.9, -9999.0)

# Physically admissible ranges. Anything outside becomes NaN BEFORE imputation.
# Reviewer 2 (basic reporting, bullet 3) asked for detail on missing handling;
# these bounds are reported in the revised Section 3.3.
PLAUSIBLE_RANGE: Dict[str, Tuple[float, float]] = {
    "AIR_TEMPERATURE": (-15.0, 60.0),            # degC
    "AIR_TEMPERATURE_DEW_POINT": (-40.0, 40.0),  # degC
    "VISIBILITY_DISTANCE": (0.0, 20000.0),       # m
    "ATMOSPHERIC_SEA_LEVEL_PRESSURE": (900.0, 1080.0),  # hPa
    "WIND_SPEED_RATE": (0.0, 75.0),              # m/s
    "WIND_DIRECTION_ANGLE": (0.0, 360.0),        # deg
    "SKY_CEILING_HEIGHT": (0.0, 22000.0),        # m
}

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
LAGS: Tuple[int, ...] = (1, 2, 3, 6, 12, 24)
ROLL_WINDOWS: Tuple[int, ...] = (3, 6, 12, 24)

# Reviewer 1 (experimental design, point 2): visibility varies far faster than
# T or p, so we additionally build a SHORT-LAG-ONLY visibility feature set and
# report the ablation. See ablations.run_visibility_lag_ablation().
VIS_SHORT_LAGS: Tuple[int, ...] = (1, 2, 3, 6)
VIS_SHORT_ROLL: Tuple[int, ...] = (3, 6)

# --------------------------------------------------------------------------
# Forecast setup
# --------------------------------------------------------------------------
# Reviewer 1 (experimental design, point 3) asked why only h=1 and h=6.
# We now evaluate the full set and report 1/3/6 in the main tables.
HORIZONS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
HORIZONS_MAIN: Tuple[int, ...] = (1, 3, 6)

SPLIT_FRACTIONS = (0.70, 0.10, 0.20)  # train / val / test, chronological

# Reviewer 2 (basic reporting, bullet 1): the 30% subsample for the neural
# models is REMOVED. Kept here only so the old behaviour can be reproduced for
# the "effect of subsampling" appendix table.
LEGACY_DL_SUBSAMPLE = 0.30
DL_SUBSAMPLE = 1.0  # production setting for the revision

# --------------------------------------------------------------------------
# Scaling (Reviewer 1, experimental design, point 4)
# --------------------------------------------------------------------------
# Variables whose marginal distribution is strongly skewed. We compare three
# strategies and report the sensitivity table.
SKEWED_VARS = ["VISIBILITY_DISTANCE", "WIND_SPEED_RATE", "SKY_CEILING_HEIGHT"]
SCALING_STRATEGIES = ("standard", "robust", "log1p_standard")
SCALING_PRODUCTION = "log1p_standard"  # decided in Step 5; update after run

# --------------------------------------------------------------------------
# Extreme-event definitions (Reviewer 1, experimental design, point 7)
# --------------------------------------------------------------------------
# Fog: ABSOLUTE WMO/Han et al. (2024) criterion, not a percentile.
FOG_VISIBILITY_M = 1000.0
FOG_MIN_DURATION_H = 1
# Optional stricter variant used for a sensitivity check
FOG_DENSE_VISIBILITY_M = 200.0

# Heat: relative percentile criterion computed per station and per calendar day
# window, following the WMO (2023) guidance and Perkins & Alexander style
# definitions. A "heatwave" additionally requires persistence.
HEAT_PERCENTILE = 0.95
HEAT_WINDOW_DAYS = 15           # +/- days around the calendar day for the climatology
HEAT_MIN_DURATION_DAYS = 3      # consecutive days above threshold -> heatwave
# If HEAT_MIN_DURATION_DAYS is not met we must call the sample
# "extreme high temperature", NOT "heatwave". enforced in extremes.py
HEAT_LABEL_IF_NO_PERSISTENCE = "extreme high temperature"

N_CASE_STUDIES = 200  # retained from the original submission

# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
SHAP_BACKGROUND_N = 2000
SHAP_SAMPLE_N = 20000       # rows used for global mean|SHAP|
SHAP_TOPK_REPORT = 20       # Reviewer 1 asked for the selection rule -> here
LIME_N_FEATURES = 15
LIME_N_SAMPLES = 5000
LIME_N_INSTANCES = 200      # instances aggregated per target x horizon
ATTENTION_N_SAMPLES = 5000

# Reviewer 2 (basic reporting, bullet 5): the ">0.90 agreement" claim must be
# defined. These are the metrics that now back it.
AGREEMENT_TOPK = (5, 10, 15)
AGREEMENT_METRICS = ("jaccard", "spearman", "rbo")
RBO_P = 0.9

# --------------------------------------------------------------------------
# Statistics (Reviewer 2, basic reporting, bullet 7)
# --------------------------------------------------------------------------
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "1000"))
BOOTSTRAP_BLOCK_H = 24 * 7   # one-week moving blocks, respects autocorrelation
CI_LEVEL = 0.95
DM_LOSS = "squared"


@dataclass
class TreeParams:
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 2000       # was 200; early stopping now decides
    early_stopping_rounds: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    random_state: int = SEED


@dataclass
class DLParams:
    """Reviewer 2 (basic reporting, bullet 2): the deep models were
    under-tuned relative to the trees. These are the DEFAULTS; the search
    space in tuning.DL_SEARCH_SPACE is what is actually explored, with an
    identical trial budget to the tree search."""
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    seq_len: int = 24
    weight_decay: float = 1e-5
    random_state: int = SEED


@dataclass
class TransformerParams(DLParams):
    d_model: int = 128
    n_heads: int = 8
    ff_dim: int = 256
    n_blocks: int = 2


@dataclass
class RunConfig:
    targets: List[str] = field(default_factory=lambda: list(TARGETS.keys()))
    horizons: List[int] = field(default_factory=lambda: list(HORIZONS_MAIN))
    tree: TreeParams = field(default_factory=TreeParams)
    dl: DLParams = field(default_factory=DLParams)
    tr: TransformerParams = field(default_factory=TransformerParams)
    dl_subsample: float = DL_SUBSAMPLE
    scaling: str = SCALING_PRODUCTION
    seed: int = SEED

    def to_dict(self):
        return asdict(self)


CONFIG = RunConfig()
