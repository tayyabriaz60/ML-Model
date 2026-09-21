"""
Step 4: chronological splits, scaling strategies, and the model zoo.

Two reviewer-driven changes versus the submitted version:
  * Reviewer 2, bullet 1 - the 30 % subsample for LSTM/GRU/Transformer is gone.
    All model families now see the same full training set.
  * Reviewer 2, bullet 2 - the deep models get a hyperparameter search with a
    trial budget at least equal to the tree search, and the search space is
    reported in the manuscript (new Table S2).
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, fracs=C.SPLIT_FRACTIONS):
    """Global chronological split on time, identical cut dates for all stations.

    A per-station split would put different calendar periods in the test set
    for different stations and break the "evaluated on future time periods"
    claim. Cut dates are written out and quoted in the revised Section 4.4.
    """
    t = pd.to_datetime(df[C.COL_TIME], utc=True)
    order = np.argsort(t.to_numpy())
    sorted_t = t.iloc[order]
    i1 = int(fracs[0] * len(sorted_t))
    i2 = int((fracs[0] + fracs[1]) * len(sorted_t))
    cut1, cut2 = sorted_t.iloc[i1], sorted_t.iloc[i2]
    train = df.loc[t < cut1]
    val = df.loc[(t >= cut1) & (t < cut2)]
    test = df.loc[t >= cut2]
    meta = {"cut_train_val": str(cut1), "cut_val_test": str(cut2),
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    with open(os.path.join(C.OUT_TABLES, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return train, val, test, meta


def purged_blocked_cv(df: pd.DataFrame, n_splits: int = 5, embargo_h: int = 24):
    """Blocked CV with an embargo, for the stability checks in Section 6.

    The embargo removes `embargo_h` hours either side of each boundary so that
    a 24 h lag feature in the validation block cannot see a training target.
    """
    clock = pd.to_datetime(df[C.COL_TIME], utc=True)
    t = np.sort(clock.unique())
    bounds = np.array_split(t, n_splits + 1)
    emb = pd.Timedelta(hours=embargo_h)
    for k in range(1, n_splits + 1):
        tr_end = pd.Timestamp(bounds[k - 1][-1])
        va = bounds[k]
        tr_mask = clock <= (tr_end - emb)
        va_mask = (clock >= pd.Timestamp(va[0])) & (clock <= pd.Timestamp(va[-1]))
        yield np.flatnonzero(tr_mask), np.flatnonzero(va_mask)


# --------------------------------------------------------------------------
# Scaling  (Reviewer 1, exp. design 4)
# --------------------------------------------------------------------------
class Scaler:
    """Wraps the three strategies compared in the sensitivity table, and keeps
    the parameters needed to BACK-TRANSFORM predictions into native units
    (Reviewer 1, basic reporting 8)."""

    def __init__(self, strategy: str = C.SCALING_PRODUCTION,
                 skewed=None):
        self.strategy = strategy
        self.skewed = set(skewed or C.SKEWED_VARS)
        self.params: dict = {}

    @staticmethod
    def _needs_log(col: str, skewed: set) -> bool:
        return any(col == s or col.startswith(s + "_") for s in skewed)

    def fit(self, X: pd.DataFrame):
        for c in X.columns:
            v = X[c].astype(float)
            log = self.strategy == "log1p_standard" and self._needs_log(c, self.skewed)
            if log:
                v = np.log1p(np.clip(v, 0, None))
            if self.strategy == "robust":
                centre = float(v.median())
                scale = float(v.quantile(0.75) - v.quantile(0.25)) or 1.0
            else:
                centre = float(v.mean())
                scale = float(v.std()) or 1.0
            self.params[c] = {"log": log, "centre": centre, "scale": scale}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            p = self.params[c]
            v = X[c].astype(float)
            if p["log"]:
                v = np.log1p(np.clip(v, 0, None))
            out[c] = (v - p["centre"]) / p["scale"]
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X): return self.fit(X).transform(X)

    def inverse_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        """Back-transform a scaled target to native units (m, degC, hPa)."""
        p = self.params[col]
        v = y_scaled * p["scale"] + p["centre"]
        return np.expm1(v) if p["log"] else v

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"strategy": self.strategy, "params": self.params}, f)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def persistence_predict(df: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    """y_hat(t+h) = y(t), taken from the within-station column built in
    features.add_targets. No shifting happens here, so no chance of a
    cross-station slip."""
    return df[f"persist_{target}_h{horizon}"].values


# --------------------------------------------------------------------------
# Tree models
# --------------------------------------------------------------------------
def fit_xgboost(Xtr, ytr, Xva, yva, params: C.TreeParams | None = None):
    import xgboost as xgb
    p = params or C.TreeParams()
    m = xgb.XGBRegressor(
        max_depth=p.max_depth, learning_rate=p.learning_rate,
        n_estimators=p.n_estimators, subsample=p.subsample,
        colsample_bytree=p.colsample_bytree, min_child_weight=p.min_child_weight,
        reg_lambda=p.reg_lambda, random_state=p.random_state,
        early_stopping_rounds=p.early_stopping_rounds,
        tree_method="hist", n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lightgbm(Xtr, ytr, Xva, yva, params: C.TreeParams | None = None):
    import lightgbm as lgb
    p = params or C.TreeParams()
    m = lgb.LGBMRegressor(
        max_depth=p.max_depth, learning_rate=p.learning_rate,
        n_estimators=p.n_estimators, subsample=p.subsample,
        colsample_bytree=p.colsample_bytree,
        min_child_samples=max(5, int(p.min_child_weight)),
        reg_lambda=p.reg_lambda, random_state=p.random_state, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(p.early_stopping_rounds, verbose=False)])
    return m


def fit_linear(Xtr, ytr, *_):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=C.SEED).fit(Xtr, ytr)


# --------------------------------------------------------------------------
# Sequence models
# --------------------------------------------------------------------------
def make_sequences(df: pd.DataFrame, feat_cols, y_col, seq_len: int):
    """Materialise (N, seq_len, F). Prefer `window_dataset` — this can be 20–40 GB."""
    ds = window_dataset(df, feat_cols, y_col, seq_len)
    if len(ds) == 0:
        return np.empty((0, seq_len, len(feat_cols)), np.float32), np.array([]), np.array([])
    Xs, ys, idx = [], [], []
    for i in range(len(ds)):
        x, y = ds[i]
        Xs.append(np.asarray(x, np.float32))
        ys.append(float(y))
        idx.append(ds.row_index[i])
    return np.stack(Xs), np.asarray(ys, np.float32), np.asarray(idx)


class WindowDataset:
    """Windows over per-station (T, F) arrays — not a (N, seq, F) copy."""

    def __init__(self, pieces, seq_len):
        self.pieces = pieces
        self.seq_len = seq_len
        self._index = []
        self.row_index = []
        for pi, (X, y, idx) in enumerate(pieces):
            finite_y = np.isfinite(y)
            for t in range(seq_len, len(X)):
                if not finite_y[t]:
                    continue
                w = X[t - seq_len:t]
                if not np.isfinite(w).all():
                    continue
                self._index.append((pi, t))
                self.row_index.append(idx[t])

    def __len__(self):
        return len(self._index)

    def __getitem__(self, j):
        pi, t = self._index[j]
        X, y, _ = self.pieces[pi]
        return X[t - self.seq_len:t], np.float32(y[t])


def window_dataset(df: pd.DataFrame, feat_cols, y_col, seq_len: int) -> WindowDataset:
    pieces = []
    for _, g in df.groupby(C.COL_STATION, sort=True):
        g = g.sort_values(C.COL_TIME)
        X = g[feat_cols].to_numpy(np.float32, copy=False)
        y = g[y_col].to_numpy(np.float32, copy=False)
        pieces.append((X, y, g.index.to_numpy()))
    return WindowDataset(pieces, seq_len)


def dl_params_from_search(kind: str, trial: dict):
    """Map DL_SEARCH_SPACE onto LSTM/GRU vs Transformer (d_model / n_blocks)."""
    hidden = int(trial.get("hidden", 128))
    layers = int(trial.get("layers", 2))
    shared = dict(
        dropout=float(trial.get("dropout", 0.2)),
        lr=float(trial.get("lr", 1e-3)),
        seq_len=int(trial.get("seq_len", 24)),
        batch_size=int(trial.get("batch_size", 512)),
        hidden=hidden,
        layers=layers,
    )
    if kind == "transformer":
        n_heads = 8
        while hidden % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        return C.TransformerParams(
            **shared,
            d_model=hidden,
            n_heads=n_heads,
            n_blocks=layers,
            ff_dim=max(128, hidden * 2),
        )
    return C.DLParams(**shared)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _RNN:
    def __init__(self, kind="lstm", n_features=1, p: C.DLParams | None = None):
        torch, nn = _torch()
        p = p or C.DLParams()
        self.p = p
        Cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.net = nn.Sequential()
        self.rnn = Cell(n_features, p.hidden, num_layers=p.layers,
                        batch_first=True, dropout=p.dropout if p.layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(p.dropout), nn.Linear(p.hidden, 1))
        self.module = nn.ModuleList([self.rnn, self.head])

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_sequence_model(kind: str, n_features: int, p=None):
    """kind in {'lstm','gru','transformer'}. Returns an nn.Module."""
    torch, nn = _torch()

    if kind in ("lstm", "gru"):
        p = p or C.DLParams()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                Cell = nn.LSTM if kind == "lstm" else nn.GRU
                self.rnn = Cell(n_features, p.hidden, num_layers=p.layers,
                                batch_first=True,
                                dropout=p.dropout if p.layers > 1 else 0.0)
                self.head = nn.Sequential(nn.Dropout(p.dropout),
                                          nn.Linear(p.hidden, 1))

            def forward(self, x):
                o, _ = self.rnn(x)
                return self.head(o[:, -1, :]).squeeze(-1)

        return Net()

    p = p or C.TransformerParams()

    class TNet(nn.Module):
        """Transformer encoder with attention weights exposed for Section 5.3.

        `last_attention` holds the averaged head weights of the final block so
        that explain.attention_importance() reports genuine attention rather
        than a perturbation proxy. The submitted manuscript labelled
        perturbation importance as 'attention weights' (Figs 10-12) - that is
        corrected here, and both quantities are now reported separately.
        """

        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_features, p.d_model)
            self.pos = nn.Parameter(torch.zeros(1, p.seq_len, p.d_model))
            self.blocks = nn.ModuleList([
                nn.MultiheadAttention(p.d_model, p.n_heads, dropout=p.dropout,
                                      batch_first=True)
                for _ in range(p.n_blocks)])
            self.ffs = nn.ModuleList([
                nn.Sequential(nn.Linear(p.d_model, p.ff_dim), nn.GELU(),
                              nn.Dropout(p.dropout), nn.Linear(p.ff_dim, p.d_model))
                for _ in range(p.n_blocks)])
            self.norms1 = nn.ModuleList([nn.LayerNorm(p.d_model) for _ in range(p.n_blocks)])
            self.norms2 = nn.ModuleList([nn.LayerNorm(p.d_model) for _ in range(p.n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(p.d_model), nn.Dropout(p.dropout),
                                      nn.Linear(p.d_model, 1))
            self.last_attention = None

        def forward(self, x):
            h = self.inp(x) + self.pos[:, :x.shape[1], :]
            for att, ff, n1, n2 in zip(self.blocks, self.ffs, self.norms1, self.norms2):
                a, w = att(h, h, h, need_weights=True, average_attn_weights=True)
                self.last_attention = w.detach()
                h = n1(h + a)
                h = n2(h + ff(h))
            return self.head(h[:, -1, :]).squeeze(-1)

    return TNet()


def _as_loader(data, batch_size, shuffle, drop_last):
    torch, _ = _torch()

    def _collate(batch):
        xs, ys = zip(*batch)
        return (torch.as_tensor(np.stack(xs), dtype=torch.float32),
                torch.as_tensor(np.asarray(ys), dtype=torch.float32))

    if isinstance(data, WindowDataset):
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            collate_fn=_collate, num_workers=0)
    X, y = data
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(np.asarray(X), dtype=torch.float32),
        torch.as_tensor(np.asarray(y), dtype=torch.float32))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_sequence_model(model, train_data, val_data, p=None, device=None):
    """train_data / val_data: WindowDataset or (X, y) arrays."""
    torch, nn = _torch()
    p = p or C.DLParams()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.MSELoss()
    trl = _as_loader(train_data, p.batch_size, True, True)
    val = _as_loader(val_data, 4096, False, False)

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(p.max_epochs):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(xb.to(device)), yb.to(device)).item()
                                for xb, yb in val]))
        hist.append(vl); sched.step(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= p.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best, "epochs_run": len(hist), "history": hist}


def predict_sequence(model, data, device=None, batch=4096):
    torch, _ = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if isinstance(data, WindowDataset):
        loader = _as_loader(data, batch, False, False)
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                out.append(model(xb.to(device)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])
    X = np.asarray(data)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------
# Hyperparameter search  (Reviewer 2, bullet 2)
# --------------------------------------------------------------------------
DL_SEARCH_SPACE = {
    "hidden": [64, 128, 256],
    "layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
    "seq_len": [12, 24, 48],
    "batch_size": [256, 512],
}
TREE_SEARCH_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}
import os as _os

# Identical budget for BOTH families. This is the fairness claim that answers
# Reviewer 2's bullet 2, and it is the reason this is one constant rather than
# two: the reviewer's objection was that the sequence models were tuned less
# than the trees, not that either was tuned lightly in absolute terms.
#
# The value may be reduced under a compressed schedule (see the five-day plan,
# Section 0) but it must be reduced for EVERY family together. Whatever value
# is used is written to outputs/run_meta.json and reaches the manuscript and
# the response letter through the [[N_TRIALS]] placeholder, so the text always
# states the budget that was actually spent. Never edit that number by hand.
#
#   export N_TRIALS=24   # before launching the search
N_TRIALS = int(_os.environ.get("N_TRIALS", 40))
if N_TRIALS < 12:
    raise ValueError(
        f"N_TRIALS={N_TRIALS} is too small to support the equal-budget claim "
        "made in Section 4.5. Raise it or remove that claim from the manuscript."
    )


def random_search(space: dict, n_trials: int = N_TRIALS, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    seen = set()
    for _ in range(n_trials * 5):
        cand = tuple(rng.choice(len(v)) for v in space.values())
        if cand in seen:
            continue
        seen.add(cand)
        yield {k: space[k][i] for k, i in zip(space.keys(), cand)}
        if len(seen) >= n_trials:
            return
