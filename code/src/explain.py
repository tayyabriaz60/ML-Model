"""
Step 7: explainability, with the two things the reviewers found missing.

(a) DIRECTION and physical meaning, not just magnitude
    Reviewer 1 (validity, point 1): the SHAP discussion stops at "feature X had
    high importance". `signed_shap_summary()` returns, for every top feature,
    the correlation between the feature value and its SHAP value. That gives a
    sign ("high dew-point depression pushes visibility up") which the revised
    Section 5.1 interprets physically.

(b) A DEFINED agreement metric
    Reviewer 2 (bullet 5): the ">0.90 agreement between SHAP, LIME and
    attention" claim has no method behind it anywhere in the submitted paper.
    `cross_method_agreement()` defines it three ways - Jaccard overlap of the
    top-k sets, Spearman rank correlation of the shared-feature rankings, and
    rank-biased overlap - and reports all of them per target x horizon. If the
    numbers come out below 0.90, the claim is corrected in the manuscript
    rather than defended.

Also corrected here: the submitted Figs 10-12 are labelled "Transformer
Attention Weights" but their x-axis reads "Perturbation Importance". Those are
different quantities. Both are now computed and reported under their correct
names.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def shap_values_tree(model, X: pd.DataFrame, n=C.SHAP_SAMPLE_N, seed=C.SEED):
    import shap
    Xs = X.sample(min(n, len(X)), random_state=seed)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xs)
    return Xs, np.asarray(sv)


def global_importance(sv: np.ndarray, cols) -> pd.DataFrame:
    imp = np.abs(sv).mean(axis=0)
    t = pd.DataFrame({"feature": list(cols), "mean_abs_shap": imp})
    t["share_pct"] = 100 * t.mean_abs_shap / t.mean_abs_shap.sum()
    t = t.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t["cumulative_share_pct"] = t.share_pct.cumsum()
    return t


def signed_shap_summary(sv: np.ndarray, X: pd.DataFrame, topk=C.SHAP_TOPK_REPORT):
    """Direction of effect for each top feature (Reviewer 1, validity 1)."""
    imp = global_importance(sv, X.columns).head(topk)
    rows = []
    for _, r in imp.iterrows():
        j = list(X.columns).index(r.feature)
        x, s = X.iloc[:, j].values.astype(float), sv[:, j]
        ok = np.isfinite(x) & np.isfinite(s)
        rho = float(np.corrcoef(x[ok], s[ok])[0, 1]) if ok.sum() > 10 else np.nan
        rows.append({
            "feature": r.feature,
            "mean_abs_shap": float(r.mean_abs_shap),
            "share_pct": float(r.share_pct),
            "value_shap_corr": rho,
            "direction": ("higher value -> higher prediction" if rho > 0.1 else
                          "higher value -> lower prediction" if rho < -0.1 else
                          "non-monotonic / interaction-dominated"),
            "shap_range": [float(np.nanpercentile(s, 1)), float(np.nanpercentile(s, 99))],
        })
    return pd.DataFrame(rows)


def topk_coverage(imp: pd.DataFrame, ks=(5, 10, 20, 30, 50)) -> dict:
    """Reviewer 1 (exp. design 1): the criterion for showing only some of the
    122 features. We now state it: the top-20 by mean|SHAP|, which capture the
    share reported here."""
    return {f"top_{k}_share_pct": float(imp.head(k).share_pct.sum()) for k in ks}


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def lime_importance(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                    n_instances=C.LIME_N_INSTANCES,
                    n_features=C.LIME_N_FEATURES,
                    n_samples=C.LIME_N_SAMPLES, seed=C.SEED) -> pd.DataFrame:
    """Aggregate LIME weights over many instances.

    The submitted paper showed a SINGLE instance per target (Figs 7-9) and then
    compared it with a GLOBAL SHAP ranking. That is not a like-for-like
    comparison and is part of why Fig. 11 and Fig. 16 disagreed. We now
    aggregate |weight| over `n_instances` stratified instances so that the LIME
    ranking is global in the same sense as the SHAP ranking.
    """
    from lime.lime_tabular import LimeTabularExplainer
    rng = np.random.default_rng(seed)
    expl = LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns),
        mode="regression", discretize_continuous=True, random_state=seed)
    idx = rng.choice(len(X_explain), size=min(n_instances, len(X_explain)),
                     replace=False)
    acc = {}
    for i in idx:
        e = expl.explain_instance(X_explain.values[i], model.predict,
                                  num_features=n_features,
                                  num_samples=n_samples)
        for fid, w in e.as_map()[0]:
            acc.setdefault(X_train.columns[fid], []).append(abs(w))
    rows = [{"feature": k, "mean_abs_lime_weight": float(np.mean(v)),
             "n_instances_selected": len(v)} for k, v in acc.items()]
    t = pd.DataFrame(rows).sort_values("mean_abs_lime_weight", ascending=False)
    t["share_pct"] = 100 * t.mean_abs_lime_weight / t.mean_abs_lime_weight.sum()
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------
# Attention vs perturbation importance
# --------------------------------------------------------------------------
def attention_weights(model, X_seq: np.ndarray, n=C.ATTENTION_N_SAMPLES,
                      device=None) -> pd.DataFrame:
    """TRUE attention: mean weight assigned to each lag position by the final
    encoder block. This is a distribution over TIME STEPS, not over features -
    a distinction the submitted Figs 10-12 blurred."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    X = X_seq[:min(n, len(X_seq))]
    mats = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            model(torch.tensor(X[i:i + 1024]).to(device))
            w = model.last_attention  # (B, L, L)
            mats.append(w[:, -1, :].cpu().numpy())  # query = last step
    A = np.concatenate(mats)
    L = A.shape[1]
    return pd.DataFrame({
        "lag_h": list(range(L - 1, -1, -1)),
        "mean_attention": A.mean(axis=0),
        "sd_attention": A.std(axis=0),
    })


def perturbation_importance(predict_fn, X_seq: np.ndarray, y: np.ndarray,
                            feature_names, n=5000, seed=C.SEED) -> pd.DataFrame:
    """Permutation importance over FEATURES for a sequence model. This is what
    the submitted Figs 10-12 actually plotted; it is retained but renamed."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_seq), size=min(n, len(X_seq)), replace=False)
    X, yy = X_seq[idx], y[idx]
    base = np.mean((predict_fn(X) - yy) ** 2)
    rows = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
        rows.append({"feature": name,
                     "perturbation_importance": float(np.mean((predict_fn(Xp) - yy) ** 2) - base)})
    t = pd.DataFrame(rows).sort_values("perturbation_importance", ascending=False)
    t["share_pct"] = 100 * t.perturbation_importance.clip(lower=0) / \
        max(t.perturbation_importance.clip(lower=0).sum(), 1e-12)
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------
# THE AGREEMENT METRIC  (Reviewer 2, bullet 5; Reviewer 1, validity 4)
# --------------------------------------------------------------------------
def _jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else np.nan


def _rbo(l1, l2, p=C.RBO_P) -> float:
    """Rank-biased overlap: top-weighted, handles non-identical vocabularies."""
    s, S1, S2, acc = 0.0, set(), set(), 0.0
    for d in range(1, max(len(l1), len(l2)) + 1):
        if d <= len(l1): S1.add(l1[d - 1])
        if d <= len(l2): S2.add(l2[d - 1])
        acc += (len(S1 & S2) / d) * (p ** (d - 1))
    return float((1 - p) * acc)


def cross_method_agreement(rankings: dict, ks=C.AGREEMENT_TOPK) -> dict:
    """`rankings` maps method name -> list of features, best first.

    Returns Jaccard@k, Spearman on the shared support, and RBO for every
    method pair. The revised manuscript reports the MEDIAN Jaccard@10 as the
    headline agreement figure and states the metric explicitly, replacing the
    undefined '>0.90 agreement' claim.
    """
    from scipy.stats import spearmanr
    methods = list(rankings)
    out = {"pairs": {}, "ks": list(ks)}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            ra, rb = rankings[a], rankings[b]
            rec = {f"jaccard@{k}": round(_jaccard(ra[:k], rb[:k]), 4) for k in ks}
            rec["rbo"] = round(_rbo(ra, rb), 4)
            shared = [f for f in ra if f in rb]
            if len(shared) >= 3:
                pa = [ra.index(f) for f in shared]
                pb = [rb.index(f) for f in shared]
                rho, pval = spearmanr(pa, pb)
                rec["spearman_shared"] = round(float(rho), 4)
                rec["spearman_p"] = round(float(pval), 6)
                rec["n_shared"] = len(shared)
            out["pairs"][f"{a}__vs__{b}"] = rec
    js = [v[f"jaccard@10"] for v in out["pairs"].values() if f"jaccard@10" in v]
    out["median_jaccard@10"] = float(np.median(js)) if js else np.nan
    out["min_jaccard@10"] = float(np.min(js)) if js else np.nan
    return out


def agreement_divergence_table(rankings: dict, k: int = 10) -> pd.DataFrame:
    """Reviewer 1, validity 4: separate the features that ALL methods agree on
    from those where they disagree, so the divergences can be explained."""
    tops = {m: list(r[:k]) for m, r in rankings.items()}
    universe = sorted(set().union(*tops.values()))
    rows = []
    for f in universe:
        ranks = {m: (t.index(f) + 1 if f in t else None) for m, t in tops.items()}
        n_in = sum(v is not None for v in ranks.values())
        rows.append({
            "feature": f,
            **{f"rank_{m}": ranks[m] for m in tops},
            "n_methods_selecting": n_in,
            "status": ("consensus" if n_in == len(tops) else
                       "partial" if n_in > 1 else "method-specific"),
        })
    return (pd.DataFrame(rows)
            .sort_values(["n_methods_selecting", "feature"], ascending=[False, True])
            .reset_index(drop=True))


def save_explainability(target: str, horizon: int, payload: dict):
    d = os.path.join(C.OUT_XAI, f"{target}_h{horizon}")
    os.makedirs(d, exist_ok=True)
    for k, v in payload.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(d, f"{k}.csv"), index=False)
        else:
            with open(os.path.join(d, f"{k}.json"), "w") as f:
                json.dump(v, f, indent=2, default=float)
