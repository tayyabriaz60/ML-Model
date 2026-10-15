#!/usr/bin/env python
"""Client acceptance: CI contains point, 63/63 parity, seven model families in per_station."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C

CI = os.path.join(C.OUT_TABLES, "bootstrap_ci.csv")
PS = os.path.join(C.OUT_TABLES, "per_station_metrics.csv")
PARITY = os.path.join(C.OUT_TABLES, "bootstrap_table_parity_report.txt")
PREFLIGHT = os.path.join(C.OUT_TABLES, "tree_preflight_report.txt")
OUT = os.path.join(C.OUT_TABLES, "milestone_acceptance_report.txt")
FAMILIES = (
    "Persistence", "LinearRegression", "XGBoost", "LightGBM",
    "LSTM", "GRU", "Transformer",
)
HORIZONS = list(C.HORIZONS_MAIN)


def _tables_root() -> str:
    return os.environ.get("TABLE_RESULTS_DIR", os.path.join(C.OUT_TABLES, "main_h136"))


def _check_ci_contains(ci: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    bad = []
    for _, r in ci.iterrows():
        lo, hi, pt = float(r.ci_low), float(r.ci_high), float(r.rmse)
        if pt < lo - 1e-6 or pt > hi + 1e-6:
            bad.append({**r.to_dict(), "reason": "point outside CI"})
    rep = pd.DataFrame(bad)
    return len(ci) - len(bad), rep


def _check_parity(ci: pd.DataFrame) -> tuple[int, int]:
    parts = []
    for target in C.CONFIG.targets:
        p = os.path.join(_tables_root(), f"table_results_{target}.csv")
        parts.append(pd.read_csv(p))
    ref = pd.concat(parts, ignore_index=True)
    ref = ref[ref.horizon_h.isin(HORIZONS)]
    n_ok = 0
    for _, r in ref.iterrows():
        m = ci[(ci.target == r.target) & (ci.horizon_h == r.horizon_h)
               & (ci.model.str.lower() == str(r.model).lower())]
        if len(m) and round(float(m.iloc[0].rmse), 4) == round(float(r.RMSE), 4):
            n_ok += 1
    return n_ok, len(ref)


def _check_preflight() -> tuple[int, int, str]:
    """18/18 = plausible trees loaded (XGB from .json); not FAIL / LOAD_ERR."""
    if not os.path.exists(PREFLIGHT):
        return 0, 18, "missing tree_preflight_report.txt"
    text = open(PREFLIGHT, encoding="utf-8").read()
    first = text.split("\n", 1)[0].strip()
    # Prefer explicit summary line from pod_tree_preflight.py
    if "tree preflight:" in first and "/18" in first:
        import re
        m = re.search(r"tree preflight:\s*(\d+)/18", first)
        if m:
            n = int(m.group(1))
            return n, 18, first
    return 0, 18, first or "unparsed preflight report"


def _check_families(ps: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = []
    for target in C.CONFIG.targets:
        for h in HORIZONS:
            sub = ps[(ps.target == target) & (ps.horizon_h == h)]
            have = set(sub.model.unique())
            for fam in FAMILIES:
                if fam not in have:
                    missing.append(f"{target} h={h} missing {fam}")
    return len(missing) == 0, missing


def main() -> int:
    if not os.path.exists(CI) or not os.path.exists(PS):
        raise SystemExit(f"missing {CI} or {PS}")
    ci = pd.read_csv(CI)
    ci = ci[ci.horizon_h.isin(HORIZONS)]
    ps = pd.read_csv(PS)
    n_ci_ok, ci_bad = _check_ci_contains(ci)
    n_par, n_ref = _check_parity(ci)
    fam_ok, fam_miss = _check_families(ps)
    n_pf, n_pf_ref, pf_line = _check_preflight()
    pf_ok = n_pf == n_pf_ref
    lines = [
        "Milestone acceptance (client T5.R1)",
        f"1) CI contains point: {n_ci_ok}/{len(ci)} ok",
        f"2) Bootstrap RMSE vs table_results: {n_par}/{n_ref} exact (4 dp)",
        f"3) per_station seven families each target-h: {'PASS' if fam_ok else 'FAIL'}",
        f"4) Tree preflight: {n_pf}/{n_pf_ref} {'PASS' if pf_ok else 'FAIL'} — {pf_line}",
        "",
    ]
    if len(ci_bad):
        lines.append("CI violations (first 10):")
        lines.append(ci_bad.head(10).to_string(index=False))
        lines.append("")
    if not fam_ok:
        lines.append("Missing model families (first 20):")
        lines.extend(fam_miss[:20])
    text = "\n".join(lines) + "\n"
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    ok = (n_ci_ok == len(ci) and n_par == n_ref and fam_ok and pf_ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
