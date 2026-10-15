#!/usr/bin/env python
"""Audit extracted T5_milestone_compute.zip (or staging dir) before client delivery."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

FAMILIES = (
    "Persistence", "LinearRegression", "XGBoost", "LightGBM",
    "LSTM", "GRU", "Transformer",
)
HORIZONS = (1, 3, 6)
TARGETS = ("temperature", "visibility", "pressure")


def _load_tables(root: Path) -> pd.DataFrame:
    parts = []
    for t in TARGETS:
        p = root / f"table_results_{t}.csv"
        if not p.is_file():
            raise FileNotFoundError(p)
        parts.append(pd.read_csv(p))
    ref = pd.concat(parts, ignore_index=True)
    return ref[ref.horizon_h.isin(HORIZONS)].copy()


def audit(root: Path) -> int:
    ci_p = root / "bootstrap_ci.csv"
    ps_p = root / "per_station_metrics.csv"
    acc_p = root / "milestone_acceptance_report.txt"
    if not ci_p.is_file():
        raise SystemExit(f"missing {ci_p}")
    if not ps_p.is_file():
        raise SystemExit(f"missing {ps_p}")

    ci = pd.read_csv(ci_p)
    ci = ci[ci.horizon_h.isin(HORIZONS)]
    ps = pd.read_csv(ps_p)
    ref = _load_tables(root)

    n_ci = len(ci)
    if n_ci != 63:
        print(f"FAIL: bootstrap_ci has {n_ci} main-horizon rows (expected 63)")

    ci_bad = []
    for _, r in ci.iterrows():
        lo, hi, pt = float(r.ci_low), float(r.ci_high), float(r.rmse)
        if pt < lo - 1e-6 or pt > hi + 1e-6:
            ci_bad.append(r)

    parity_rows = []
    n_match = 0
    n_n_same = 0
    for _, r in ref.iterrows():
        m = ci[
            (ci.target == r.target)
            & (ci.horizon_h == r.horizon_h)
            & (ci.model.str.lower() == str(r.model).lower())
        ]
        tbl_n = int(r.n) if "n" in r.index and pd.notna(r.n) else None
        boot_n = None
        if len(m):
            boot_n = int(m.iloc[0].n) if "n" in m.columns and pd.notna(m.iloc[0].n) else None
        boot = float(m.iloc[0].rmse) if len(m) else None
        tbl = float(r.RMSE)
        ok = boot is not None and round(boot, 4) == round(tbl, 4)
        if ok:
            n_match += 1
        if tbl_n is not None and boot_n is not None and tbl_n == boot_n:
            n_n_same += 1
        parity_rows.append({
            "target": r.target, "h": r.horizon_h, "model": r.model,
            "table_rmse": tbl, "boot_rmse": boot, "table_n": tbl_n, "boot_n": boot_n,
            "parity_ok": ok,
        })

    fam_miss = []
    for target in TARGETS:
        for h in HORIZONS:
            sub = ps[(ps.target == target) & (ps.horizon_h == h)]
            have = set(sub.model.unique())
            for fam in FAMILIES:
                if fam not in have:
                    fam_miss.append(f"{target} h={h} missing {fam}")

    lr_rows = len(ps[ps.model == "LinearRegression"])
    persist_rows = len(ps[ps.model == "Persistence"])

    print("=" * 60)
    print("PRE-DELIVERY AUDIT (client T5.R1)")
    print("=" * 60)
    print(f"Directory: {root}")
    if acc_p.is_file():
        print("\n--- milestone_acceptance_report.txt ---")
        print(acc_p.read_text(encoding="utf-8").strip())

    print("\n--- Test 1: CI contains point ---")
    print(f"  {n_ci - len(ci_bad)}/{n_ci} ok")
    if ci_bad:
        print("  FAIL rows (must be 0 before delivery):")
        print(pd.DataFrame(ci_bad).head(10).to_string(index=False))

    print("\n--- Test 2: bootstrap RMSE == table_results (4 dp) ---")
    print(f"  {n_match}/63 ok")
    if n_match != 63:
        bad = [x for x in parity_rows if not x["parity_ok"]]
        print(f"  MISMATCH count: {len(bad)}")
        print("  First 15 mismatches:")
        print(pd.DataFrame(bad).head(15).to_string(index=False))
        print("\n  n same on mismatches (table_n vs boot_n when both present):")
        mism = pd.DataFrame([x for x in parity_rows if not x["parity_ok"]])
        if "boot_n" in mism.columns and mism["boot_n"].notna().any():
            print(mism[["target", "h", "model", "table_n", "boot_n"]].head(15).to_string(index=False))
        else:
            print("  (bootstrap_ci has no n column — add table_n from table_results for client trace)")

    print("\n--- Test 3: seven families per target-h ---")
    print(f"  {'PASS' if not fam_miss else 'FAIL'}")
    if fam_miss:
        print("  ", fam_miss[:15])
    print(f"  per_station rows: {len(ps)} (expect 1404 with LR+relaxed persistence)")
    print(f"  LinearRegression rows: {lr_rows}, Persistence rows: {persist_rows}")

    print("\n--- Zip contents sanity ---")
    for name in (
        "bootstrap_table_parity_report.txt", "ci_mask_report.txt",
        "station_coverage_note.txt", "tree_preflight_report.txt",
        "dl_pressure_h1_note.txt", "run_env.txt",
    ):
        print(f"  {name}: {'OK' if (root / name).is_file() else 'MISSING'}")

    ship = (
        len(ci_bad) == 0
        and not fam_miss
        and n_ci == 63
    )
    ship_ideal = ship and n_match == 63
    print("\n--- VERDICT ---")
    if ship_ideal:
        print("  READY: all three client tests pass (ideal delivery).")
        return 0
    if ship and n_match != 63:
        print("  PARTIAL: CI + families OK; parity NOT 63/63 — client may reject test #2")
        print("  unless they accept honest mismatch (they said they would trace n).")
        print("  For 63/63 parity you need table-matching checkpoints (not pod refit).")
        return 2
    print("  DO NOT SHIP: fix CI or per_station first.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to extracted zip folder or .zip file")
    args = ap.parse_args()
    p = Path(args.path)
    if p.suffix.lower() == ".zip":
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(p) as zf:
                zf.extractall(td)
            return audit(Path(td))
    return audit(p)


if __name__ == "__main__":
    raise SystemExit(main())
