# Revision package — manuscript under major revision

> Author, affiliation, journal and manuscript number are withheld from this
> copy. Where the documents show `[WITHHELD]`, that is deliberate.

Code for the major revision of *"Explainable short-horizon hourly meteorological
forecasting models for temperature, visibility, and atmospheric pressure using
SHAP, LIME, and attention."*

> **Scope note.** Two components of the submitted paper are withdrawn in this
> revision rather than repaired: the temporal saliency analysis and the XAI
> dashboard. Nothing in this package generates them. See `WITHDRAWN` in
> `src/figures.py` and Section 9 of the manuscript for the reasoning. If a
> stage appears to be missing relative to the submitted paper, this is why.

## Running on a compressed schedule

The five-day plan overlaps human work with three overnight compute windows. Two
things in this package are tuned for that:

**The search budget is one constant, set by environment variable.**

```bash
export N_TRIALS=24        # applies to trees AND sequence models
python run_all.py --stage tune --parallel 9
```

`models.N_TRIALS` defaults to 40 and may be reduced, but only for every family
at once. The equal budget is what answers Reviewer 2's bullet 2; an unequal one
re-creates the exact criticism. The value used is written to
`outputs/run_meta.json` and reaches the manuscript and the letter through the
`[[N_TRIALS]]` placeholder, so the text states the budget actually spent. Do not
type that number anywhere.

**The nine tuning jobs are independent.** Three architectures x three targets,
no shared state. On one GPU that is about three days and it blocks everything
downstream; on three or more it fits a single night. Confirm your concurrency
quota before the first evening, not after.

## Before anything else

Run the audit. It is a **decision gate**, not a formality.

```bash
pip install -r requirements.txt
export KAPSARC_RAW=/path/to/raw
python run_all.py --stage audit
cat outputs/audit/audit_all_targets.json
```

The submitted Table 5 reports, for 1 h air temperature, a persistence baseline
with RMSE = 0.8864 °C, MAE = 0.0568 °C and R² = −0.5144. Those three numbers
are mutually inconsistent: an RMSE/MAE ratio of ~15.6 means a handful of
enormous errors dominate, and R² = −0.51 implies an hourly autocorrelation of
about 0.24, which no station temperature series has. The most likely cause is
that lag, rolling and persistence features were built on a frame sorted by
timestamp with all 29 stations interleaved — no `groupby(STATION)` — so `lag_1`
is the previous **row** (a different station in the same hour) rather than the
previous **hour**. That also explains the pattern across targets: sea-level
pressure is spatially coherent so it survived at R² ≈ 0.98, visibility saturates
at 10 km so it survived at ≈ 0.84, and temperature — which differs by 15 °C
between coastal and interior desert sites at the same hour — collapsed.

`src/data.audit_station_grouping()` computes persistence both ways and reports
whether the ungrouped variant reproduces the submitted numbers.

* **Verdict CONFIRMED** → every table, figure and SHAP result is regenerated.
  Budget the full 8-week plan.
* **Verdict NOT CONFIRMED** → investigate `audit_units.json` next (are Table 5's
  "°C" values actually z-scores?) before touching the manuscript.

## Stages

| Stage | Command | Produces | Reviewer points |
|---|---|---|---|
| 1 Audit | `--stage audit` | `outputs/audit/*.json` | R2-4, R2-6 |
| 2 Features | `--stage features` | `features.parquet`, VIF, collinearity, autocorrelation | R1-ED1, R1-ED2, R1-V2 |
| 3 Tune | `--stage tune` | `tuning_*.csv`, `best_hyperparameters.json` | R2-2 |
| 4 Train | `--stage train` | Tables 5–8, bootstrap CIs, DM tests | R2-1, R2-7, R2-8, R1-BR8 |
| 5 Explain | `--stage explain` | SHAP/LIME/attention, **agreement metric** | R2-5, R1-V1, R1-V4 |
| 6 Extremes | `--stage extremes` | fog/heat events, regime labels, Table 9 | R1-ED7, R1-ED8, R1-V3 |
| 7 Ablations | `--stage ablations` | four ablation tables | R1-ED1/2/3/4, R2-1 |
| 8 Figures | `--stage figures` | 13 consolidated figures | R1-BR7, R2-9 |
| 9 Numbers | `--stage numbers` | `manuscript_numbers.json` | — |

## The numbers bridge

Stage 9 writes every manuscript-bound value to `outputs/manuscript_numbers.json`
under a stable key. The manuscript source uses `[[KEY]]` placeholders and
`build_manuscript.py` substitutes them. Nothing is retyped by hand.

This exists because the submitted version reports **three different values for
one quantity** — fog-regime importance of current temperature is 0.35 in
Table 9, roughly 0.85 in Fig. 16 and roughly 0.37 in Fig. 22. Generating the
table, the figure and the sentence from one source makes that impossible.

Check coverage before submitting:

```bash
python -m src.export_numbers --check ../manuscript/manuscript_master.md
```

## Hard gates

Two assertions abort the pipeline rather than warn:

1. `features.assert_no_cross_station_leakage()` — `lag_1` must equal the
   within-station previous hour, and the first row of every station must have a
   null lag.
2. Rolling statistics are computed on `.shift(1)` values. A window that includes
   the current observation leaks the present into a "history" feature.

## What changed versus the submitted pipeline

- `groupby(STATION)` on every temporal operator (was the likely root cause)
- neural models train on 100 % of the data, not 30 % (R2-1)
- 40-trial random search for **both** tree and DL families (R2-2)
- metrics in native units, plus a unit-free skill score vs persistence (R1-BR8/9, R2-8)
- test metrics computed twice: all rows, and observed-target rows only (R2-3)
- fog = visibility < 1 km absolute; heatwave requires 3-day persistence (R1-ED7)
- cross-method agreement defined as Jaccard@k / Spearman / RBO (R2-5)
- block bootstrap CIs, per-station variance, Diebold–Mariano tests (R2-7)
- horizons 1–6 h evaluated; 1/3/6 reported (R1-ED3)
- true attention weights separated from perturbation importance (Figs 10–12 fix)
