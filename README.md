# T5.R1 — explainable meteorological forecast revision

Working tree for the major revision of *Explainable short-horizon hourly meteorological forecasting models for temperature, visibility, and atmospheric pressure using SHAP, LIME, and attention.*

Client contract: **ML Model Test and Enhancement T5.R1**, 5 working days, four deliverables. Follow `briefing/revision_run_sheet.html` and `briefing/REVISION_WORK_ORDER.pdf` (PDF is local-only).

## Deliverables

| # | File | Source |
|---|---|---|
| 1 | `code/` rerun | `code/run_all.py` |
| 2 | `manuscript_marked_up.docx` | `python build_manuscript.py` after Stage 9; then Word mark-up |
| 3 | `manuscript_clean.docx` | same filled text, clean copy |
| 4 | `response_letter.docx` | `python build_letter.py` after Stage 9 |

Do not type numbers by hand. Stage 9 writes `manuscript_numbers.json`; the letter and manuscript substitute `[[KEY]]`.

Word copies (re-run after Stage 9):

```bash
cd code
python build_docx.py
```

Outputs: `manuscript/manuscript_clean.docx`, `manuscript/manuscript_marked_up.docx`, `letter/response_letter.docx`, and copies in `deliverables/`. Yellow highlight = changed diagnostic or an unfilled `[[KEY]]`.

## What this repo contains

```
briefing/     run-sheet + engineering rules (PDFs stay on disk, not on GitHub)
code/         revision pipeline (run_all.py + src/)
data/         README only; raw CSVs are local
notes/        problem statement + changelog
```

The 458 MB `Weather-suaida-project.zip` is **not** this paper’s pipeline. It is a QC/anomaly project on the same KAPSARC source. Use it only as a data pointer. Do not run its stage-1…8 QC scripts for this revision.

## Clone on RunPod

The 5 GB historical zip and `features.parquet` are **not** on GitHub. Upload `saudi-hourly-weather-data_Historical.zip` to the pod separately.

```bash
git clone https://github.com/tayyabriaz60/ML-Model.git
cd ML-Model/code
pip install -r requirements.txt
export N_TRIALS=24
export KAPSARC_RAW=/path/to/saudi-hourly-weather-data_Historical.zip
python run_all.py --stage audit
python run_all.py --stage features
python run_all.py --stage tune
```

If audit + features already ran locally, copy `code/data/processed/*.parquet` onto the pod and start at `--stage tune`.

## Data status

Production source: `saudi-hourly-weather-data_Historical.zip` (9.27M rows, 88 stations, visibility in metres).
Stage 1 kept **29 stations**, 1 Jan 2009 – 24 May 2019, 2.83M rows. Audit did **not** confirm Table 5 as a cross-station lag error — inspect `outputs/audit/audit_units.json`.

## Run order (do not skip Stage 1)

Audit is a gate. Confirmed cross-station lag contamination → regenerate every table, figure, and SHAP result. Not confirmed → inspect `outputs/audit/audit_units.json` before editing the manuscript.

## Hard rules

- Every lag / rolling / persistence operator is `groupby(station)`.
- Metrics in native units (°C, m, hPa). Do not inverse-transform already-native `y`.
- Identical `N_TRIALS` for trees and sequence models.
- Withdraw XAI dashboard and temporal saliency; do not regenerate them.
- GitHub copy is public: manuscript PDF, reviewer comments, and response letter are gitignored.
