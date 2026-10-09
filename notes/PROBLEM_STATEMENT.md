# Client problem statement (locked)

Contract: **ML Model Test and Enhancement T5.R1** — 5 working days.
Paper: explainable 1–6 h forecasts of temperature, visibility, and sea-level pressure (SHAP / LIME / attention).

## Job as stated on Upwork

Fix a feature-engineering bug, rerun the supplied Python pipeline end to end, and deliver:

1. Corrected codebase + rerun (`code/`)
2. Marked-up manuscript (`manuscript_marked_up.docx`)
3. Clean manuscript (`manuscript_clean.docx`)
4. Point-by-point response letter (`response_letter.docx`)

Text is already drafted. Numbers auto-fill from the run via `[[KEY]]`. Follow the HTML run-sheet checklist.

## Root cause — client hypothesis vs audit

Client / work-order story: lags built without `groupby(station)`, so Table 5 is cross-station contamination.

**Audit (see `AUDIT_VERDICT.md`): NOT CONFIRMED.** Within-station 1 h temperature persistence is R² 0.97 in native °C. Table 5 (RMSE 0.8864, MAE 0.0568, R² −0.5144) is a scaled-unit / reporting defect. Pressure is 78 % missing. Still regenerate everything; do not write the ungrouped-lag story to the editor.

Submitted Table 5 is still internally impossible as a set of °C metrics:

- RMSE = 0.8864 °C
- MAE = 0.0568 °C
- R² = −0.5144

Second defect: three different SHAP values for one fog-regime quantity (Table 9 / Fig. 16 / Fig. 22). Fix is one shared values file, not hand edits.

## What the last zip is

Client: *“Please find the requested data scripts file. The scripts has some other unrelated approaches for another works on the same dataset.”*

`Weather-suaida-project.zip` is a **QC / drift / autoencoder** repo on KAPSARC hourly weather. Unrelated stages must not be run for this revision.

Usable extract inside it:

- 4 stations, 2018-01-01 → 2019-05-24, 63,266 rows
- Headers: `station_name`, `datetime`, `air_temperature`, `dew_point`, `visibility`, `sea_level_pressure`, `wind_speed`, `wind_direction`, `sky_ceiling`
- Visibility range **0.0–0.9** (paper expects metres, 0–20 000). Do not map this column into `VISIBILITY_DISTANCE` until the unit is proven.
- Temperature °C and pressure hPa look plausible.

Full KAPSARC download script in that zip hits `https://datasource.kapsarc.org/api/records/1.0/search/` dataset `saudi-hourly-weather-data`, ~5.2 GB, 1000-row API batches. Paper-scale 29-station / 2009–2019 file is **not** in the zip. The 3–4 GB `stage*_flagged_data.csv` files are QC outputs, not the modelling table.

## Reviewer work that needs a rerun vs prose only

Rerun / numbers: ~24 of 36 points (audit, native units, full-data DL, equal `N_TRIALS`, SHAP/LIME agreement, fog/heat definitions, ablations, CIs/DM tests, figures).

Prose / structure only: ~12 points (Intro vs Related Work, citations, terminology, abbreviations, withdrawals of dashboard and saliency).

Withdraw, do not repair: XAI dashboard, temporal saliency.

## Builders (added 2026-09-23)

- `manuscript/manuscript_master.md` — revision source; edit this, not the filled copy
- `code/build_manuscript.py` — substitutes `[[KEY]]` → `manuscript/manuscript_filled.md`
- `code/build_letter.py` — same for `letter/letter_text.txt` → `letter/response_letter_filled.txt`

Word `.docx` packaging still waits until Stage 9 writes a complete `manuscript_numbers.json`. Do not type numbers.

## Execution order

1. Confirm data coverage and units (blocker).
2. Stage 1 audit on the real 29-station hourly file.
3. Stages 2–9 in order (`N_TRIALS=24` unless the work order is overridden).
4. Substitute numbers into letter + manuscript. Never type them.
