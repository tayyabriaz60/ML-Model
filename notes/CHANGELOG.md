# Run log

| Date | Change | Why | Affects |
|---|---|---|---|
| 2026-10-10 | Client approved pod eval plan; station note documents NaN pressure-block caveat + AL KHARJ; `h1_xgb_lgb_ci_overlap.py` | Jarwal Ray milestone T5.R1 scope lock | deliverable zip / station_coverage_note.txt |
| 2026-09-21 | Working tree isolated in `T5_revision/` | Single place to execute the contract | organisation only |
| 2026-09-21 | Pipeline wiring: native-unit metrics, WindowDataset, DL tune/train, ablations, run_meta | Scaffold was not runnable as written | all regenerated tables/figures once data exists |
| 2026-09-21 | COLUMN_MAP + 2009–2019 window + zip chunked load | Client sent Historical.zip; avoid loading 1946–2019 88-station grid | Stage 1 audit |
| 2026-09-22 | Stage 2 features: float32, no full-frame copies | 2.56M-row copy OOM’d; leakage assert still passed (122 features) | Stage 2 outputs |
| 2026-09-22 | Audit verdict + letter Finding A rewrite (no GPU) | Ungrouped-lag story disproved; letter must not repeat it | letter / R2-4 |
| 2026-09-23 | Tree trial logs; XGB GPU then CPU dump; train uses tuned HP | Tune was silent; Stage 4 ignored `best_hyperparameters.json` | Stage 3–4 |
| 2026-09-23 | Scaling ablation no longer inverse-transforms native y | Would KeyError / garbage RMSE | Stage 7 |
| 2026-09-23 | Stage 8 loads SHAP/LIME/agreement/regime figures | Only ACF + CI were wired | Stage 8 |
| 2026-09-23 | `build_letter.py`, `build_manuscript.py`, `manuscript_master.md` | Missing builders blocked Word deliverables | letter + manuscript |
| 2026-09-23 | Letter R2-4 + README: units/missingness, not confirmed lag bug | Comment 4 still had the disproved story | letter / docs |
| 2026-09-23 | Vectorize heat threshold + regime labels; Stage 6 ran locally | Row-wise 2.5M masks hung; now 2888 fog / 1512 heat / 156 heatwaves | letter FOG_/HEAT_/NORMAL_ keys |
| 2026-09-23 | `run_all.py --stage train --arch tree`; `build_docx.py` | 3-day delivery: local CPU tables + Word shells while pod tunes | deliverables 2–4 |
| 2026-09-25 | SQLite trial store + `--clock-trial` + `--arch dl` | Client: hour-1 wall-clock + crash-resume; do not redo trees | Stage 3 / RunPod |
