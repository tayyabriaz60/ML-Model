# Run log

| Date | Change | Why | Affects |
|---|---|---|---|
| 2026-09-21 | Working tree isolated in `T5_revision/` | Single place to execute the contract | organisation only |
| 2026-09-21 | Pipeline wiring: native-unit metrics, WindowDataset, DL tune/train, ablations, run_meta | Scaffold was not runnable as written | all regenerated tables/figures once data exists |
| 2026-09-21 | COLUMN_MAP + 2009–2019 window + zip chunked load | Client sent Historical.zip; avoid loading 1946–2019 88-station grid | Stage 1 audit |
| 2026-09-22 | Stage 2 features: float32, no full-frame copies | 2.56M-row copy OOM’d; leakage assert still passed (122 features) | Stage 2 outputs |
