# Stage 1 audit — locked (2026-09-21)

Source: `code/outputs/audit/*.json` after cleaning
`saudi-hourly-weather-data_Historical.csv` to 2009-01-01–2019-05-24,
`MIN_STATION_RECORDS = 30000`.

## Coverage

- Raw archive: 9,270,697 rows, 88 stations, 1946–2019, visibility in metres
- Study window kept: 2,830,621 rows → **29 stations**
- Leakage assert (Stage 2): **passed** (`lag_1` = same station, previous hour)
- Features: **122**

## Temperature persistence (native °C), 1 h

| Pipeline | n | RMSE | MAE | R² | lag-1 r |
|---|---|---|---|---|---|
| Grouped (correct) | 2,301,609 | 1.49 °C | 1.11 °C | **0.973** | 0.987 |
| Ungrouped (cross-station) | 2,160,244 | 6.31 °C | 4.88 °C | 0.525 | 0.761 |
| Submitted Table 5 | — | 0.8864 | 0.0568 | **−0.5144** | — |

`ungrouped_matches_submission = false`

Submitted RMSE/MAE ratio ≈ 15.6. Correct grouped ratio ≈ 1.34.

**Do not tell the editor that z-scoring produced Table 5.** Affine rescaling leaves R² and RMSE/MAE unchanged. Z-scoring observed persistence would give RMSE ≈ 0.16, not 0.8864.

**Imputed-target persistence (client 2026-09-23) — NOT REPRODUCED.**
Temperature missingness on the hourly grid: **8.46%** (216,535 / 2,558,212). Pressure is 78.18%.
Scoring 1 h persistence on frames that mean-impute the *target* (global / station / station-month, with and without 6 h ffill+interp) never collapses MAE to 0.0568 or R² to −0.51. Best-matching imputed variant still has MAE ≈ 1.05 °C, R² ≈ 0.97, ratio ≈ 1.4. See `audit_imputed_target_persistence.json`.

The submitted Table 5 computation **cannot be reconstructed** from this extract. Report corrected observed-target values. Do not assert a mechanism.

## Other gates

- Pressure missing **78%** on the hourly grid (ABHA 100%). Grouped persistence n = 70.
  Submitted pressure R² ≈ 0.98 cannot be observed-target skill.
- Visibility persistence (metres): grouped RMSE ≈ 958 m, R² ≈ 0.80.

## Letter / paper rule

Regenerate all tables and figures anyway (native units, groupby, no target mean-impute, equal `N_TRIALS`).
Rewrite Finding A as **units + missingness**, not as a confirmed ungrouped-lag bug.
