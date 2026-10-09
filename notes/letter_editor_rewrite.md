# Replacement: Letter to the Editor, Finding A

Paste this in place of the three “cross-station lag” paragraphs in
`letter/letter_text.txt` (lines 7–9). Do not type later table numbers;
leave `[[KEY]]` as-is.

---

A correction that precedes everything else. Reviewer 2 asked for a sanity explanation of the negative persistence R² for temperature at a 1-hour lead. That question was the most valuable comment in either report. The three values in the submitted Table 5 for 1-hour temperature persistence — RMSE = 0.8864 °C, MAE = 0.0568 °C, R² = −0.5144 — cannot describe one physical series: the RMSE-to-MAE ratio of about 15.6 requires a few enormous errors to dominate, and R² = −0.51 implies a lag-1 autocorrelation of about 0.24, which no hourly station temperature series has.

The Stage 1 audit on the 29-station, 1 January 2009 – 24 May 2019 extract does **not** reproduce those three numbers as a cross-station lag error. Persistence computed within station in native °C is RMSE 1.49 °C, MAE 1.11 °C, R² 0.97, lag-1 r = 0.99. Persistence computed on a time-sorted frame with stations interleaved is worse (RMSE 6.31 °C, R² 0.53) but still nothing like Table 5. The submitted figures are the scale of z-scored residuals, not degrees Celsius. Sea-level pressure is missing on 78 % of the hourly grid (100 % at some inland stations); a reported pressure R² near 0.98 is therefore not observed-target skill.

The revision pipeline now (i) reports every headline metric in native units after inverting the scaler, (ii) applies every lag, roll and persistence operator inside `groupby(station)` and aborts if that test fails, (iii) never mean-imputes targets, and (iv) evaluates test metrics on observed-target rows as well as on the filled grid. Every table, figure and SHAP/LIME result is regenerated from one `manuscript_numbers.json`. Sections 4.1 and 6.1 document this diagnostic. I am grateful the review process caught the inconsistency before publication.

---

Keep the rest of the letter (withdrawals, `[[N_TRIALS]]`, point-by-point). Fill keys only after Stage 9.
