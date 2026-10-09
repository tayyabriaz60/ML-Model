# Explainable Short-Horizon Hourly Meteorological Forecasting Models for Temperature, Visibility, and Atmospheric Pressure Using SHAP, LIME, and Attention

[AUTHOR NAME WITHHELD]
[AFFILIATION WITHHELD]
Manuscript [ID WITHHELD] — major revision

Every numeric claim below is a placeholder filled from `outputs/manuscript_numbers.json`. Do not type table numbers by hand.

---

## Abstract

We revise a station-local 1–6 h forecast of air temperature, horizontal visibility and sea-level pressure for 29 Saudi meteorological stations over 2009-01-01 to 2019-05-24 (2,558,212 hourly rows after reindexing). Headline metrics are in native units and are scored on observed-target rows. LSTM, GRU and a transformer train on the full set with the same 24-trial budget as the trees. Sea-level pressure is missing on 78.18 % of the hourly grid; that coverage finding is reported separately from the Table 5 withdrawal. Submitted 1 h temperature persistence (RMSE 0.8864 °C, MAE 0.0568 °C, R² = −0.5144) is withdrawn: cross-station lag, unit scaling, and imputed-target scoring were systematically excluded, and the values cannot be reconstructed. Corrected persistence is R² = 0.9734 (predicted 2r−1 = 0.9734 from r = 0.9867).

The revision reports every headline metric in native units (°C, m, hPa), applies every lag, roll and persistence operator inside `groupby(station)`, never mean-imputes targets, and evaluates test rows whose targets were observed as well as the filled grid. LSTM, GRU and a transformer train on the full training set. Every family receives an identical 24-trial search. The phrase “>0.90 agreement” is withdrawn and replaced by Jaccard, Spearman and rank-biased overlap. Fog is visibility below 1 000 m; a heatwave requires a station- and day-of-year threshold sustained for three days. The XAI dashboard and the temporal saliency analysis are withdrawn.

Best 1 h models: temperature LightGBM (R² = 0.9918, RMSE 0.7993 °C); visibility LightGBM (R² = 0.8212); pressure XGBoost (R² = 0.9882). Pressure skill is reported relative to persistence (R² = 0.9591).

**Keywords.** short-horizon forecast; SHAP; LIME; attention; persistence; missing data; Saudi Arabia

---

## 1. Introduction

Hourly forecasts of temperature, visibility and pressure at 1–6 h leads support aviation, road safety and heat-health warnings. A station-local model that uses only a site’s own recent history is a lower bound on what is possible without a numerical weather prediction (NWP) grid. This paper reports that lower bound, with explainability, on a 29-station KAPSARC hourly archive.

### 1.1 Scope and contributions

We forecast three targets at horizons of 1, 3 and 6 h (full 1–6 h evaluation in the supplement):

1. A leakage-gated feature pipeline: every temporal operator is computed within station and the pipeline aborts if a one-hour lag is not the previous hour at the same site.
2. An equal-budget comparison of linear, boosted-tree and sequence models (24 trials each; no 30 % neural subsample).
3. Quantified cross-method agreement (Jaccard@k, Spearman, RBO) in place of an undefined “>0.90” claim.
4. WMO-aligned fog and heat definitions, with the residual “normal” class described rather than assumed benign.

### 1.2 What changed in this revision

The submitted Table 5 temperature persistence triple is internally impossible as native °C. Section 4.1 and Section 6.1 document the audit. Related Work is re-scoped to methodological neighbours. Fifteen off-domain citations are replaced. Terminology is consistent (visibility in metres; abbreviations at first use). Twenty-four figures are consolidated to eleven.

### 1.3 Removal note — title and scope

Two submitted components are withdrawn rather than repaired: the temporal saliency analysis (old Section 5.4, Figs 13–15) and the XAI dashboard (old Section 7.6, Fig. 24). Neither is regenerated. See Section 5.3 and Section 7.5. The title no longer implies a deployed dashboard.

---

## 2. Related work

Section 2 is organised by methodological characteristic, not by topic importance.

### 2.1 Persistence and short-horizon skill

==At a 1 h lead, persistence is a strong baseline for surface variables whose lag-1 autocorrelation is high (temperature and pressure in this archive). Skill must be reported against that baseline using the unit-free score SS = 1 − MSE_model / MSE_persistence (Section 6.2.3), not raw R² alone. Visibility is different: the reporting ceiling and dust-haze tail compress dynamic range, so persistence can look strong in R² while still leaving large absolute errors in metres (Section 6.2.2).==

### 2.2 Tree ensembles versus sequence models

Gradient-boosted trees on lagged tabular features remain competitive with LSTM/GRU/transformer architectures on regular hourly series. Fair comparison requires the same data and the same search budget (Section 4.5).

### 2.3 Attribution methods

SHAP, LIME and attention answer different questions. Comparing a ranking over features with a distribution over time steps is a category error (Section 5.4).

### 2.4 Extremes in arid climates

Percentile-only “fog” labels in a visibility climate that sits at the reporting ceiling identify dust haze, not condensation fog (Section 7.1).

---

## 3. Data

### 3.1 Source and window

The modelling table is built from `saudi-hourly-weather-data_Historical.csv`. The archive contains 88 stations from 1946; the study window is 2009-01-01 to 2019-05-24. After the window cut and a floor of 30 000 records per station, 29 stations remain (2,558,212 rows on the complete hourly grid).

### 3.2 Variables

Targets: air temperature (°C), visibility distance (m), atmospheric sea-level pressure (hPa). Predictors: dew-point temperature, wind speed, wind direction, sky-ceiling height. Sentinel codes (9999.9, 99999, 999999, …) are set to missing before range screening.

### 3.3 Missingness

Per-variable missingness on the hourly grid: temperature 8.46 %; visibility 32.58 %; sea-level pressure 78.18 %. Gap-length distributions are in Supplementary Table S1. A station is reindexed onto a complete hourly clock before any gap statistic is computed.

---

## 4. Methods

### 4.1 Diagnostic (submitted Table 5)

The submitted 1 h temperature persistence row (RMSE 0.8864 °C, MAE 0.0568 °C, R² = −0.5144) is withdrawn. Three accounts were tested and systematically excluded.

- Cross-station lag. Persistence on a time-sorted interleaved frame gives R² = 0.5251, not −0.51.
- Unit scaling. R² and the RMSE/MAE ratio are invariant to affine rescaling, so z-scoring cannot produce those two anomalies.
- Imputed targets. Persistence scored on a mean-filled temperature frame, including imputed target rows, does not reproduce the triple. Temperature is missing on only 8.46 % of the hourly grid.

The submitted values cannot be reconstructed from this extract. Corrected within-station persistence on observed temperature is RMSE 1.4894 °C, MAE 1.1127 °C, R² = 0.9734, lag-1 r = 0.9867.

Those numbers agree with themselves three ways. For a lag-1 forecast, R² ≈ 2r − 1: r = 0.9867 predicts 0.9734, and the measured R² is 0.9734. The RMSE/MAE ratio is 1.34, a normal error shape. The submitted triple agreed in none of these ways.

The revision reports native units, never mean-imputes targets, applies every temporal operator inside `groupby(station)` (and aborts on failure), and scores test metrics on observed-target rows. All tables and figures are regenerated from one `manuscript_numbers.json`.

### 4.2 Features

122 predictors: lags {1, 2, 3, 6, 12, 24} h, rolling windows {3, 6, 12, 24} h on the shifted series, cyclical calendar encodings, and a small set of derived terms. Pairwise |r| > 0.95: 97 pairs; mean |r| = 0.146 (Supplementary Fig. S1).

#### 4.2.1 Feature-count ablation

Retraining on the top k SHAP features (Table 5): the top 20 retain 99.9 % of full-model R² at a 3.81 % RMSE penalty.

#### 4.2.2 Visibility lags

Short-lag-only (1, 2, 3, 6 h + 3/6 h rolls) versus the full set (Table 6): R² = 0.4982 versus 0.5032 at 1 h, and 0.1208 versus 0.1250 at 6 h.

### 4.3 Horizons

We evaluate 1–6 h and report 1, 3 and 6 h in the main tables. Sequence models see raw windows of length {12, 24, 48} h; engineered lags are an ablation, not a hidden extra channel.

### 4.4 Split

A single global chronological split, identical cut dates at every station: train / validation / test = 70 / 10 / 20. Cuts: 2016-03-06 and 2017-03-22 (n = 419,995 / 60,002 / 120,003). Blocked cross-validation with a 24 h embargo is used only for stability checks.

### 4.5 Models and search

Families: persistence, ridge, XGBoost, LightGBM, LSTM, GRU, transformer. Neural models train on 100 % of the training set. Every family receives 24 random-search trials (Supplementary Table S2). Trees: depth, learning rate, minimum child weight, row/column subsample, L2. Sequence models: hidden width {64, 128, 256}, depth {1, 2, 3}, dropout, learning rate, sequence length, batch size; up to 100 epochs with early stopping (patience 10).

The original 30 % neural subsample is reproduced only as an ablation: the R² gap versus full data is -0.0006 (Supplementary Table S5).

### 4.6 Scaling

Features are scaled with the production strategy log1p_standard. Targets remain in native units. A three-way sensitivity (standard / robust / log1p+standard) is reported in Table 8.

### 4.7 Metrics

Headline metrics are RMSE, MAE, bias and R² in native units, plus the unit-free skill score SS = 1 − MSE_model / MSE_persistence. Test metrics are computed on all finite rows and on rows whose target was observed. Uncertainty: moving-block bootstrap (block = 7 days, 1 000 resamples) and Diebold–Mariano tests against persistence. Per-station distributions are reported for all 29 sites.

---

## 5. Explainability

SHAP estimates the contribution of each feature to a prediction from a cooperative-game value. LIME fits a local linear surrogate around an instance. We aggregate |LIME weight| over a stratified sample so the LIME ranking is global in the same sense as mean |SHAP|. Attention, when used, is a distribution over input time steps, not over feature names.

LIME and SHAP in the main text are computed on the fitted XGBoost model at each reported horizon. Attention is reported only for the transformer, and only as a function of lag position.

### 5.1 Direction (SHAP)

For each top feature we report the correlation between the feature value and its SHAP value (Fig. 2). Positive correlation means a higher feature value raises the prediction.

#### 5.1.1 Reordering with lead time

Fig. 1 (1 h vs 6 h) and Fig. 6 (autocorrelation) show how the leading drivers shift as the horizon lengthens. Beyond a few hours, station-local memory is a weaker constraint.

### 5.2 LIME

==The submitted Figs 7–9 illustrated a single LIME instance; that instance is not reproduced here. Fig. 3 reports an aggregated ranking: for each target and horizon we draw 200 stratified test rows, fit local linear surrogates around the fitted XGBoost model, and average absolute LIME weights by feature. The ranking is therefore global in the same sense as mean |SHAP|, but it remains model-specific (XGBoost at that horizon).==

### 5.3 Removal note — temporal saliency

Old Section 5.4 and Figs 13–15 (temporal saliency) are removed. They restated the SHAP lag ranking under a second name and fed the attention/saliency conflation. The memory-window check is now Fig. 6: transformer attention versus the empirical autocorrelation of each target.

### 5.4 Agreement and divergence

The claim “>0.90 agreement between SHAP, LIME and attention” is withdrawn. We report, per target and horizon, on the same test partition:

- Jaccard overlap of the top-k feature sets, k ∈ {5, 10, 15}
- Spearman rank correlation on the shared support
- Rank-biased overlap with p = 0.9

Median Jaccard@10: temperature 0.111; visibility 0.333; pressure 0.000 (Table 13, Fig. 7). ==Attention over lag positions is excluded from feature-set overlap because it is a distribution over time steps, not over feature names; permutation importance (Fig. 5) enters instead.==

==**Pressure caveat (R2-5).** Sequence models for pressure at h = 1 are worse than persistence (GRU skill -7.811; LSTM skill -6.081). The headline median Jaccard@10 for pressure (0.000) is driven by SHAP versus LIME on the tree model and by perturbation pairs on degenerate sequence predictors; SHAP versus LIME alone is 0.250. We do not interpret 0.000 overlap as agreement between SHAP and LIME on pressure; we report the pair breakdown in Table 13 and Fig. 7.==

Table 14 lists consensus versus method-specific features.

---

## 6. Results

### 6.1 Persistence baseline after the diagnostic

Submitted Table 5 is withdrawn. Cross-station lag, unit scaling, and imputed-target scoring were systematically excluded (Section 4.1). The values cannot be reconstructed from this extract.

Corrected 1 h temperature persistence on observed rows: R² = 0.9734, RMSE 1.4894 °C, MAE 1.1127 °C, lag-1 r = 0.9867. R² ≈ 2r − 1 predicts 0.9734 from that r; the RMSE/MAE ratio is 1.34. The corrected numbers agree three ways. The submitted numbers agreed no ways.

Sea-level pressure missingness (78.18 % of the hourly grid) is a separate coverage finding and is not an explanation of Table 5. Pressure results in Section 6.2.3 are reported against persistence on observed-target rows.

### 6.2 Headline skill (observed targets)

Tables 5–7 report native-unit metrics on observed-target test rows. Summary (Table 8):

Temperature, 1 h — best LightGBM, R² 0.9918, RMSE 0.7993 °C.
Visibility, 1 h — best LightGBM, R² 0.8212, RMSE 926.4935 m.
Pressure, 1 h — best XGBoost, R² 0.9882, RMSE 0.7423 hPa.

#### 6.2.1 Ordering is not a scaling artefact

R² is invariant to affine rescaling. Cross-variable RMSE is not comparable; skill versus persistence is the cross-variable comparator.

#### 6.2.2 Visibility in metres

All visibility RMSE/MAE values are metres, not scaled units.

#### 6.2.3 Pressure versus persistence

Persistence alone reaches R² = 0.9591 at 1 h. The learned improvement to 0.9882 corresponds to skill 0.711. Whether that increment justifies replacing a carry-forward rule is a separate operational question; the present evidence does not obviously answer it affirmatively.

### 6.3 Uncertainty and stations

Bootstrap CIs are in Fig. 9 and Supplementary Table S7. Diebold–Mariano tests against persistence are in Supplementary Table S8. Per-station R² spread is in Fig. 10.

#### 6.3.1 Arid-climate scope

Results are for this network and climate. Visibility sits near a reporting ceiling for many hours; pressure coverage is uneven. Claims are not generalised to humid or mid-latitude networks.

### 6.4 Longer leads

At 6 h the best temperature model is XGBoost (R² 0.9699); visibility LightGBM (R² 0.3970); pressure LightGBM (R² 0.9773). Skill falls as the diurnal cycle and unobserved advection dominate.

---

## 7. Extremes and regimes

### 7.1 Definitions

Fog: horizontal visibility below 1 000 m (WMO). The absolute rule identifies 2,888 episodes (median duration 2.0 h; median minimum visibility 500 m), of which 592 reach dense fog (< 200 m). A percentile-only subset is labelled extreme low visibility, not fog.

Heat: station- and calendar-day 95th percentile of daily maximum temperature in a ±15-day window. A heatwave requires three consecutive days above that threshold: 156 episodes, median duration 3.0 days. Shorter exceedances (1,356) are extreme high temperature, not heatwaves.

The normal class is hours meeting neither criterion: 97.59 % of the record, median visibility 10000 m, of which 3.65 % are still below 5 km (dust haze). Supplementary Table S12.

### 7.2 Regime-conditioned attribution

Table 9 and Fig. 8 are built from one `regime_shap_table` object so a single fog-regime quantity cannot take three values.

### 7.3 Case studies and matched versus cross-regime design

==Primary comparisons pair each target with its matched extreme regime: visibility with fog hours (visibility below 1 000 m), temperature with heatwave hours (three-day sustained exceedance of the local calendar-day 95th percentile). Cross-regime cells (e.g. temperature attribution during fog, visibility skill during heat) are descriptive only; they are not used to claim that the same driver ordering “generalises” across regimes. Fig. 11 summarises episode duration and intensity for the longest fog and heat events; full time-series panels are in Supplementary Fig. S4 when the original hourly trace is available at the station.==

### 7.4 Agreement under extremes

Section 5.4 statistics are not restated here. Regime skill is in Supplementary Table S10.

### 7.5 Removal note — XAI dashboard

Old Section 7.6 and Fig. 24 are removed. The dashboard was never deployed, instrumented or user-tested. Retaining it under a weaker label would have kept a claim without evidence. Future work is listed in Section 9.

---

## 8. Discussion

==**Which target is hardest?** Using skill versus persistence at h = 1, visibility is the limiting target (LightGBM SS 0.171; persistence baseline R² 0.7845). Temperature and pressure tie for ease of short-horizon prediction (temperature SS 0.705; pressure SS 0.711). ~~Temperature remains the hardest of the three targets once persistence is computed correctly.~~ Raw R² alone is misleading for visibility because many hours sit at the reporting ceiling; native-unit RMSE in metres and SS against persistence are the honest comparators (Section 6.2).==

==Visibility errors reflect dust haze and ceiling effects as much as condensation fog; the WMO 1 000 m fog rule identifies 2,888 episodes, but the “normal” class still spends 3.65 % of hours below 5 km visibility. Pressure must be read against 78.18 % hourly missingness and persistence R² = 0.9591 at 1 h; incremental tree skill (0.711) is modest relative to that baseline.==

==After 3–6 h, all three targets degrade as advection and unobserved synoptic forcing dominate; that ceiling is structural for station-local models, not evidence of under-tuning (equal 24-trial budget per family). The phrase “very high skill” is not used anywhere in this revision.==

---

## 9. Conclusions

We reported a corrected, native-unit, leakage-gated 1–6 h station-local forecast for three surface variables, with quantified attribution agreement and WMO-aligned extreme definitions. Two unsupported components — a temporal saliency analysis and an XAI dashboard — were withdrawn.

Future work: multi-station or gridded predictors for temperature beyond 3–6 h; user-tested operational interfaces, if any, as a separate evaluation; denser pressure observations before claiming high pressure skill.

---

## Figures

- Fig. 1 SHAP global importance, targets × {1 h, 6 h}
- Fig. 2 SHAP direction / dependence
- Fig. 3 Aggregated LIME
- Fig. 4 Transformer attention over lag positions
- Fig. 5 Perturbation importance
- Fig. 6 Autocorrelation versus lag
- Fig. 7 Cross-method agreement
- Fig. 8 Regime-conditioned SHAP (same source as Table 9)
- Fig. 9 Performance with bootstrap CIs
- Fig. 10 Per-station skill
- Fig. 11 Fog and extreme-heat case studies

Withdrawn: old Figs 13–15 (saliency), old Fig. 24 (dashboard).

---

## References

==Almazroui, M., et al. (2022). [Title and venue per submitted citation key — temperature extremes in the Arabian Peninsula.] Used to support the revised statement on heat extremes in Section 7.1.==

Al-Khalaf, F. A., & Al-Awadi, F. M. (2020). Machine learning for weather forecasting: review. *Applied Sciences*, 10(11), 3999.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4765–4774.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. *KDD*, 1135–1144.

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 5998–6008.

World Meteorological Organization (WMO). (2023). *International Cloud Atlas* — fog definition (horizontal visibility below 1 000 m).

==Fifteen off-domain citations from the submitted Related Work are removed or replaced by the methodological references above; the full mapping is in the point-by-point response letter (R2-11).==

---

## Data and code

Pipeline: `code/run_all.py`. Numbers: `outputs/manuscript_numbers.json`. This file is the only manuscript source; rebuild with `python build_manuscript.py`. ==Restore `data/processed/features.parquet` and run `python regenerate_headline_eval.py` to refresh bootstrap CIs (observed-target mask) and per-station metrics for all targets.==
