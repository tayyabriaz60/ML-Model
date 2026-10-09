### 5.4 Agreement and divergence

The claim “>0.90 agreement between SHAP, LIME and attention” is withdrawn. We report, per target and horizon, on the same test partition:

- Jaccard overlap of the top-k feature sets, k ∈ {5, 10, 15}
- Spearman rank correlation on the shared support
- Rank-biased overlap with p = 0.9

Median Jaccard@10: temperature [[AGREE_TEMPERATURE_H1_MEDIAN_J10]]; visibility [[AGREE_VISIBILITY_H1_MEDIAN_J10]]; pressure [[AGREE_PRESSURE_H1_MEDIAN_J10]] (Table 13, Fig. 7). ==Attention over lag positions is excluded from feature-set overlap because it is a distribution over time steps, not over feature names; permutation importance (Fig. 5) enters instead.==

==**Pressure caveat (R2-5).** Sequence models for pressure at h = 1 are worse than persistence (GRU skill [[PRESSURE_H1_GRU_SKILL]]; LSTM skill [[PRESSURE_H1_LSTM_SKILL]]). The headline median Jaccard@10 for pressure ([[AGREE_PRESSURE_H1_MEDIAN_J10]]) is driven by SHAP versus LIME on the tree model and by perturbation pairs on degenerate sequence predictors; SHAP versus LIME alone is [[AGREE_PRESSURE_H1_SHAP_LIME_J10]]. We do not interpret 0.000 overlap as agreement between SHAP and LIME on pressure; we report the pair breakdown in Table 13 and Fig. 7.==

Table 14 lists consensus versus method-specific features.
