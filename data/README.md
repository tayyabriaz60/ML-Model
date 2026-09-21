# Data

Production path expected by `code/src/config.py`: `KAPSARC_RAW` or `code/data/raw`.

Paper-scale file (not in this folder yet):

- 29 stations
- 1 Jan 2009 – 24 May 2019
- ~2,775,657 complete hourly rows after cleaning
- Visibility in **metres**

## Local extract from the client zip (not production)

`raw/saudi_weather_historical_subset.csv` — 4 stations, 2018–2019, 63,266 rows.

Observed ranges on that extract:

- `air_temperature` 3–49 °C (plausible)
- `sea_level_pressure` 990.9–1033.0 hPa (plausible)
- `visibility` **0.0–0.9** (not metres; do not feed this into `VISIBILITY_DISTANCE` until the unit is proven)

The 458 MB zip also contains 3–4 GB QC flagged CSVs. Do not extract them. Do not run that repo’s anomaly stages for this revision.
