# SOLVIA project log

Every decision and threshold, with the reason. Newest entries at the bottom.

## 2026-10-28 — Phase 0: setup and data audit

### Folder restructure (approved by project owner)
- Project root kept as `Solar Energy` (owner's choice; the Python package is still named `solvia`).
- `Raw/` moved to `data/raw/` and renamed per the plan:
  - `household_power_consumption.txt` → `uci_household_power/household_power_consumption.txt`
  - `India Map Solar Irradiance.xlsx` → `india_solar_irradiance/india_solar_irradiance.xlsx`
  - `Electricity Demand, Solar and Wind Generation Data/` → `electricity_demand_solar_wind/`
  - `Solar_PV/` → `to_identify/solar_pv/`
  - `chstgpt datasets/` → `to_identify/chatgpt_datasets/`
  - `cluade datasets/` → `to_identify/claude_datasets/`
  - `dataset_1/` → `to_identify/dataset_1/`
- Files *inside* the `to_identify/` folders were not renamed; final names are proposed in
  `reports/data_audit.md` and wait for approval.

### Environment
- Python 3.14.6 (meets the 3.11+ requirement), venv at `.venv/`, versions pinned in `requirements.txt`.
- XGBoost not installed: HistGradientBoostingRegressor covers gradient boosting, and adding
  XGBoost on macOS needs libomp. Revisit only if needed.

### Initial thresholds (configs/config.yaml)
| Setting | Value | Why |
|---|---|---|
| `random_state` | 42 | reproducibility |
| `consumption.min_daily_coverage` | 0.90 | a day with <90% of minutes would under-count daily kWh |
| `consumption.max_interpolate_days` | 2 | longer gaps are not filled, to avoid inventing data |
| `consumption.split` | train ≤ 2008, valid 2009, test 2010 | chronological; whole years keep seasonality balanced |
| `monitoring.ratio_threshold` | 0.85 | starting point from the plan; to be checked against the data |
| `monitoring.min_consecutive_days` | 3 | one bad day is usually weather or noise |
| `energy_balance.default_performance_ratio` | 0.75 | typical rooftop PV system PR; user-editable |

Audit findings: see `reports/data_audit.md`.

### Audit outcome (details in reports/data_audit.md)
- Measured PV generation: Kaggle plants only (34 days). No long-history PVDAQ time series present.
- Plant 1 DC_POWER is ~10.2x AC_POWER (Plant 2: 1.02x). Unit inconsistency documented, not
  rescaled; AC_POWER is the target and DC_POWER is never a feature.
- Plant 2 has 3,774 daylight rows with zero AC output; this is useful for monitoring but
  needs a documented filter when training the expected-output model (decision D5).
- NCA radiation file judged unusable (night offset, 1023 saturation, 9,120 conflicting duplicate timestamps).
- PV360 labelled SYNTHETIC (provisional): provenance unverifiable; PV output also curtailed.
- Grid-India data is grid-level; never treated as household data.
- `to_identify/dataset_1/` is entirely duplicate files.
- STOPPED after Phase 0, waiting for decisions D1–D6.

## 2026-10-28 — Phase 0 decisions (project owner asked Claude to choose)
- D1: renames applied as proposed in data_audit.md §5. Duplicates moved to `data/raw/_duplicates/`,
  **not deleted** (reversible).
- D2: PV360 stays SYNTHETIC (provisional). D4: modelled capacity factors not used.
  D6: Grid-India not used.
- D5: Plant inverter rows with AC = 0 at irradiation > 0.1 kW/m² are *flagged*
  (`flag_off_in_daylight`), kept for monitoring, excluded only when training the expected-output model.
- D3: deferred to the energy-balance step (Phase 4E); external download still needs approval.

## 2026-10-28 — Phase 1: loading and cleaning
Run: `python -m solvia.pipeline --steps clean` → `data/interim/*.parquet`, `reports/cleaning_report.md`.

Decisions:
- **Daily household kWh = mean(Global_active_power) × 24** instead of sum/60. Identical on complete
  days; on days with 90–99% coverage, sum/60 would under-count by up to 10%.
- Partial first (2006-12-16) and last (2010-11-26) days trimmed → series is 2006-12-17 … 2010-11-25 (1,440 days).
- 24 days below 90% coverage set to missing; 3 days in gaps of ≤ 2 days interpolated
  (2009-08-13, 2010-03-20, 2010-03-21); 19 days in 5 longer gaps (3–6 days) left missing.
- Solar: 4 Plant 1 rows had no weather reading and were dropped by the inner join. Nothing else was removed.
- **AC_POWER unit confirmed as kW**: Σ AC × 0.25 vs the logger's DAILY_YIELD has median ratio
  1.0001 (Plant 1) and 0.9999 (Plant 2); 96.8% and 96.5% of inverter-days agree within 5%.
- Plant 2's DAILY_YIELD counter sometimes keeps the previous day's total after midnight, so the
  logger daily total is taken from afternoon rows (hour ≥ 12). Using the daily max had
  wrongly shown only 77.5% agreement.
- NITI districts: 47 exact duplicates removed; 29 districts listed with conflicting values
  averaged (spread stored in `source_spread_kwh_m2`).
- Tests: `tests/test_cleaning.py` (7 tests) pass.

## 2026-10-28 — Phases 2 and 3 (EDA and feature engineering)
Run: `python -m solvia.pipeline --steps eda features`. Full checklist: `docs/ROADMAP.md`.

EDA findings that shaped decisions (numbers from `reports/eda_summary.md`):
- Consumption: winter days average 34.0 kWh vs 17.4 kWh in summer; weekends +19% vs weekdays;
  autocorrelation lag 1 r = 0.61, lag 7 r = 0.55 → lag 1/2/7/14 and weekly features justified.
  August varies most between years (6.6 kWh/day in Aug 2008), which lag features cannot foresee.
- Distribution is right-skewed (skew 0.57) → **MAE is the headline metric** for model selection.
- Solar: operating rows correlate r = 0.986 (Plant 1) / 0.948 (Plant 2) with irradiation; output
  falls about 0.4–0.45% per °C of module temperature → module temperature kept as a feature.
- Plant 2 inverters are the same size (peak 1,173–1,385 kW). Low inverters have the most
  off-in-daylight intervals (r = −0.88), so their gap is lost output. Peer comparison is valid.
- On 24 days many Plant 2 inverters drop together, which pulls the peer median down →
  **monitoring will combine peer comparison with model-expected output** (not peers alone).
- **Threshold 0.85 kept**: the Plant 1 5th percentile of the peer ratio is 0.92, so 0.85 lies
  outside normal variation (flags 0.9% of Plant 1 and 32.0% of Plant 2 inverter-days).
- Rain: 85–89% of yearly rain falls Jun–Sep; 6 (Vadodara) / 3 (Delhi) months under 10 mm →
  the "no recent rain" condition in the cleaning rule matters mostly outside the monsoon.

Feature decisions:
- Consumption: 18 features (lags 1/2/7/14, rolling mean/std 7 & 30 on the series shifted by
  1 day, same-weekday mean of the last 4 weeks, 7-vs-30-day trend, yesterday's water-heater/AC
  sub-meter kWh and peak power, day of week, month, weekend, French public holiday, day-of-year
  sin/cos). Rolling windows need ≥ 80% of days present.
- French holidays computed in code (Easter algorithm) instead of installing the `holidays`
  package (rule: ask before installing anything outside requirements.txt).
- 97 of 1,440 days dropped (first 30 days, days around missing gaps) → 1,343 rows:
  train 709 (2007-01-10 … 2008-12-31), valid 352 (2009), test 282 (2010-01-01 … 2010-11-25).
- Solar: 6 features (irradiation, module temp, ambient temp, module − ambient, hour sin/cos);
  forbidden output-derived columns enforced in code. Split by whole days: train 24
  (15 May–7 Jun), valid 5 (8–12 Jun), test 5 (13–17 Jun). `train_eligible` excludes
  off-in-daylight rows (decision D5): 2,848 of 94,276 training rows.
- Linear models: one-hot day of week and month + StandardScaler inside the Pipeline; tree
  models use raw columns.
- Tests: 12 pass (`tests/test_cleaning.py`, `tests/test_features.py`).

## 2026-10-28 — Weather for the household (owner approved the download)
- Downloaded NASA POWER daily T2M / T2M_MAX / T2M_MIN for Sceaux (48.78 N, 2.29 E),
  2006-12-01 … 2010-11-30 → `data/raw/nasa_power/sceaux_48.78N_2.29E_daily_2006-2010.csv`.
  Checked before use: 1,461 days, full calendar, 0 missing values, min ≤ mean ≤ max on every day,
  mean 10.5 °C (range −6.8 … 26.2 °C).
- Why: heating is the main winter driver. Yesterday's temperature correlates r = −0.56 with
  daily consumption; days below 5 °C average 33.1 kWh vs 17.4 kWh above 18 °C; about
  0.81 kWh extra per 1 °C colder below the heating base (figure c08).
- Heating degree-days use the French DJU base of 18 °C (`consumption.hdd_base_c`).
- New causal features: `t2m_lag_1`, `t2m_min_lag_1`, `t2m_mean_3d`, `hdd_lag_1`, `hdd_sum_7d`
  (23 consumption features in total). Split sizes unchanged (709 / 352 / 282).
- **Forecast-proxy experiment**: `t2m_target_day`, `hdd_target_day` are the observed temperature
  of the target day, a stand-in for a next-day weather forecast. They are excluded from the main
  model and will only be reported as a separately labelled Phase 4 experiment.
- The Indian-cities NASA download for the energy balance was **not** approved yet; not downloaded.

## 2026-10-28 — India first: 48-city weather, NITI validation, plant location, energy balance
Owner asked to prioritise India and download as much Indian weather data as possible.
- **Download**: NASA POWER daily point data for 48 cities covering every region (list and reasons in
  `configs/india_locations.yaml`), 2001-01-01 … 2025-12-31 (CERES irradiance starts 2001), 9 variables:
  GHI, clear-sky GHI, clearness index, T2M mean/max/min, corrected rain, RH, wind. 26 MB.
  `python -m solvia.data.download_nasa_power` validates each file (NASA header, 9,131 rows) before
  saving and writes `_manifest.json` (URL, MD5, time).
- **Cleaning**: 438,288 rows, 0 missing, 0 min/mean/max temperature inconsistencies, 0 days with
  GHI > 1.05 × clear-sky. Monthly climatology keeps 10th/90th percentile months for ranges.
- **Validation vs NITI Aayog**: Kheda 2009–2023 monthly shape r = 0.88, NITI = 0.90 × NASA on average;
  December 2024 over 33 matching districts NITI = 0.67 × NASA (rank agreement 0.90). The single-month
  NITI file reads low, so NASA POWER is used for the energy balance; NITI is kept for ranking only.
- **Kaggle plant location (estimate)**: correlating each plant's measured daily irradiation with each
  city's NASA GHI over 15 May – 17 Jun 2020, both plants match the southern Deccan plateau best:
  Anantapur r = 0.65 (Plant 1) / 0.76 (Plant 2), temperature r = 0.87 / 0.86, vs a median of
  0.12 / 0.32 over all cities. Labelled as an estimate everywhere; NASA rain at Anantapur (84 mm over
  the 34 days) is stored as `plant_estimated_weather` ("ESTIMATE") for the rain-effect analysis.
- **Superseded**: the two short Vadodara/Delhi NASA files and the old `location_irradiance` table.
- **Energy balance (Phase 4E)** implemented in `rules/energy_balance.py`: monthly kWp × GHI × PR (0.75),
  poor/good-month range, kWp needed. Sanity check: 3 kWp gives ~1,567 kWh/kWp/yr in Jaisalmer and
  ~1,327 in Delhi, in line with typical Indian rooftop yields. 4 tests.
- Phase 4 order changed to India first: B (solar model) → C (monitoring) → D (maintenance) → A (consumption).
- Tests: 16 pass.

## 2026-10-28 — All remaining datasets downloaded; second cleaning / preprocessing / feature pass
Owner: "first download all the datasets that are required for the project", then redo cleaning,
preprocessing and feature engineering over old and new datasets together.

**Downloaded** (each in its own `data/raw/` folder with a manifest; nothing existing was changed):
- **Prayas eMARC smart meters** (Harvard Dataverse doi:10.7910/DVN/YJ5SP1, v1.1): household info
  (273 deployments), daily kWh (111,280 rows, 216 meters, 128 households), 15-minute block load
  (12,702,801 rows, 368 MB). All three MD5s equal Dataverse's. Licence CC0 on Dataverse; terms of use
  ask for non-commercial academic use and acknowledgement of Prayas (Energy Group). **This is measured
  Indian household consumption**, so objective A can now be done for India first.
- NASA POWER daily 2001–2025 for **Aurangabad** and **Gonda** (eMARC regions; now 50 cities).
- NASA POWER **full 9-variable Sceaux file** (`..._full.csv`), same T2M as before plus GHI, rain, RH, wind.
- NASA POWER **1° grid**, 169 cells over 10–22 °N, 72–84 °E, May–June 2020 (plant-location estimate).
- **French school holidays** 1990–2028 (Etalab compilation, Licence Ouverte). The official
  data.education.gouv.fr calendar has no mainland records before 2013, so the 2006–2010 dates could
  not be cross-checked; Sceaux (académie de Versailles) is taken as zone C.

**eMARC decisions** (`configs/config.yaml: emarc`):
- Each meter is its own series. Households are never added together into one "household"; a later
  model may learn across meters, with the meter id kept and per-meter metrics reported.
- Info sheet = master metadata. One conflict: H128 (D0239/D0240) is "Water Heaters but no AC" in the
  daily file but "With Air Conditioners" in the info sheet → info sheet used, `meta_conflict` flag set.
- 7 exact duplicate rows dropped (H077, Feb 2019). Complete calendar per meter (6,980 unreported days added as missing).
- Zero days: 1,235 in total. Runs of ≥ 3 consecutive zero days (1,103 meter-days) → missing (meter or
  supply off, or house empty); shorter runs kept as real (full-day power cuts happen, especially in UP).
- Outliers: 81 days above Q3 + 6 × IQR of their meter are flagged, not removed; the largest are an AC
  home in May (up to 44.6 kWh), which is plausible.
- Gaps ≤ 2 days interpolated (264 meter-days); interpolated days are never used as targets.
- 15-minute blocks (0..95): daily energy = mean kW × 24 agrees with the cumulative-meter daily file,
  median ratio 0.997, 97.4 % of days within ±10 %. Load-shape features need ≥ 90 of 96 blocks.
- Model-eligible: mains meters with ≥ 180 measured days → **112 of 116** meters, 58,440 measured meter-days.
- Weather: NASA POWER at the mapped city (Pune city and rural Pune → Pune; Aurangabad; Kanpur rural → Kanpur; Gonda).
  Cooling degree-days base 24 °C.
- Split by date: train ≤ 2019-06-30 < valid ≤ 2019-12-31 < test (2020-01-01 … 2020-06-30).
  Rows 19,896 / 16,708 / 16,052 (84 / 107 / 104 meters). Test is labelled `normal` (7,805) vs
  `lockdown` (from 2020-03-25, 8,247) and will be reported separately: median use of 54 comparable
  homes was +7 % in the 2020 lockdown window vs the same days of 2019 (not weather-adjusted).
- 29 India features: history (10), load shape (4), calendar incl. Indian national days and major
  festivals (6), weather (7), household type and region (2). Target-day weather kept apart as a
  forecast-proxy experiment, as for France.

**UCI updates**: weather from the full Sceaux file; new features `is_school_holiday` (zone C) and
`ghi_lag_1` → 25 consumption features. Split sizes unchanged (709 / 352 / 282).

**Plant location on the 1° grid** (figure i08): r-weighted centre of the 5 best cells is 13.6 °N 77.0 °E
(Plant 1, best r = 0.70, less certain) and 14.6 °N 77.2 °E (Plant 2, best r = 0.78), consistent with the
Anantapur city match. Still an estimate; the city match remains the source of `plant_estimated_weather`.

**EDA findings (eMARC)**: median household 3.9 (Basic) / 4.7 (water heater) / 5.8 (AC) kWh/day; use peaks
in May–Aug (1.36–1.47× the home's average), opposite to France; above 30 °C each extra degree adds
~2.9 % (Basic) to ~6.2 % (AC); median per-meter correlation with daily max temperature 0.45. Only 25–29 %
of use falls in 09:00–16:00, so most rooftop solar would be exported without a battery.

Tests: 25 pass.

## 2026-10-28 — Phase 4B: expected-solar model
Command: `python -m solvia.pipeline --steps model_solar` → `reports/solar_model_report.md`, `reports/metrics/solar_*`,
`models/solar_expected_output.joblib` + `.json`. Validation split only; **test not touched** (Phase 5).
- **Decision D7, healthy reference**: an inverter-day is healthy if its specific yield (kWh per kWh/m²) is
  ≥ 0.90 × the plant's 90th-percentile inverter on the same day (same-day comparison removes weather; the
  p90 is robust to plant-wide faults; peer median is not, Plant 2 medians fall to 732 on 24 May while its p90
  stays at 1,352). 0.90 = Plant 1 training 5th percentile (0.908), rounded down. Healthy share of daytime
  rows: Plant 1 96.5 % (train) / 95.5 % (valid); Plant 2 52.4 % / 69.2 %. The label uses actual output, so it
  only selects training rows and an evaluation subset; it is never a feature.
- Model domain: daytime rows (irradiation > 0); night → 0 kW by rule (only 14 of 59,374 night rows had any power).
- Candidates: k × irradiation baseline, Linear Regression, Random Forest (200 trees), HistGradientBoosting (300 iterations);
  trained on all operating rows or on the healthy reference; one pooled model or one model per plant (16 combinations).
- Tuning: GridSearchCV with 4-fold expanding-window CV over whole training days (8→12→16→20 train days,
  4 validation days each, no shared days; verified).
- **Selected: HistGradientBoosting, per plant, healthy reference** (learning rate 0.05, 15 leaves, min 100
  samples per leaf for both plants). Validation, healthy inverter-days, both plants: daily MAE **122.5 kWh**
  per inverter-day (MAPE 1.96 %, R² 0.991); 15-minute daytime MAE 33.6 kW (R² 0.961).
  Baseline k × irradiation: 311.1 kWh (5.55 %); the selected model is 61 % lower.
- Healthy-reference training beats all-operating training for every model type, **even when scored on all
  operating inverter-days** (Plant 2 daily MAE 218.0 vs 275.1 kWh): faulty inverters make the target noisy.
- Cross-plant (healthy validation days): Plant 1 → Plant 2 daily MAE 152.7 kWh vs 118.1 own; Plant 2 → Plant 1
  190.9 vs 125.6. The shape generalises (R² ≥ 0.95) but each plant's own model is better, so per-plant
  models are kept; a new plant would need a few healthy days of its own data.
- Reproducibility: re-running the step gives identical metrics (random_state = 42).

## 2026-10-28 — Phases 4C (monitoring) and 4D (maintenance)
Command: `python -m solvia.pipeline --steps monitor maintenance` → `reports/monitoring_report.md`, figures m01–m04,
`data/interim/monitoring_*.parquet`, `maintenance_recommendations.parquet`, `reports/metrics/monitoring_*`, `maintenance_summary.json`.
- **Out-of-sample expected output** (leakage check 10): the 4B configuration is refitted for every day on
  earlier days only (expanding window, 7-day warm-up) → monitored 2020-05-22 … 06-17, 27 days, 1,160 inverter-days.
  Healthy inverter-days: median PR 1.007 (Plant 1), 0.981 (Plant 2); Plant 1 5th–95th pct 0.942–1.038.
- Rules: PR < 0.85 = low; peer outlier = robust z < −3 **and** PR < 0.95 (new `peer_max_pr`: Plant 1's spread is so
  tight (daily MAD 0.017) that z = −3 is ~5 % below peers; before the gate 48 of 54 Plant 1 flags were that alone);
  alert = low ≥ 3 consecutive days. Status: Plant 1 544 ok / 50 watch / 0 alert (41 of the 50 from its two
  chronically weaker inverters, peer medians 0.92–0.93); Plant 2 354 / 130 / 82, alerts on 14 of 22 inverters.
- **Simulated faults** (healthy Plant 1 inverter-days scaled down): false alarms 0 %; one day flags 5.4 % of
  10 % losses, 40.4 % of 15 %, 99.8 % of 20 %, 100 % of ≥ 30 %.
- Isolation Forest (same flag rate 18.6 %): Jaccard 0.846 with the PR rule; kept as a comparison only.
- **Finding: Plant 2's loss is inverters switching off in bright conditions, not dust.** Plant PR 0.797 but
  0.959 while inverters run ("running PR", new); shutdowns explain 85 % of its lost energy. Off share rises with
  irradiation (often-low inverters 57 % of intervals above 1.0 kW/m², other Plant 2 inverters 26 %, Plant 1 ~1 %),
  peaks 10:00–13:00, module temperature the same off vs on (56.0 vs 55.2 °C), median 7 of 22 inverters off together
  (not a plant outage). Cause (grid over-voltage / overload / over-temperature trips or curtailment) is not
  identifiable from the data → recommendation: read the inverters' event logs.
- **Rain effect (estimate)**: Plant 2 PR is higher after rain (0.905 vs 0.775) but those days are dimmer
  (4.04 vs 5.71 kWh/m²) and dim days have fewer shutdowns; running PR shows no link to rain (r = 0.01 Plant 2,
  0.19 Plant 1). No evidence of soiling or rain washing in 27 days; with 5 rainy days, weak evidence either way.
- **4D rules** (`rules/maintenance.py`): R1 offline ≥ 1 h in daylight; R2 low ≥ 3 days while not plant-wide
  (shutdowns → event log / trip settings, steady shortfall → strings); R3 plant-wide loss (rain decides dust vs
  other); R4 falling *running* PR (so trips never trigger a cleaning advice); R5 sensor check; R6 dry-month
  cleaning (Anantapur dry months from climatology: Jan, Feb); R7 inverter not reporting (added after finding 4
  Plant 2 inverters silent 20–28 May); R0 no action. Fired: Plant 1 R1 on 6 plant-days, R0 on 25; Plant 2 R1 126,
  R2 79 (14 inverters, all shutdown type), R7 28 (4 inverters × 7 days), R0 1. R3, R4, R5 did not fire.
- An earlier version of R4 used plant PR and advised cleaning Plant 2 on 4 days; it was wrong (trips, not dust)
  and was changed to running PR before any results were reported.
- Tests: 42 pass (monitoring 5, maintenance 6 new).

## 2026-10-28 — Phase 4A (next-day consumption, India first, then France) and 4E upgrade (direct solar use)
Commands: `python -m solvia.pipeline --steps model_consumption_india model_consumption_france energy_balance`
→ `reports/consumption_{india,france}_report.md`, `reports/metrics/consumption_*`, `models/consumption_{india,france}.joblib` + `.json`,
`reports/energy_balance_report.md`, `reports/metrics/energy_balance_examples.json`, `data/interim/energy_balance_shapes.parquet`.
- **India (eMARC, 107 meters in validation, one model across meters, each row is one meter-day; no pooling into one household).**
  Train 19,896 meter-days (to 2019-06-30), validation 16,708 (2019-07-01 … 12-31); test (2020, includes lockdown) untouched.
  Selected **Ridge (alpha 1000): validation MAE 0.716 kWh/day** (R² 0.875, MAPE 15.9 % on days ≥ 1 kWh), 6.1 % lower than
  the best baseline (mean of last 7 days, 0.763); HGB on the ratio to the meter's 30-day mean 0.726, Random Forest 0.733,
  HGB 0.749; yesterday 0.798, same day last week 1.060. Median per-meter MAE 0.579 (baseline 0.633); the model beats
  the baseline on 78 % of meters. Meters not seen in training: MAE 0.841 vs baseline 0.899 (seen: 0.673 vs 0.724).
  Gonda is hardest (1.269); Pune / Pune city easiest (0.58 / 0.56).
- **CV caveat (India):** the first expanding-window fold trains on only 105 days of 36 meters (Jan–May 2018) and is scored on
  the following summer. Ridge extrapolates there (fold MAE 2.28 vs 0.89 Random Forest); on later folds it matches the tree models.
  This fold pushed Ridge's alpha to 1000. Selection stayed on validation MAE as planned, but Ridge is sensitive to
  short history: a new deployment with little data should be checked before relying on it.
- Ridge's alpha grid was widened from ≤ 100 to ≤ 1000 after both tracks picked the edge value (training-split CV only;
  France still picks 100).
- **France (UCI, one household):** train 709 days, validation 352. Selected **Ridge (alpha 100): MAE 4.566 kWh/day**
  (R² 0.555, MAPE 18.4 %), 5.4 % lower than the 7-day mean (4.825); Random Forest 4.592; HGB 5.158 (worse than the
  baseline: too little data for boosting).
- **Prediction intervals (quantile HGB 10th/90th):** India covers 75.6 % of validation days (target 80 %, mean width 2.11 kWh);
  France only **54.8 %** (width 9.02 kWh): the quantile models are over-confident, badly so on 709 training days. Not
  fixed yet; calibrating the interval (e.g. conformal adjustment on validation, coverage checked on test) is a Phase 5 option.
- **Forecast-proxy experiment (labelled; target-day observed weather as an upper bound for a forecast):** India MAE
  0.716 → 0.714, France 4.566 → 4.592. Same-day weather adds almost nothing beyond yesterday's weather and the history.
- **4E upgrade, direct use:** measured shapes, 15-minute: solar from healthy Kaggle inverters (81 % of energy 09:00–16:00), load per
  eMARC household type. Typical Pune home of each type (eMARC median daily use 3.94 / 4.65 / 5.79 kWh), sized to its
  yearly use (1.5 / 1.5 / 2.0 kWp), no battery: panels cover 41.3 / 47.4 / 43.0 % of use directly; 28.6 / 38.6 / 32.7 % of the
  panels' output is used in the home, the rest exported. All solar figures are estimates.
- Bug fixed during the run: the per-meter breakdown overwrote the selected-model name in the summary (report crashed);
  keys renamed, results unchanged. Per-fold CV MAE now stored and shown in the reports.
- Tests: 48 pass (consumption 4, energy-balance direct use 2 new).

## 2026-10-28 — Phase 5 (single test-set evaluation)
Command: `python -m solvia.pipeline --steps evaluate` → `reports/evaluation_report.md`, `reports/model_card_*.md` (3),
figures p01–p09, `reports/metrics/evaluation_*.csv`, `evaluation_summary.json`, `test_run_log.json`; `test_metrics` added to `models/*.json`.
- **Decided before opening the test split:** score the saved Phase 4 models unchanged (trained on train, chosen on
  validation); refit the other candidates with their stored settings only to fill the tables; permutation importance
  on validation; record model-file hashes and refuse a later test run with different models (retraining was checked to
  give byte-identical files, so a normal pipeline re-run passes the guard).
- **Interval fix before the test run:** conformalized quantile regression. On validation, each end of the 10th–90th
  interval is moved out by the amount needed for 80 % coverage (India +0.08 kWh, France +2.68 kWh), stored in the model.
- First test run 2026-10-28 03:52 UTC. Re-runs afterwards only changed figure layout and model-card text (same model files, same numbers).
- **Solar (5 test days, 13–17 Jun):** healthy inverter-days, daily MAE **176.8 kWh** (MAPE 2.76 %, R² 0.964) vs 122.5 on
  validation; baseline k × irradiation 365.6. 15-minute daytime MAE 33.0 kW (validation 33.6). Random Forest was
  slightly lower on test (174.2) but higher on validation (136.5); selection is not revisited. Error peaks at midday
  and is higher for Plant 2 (p03).
- **India (test Jan–Jun 2020, 104 meters, 28 never seen in training):** Ridge MAE **0.711 kWh/day** (validation 0.716),
  R² 0.909; baseline (7-day mean) 0.746 → 4.7 % better. Normal days 0.673 (baseline 0.694), **lockdown 0.748 (0.796)**:
  the model stays better than the baseline through a behaviour change it never saw. New meters 0.897 vs 0.912 baseline
  (seen 0.648 vs 0.690). Beats the baseline on 70 % of meters. Gonda / Aurangabad hardest (≈ 1.0), Pune city easiest (0.558).
  80 % interval: raw 77.2 % → calibrated **81.2 %** (width 2.57 kWh); normal days 78.8 %, lockdown 83.5 %.
- **France (test 2010, 282 days):** Ridge MAE **4.259 kWh/day** (validation 4.566), R² 0.490; baseline 4.375 → 2.7 % better.
  Random Forest 4.034 on test (validation 4.592): with one household the ranking between the two is within noise.
  80 % interval: raw 64.2 % → calibrated **83.0 %** (width 14.31 kWh). October is the worst month (MAE 6.6).
- Permutation importance (validation): solar is almost entirely irradiation; India lag_1 ≫ lag_2, 7- and 30-day means;
  France is_weekend, school holidays, same-weekday mean (Ridge spreads credit over correlated lags).
- Leakage table complete (rows 9, 9b ✅; 9c disclosed: monitoring thresholds were set on a window that contains the solar test days).
- Tests: 52 pass (conformal 2, test guard 2 new).
- **Stop here for the owner's review before Phase 6 (dashboard).**

## 2026-10-28 — Phase 6 (Streamlit dashboard), after the owner's review of Phase 5
Command: `streamlit run app/Home.py`. Code: `app/Home.py`, `app/pages/1_…7_*.py`, `app/common.py`, data layer
`src/solvia/dashboard/data.py` (reads pipeline outputs and applies the saved models; never trains, never writes).
- Pages: Overview · Consumption Forecast (India first, France) · Solar Generation · Performance Monitor · Maintenance ·
  Energy Balance · Model Performance · SYNTHETIC demo.
- Labels: every section carries MEASURED / MODEL / ESTIMATE / SYNTHETIC badges with their meaning. Colors follow the
  kind of number (measured blue, model orange, baseline gray); status and severity always icon + text.
- Consumption shows validation and test days only (out-of-sample); India is one meter at a time (region → type → meter),
  flags meters not seen in training, and lists only meters with data in the chosen split (first version opened on a
  meter with no test days; fixed). Solar warns when training days (in-sample) are selected.
- Energy balance runs the Phase 4E functions live for any of the 50 cities, household type (eMARC load shape), daily use,
  kWp (default: smallest 0.5 kWp step covering the yearly use) and performance ratio; ESTIMATE banner throughout.
- PV360 is read from `data/raw/unverified/…/PV360_data.zip` without changes and shown only on the SYNTHETIC page
  (red banner, watermark on the chart).
- Inverter ids shown as P1-01 … P2-22 (sorted source ids within each plant).
- Checks: the data layer reproduces Phase 5 (India test MAE 0.711, coverage 81.2 %; France 4.259, 83.0 %; Pune Basic
  self-sufficiency 41.3 %); all 8 pages and their main controls run without exceptions in Streamlit AppTest; the server
  starts and serves pages (HTTP 200). Tests: 62 pass (dashboard 10 new). No new packages (streamlit, plotly were already pinned).

## 2026-10-28 — Phase 7 (tests and documentation)
- **End-to-end run:** `python -m solvia.pipeline` (clean → eda → features → model_solar → monitor → maintenance →
  energy_balance → model_consumption_india → model_consumption_france → evaluate) finished in 291 s (wall clock),
  exit 0. Every model file is byte-identical to the one scored in Phase 5 (SHA-256 compared before/after), so the
  test guard accepted the run; the SHA-256 of all 307 files in `data/raw/` is unchanged. eMARC cleaning reproduced
  the same counts (216 deployments, 112 model-eligible meters).
- **New tests:** `test_model_loading.py` (metadata complete and scikit-learn version matches; solar model 0 at night and
  never negative; consumption models give finite, non-negative forecasts and ordered intervals on test rows; model
  features match the feature code). Total **68 tests pass**.
- Checks of the engineering rules: no `print` in `src/` or `app/`; every `random_state` is 42.
- **README rewritten** for readers of the project: results table (test split), setup, commands, dashboard, method,
  layout, data sources and licences, limitations (complete), acknowledgements (Prayas credited).
- `docs/NEXT_SESSION_PROMPT.md` marked as historical with a current first-task section.
- Not done (needs the owner): git repository; checking the hand-entered Indian festival dates against an official
  list; confirming the Kaggle licence; optional notice to Prayas.

## 2026-10-28 — Open items closed
- **Indian festival dates verified.** All 12 dates of the eMARC period match the DoPT holiday lists for central
  government offices (Delhi/New Delhi): Holi 2018-03-02, 2019-03-21, 2020-03-10; Idu'l Fitr 2018-06-16,
  2019-06-05, 2020-05-25; Dussehra 2018-10-19, 2019-10-08; Diwali 2018-11-07, 2019-10-27 (Annexure I, gazetted);
  Ganesh Chaturthi 2018-09-13, 2019-09-02 (Annexure II, restricted). Sources: DoPT orders as reproduced on
  staffnews.in (2018 OM F.No.12/3/2017-JCA-2; 2020 Delhi list) and gconnect.in (2019). No code or data change needed.
- **Kaggle licence confirmed** from Kaggle's dataset metadata: "Data files © Original Authors" (owner Ani Kannal,
  version 1, 11,233,523 bytes = our archive). Not an open licence: academic use with credit; the raw files are not
  redistributed (data/raw/ is excluded from git).
- Prayas notice left to the owner (it is an email from them).
