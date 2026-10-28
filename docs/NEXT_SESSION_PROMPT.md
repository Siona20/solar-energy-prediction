# SOLVIA: project status and instructions for the next session

> Paste everything below into a new Claude Code session opened in the project folder.
> **Updated 2026-10-28: ALL PHASES 0–7 ARE COMPLETE.** Sections 6–7 below describe the plan as it was before
> Phase 4 and are kept for history; the current state is in `README.md`, `docs/ROADMAP.md` and `docs/PROJECT_LOG.md`.
> Test results: `reports/evaluation_report.md`. Section 9 has the current first task.

---

## 1. Who I am and what this project is

I am a final-year B.Tech CSE (Data Science & ML) student. My project is **SOLVIA, the Smart Solar Energy
Forecasting & Maintenance System**. It does five things:

- **A.** Forecasts next-day household electricity consumption.
- **B.** Estimates the solar generation a plant *should* produce from the weather (expected output).
- **C.** Flags days when actual solar output falls below that expectation (performance monitoring).
- **D.** Gives transparent, rule-based maintenance recommendations with reasons (cleaning, inspection, inverter check).
- **E.** Estimates monthly energy surplus or shortfall for a user scenario (kWp, city, daily use).

All five are shown in a Streamlit dashboard (Phase 6). **Priority: India first.** Indian data and models come
before the French (UCI) household, which is kept as a secondary, comparison track.

Project root: `/Users/samarthmaru/Desktop/Solar Energy ` (the folder name ends with a **space**; do not rename it).

---

## 2. Rules you must follow (from my original brief, non-negotiable)

1. **Never modify, move, rename or delete anything in `data/raw/` without my approval.** New downloads go in new folders or under new file names.
2. **Never combine datasets from different locations, systems or periods as if they were one household.** Each eMARC meter is its own series. UCI (France) is never merged with Indian data. The energy balance is a *scenario*, not one real home.
3. Synthetic or unverified data (PV360) is labelled **SYNTHETIC**: dashboard testing only, never in reported results.
4. Never present irradiance-based estimates as measured generation. Label every estimate.
5. **Report only numbers produced by code you actually ran.** If something fails or the data cannot support an objective, tell me instead of working around it silently.
6. Ask before installing anything outside `requirements.txt`, downloading external data, or deleting any file.
7. Engineering rules:
   - Use `random_state = 42` everywhere.
   - Build every path with pathlib from `configs/config.yaml`.
   - Use logging, not `print`, inside `src/`.
   - Use **chronological splits only**.
   - **Use the test set once**, in Phase 5.
   - Keep scalers and encoders inside sklearn Pipelines.
8. **STOP after Phase 5** and show me the results before building the dashboard.
9. Match the existing code style (src layout, small functions, `CleaningLog` for every data step, computed takeaways in figures).

---

## 3. Environment and commands

- Python 3.14 in `.venv`, package installed with `pip install -e .`
- Libraries: pandas 3.1, numpy, scikit-learn 1.9.1, pyarrow, matplotlib (Agg backend), openpyxl, pytest. Versions are pinned in `requirements.txt`.

| Task | Command |
|---|---|
| Full pipeline | `python -m solvia.pipeline` (steps: `clean`, `eda`, `features`) |
| One step | `python -m solvia.pipeline --steps clean` |
| Re-download NASA weather | `python -m solvia.data.download_nasa_power` (skips files that already exist) |
| Phase 0 audit | `python -m solvia.data.audit` |
| Tests | `pytest` (**25 pass**) |

---

## 4. Folder structure (current)

```
configs/config.yaml            paths, thresholds, split dates, emarc settings, random_state
configs/india_locations.yaml   50 Indian cities + Sceaux extra point + 1° plant search grid
data/raw/                      READ-ONLY originals (see section 5)
data/interim/                  cleaned parquet tables
data/processed/                model-ready features: consumption_features, india_consumption_features, solar_features_15min
docs/PROJECT_LOG.md            every decision, threshold and number, with reasons
docs/ROADMAP.md                checklist of all phases (ticked = done)
notebooks/                     02_eda_consumption, 03_eda_india, 04_eda_solar_plants, 05_eda_india_households
src/solvia/
  config.py                    load_config(), project_path(), get_path()
  data/                        loaders: uci, kaggle_solar, nasa_power, niti, emarc, download_nasa_power, audit
  preprocessing/               household, solar, india_weather, irradiance, emarc, cleaning_log
  features/                    consumption (UCI), india_consumption (eMARC), solar, calendar, preprocess (make_pipeline)
  eda/                         consumption, solar, india, india_households, irradiance, report
  rules/energy_balance.py      objective E (done)
  viz.py                       shared palette and style
  pipeline.py                  step_clean / step_eda / step_features
models/                        (empty: Phase 4/5)
reports/                       data_audit, cleaning_report, eda_summary, features_report, leakage_checks, figures/
tests/                         test_cleaning, test_features, test_energy_balance, test_india
app/                           (empty: Phase 6 Streamlit)
```

---

## 5. Datasets

| Dataset | Folder in `data/raw/` | Role |
|---|---|---|
| **Prayas eMARC smart meters (India)**: 216 meters, 128 homes, Pune / Aurangabad / Kanpur rural / Gonda, 2018-01 to 2020-06. Daily kWh + 15-min block load (12.7 M rows) + household info. DOI 10.7910/DVN/YJ5SP1 | `prayas_emarc/` | **Indian household consumption (objective A, India)**. Non-commercial academic use; must credit Prayas (Energy Group). |
| **Kaggle Solar Power Generation**: 2 Indian plants × 22 inverters, 15-min, 2020-05-15 to 06-17 (34 days) | `kaggle_solar_plants/archive.zip` | **The only measured PV generation** (objectives B, C, D) |
| **NASA POWER daily, 50 Indian cities, 2001–2025**, 9 variables (GHI, clear-sky GHI, clearness, T2M mean/max/min, rain, RH, wind) | `nasa_power/india_cities/` | India weather, energy balance, eMARC weather |
| NASA POWER 1° grid, 169 cells, May–Jun 2020 | `nasa_power/plant_search_grid_2020/` | Plant-location estimate |
| UCI household, Sceaux, France, 1-min, 2006–2010 | `uci_household_power/` | Consumption (France, secondary) |
| NASA POWER Sceaux full (9 vars) 2006–2010 | `nasa_power/..._full.csv` | UCI weather features |
| French school holidays 1990–2028 (Etalab) | `french_school_holidays/` | UCI `is_school_holiday` (zone C) |
| NITI Aayog ICED insolation | `india_solar_irradiance/` | Validation of NASA (Kheda r = 0.88) and district ranking |
| Grid-India, modelled PV capacity factors, NREL PVDAQ metadata | various | Audited, not used |
| PV360 (Philippines) | `unverified/` | **SYNTHETIC (provisional)**: dashboard testing only |
| NCA Sardar Sarovar | `unusable/` | Unusable (sensor faults) |

---

## 6. What has been done (Phases 0–3 ✅ for all datasets)

### Phase 0: audit ✅
- Every raw file profiled and classified (`reports/data_audit.md` plus two addenda).
- Duplicates moved to `data/raw/_duplicates/`.

### Phase 1: cleaning ✅ (`reports/cleaning_report.md`)

**UCI (France)**
- Day-first dates; `?` read as missing.
- Daily kWh = mean(kW) × 24, so the total is coverage-adjusted.
- Days with less than 90% minute coverage set to missing; gaps of 2 days or less interpolated.

**Kaggle solar plants**
- Per-file date formats; kWh = AC kW × 0.25.
- Plant 1 DC is about 10× AC. This is a unit issue; it is documented and DC is never used.
- DAILY_YIELD counter fix: read it from rows at hour ≥ 12 (agreement went from 77.5% to 96.5%).
- `flag_off_in_daylight` (AC = 0 while irradiation > 0.1): kept for monitoring, excluded from training (`train_eligible`).
- Daily table per inverter and per plant.
- Plant 2 under-performance confirmed as real lost output.

**NASA India**
- 0 missing values; consistency checks pass.
- Monthly climatology with p10/p90 months; one summary row per city.

**Kaggle plant location (ESTIMATE)**
- Southern interior Deccan. Best city: Anantapur (r = 0.65 / 0.76).
- 1° grid centre: 13.6°N 77.0°E (Plant 1, less certain) and 14.6°N 77.2°E (Plant 2).
- NASA rain at Anantapur is stored as "estimated rainfall".

**eMARC**
- Info sheet used as master metadata (H128 type conflict flagged).
- 7 duplicates dropped; complete calendar per meter.
- Runs of 3 or more zero days → missing; single zero days kept as real power cuts.
- 81 high days flagged, not removed.
- Gaps of 2 days or less interpolated, and never used as targets.
- 15-min blocks agree with the daily file (median ratio 0.997).
- Appliance meter joined to its home's mains meter.
- **112 of 116 mains meters model-eligible** (180 or more measured days).

### Phase 2: EDA ✅ (`reports/eda_summary.md`, figures in `reports/figures/`)

| Figures | Topic |
|---|---|
| c01–c08 | UCI: yearly cycle, weekly pattern, sub-metering, temperature effect (about 0.81 kWh per °C colder) |
| s01–s07 | Solar: power curves, AC vs irradiation, inverter peer ratio (threshold 0.85 checked against data) |
| i01–i08 | India: city ranking, monthly heatmap, rain and dry months, NASA vs NITI, plant location (city + grid) |
| e01–e06 | eMARC (findings below) |

eMARC findings:
- Median daily use: 3.9 / 4.7 / 5.8 kWh/day for Basic / water-heater / AC homes.
- Use peaks May–Aug at 1.36–1.47× the home's average.
- Above 30 °C, each extra degree adds about 2.9% (Basic) to 6.2% (AC).
- Only 25–29% of use falls in 09:00–16:00, the solar window.
- Lockdown: median use +7% compared with the same days of 2019.

### Phase 3: features ✅ (`reports/features_report.md`, `reports/leakage_checks.md` rows 1–7e ✅)

**UCI: 25 features, split 709 / 352 / 282**
- Features: lags 1/2/7/14, rolling mean/std 7/30, same-weekday mean, trend, sub3 and peak lag, calendar plus French and school holidays, past temperature / HDD / GHI.
- Split by date: train ≤ 2008, valid 2009, test 2010.

**India eMARC: 29 features, 52,656 rows**
- Features: same history features, load shape (peak, base, evening share), Indian holidays, CDD (base 24 °C), temperature, humidity, rain, household type and region.
- Split by date:

  | Split | Dates | Rows | Meters |
  |---|---|---|---|
  | train | up to 2019-06-30 | 19,896 | 84 |
  | valid | 2019-07 to 2019-12 | 16,708 | 107 |
  | test | 2020-01 to 2020-06 | 16,052 | 104 |

- Test rows are labelled `normal` (7,805) or `lockdown` (8,247, from 2020-03-25).

**Solar: 6 features, split 24 / 5 / 5 whole days**
- Features: irradiation, module temp, ambient temp, module − ambient, hour sin/cos.
- `FORBIDDEN` output columns are blocked in code.

**Forecast-proxy features (target-day weather)** are kept outside `FEATURES`, for one labelled experiment only.

### Objective E: energy balance ✅
- `rules/energy_balance.py`: kWp × GHI × PR (0.75) per month, with p10/p90 range and kWp needed to cover use.
- Sanity check: about 1,567 kWh/kWp/yr in Jaisalmer and about 1,327 in Delhi.

---

## 7. What is left (in this order)

### Phase 4: modelling, India first. **START HERE with 4B.**

**4B. Expected-solar model** (`data/processed/solar_features_15min.parquet`, train on `train_eligible` rows)
1. Baseline: `AC = k × irradiation`, with k fitted on train.
2. Models: Linear Regression, Random Forest, HistGradientBoosting, all built with `features.preprocess.make_pipeline`.
3. Metrics on **daytime rows only** (irradiation > 0) and on **daily kWh per inverter**: MAE, RMSE, R².
4. Cross-plant check: train on Plant 1, test on Plant 2, and the reverse.
5. Tuning with **grouped-by-day time-series CV inside train only** (leakage row 8).
6. Compare training on all operating rows vs a "healthy reference" (inverters close to their peers); pick by validation.
7. Save to `models/` with joblib plus a metadata JSON (features, dates, metrics, library versions).

**4C. Performance monitoring**
- Use only **out-of-sample** expected output (leakage row 10).
- Performance ratio = actual / expected, plus a robust peer z-score (threshold −3).
- Low = ratio < 0.85 for **3 or more consecutive days** (config `monitoring`).
- Combine peer ratio with model-expected output, because a plant-wide drop pulls the peer median down.
- Compare the rules with Isolation Forest.
- Rain effect: does output per unit of sunlight change after estimated rain days? Label it an estimate.

**4D. Maintenance rules** (`src/solvia/rules/maintenance.py` + tests). Each rule outputs an action, a reason and its evidence:
- Gradual decline across 60% or more of inverters → likely soiling → recommend cleaning (consider estimated rain and dry months).
- A sudden drop on a single inverter → inverter/string check.
- `flag_off_in_daylight` → inverter outage.
- Config `maintenance`: `days_since_cleaning` 30, `recent_rain_mm` 5.

**4A-India. eMARC consumption forecast** (`data/processed/india_consumption_features.parquet`)
1. Three baselines: yesterday, same weekday last week, 7-day mean.
2. Models: Ridge, Random Forest, HistGradientBoosting, trained as one model across meters. Meter-level features only; meters are never summed.
3. One-hot encode `household_type` and `region` inside the Pipeline.
4. Select by validation MAE.
5. Report overall **and per-meter** MAE.
6. Test is split into normal vs lockdown days, and into meters seen in train vs new meters.
7. Extras: quantile HistGB intervals (p10–p90); forecast-proxy experiment, clearly labelled.

**4A-France. UCI consumption**: the same procedure on `consumption_features.parquet`, with TimeSeriesSplit tuning.

**4E upgrade.** Add the measured solar-hours share (eMARC e05) to the energy balance, so it shows direct self-consumption vs export, not only daily totals.

### Phase 5: evaluation, then STOP for my review
- Metrics tables (MAE, RMSE, R², MAPE) for every baseline and model, on validation and **one** test run.
- Plots: predicted vs actual, residuals over time, error by month and weekday, permutation importance.
- Complete `reports/leakage_checks.md` rows 8–11.
- Write a model card per model in `reports/`.
- **Stop and show me the results.**

### Phase 6: Streamlit dashboard (only after my approval)
- Pages: Overview · Consumption Forecast (India / France) · Solar Generation · Performance Monitor · Maintenance · Energy Balance · Model Performance.
- Load saved models only; label every estimate and assumption; PV360 only as SYNTHETIC demo data.

### Phase 7: tests and documentation
- Tests for maintenance rules, model loading and monitoring logic.
- Final README: setup, phases, dashboard, sources and licences, limitations, acknowledgements.
- `python -m solvia.pipeline` runs everything, including a `train` / `evaluate` step.
- Optional: git repository with meaningful commits (offered, not set up yet).

---

## 8. Open items I still need to decide or check
- ✅ Done 2026-10-28: Indian festival dates checked against the DoPT central-government holiday lists 2018–2020 (all 12 match).
- Optionally tell Prayas we use eMARC (esmi_energy@prayaspune.org).
- ✅ Done 2026-10-28: Kaggle licence = "Data files © Original Authors" (academic use with credit; do not redistribute the raw files).
- The French school holidays could not be cross-checked for 2006–2010.
- Known limits:
  - The solar data covers only 34 days, so no seasonal claims.
  - The solar model maps *observed* weather to output.
  - The plant location is an estimate.
  - The eMARC test period includes the COVID lockdown.

---

## 9. Your first task in this session
All phases are done. Do not retrain or re-score the test split unless I ask (the guard in
`src/solvia/models/evaluation.py` refuses changed models; never delete `reports/metrics/test_run_log.json` without my approval).
1. Read `README.md`, `docs/ROADMAP.md` and the last entries of `docs/PROJECT_LOG.md`.
2. Run `pytest` (68 tests) and, if asked, `python -m solvia.pipeline` (about 5 minutes, deterministic).
3. Remaining optional items: section 8 (festival dates, Kaggle licence, Prayas notice) and a git repository
   (only if I approve it).
