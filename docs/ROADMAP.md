# SOLVIA roadmap and checklist

Every step from the original project brief, plus the extra fine-tuning steps added on
2026-10-28 (marked ➕). Ticked items are implemented and reproducible with the commands in README.md.

## Phase 0 — Setup and data audit ✅
- [x] Folder structure, `.venv`, `requirements.txt`, `pyproject.toml`, `config.yaml`, `.gitignore`, READMEs, project log
- [x] Inventory and profile every raw file → `reports/data_audit.md`
- [x] Classify datasets (measured PV / household / weather / grid / synthetic / unusable)
- [x] Final dataset names; duplicates moved to `data/raw/_duplicates/`

## Phase 1 — Loading and cleaning ✅
- [x] One loader per dataset (`src/solvia/data/`)
- [x] Household: day-first parsing, `?` → NaN, coverage-adjusted daily kWh, sub-metering kWh, ≥ 90% coverage rule, gaps ≤ 2 days interpolated and listed
- [x] Solar: per-file datetime formats, duplicates, negatives, night power, DC/AC ratio documented, join to weather, kWh = kW × 0.25, daily per inverter and plant
- [x] DAILY_YIELD cross-check (and fix for the stale counter after midnight)
- [x] Irradiance tidied, one row per location
- [x] Row-count log → `reports/cleaning_report.md`; tests for parsing and daily maths

## Phase 2 — EDA ✅
- [x] Consumption: trend, weekly and monthly seasonality, weekday vs weekend, sub-metering shares, missing-data map, distribution, autocorrelation
- [x] Solar: daily power curves, AC vs irradiation, module-temperature effect, inverter comparison, Plant 1 vs Plant 2, inverter × day heatmap
- [x] Irradiance: Indian locations ranked by solar potential
- [x] ➕ Monthly irradiance and rainfall climate for Vadodara / Delhi / Kheda (context for cleaning rules)
- [x] ➕ Distribution of each inverter's output vs its peers, to check the 0.85 threshold against real data
- [x] Every figure: title, axis labels with units, one-line takeaway (notebooks + `reports/eda_summary.md`)

## Phase 3 — Feature engineering ✅
- [x] Consumption: lags 1, 2, 7, 14; rolling mean/std 7 and 30 (shift(1)); day of week, month, weekend, day-of-year sin/cos; French public holidays
- [x] ➕ Extra consumption features: same-weekday mean of the last 4 weeks, short-vs-long trend (7-day vs 30-day mean), yesterday's water-heater/AC sub-meter kWh and peak power
- [x] Solar: irradiation, module temp, ambient temp, module − ambient, hour sin/cos
- [x] Leakage guard: yield and DC columns can never become features (enforced in code + tests)
- [x] Chronological split labels stored with the features (consumption by date, solar 70/15/15 by day)
- [x] Scalers inside scikit-learn Pipelines, fitted on training data only
- [x] ➕ `reports/leakage_checks.md` started in this phase, not at the end

## ➕ India data (added 2026-10-28, owner: "India first") ✅
- [x] NASA POWER daily weather for 48 Indian cities, 2001–2025, 9 variables (reproducible downloader + manifest)
- [x] Cleaned → daily table, monthly climatology (with 10th/90th percentile months), one row per city
- [x] Validated NASA against NITI Aayog (Kheda long-term r = 0.88; Dec-2024 districts rank agreement 0.90)
- [x] Estimated the Kaggle plants' region by matching their daily sunlight to every city (southern Deccan, best match Anantapur)
- [x] Estimated rainfall for the plants' 34 days (labelled estimate), unlocking objective 3's rainfall analysis
- [x] India EDA figures i02–i07 and notebook `03_eda_india.ipynb`

## ➕ All remaining datasets + second cleaning / feature pass (2026-10-28) ✅
- [x] Prayas eMARC Indian smart meters downloaded (3 files, MD5 = Dataverse) → loader, cleaning, 15-minute block features, cross-check (median ratio 0.997)
- [x] eMARC cleaning: duplicates, complete calendar per meter, ≥ 3-day zero runs → missing, outlier flags (kept), ≤ 2-day interpolation, appliance meter joined, ≥ 180 measured days = model-eligible (112 meters)
- [x] India consumption features per meter (history, load shape, CDD/temperature/humidity/rain, Indian holidays, type/region), lockdown period label
- [x] NASA POWER for Aurangabad and Gonda (now 50 cities); full 9-variable Sceaux file; 1° plant search grid (169 cells)
- [x] French school holidays (zone C) → UCI `is_school_holiday`; yesterday's GHI → UCI `ghi_lag_1`
- [x] EDA: eMARC figures e01–e06 + notebook `05_eda_india_households.ipynb`; plant-location grid map i08
- [x] Tests: 25 passing (eMARC zero runs, gaps, duplicates, block maths, per-meter no-leakage, Indian holidays, file-name tags)

## Phase 4 — Modelling (India tracks first)
- [x] E. Energy balance: kWp × GHI × PR per month for 50 cities, poor/good-month range, kWp needed (`rules/energy_balance.py`, tests)
- [x] B. Solar: k × irradiation baseline; Linear Regression, Random Forest, HistGradientBoosting; daytime and daily-kWh metrics; train on one plant, test on the other
- [x] ➕ B. Hyperparameter tuning with grouped-by-day time-series CV
- [x] ➕ B. Train on all operating rows vs on a healthy reference (Plant 1 / inverters matching peers); pick by validation
- [x] C. Performance monitoring: out-of-sample expected output only, performance ratio, robust peer z-score, 3-day rule, Isolation Forest comparison
- [x] ➕ C. Rain effect: does output per unit of sunlight change after estimated rain days? (labelled estimate)
- [x] D. Maintenance rules with reasons and evidence, using estimated rainfall and dry-month climate (R0–R7)
- [x] ➕ C. Running PR (shutdowns vs dust), simulated-fault detection check, inverter-not-reporting rule
- [x] A-India. Consumption (eMARC, India first): 3 baselines; Ridge, Random Forest, HistGradientBoosting as one model across meters (meter-level features, no pooling into one household); per-meter MAE; test split into normal vs lockdown days (Phase 5); seen vs new meters
- [x] ➕ E. Energy balance: add the measured share of use in solar hours (e05) so the balance shows direct self-consumption, not only daily totals
- [x] A. Consumption (France): 3 baselines; Ridge, Random Forest, HistGradientBoosting; TimeSeriesSplit tuning; select by validation MAE
- [x] ➕ A. Prediction intervals (quantile HistGradientBoosting, 10th–90th percentile)
- [x] ➕ A. Experiment: target-day temperature as a stand-in for a weather forecast (labelled, compared with the causal model)
- [x] ➕ NASA POWER daily temperature for Sceaux 2006–2010 added as consumption features

## Phase 5 — Evaluation (STOP for review)
- [x] Metrics tables (MAE, RMSE, R², MAPE) for every baseline and model, validation + test
- [x] Predicted vs actual, residuals over time, error by month and weekday, permutation importance
- [x] `reports/leakage_checks.md` completed
- [x] Models saved with joblib + metadata JSON (dates, features, metrics, library versions)
- [x] ➕ Model card per model (what it is for, data, limits) in `reports/`
- [x] ➕ Conformal calibration of the 80 % intervals on validation; one-time test guard (model file hashes)

## Phase 6 — Streamlit dashboard
- [x] Overview · Consumption Forecast · Solar Generation · Performance Monitor · Maintenance · Energy Balance · Model Performance
- [x] Loads saved models only; every estimate and assumption labelled
- [x] ➕ SYNTHETIC demo page (PV360, read from the raw zip, banner + watermark, not used in results)
- [x] ➕ Page smoke tests with Streamlit AppTest; data layer reproduces the Phase 5 numbers

## Phase 7 — Tests and documentation
- [x] pytest: parsing ✅, daily kWh ✅, no future rows in lags ✅, solar feature list ✅, holidays ✅, splits ✅, maintenance rules ✅, energy balance ✅, test guard ✅, model loading ✅, monitoring ✅, dashboard pages ✅ (68 tests)
- [x] README: setup, phases, dashboard, sources and licences, limitations
- [x] `python -m solvia.pipeline` runs everything (raw → test evaluation in 4 min 51 s; byte-identical models; raw data unchanged)
- [x] ➕ git repository (local, branch main; history grouped by phase, created at the end; no remote)
