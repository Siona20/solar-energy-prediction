# SOLVIA: Smart Solar Energy Forecasting & Maintenance System

Final-year B.Tech CSE (Data Science & ML) project.

SOLVIA forecasts a home's electricity use for the next day, estimates what a solar plant *should* produce
in the measured weather, detects inverters that produce less than expected, recommends maintenance with a
reason and evidence for each action, and estimates how much of a home's use rooftop solar could cover.
Everything is shown in a Streamlit dashboard. **Indian data comes first**; one French household is kept
as a comparison track.

> **Status:** all phases complete (0–7, plus Phase 8: accuracy improvements, see below). Progress and decisions: [docs/ROADMAP.md](docs/ROADMAP.md),
> [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).
> Reports: [data audit](reports/data_audit.md) · [cleaning](reports/cleaning_report.md) · [EDA](reports/eda_summary.md) ·
> [features](reports/features_report.md) · [solar model](reports/solar_model_report.md) ·
> [monitoring](reports/monitoring_report.md) · [consumption India](reports/consumption_india_report.md) ·
> [consumption France](reports/consumption_france_report.md) · [energy balance](reports/energy_balance_report.md) ·
> [**test evaluation**](reports/evaluation_report.md) · [leakage checks](reports/leakage_checks.md)

## Results (test split, scored once)

Models were chosen on validation; the test split was then scored once (Phase 5). Model file hashes are
recorded in `reports/metrics/test_run_log.json`, and scoring changed models on the test split is refused.

| Objective | Data | Selected model | Test result | Simple baseline |
|---|---|---|---|---|
| A. Next-day consumption, **India** | Prayas eMARC, 104 meters, Jan–Jun 2020 | equal-weight ensemble (Ridge + Random Forest + absolute-loss gradient boosting on the ratio to the home's 30-day mean), one model across meters, refitted on train + validation (Phase 8) | MAE **0.679 kWh/day** (R² 0.913); lockdown days 0.712. First test run (Phase 5, Ridge): 0.711 | 7-day mean 0.746 |
| A. Next-day consumption, France | UCI, one home, 2010 | Ridge | MAE **4.259 kWh/day** (R² 0.490) | 7-day mean 4.375 |
| A. 80 % prediction interval | same | quantile gradient boosting + conformal adjustment (validation) | coverage **83.0 %** India, **83.0 %** France | — |
| B. Expected solar output | Kaggle, 2 Indian plants, 5 test days | gradient boosting per plant, trained on healthy inverters | daily error **176.8 kWh per inverter-day (2.76 %)** | k × irradiation 365.6 kWh |
| C. Performance monitoring | Kaggle plants, 27 days | out-of-sample expected output, PR rules | 0 % false alarms and 99.8 % of 20 % losses caught in one day (simulated faults) | — |
| D. Maintenance | monitoring results + estimated rain | rules R0–R7 with evidence | Plant 2: losses are inverter shutdowns in bright sun (PR 0.797 vs 0.959 while running), not dust → read event logs | — |
| E. Energy balance (estimate) | NASA POWER 2001–2025 + measured load and PV shapes | kWp × GHI × PR, 15-minute direct use | typical Pune home sized to its yearly use covers **41–47 %** of its use directly (no battery) | — |

**Phase 8 (accuracy improvements).** Eight changes were tried on train/validation only
([reports/improvement_experiments.md](reports/improvement_experiments.md)). Only two helped India beyond noise: an
equal-weight ensemble with fixed members (validation 0.716 → 0.705 kWh/day; day-bootstrap 95 % CI of the gain
0.008–0.015) and refitting on the most recent data (train + validation). With the owner's approval the test split was
scored a second time for the changed India model: 0.711 → **0.679** kWh/day (the ensemble alone 0.697; refitting
the rest); the model now beats the baseline on 86 % of meters (was 70 %). Because the test split had been seen once
before, the second number may be slightly optimistic; the first run is kept in `reports/archive_phase5/`. France
(ensemble gain not significant) and solar (gains within noise on 5 validation days) were left unchanged.

Every number above was produced by the code in this repository. The full tables, with per-meter, per-region and
lockdown breakdowns, are in [reports/evaluation_report.md](reports/evaluation_report.md), with one model card per
model in `reports/model_card_*.md`.

## Setup (macOS, zsh, Python 3.14)

```bash
cd "$HOME/Desktop/Solar Energy "      # the folder name ends with a space
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # pinned versions
pip install -e .
```

The raw data is not in the repository. Download it into `data/raw/` as listed in [data/README.md](data/README.md);
NASA POWER weather can be fetched with `python -m solvia.data.download_nasa_power`.

## Running

| Step | Command |
|---|---|
| Everything, from raw data to the test evaluation (about 5 minutes) | `python -m solvia.pipeline` |
| Phase 0 data audit | `python -m solvia.data.audit` |
| Phase 1 cleaning · 2 EDA figures · 3 features | `python -m solvia.pipeline --steps clean eda features` |
| Phase 4 training (models, monitoring, maintenance, energy balance) | `python -m solvia.pipeline --steps model_solar monitor maintenance energy_balance model_consumption_india model_consumption_france` |
| Phase 5 test evaluation (once; refuses changed models) | `python -m solvia.pipeline --steps evaluate` |
| Phase 8 improvement experiments (validation only; not in the default run) | `python -m solvia.pipeline --steps improve` |
| Latest data, 2024–25 (Grid-India + NASA POWER; context only, no model) | `python -m solvia.pipeline --steps recent` |
| "My home tomorrow" models: weather (1–21 days), home solar units, home use from 7 days of units | `python -m solvia.pipeline --steps model_weather model_pv_home model_home_simple` |
| Phase 6 dashboard | `streamlit run app/Home.py` → http://localhost:8501 |
| Forecast for my home (tomorrow and the next 7 days) | `streamlit run app/simple/app.py` |
| Tests | `pytest` |

Training is deterministic (`random_state = 42`): re-running the pipeline produces byte-identical model files, so
the test-set guard accepts a full re-run.

## Forecast for my home (tomorrow and the next 7 days)

The page `streamlit run app/simple/app.py`. Enter the place (any of 50 cities, or latitude/longitude), the
solar system size in kW, optionally the units it gives on a clear sunny summer day (e.g. 12 kW → 60 units), the type
of home and the units used recently (one typical number or the last 7 days). It fetches the latest NASA POWER weather
for that point (cached in `data/cache/`, never in `data/raw/`) and chains three models:

| Model | What it predicts | Accuracy |
|---|---|---|
| Weather (`reports/weather_model_report.md`) | Max / min / mean temperature and sunlight, 1–21 days ahead, any Indian point | Test 2022–2025, max temperature: ±1.02 °C one day ahead (normal for the date: ±1.76), ±1.59 °C 8–14 days ahead; sunlight 8+ days ahead is no better than the normal |
| Home solar units (`reports/pv_home_report.md`) | kWh per day = kW × sunlight × losses × heat loss (0.4 %/°C); panel heating measured on the Kaggle plants | Daily error on validation days 4.05 % (5.63 % without the temperature effect); ESTIMATE for homes, calibrated by the user's sunny-day units |
| Home use (`reports/home_use_simple_report.md`) | Units per day for the next 7 days from the last 7 days, type of home, date and temperature | Validation (Jul–Dec 2019): ±0.68 units/day for tomorrow, ±0.89 on day 7 (last 7 days' average: 0.74 / 0.91); eMARC test split not used |

NASA POWER publishes new days about a week (temperature) to 10 days (sunlight) late, so "tomorrow" is 8–10 days
after the latest data and its weather is close to the normal for the date. A real weather-forecast feed (for example
Open-Meteo) would sharpen tomorrow's numbers; it is not used because it is a different external source.

## Dashboard

| Page | Shows |
|---|---|
| Overview | headline test results and what each label means |
| Consumption Forecast | one Indian home at a time (region → type → meter) or the French home: out-of-sample next-day forecasts, 80 % interval, baseline, lockdown vs normal days |
| Solar Generation | measured vs expected output per plant or inverter, irradiation, daily actual ÷ expected |
| Performance Monitor | PR heatmap (inverter × day), plant PR vs running PR, estimated rain, inverter status by day |
| Maintenance | the day's recommendations with action, reason and evidence; history by rule |
| Energy Balance | scenario for any of 50 Indian cities: monthly solar vs use, direct use, export, grid import |
| Model Performance | Phase 5 tables and figures, model cards, leakage checks |
| SYNTHETIC demo | PV360 data of unverified origin, for layout testing only |

Every chart carries a label: **MEASURED** (meter or inverter reading), **MODEL** (saved model output),
**ESTIMATE** (calculated from long-term weather and assumptions) or **SYNTHETIC** (unverified, never in results).
The dashboard only loads saved models; it never trains or writes anything.

## Method in brief

- **Splits are chronological.** India: train to 2019-06-30, validation 2019-07 to 12, test 2020-01 to 06 (lockdown
  from 2020-03-25 reported separately). France: train 2007–2008, validation 2009, test 2010. Solar: whole days,
  24 / 5 / 5.
- **No leakage by construction.** Every consumption feature comes from earlier days of the same meter; solar
  features are weather and time only (output-derived columns are blocked in code); scalers and encoders live
  inside scikit-learn pipelines; tuning uses expanding-window CV over whole training days. Each risk, its
  prevention and the test that checks it are listed in [reports/leakage_checks.md](reports/leakage_checks.md).
- **Baselines first.** Every model is compared with yesterday / same day last week / 7-day mean (consumption) and
  k × irradiation (solar).
- **Healthy reference (solar).** The expected-output model is trained only on inverter-days within 10 % of the
  plant's best inverters that day, so faults are not learned as normal.
- **Monitoring is out-of-sample.** The expected output for day *d* comes from a model refitted on days before *d*.
  Running PR (output while the inverter is on) separates shutdowns from slow losses such as dust.
- **Households are never combined.** Each eMARC meter is its own series, India and France are separate tracks,
  and the energy balance is a scenario, not a real home.

## Project layout

```
configs/config.yaml           paths, split dates, thresholds, random_state (each threshold explained in PROJECT_LOG)
configs/india_locations.yaml  50 NASA POWER cities
data/raw/                     original downloads (read-only, not in git)
data/interim/                 cleaned data, monitoring and maintenance tables (parquet)
data/processed/               model-ready feature tables (parquet)
src/solvia/data/              loaders and downloaders
src/solvia/preprocessing/     cleaning
src/solvia/features/          feature building, leakage guards
src/solvia/models/            solar, consumption, monitoring, evaluation
src/solvia/rules/             maintenance rules, energy balance
src/solvia/dashboard/         dashboard data layer
src/solvia/pipeline.py        one command for every step
app/                          Streamlit dashboard (Home.py + pages/)
models/                       saved models (.joblib) + metadata (.json)
reports/                      reports, metrics, figures, model cards
notebooks/                    exploration only
tests/                        pytest
docs/                         project log, roadmap
```

## Data sources and licences

Full list with paths, links, download dates and MD5 manifests: [data/README.md](data/README.md).

| Data | Use | Licence / terms |
|---|---|---|
| Prayas (Energy Group) eMARC smart meters, Maharashtra and Uttar Pradesh, 2018–2020 (doi:10.7910/DVN/YJ5SP1) | Indian household consumption | CC0 on Dataverse; terms: non-commercial academic use, acknowledge Prayas (Energy Group) |
| Kaggle "Solar Power Generation Data" (2 Indian plants, 34 days) | the only measured PV generation | "Data files © Original Authors" (no open licence): academic use with credit, raw files not redistributed |
| NASA POWER daily weather: 50 Indian cities 2001–2025, Sceaux 2006–2010, 1° grid May–Jun 2020 | weather features, energy balance, rain estimate, plant-location estimate | free and open; NASA asks for citation |
| UCI Individual household electric power consumption (Sceaux, 2006–2010) | French comparison household | CC BY 4.0 |
| French school holidays (Etalab compilation) | calendar feature, France | Licence Ouverte |
| NITI Aayog ICED irradiance | cross-check of NASA POWER | Government of India open data |
| PV360 (Philippines) | SYNTHETIC demo page only | unverified source |

## Limitations

- **Different places.** The households (India: eMARC homes in Maharashtra and Uttar Pradesh; France: one home in
  Sceaux) and the solar plants (two utility plants, location estimated as the southern Deccan near Anantapur) are
  not the same systems. The energy balance is a scenario tool, not one real home with its own panels.
- **Short solar record.** 34 days (15 May – 17 Jun 2020), 5 of them in the test split: the solar results are a
  prototype check, with no seasonal claim. A new plant needs its own healthy days; on the other plant the model
  is worse.
- **Observed, not forecast, weather.** The solar model maps measured weather to output; consumption uses
  yesterday's weather. Using the target day's observed weather as a stand-in for a perfect forecast improved India's
  validation MAE by only 0.002 kWh/day and did not improve France.
- **Rain is an estimate** (NASA POWER at the best-matching city), and the plants have no cleaning log, so the
  dust-vs-rain conclusions rest on 27 days with 5 rainy ones.
- **Shutdown cause unknown.** Plant 2's inverters switch off in bright sun; the data cannot tell grid
  over-voltage, overload, over-temperature trips or curtailment apart.
- **Ridge and short history.** Ridge (a member of the India ensemble) was much worse on the earliest,
  shortest-history CV fold (MAE 2.28 kWh vs 0.89 for Random Forest); a new deployment should be checked for a few
  weeks before relying on it.
- **India test split used twice.** The India model was changed after the first test run (decided on validation only)
  and scored again with the owner's approval; both results are reported.
- **Selection was not revisited for France and solar.** Random Forest scored slightly lower test MAE than the
  selected model for both; the choice made on validation was kept so those test results stay unbiased.
- **Monitoring thresholds** (e.g. the peer gate PR < 0.95) were set while looking at a window that includes the
  5 solar test days; the expected-output model itself never saw them (leakage check 9c).
- **Sample.** 112 model-eligible eMARC meters in two states and one French home are not representative of all
  households.
- **Indian holidays are a short list** (national days plus Holi, Eid-ul-Fitr, Ganesh Chaturthi, Dussehra, Diwali;
  dates checked against the DoPT central-government lists 2018–2020); regional festivals are not included.

## Acknowledgements

Indian household data: **Prayas (Energy Group)**, eMARC smart-meter data, Harvard Dataverse,
doi:10.7910/DVN/YJ5SP1, used for non-commercial academic purposes. Solar plant data: Ani Kannal, "Solar Power
Generation Data", Kaggle. Weather: NASA POWER project, NASA Langley Research Center. French household data:
Georges Hébrail and Alice Bérard, UCI Machine Learning Repository. French school holidays: Etalab / Ministère de
l'Éducation nationale.
