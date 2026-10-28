# SOLVIA — Phase 0 data audit

*Generated 2026-10-28 from `python -m solvia.data.audit`. Every number in this report comes from
`reports/audit/inventory.json`, produced by that command. Nothing in `data/raw/` was modified
except the folder restructure the project owner approved (see `docs/PROJECT_LOG.md`).*

## 1. Summary

| # | Dataset | Location in `data/raw/` | Class | Verdict |
|---|---|---|---|---|
| 1 | UCI Individual Household Electric Power Consumption | `uci_household_power/` | **Household consumption** (measured) | ✅ Primary for objective 1 |
| 2 | Kaggle "Solar Power Generation Data" (2 plants) | `to_identify/chatgpt_datasets/archive.zip` (+ copies) | **Measured PV generation** + plant weather sensors | ✅ Primary for objectives 2–4 |
| 3 | NASA POWER daily/hourly, Vadodara (22.31 N, 73.18 E) | `to_identify/chatgpt_datasets/POWER_Point_*` | Weather / irradiance (satellite + reanalysis) | ✅ Energy balance, rainfall context |
| 4 | NASA POWER daily, New Delhi (28.61 N, 77.21 E) | `to_identify/claude_datasets/POWER_Point_*` | Weather / irradiance | ✅ Energy balance, rainfall context |
| 5 | NITI Aayog ICED district insolation, **Dec 2024 only** | `india_solar_irradiance/` | Irradiance | ⚠️ Usable for ranking districts; one month only |
| 6 | NITI Aayog ICED monthly avg insolation, Kheda (Gujarat) 2009–2023 | `to_identify/chatgpt_datasets/Monthly Average Insolation.xlsx` | Irradiance | ✅ One district, 12 months |
| 7 | Grid-India (NLDC) all-India demand, solar, wind | `electricity_demand_solar_wind/` | **Grid-level** | ⚠️ Not household data; optional context only |
| 8 | Modelled PV capacity factors, India grid cells, 1980–2025 | `to_identify/solar_pv/` | Modelled (not measured) | ⚠️ No region → location key; source unknown |
| 9 | NREL PVDAQ system list | `to_identify/chatgpt_datasets/systems*.csv` | Metadata only | ❌ No time series; not usable for any objective |
| 10 | NCA solar radiation, Sardar Sarovar Dam | `to_identify/chatgpt_datasets/solar_rediation_tel_hr_nca_gj_2021_2025.csv` | Irradiance | ❌ **Unusable** (sensor faults, conflicting duplicates) |
| 11 | "PV360" residential PV + battery, Capiz, Philippines | `to_identify/claude_datasets/PV360_data.zip` | Household PV + consumption | ⚠️ **UNVERIFIED → treated as SYNTHETIC** until a public source is confirmed |
| — | `to_identify/dataset_1/` | — | — | ❌ **100% duplicate** of other files (see §4) |

**Measured PV generation exists** (dataset 2), so the project can go ahead as planned.
**No long-history measured PV dataset** (e.g. NREL PVDAQ time series) is present. The two
`systems*.csv` files only list PVDAQ systems; they contain no generation data.

---

## 2. Dataset details

### 2.1 UCI household power — `uci_household_power/household_power_consumption.txt`
- 132.96 MB, `;`-separated, ASCII/UTF-8, `?` for missing. 2,075,259 rows × 9 columns.
- Columns: `Date` (dd/mm/yyyy), `Time` (HH:MM:SS), `Global_active_power`, `Global_reactive_power`,
  `Voltage`, `Global_intensity`, `Sub_metering_1..3`. All parsed with an explicit format: **0 failures**.
- Time range **2006-12-16 17:24 → 2010-11-26 21:02**, strictly 1-minute, **0 duplicate timestamps**,
  no gaps in the timestamp column (missing readings are rows of `?`, not absent rows).
- Missing: 25,979 rows (1.252%) have *every* measurement missing. By year: 2006: 4, 2007: 3,931,
  2008: 135, 2009: 4,280, **2010: 17,629** minutes.
- Source: UCI ML Repository (Hébrail & Bérard), Sceaux, France. Byte-identical copies sit inside
  two `individual+household+electric+power+consumption.zip` files in `to_identify/`.

**Unit checks (verified):**
- `Global_active_power` is in **kW**: the implied power factor P·1000 / (V·I) has median 0.9925.
  In 5.75% of rows it exceeds 1.05; `Global_intensity` is rounded to 0.2 A, which inflates the
  ratio at low loads.
- `Sub_metering_*` are **Wh per minute**: P·1000/60 − (sub1+sub2+sub3) has median 5.5 Wh and is
  below −1 Wh in only 0.002% of rows.

**Daily coverage** (1,442 calendar days): 1,418 days have ≥ 90% of minutes; 24 days are below 90%,
including 9 days with no data. The first day (27.5%) and the last day (87.7%) are partial and fall
below the threshold.
Daily kWh on good days: mean 26.19, median 25.92, min 4.17, max 79.56.

**Implication for the split:** the test year 2010 ends on 26 Nov and holds most of the missing
minutes. The planned split (train ≤ 2008, validate 2009, test 2010) still works. The test set
covers about 11 months, which will be stated.

### 2.2 Kaggle Solar Power Generation Data — `archive.zip` (4 CSVs)

| File | Rows | DATE_TIME format | Inverters | Missing 15-min slots |
|---|---|---|---|---|
| Plant_1_Generation_Data.csv | 68,778 | `%d-%m-%Y %H:%M` | 22 | median 4.27% per inverter (3,104–3,155 of 3,264) |
| Plant_1_Weather_Sensor_Data.csv | 3,182 | `%Y-%m-%d %H:%M:%S` | 1 sensor | 2.51% |
| Plant_2_Generation_Data.csv | 67,698 | `%Y-%m-%d %H:%M:%S` | 22 | median 0.15%; **one inverter has only 2,355 rows** |
| Plant_2_Weather_Sensor_Data.csv | 3,259 | `%Y-%m-%d %H:%M:%S` | 1 sensor | 0.15% |

- Both plants cover **2020-05-15 00:00 → 2020-06-17 23:45** (34 days), 15-minute data.
  PLANT_ID 4135001 and 4136001, India. The exact location is not published.
- The two plants use **different datetime formats** (confirmed). No duplicate (timestamp, inverter)
  rows and no negative values.
- Largest timestamp gap: 9 h in Plant 1 generation, 7 h 15 min in Plant 1 weather.
- Generation → weather join on DATE_TIME + PLANT_ID: 4 Plant 1 rows have no weather match; Plant 2 has none missing.

**Data-quality issues:**
1. **Plant 1 DC/AC ratio = 10.22** (5th–95th percentile 10.18–10.35) vs **1.022 for Plant 2**.
   Plant 1 `DC_POWER` is off by about 10×, a unit inconsistency. It is documented here and **not
   rescaled**. `AC_POWER` is the target, and DC_POWER is excluded from features in any case.
2. **Plant 2 has many zero-output rows in daylight:** 3,774 rows with AC_POWER = 0 while
   IRRADIATION > 0.1 (Plant 1: 63). The daytime AC–irradiation correlation is 0.656 for Plant 2 vs
   0.980 for Plant 1. This is what objective 4 is meant to detect. It also means Plant 2 rows need
   care when training the *expected-output* model (see §5, decision D5).
3. Plant 2: 14 rows with AC_POWER > 0 at IRRADIATION = 0 (dawn/dusk timing; minor). Plant 1: 0.
4. Plant 2 `TOTAL_YIELD` ranges 0 → 2.25 × 10⁹ with inconsistent counters. It is not used
   (excluded from features as a leakage risk anyway).
- **Units:** the median per-inverter maximum AC_POWER is 1,394 kW (Plant 1) and 1,366 kW (Plant 2),
  and DAILY_YIELD peaks at 9,163 and 9,873. That is consistent with AC_POWER in **kW** and
  DAILY_YIELD in **kWh** (about 1,400 kW × 6.5 equivalent sun hours). Phase 1 confirms this by
  comparing Σ AC_POWER × 0.25 with DAILY_YIELD.

### 2.3 NASA POWER (satellite / reanalysis weather)

| File | Point | Period | Rows | Parameters | −999 missing |
|---|---|---|---|---|---|
| `POWER_Point_Daily_20240101_20251231_022d31N_073d18E_LST.csv` | Vadodara 22.3072 N, 73.1812 E | 2024-01-01 → 2025-12-31, daily | 731 | T2M, T2M_MAX, T2M_MIN, ALLSKY_SFC_SW_DWN (kWh/m²/day), PRECTOTCORR (mm/day) | 0 |
| `POWER_Point_Hourly_20240101_20251231_022d31N_073d18E_LST.csv` | same | hourly | 17,544 | T2M, ALLSKY_SFC_SW_DWN (Wh/m²), PRECTOTCORR (mm/h) | 0 |
| `POWER_Point_Daily_20230101_20251231_028d61N_077d21E_LST.csv` | New Delhi 28.61 N, 77.21 E | 2023-01-01 → 2025-12-31, daily | 1,096 | T2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN | 0 |

- Each file has a 13–14-line `-BEGIN HEADER-` … `-END HEADER-` block before the CSV header.
- Mean daily GHI: Vadodara 5.01 kWh/m²/day, Delhi 4.61 kWh/m²/day. Max precipitation 149.1 mm/day
  in Vadodara and 72.5 mm/day in Delhi.
- Neither location nor period overlaps the Kaggle plants (unknown location, 2020). **NASA rainfall
  cannot be joined to the plant data**, so measured "rainfall vs solar output" analysis is not
  possible with these files (see objective 3 in §3).

### 2.4 NITI Aayog ICED irradiance (Excel)
**`india_solar_irradiance/india_solar_irradiance.xlsx`**: one sheet, 732 data rows + 3 footer
rows (copyright, `https://iced.niti.gov.in`, disclaimer). Columns: State, District, Year, Month,
`Insolation (in kWh/m²)`.
- 36 states, 641 districts, **Year = 2024 and Month = December only**. It is a single month.
- Values 29–114 kWh/m² (mean 71.2). These are **monthly totals**, not daily values.
- **47 duplicate rows** (e.g. Andaman & Nicobar, Himachal Pradesh districts listed twice).
- Cross-check: Vadodara Dec 2024 = **84 kWh/m²** here, vs **123.2 kWh/m²** from NASA POWER for the
  same month (3.97 kWh/m²/day), vs NITI's own long-term Dec mean for neighbouring Kheda of
  **105.7**. The values are about 30% below NASA POWER. They are fine for **ranking** districts
  but are not a reliable absolute annual input.

**`Monthly Average Insolation.xlsx`**: Kheda, Gujarat, 12 monthly averages for 2009–2023,
105.7–197.2 kWh/m²/month (Dec lowest, May highest). The footer has the same NITI ICED source.

### 2.5 Grid-India / NLDC — `electricity_demand_solar_wind/` (29 xlsx, 713 MB)
All-India SCADA values in **MW** (`NLDC_DEMAND`, `ALL_IND_SOLAR`, `ALL_INDIA_WIND`, thermal, hydro,
gas, nuclear, inertia). This is **grid-level data and must not be treated as a household.**

| Files | Layout | Resolution | Rows per file |
|---|---|---|---|
| Sep 2021 → Sep 2022 (13 files) | `_osi_config` + `Sheet1` (SCADA tags, 2 header rows) + `Sheet2` | **5 min** | 8,064–8,928 |
| Oct 2022 → Dec 2023 (15 files) | same | **10 s** | 241,920–267,840 |
| `January 2024- June 2025.xlsx` | `Report` sheet: Timestamp, Demand, Wind, Solar, Total Generation (MW); 3 summary rows at the end | **1 h** | 12,984 (ends **2025-06-24 23:00**, not 30 June) |

- Within each file: no duplicate timestamps, no missing values, perfectly regular intervals.
- After averaging to hourly: **33,432 hours, 2021-09-01 → 2025-06-24, no gaps and no overlaps
  between files.**
- Issues: `ALL_IND_SOLAR` is negative in 2,784 hours (min −75 MW, a night-time SCADA offset).
  `NLDC_DEMAND` reaches 0 MW in some hours (outage or logging fault). Columns H–J of `Sheet1` and
  `Sheet2` are Excel formulas derived from other columns and are ignored. The file name
  `Novemeber 2023.xlsx` has a typo.
- Peak hourly demand of 248,688 MW is consistent with India's record ~250 GW peak (2024), a good
  sign the data is genuine.

### 2.6 Modelled PV capacity factors — `to_identify/solar_pv/` (9 CSVs, 7.9 GB)
- 3 technologies (`PV_Rooftop`, `PV_Utility_scale_no_tracking`, `PV_Utility_scale_tracking`) × 3
  resource grades (`RGA`, `RGB`, `RGC`). Each folder has a CSV and a map `PV_Future_PV_cfs.png`
  titled "PV_Future – PV" showing capacity factor over an India grid.
- Each CSV: 403,248 hourly rows, **1980-01-01 → 2025-12-31**, a `time` column plus 328–332
  `region_N` columns (IDs 0–368 with gaps). The column set differs slightly between files.
  0 missing values; values 0–1 (capacity factor).
- Mean capacity factor: rooftop 0.171–0.174, fixed utility 0.205–0.208, tracking 0.245–0.249.
  Per-region mean 0.09–0.30.
- **These are modelled values (reanalysis-weather-driven), not measured generation.** The download
  has **no README and no lookup from `region_N` to latitude/longitude or state**, so a region
  can't be matched to a user's location. The source is unknown.

### 2.7 NREL PVDAQ metadata — `systems.csv`, `systems_20250729.csv`
- `systems.csv`: 158 rows (2 exact duplicates), 156 unique systems, id/name/site/lat/lon/elevation.
  0 in India.
- `systems_20250729.csv`: 1,862 rows, 1,860 unique systems, with capacity, tilt, azimuth, record
  counts and QA status (1,564 pass / 298 fail), plus 5 empty trailing columns. 1 system falls in
  an India bounding box.
- **No generation time series.** To use PVDAQ, one system's data would have to be downloaded
  separately (needs approval, and it would be US data).

### 2.8 NCA solar radiation — `solar_rediation_tel_hr_nca_gj_2021_2025.csv` ❌
- 36,909 rows, **one station** (Sardar Sarovar Dam, Narmada, Gujarat, 21.83 N 73.75 E), agency NCA,
  hourly-ish, 2022-01-21 → 2025-12-23, `Solar Radiation (Watt/m2)`.
- **Reasons it is unusable:**
  - Night-time (20:00–05:00) median is **13 W/m²**, and 20.6% of night readings exceed 50 W/m².
    The sensor has an offset or the timestamps are wrong.
  - **1023 W/m² appears 1,165 times at every hour including midnight**. 1023 is the maximum of a
    10-bit sensor, so this is saturation or a fault code.
  - **11,853 timestamps appear more than once**, and 9,120 of them have *conflicting* values.
  - Year coverage is uneven: 2022: 6,645, 2023: 14,092, 2024: 11,697, 2025: 4,475 rows.

### 2.9 "PV360" — `to_identify/claude_datasets/PV360_data.zip` ⚠️ UNVERIFIED
Claims to be one year (2024-11-01 → 2025-10-31) of 5-minute data from a 5.4 kWp PV + 15.36 kWh
battery home in Capiz, Philippines, with a README, CC BY 4.0 licence, analysis code and
**pre-computed ML results** (e.g. "day-ahead forecasting R² 0.38").

Evidence collected:

| Check | Result | Reading |
|---|---|---|
| Public source | README cites a manuscript but gives **no DOI, URL or repository** | ❌ cannot verify |
| Authorship note | README: scripting, tables and documentation "prepared with the assistance of … Claude" | ⚠️ |
| File timestamps | every zip member dated **2026-07-23 08:26** | ⚠️ packaged in one go |
| Folder it came from | `cluade datasets` (downloaded during a Claude chat) | ⚠️ |
| Rows / gaps | 92,661 rows (matches README), 87.9% at 5-min, largest gap 1 day 23 h | consistent |
| Internal sums | daily PV sum 6,299.4 kWh, consumption 5,757.6 kWh (README: 6,299 / 5,758) | consistent |
| PV shape | realistic dawn ramp, and a **sharp drop at ~11:50 to the house load** (e.g. 4,763 W → 526 W on 2025-03-15) | looks like real zero-export curtailment |
| PV at night | 5–7 W standby on 99.9% of night readings | plausible logger behaviour |
| Daily PV vs GHI correlation | 0.60 | low but possible with coarse satellite GHI |
| Max PV | 5,965 W on a 5.4 kWp array with a 5 kW inverter | ⚠️ above rating |

**Verdict:** the data *looks* like it could be genuine logger data, but its provenance cannot be
verified. Under the project's data rules it is **labelled SYNTHETIC (provisional)**. It may be used
only to test the dashboard, never for reported results, unless you can point to the public
repository/DOI you got it from. Two further points:
- Even if it is genuine, the PV output is **curtailed by the battery/zero-export controller**, so it
  measures what the house used, not what the panels could produce. It is unsuitable for the
  expected-output model and the performance monitor.
- Its archived "results" files must never be reused.

---

## 3. Mapping datasets to objectives

| Objective | Data | Can the data support it? |
|---|---|---|
| 1. Next-day household consumption | UCI (France) | ✅ Yes, 1,418 good days |
| 2. Expected solar output from weather (15-min → daily kWh) | Kaggle Plants 1 & 2 | ✅ Yes, for a 34-day prototype; *maps observed weather to output*, not a forecast |
| 3. Irradiance / temperature / **rainfall** vs output | Kaggle (irradiance, module & ambient temp) | ✅ irradiance & temperature. ❌ **Rainfall: no rainfall data for the plants' location/period.** Rainfall is shown only as NASA POWER context for Vadodara/Delhi and as a user input in rules. |
| 4. Detect under-performance | Kaggle (Plant 2 especially, 22 inverters per plant for peer comparison) | ✅ Yes |
| 5. Maintenance recommendations | Rules on top of objective 4 + user-entered rain/cleaning date | ✅ Yes (heuristics, labelled as such) |
| 6. Surplus / shortfall scenario | Irradiance: NASA POWER (Vadodara, Delhi), NITI Kheda monthly averages, NITI district ranking (Dec 2024). Consumption: user input or UCI model example | ⚠️ Yes as a **scenario tool**; only 2 cities have full-year irradiance (see D3) |
| (optional context) | Grid-India all-India demand & solar | Not required by any objective. Could be an extra "India grid solar share" chart on the Overview page. |

---

## 4. Duplicates

Verified by MD5 checksum (zip members included):

| Content | Copies |
|---|---|
| UCI `household_power_consumption.txt` | `uci_household_power/…txt` + inside `chatgpt_datasets/individual+household+electric+power+consumption.zip` + inside the identical zip in `dataset_1/` |
| Kaggle `archive.zip` | `chatgpt_datasets/archive.zip` = `claude_datasets/archive.zip` |
| Plant_1_Weather_Sensor_Data.csv | in both zips + `chatgpt_datasets/` + `dataset_1/` + `claude_datasets/…-7nwRaJ.csv` + `…-7nwRaJ (1).csv` (6 copies) |
| Plant_2_Generation_Data.csv | in both zips + `chatgpt_datasets/` + `dataset_1/` |
| NASA POWER Vadodara daily | 4 copies (`…LST.csv`, `…LST (1).csv` in both `chatgpt_datasets/` and `dataset_1/`) |
| NASA POWER Vadodara hourly, NCA csv, systems.csv, systems_20250729.csv, Monthly Average Insolation.xlsx | `chatgpt_datasets/` = `dataset_1/` |
| India Map Solar Irradiance.xlsx | `india_solar_irradiance/` = `dataset_1/` |

**Every file in `to_identify/dataset_1/` is a duplicate.** Nothing would be lost by deleting it.

---

## 5. Proposed final layout of `data/raw/` (awaiting approval)

Nothing below has been done yet. Moves are renames only; contents are unchanged.

```
data/raw/
  uci_household_power/household_power_consumption.txt            (unchanged)
  kaggle_solar_plants/archive.zip                                 ← to_identify/chatgpt_datasets/archive.zip
  nasa_power/
    vadodara_22.31N_73.18E_daily_2024-2025.csv                    ← chatgpt_datasets/POWER_Point_Daily_…022d31N…LST.csv
    vadodara_22.31N_73.18E_hourly_2024-2025.csv                   ← chatgpt_datasets/POWER_Point_Hourly_…csv
    delhi_28.61N_77.21E_daily_2023-2025.csv                       ← claude_datasets/POWER_Point_Daily_…028d61N…csv
  india_solar_irradiance/
    india_solar_irradiance.xlsx                                   (unchanged; NITI ICED, Dec 2024)
    niti_kheda_monthly_avg_2009-2023.xlsx                         ← chatgpt_datasets/Monthly Average Insolation.xlsx
  electricity_demand_solar_wind/                                  (unchanged; grid-level)
  modelled_pv_capacity_factors/                                   ← to_identify/solar_pv/  (7.9 GB)
  nrel_pvdaq_metadata/systems.csv, systems_20250729.csv           ← chatgpt_datasets/
  unverified/pv360_philippines/PV360_data.zip                     ← claude_datasets/PV360_data.zip
  unusable/nca_sardar_sarovar_solar_radiation_2022-2025.csv       ← chatgpt_datasets/solar_rediation_…csv
```

Then **delete the duplicates** (needs your explicit OK): all of `to_identify/dataset_1/`,
`claude_datasets/archive.zip`, the loose Plant CSVs, the `(1)` copies, and both copies of the UCI
zip: 18 files, 62.0 MB. After that, `to_identify/` would be empty and can be removed.

---

## 6. Decisions needed before Phase 1

- **D1. Approve §5**: the renames, and deleting the duplicates?
- **D2. PV360**: do you have its public source (DOI / Zenodo / Mendeley link)? Without one it stays
  SYNTHETIC (dashboard testing only).
- **D3. Location irradiance for the energy balance:** full-year data exists only for Vadodara and
  Delhi (NASA POWER) and Kheda (NITI averages); the NITI district file covers December only.
  Options: (a) use those 3 places + the December district ranking, clearly labelled; or (b) let me
  **download NASA POWER monthly climatology for ~15–20 Indian cities** (free and public, but an
  external download needs your approval).
- **D4. Modelled capacity factors (7.9 GB):** do you know where they came from? If there is a
  region → lat/lon file, they could improve the energy balance. Otherwise I suggest leaving them
  out of the pipeline (keep or move them off the project folder to save space).
- **D5. Plant 2 outages in training:** for the *expected-output* model I propose training on rows
  where the inverter is clearly operating (excluding AC_POWER = 0 at IRRADIATION > 0.1) and keeping
  every row for performance monitoring. Otherwise the model learns that outages are "normal". The
  exclusion would be logged with counts.
- **D6. Grid-India data:** include it as an optional context chart, or leave it out?

---

---

## Addendum (2026-10-28): datasets downloaded after the audit

| Dataset | Path | Checks | Verdict |
|---|---|---|---|
| NASA POWER daily temperature, Sceaux (household location), 2006-12-01 – 2010-11-30 | `nasa_power/sceaux_48.78N_2.29E_daily_2006-2010.csv` | 1,461 days, full calendar, 0 missing, min ≤ mean ≤ max every day | ✅ consumption weather features |
| NASA POWER daily weather, 48 Indian cities, 2001–2025 (9 variables) | `nasa_power/india_cities/` | 438,288 rows, 0 missing, 0 temperature inconsistencies, 0 days with GHI > 1.05 × clear-sky; validated against NITI (see `reports/eda_summary.md`) | ✅ **primary India weather source** |

The original short Vadodara and Delhi NASA files are superseded by the 25-year city files.

## Appendix A — every file under `data/raw/`

| Path | Size (MB) | Type | MD5 (first 8) | Duplicate? |
|---|---:|---|---|---|
| `electricity_demand_solar_wind/April 2022.xlsx` | 1.73 | .xlsx | a11a4508 |  |
| `electricity_demand_solar_wind/April 2023.xlsx` | 47.89 | .xlsx | bfde6af3 |  |
| `electricity_demand_solar_wind/August 2022.xlsx` | 1.67 | .xlsx | c5f714e9 |  |
| `electricity_demand_solar_wind/August 2023.xlsx` | 48.86 | .xlsx | 7f9126e7 |  |
| `electricity_demand_solar_wind/December 2021.xlsx` | 1.65 | .xlsx | fef976f4 |  |
| `electricity_demand_solar_wind/December 2022.xlsx` | 49.89 | .xlsx | c54e93e4 |  |
| `electricity_demand_solar_wind/December 2023.xlsx` | 49.67 | .xlsx | 91219872 |  |
| `electricity_demand_solar_wind/February 2022.xlsx` | 1.59 | .xlsx | 487cea80 |  |
| `electricity_demand_solar_wind/February 2023.xlsx` | 44.81 | .xlsx | 2f89aeec |  |
| `electricity_demand_solar_wind/January 2022.xlsx` | 1.75 | .xlsx | 0d914d3d |  |
| `electricity_demand_solar_wind/January 2023.xlsx` | 49.68 | .xlsx | be3f3f0e |  |
| `electricity_demand_solar_wind/January 2024- June 2025.xlsx` | 0.66 | .xlsx | e67e9d42 |  |
| `electricity_demand_solar_wind/July 2022.xlsx` | 1.65 | .xlsx | 09589708 |  |
| `electricity_demand_solar_wind/July 2023.xlsx` | 48.97 | .xlsx | d1c27f7f |  |
| `electricity_demand_solar_wind/June 2022.xlsx` | 1.61 | .xlsx | 0acc9cdd |  |
| `electricity_demand_solar_wind/June 2023.xlsx` | 46.67 | .xlsx | 4d0cc73f |  |
| `electricity_demand_solar_wind/March 2022.xlsx` | 1.7 | .xlsx | c55beb95 |  |
| `electricity_demand_solar_wind/March 2023.xlsx` | 49.48 | .xlsx | 49a61ce6 |  |
| `electricity_demand_solar_wind/May 2022.xlsx` | 1.68 | .xlsx | 4baaa463 |  |
| `electricity_demand_solar_wind/May 2023.xlsx` | 48.85 | .xlsx | 23fde614 |  |
| `electricity_demand_solar_wind/November 2021.xlsx` | 1.54 | .xlsx | 2d72aa77 |  |
| `electricity_demand_solar_wind/November 2022.xlsx` | 48.27 | .xlsx | 8bd3eacd |  |
| `electricity_demand_solar_wind/Novemeber 2023.xlsx` | 48.14 | .xlsx | 28a09a39 |  |
| `electricity_demand_solar_wind/October 2021.xlsx` | 1.59 | .xlsx | 186db7ae |  |
| `electricity_demand_solar_wind/October 2022.xlsx` | 47.75 | .xlsx | 7e1b2083 |  |
| `electricity_demand_solar_wind/October 2023.xlsx` | 49.49 | .xlsx | 1ae42aa2 |  |
| `electricity_demand_solar_wind/September 2021.xlsx` | 1.47 | .xlsx | e963dd21 |  |
| `electricity_demand_solar_wind/September 2022.xlsx` | 1.63 | .xlsx | 27163dfb |  |
| `electricity_demand_solar_wind/September 2023.xlsx` | 47.37 | .xlsx | 9d724b34 |  |
| `india_solar_irradiance/india_solar_irradiance.xlsx` | 0.06 | .xlsx | 939226f0 | yes |
| `to_identify/chatgpt_datasets/Monthly Average Insolation.xlsx` | 0.01 | .xlsx | 441a002a | yes |
| `to_identify/chatgpt_datasets/POWER_Point_Daily_20240101_20251231_022d31N_073d18E_LST (1).csv` | 0.03 | .csv | 9f0ca898 | yes |
| `to_identify/chatgpt_datasets/POWER_Point_Daily_20240101_20251231_022d31N_073d18E_LST.csv` | 0.03 | .csv | 9f0ca898 | yes |
| `to_identify/chatgpt_datasets/POWER_Point_Hourly_20240101_20251231_022d31N_073d18E_LST.csv` | 0.49 | .csv | ad56a128 | yes |
| `to_identify/chatgpt_datasets/Plant_1_Weather_Sensor_Data.csv` | 0.29 | .csv | 35d879ba | yes |
| `to_identify/chatgpt_datasets/Plant_2_Generation_Data.csv` | 5.81 | .csv | 11781a72 | yes |
| `to_identify/chatgpt_datasets/archive.zip` | 1.99 | .zip | 132b49fa | yes |
| &nbsp;&nbsp;↳ `Plant_1_Generation_Data.csv` (in zip, dated 2020-08-18) | 4.84 | zip member | 97fd3db0 | yes |
| &nbsp;&nbsp;↳ `Plant_1_Weather_Sensor_Data.csv` (in zip, dated 2020-08-18) | 0.29 | zip member | 35d879ba | yes |
| &nbsp;&nbsp;↳ `Plant_2_Generation_Data.csv` (in zip, dated 2020-08-18) | 5.81 | zip member | 11781a72 | yes |
| &nbsp;&nbsp;↳ `Plant_2_Weather_Sensor_Data.csv` (in zip, dated 2020-08-18) | 0.3 | zip member | e4b320ae | yes |
| `to_identify/chatgpt_datasets/individual+household+electric+power+consumption.zip` | 20.64 | .zip | 41f51806 | yes |
| &nbsp;&nbsp;↳ `household_power_consumption.txt` (in zip, dated 2012-10-12) | 132.96 | zip member | e05286cd | yes |
| `to_identify/chatgpt_datasets/solar_rediation_tel_hr_nca_gj_2021_2025.csv` | 4.91 | .csv | 76b583c3 | yes |
| `to_identify/chatgpt_datasets/systems.csv` | 0.02 | .csv | 928a091f | yes |
| `to_identify/chatgpt_datasets/systems_20250729.csv` | 0.38 | .csv | 2a2e4606 | yes |
| `to_identify/claude_datasets/POWER_Point_Daily_20230101_20251231_028d61N_077d21E_LST.csv` | 0.03 | .csv | ffc3c6a6 |  |
| `to_identify/claude_datasets/PV360_data.zip` | 0.7 | .zip | 480bbb9a |  |
| &nbsp;&nbsp;↳ `PV360_data/README.md` (in zip, dated 2026-07-23) | 0.01 | zip member | bc97aac7 |  |
| &nbsp;&nbsp;↳ `PV360_data/data/daily_with_weather.csv` (in zip, dated 2026-07-23) | 0.04 | zip member | fa2f5d4d |  |
| &nbsp;&nbsp;↳ `PV360_data/data/system_specifications.json` (in zip, dated 2026-07-23) | 0.0 | zip member | 6ea64131 |  |
| &nbsp;&nbsp;↳ `PV360_data/data/measured_5min.csv` (in zip, dated 2026-07-23) | 3.79 | zip member | 90f161a4 |  |
| &nbsp;&nbsp;↳ `PV360_data/data/nasa_power_daily.csv` (in zip, dated 2026-07-23) | 0.01 | zip member | b5804141 |  |
| &nbsp;&nbsp;↳ `PV360_data/results/economic_cashflow_25yr.csv` (in zip, dated 2026-07-23) | 0.0 | zip member | b824f7f8 |  |
| &nbsp;&nbsp;↳ `PV360_data/results/battery_sizing_sweep.csv` (in zip, dated 2026-07-23) | 0.0 | zip member | d801ab68 |  |
| &nbsp;&nbsp;↳ `PV360_data/results/monthly_performance.csv` (in zip, dated 2026-07-23) | 0.0 | zip member | e56b2558 |  |
| &nbsp;&nbsp;↳ `PV360_data/results/forecast_feature_importance.csv` (in zip, dated 2026-07-23) | 0.0 | zip member | 814fc485 |  |
| &nbsp;&nbsp;↳ `PV360_data/results/forecast_model_comparison.csv` (in zip, dated 2026-07-23) | 0.0 | zip member | c944bf78 |  |
| &nbsp;&nbsp;↳ `PV360_data/LICENSE.txt` (in zip, dated 2026-07-23) | 0.0 | zip member | e4450fcb |  |
| &nbsp;&nbsp;↳ `PV360_data/code/pv_analytics.py` (in zip, dated 2026-07-23) | 0.0 | zip member | f8c0a1c7 |  |
| `to_identify/claude_datasets/Plant_1_Weather_Sensor_Data-7nwRaJ (1).csv` | 0.29 | .csv | 35d879ba | yes |
| `to_identify/claude_datasets/Plant_1_Weather_Sensor_Data-7nwRaJ.csv` | 0.29 | .csv | 35d879ba | yes |
| `to_identify/claude_datasets/archive.zip` | 1.99 | .zip | 132b49fa | yes |
| &nbsp;&nbsp;↳ `Plant_1_Generation_Data.csv` (in zip, dated 2020-08-18) | 4.84 | zip member | 97fd3db0 | yes |
| &nbsp;&nbsp;↳ `Plant_1_Weather_Sensor_Data.csv` (in zip, dated 2020-08-18) | 0.29 | zip member | 35d879ba | yes |
| &nbsp;&nbsp;↳ `Plant_2_Generation_Data.csv` (in zip, dated 2020-08-18) | 5.81 | zip member | 11781a72 | yes |
| &nbsp;&nbsp;↳ `Plant_2_Weather_Sensor_Data.csv` (in zip, dated 2020-08-18) | 0.3 | zip member | e4b320ae | yes |
| `to_identify/dataset_1/India Map Solar Irradiance.xlsx` | 0.06 | .xlsx | 939226f0 | yes |
| `to_identify/dataset_1/Monthly Average Insolation.xlsx` | 0.01 | .xlsx | 441a002a | yes |
| `to_identify/dataset_1/POWER_Point_Daily_20240101_20251231_022d31N_073d18E_LST (1).csv` | 0.03 | .csv | 9f0ca898 | yes |
| `to_identify/dataset_1/POWER_Point_Daily_20240101_20251231_022d31N_073d18E_LST.csv` | 0.03 | .csv | 9f0ca898 | yes |
| `to_identify/dataset_1/POWER_Point_Hourly_20240101_20251231_022d31N_073d18E_LST.csv` | 0.49 | .csv | ad56a128 | yes |
| `to_identify/dataset_1/Plant_1_Weather_Sensor_Data.csv` | 0.29 | .csv | 35d879ba | yes |
| `to_identify/dataset_1/Plant_2_Generation_Data.csv` | 5.81 | .csv | 11781a72 | yes |
| `to_identify/dataset_1/individual+household+electric+power+consumption.zip` | 20.64 | .zip | 41f51806 | yes |
| &nbsp;&nbsp;↳ `household_power_consumption.txt` (in zip, dated 2012-10-12) | 132.96 | zip member | e05286cd | yes |
| `to_identify/dataset_1/solar_rediation_tel_hr_nca_gj_2021_2025.csv` | 4.91 | .csv | 76b583c3 | yes |
| `to_identify/dataset_1/systems.csv` | 0.02 | .csv | 928a091f | yes |
| `to_identify/dataset_1/systems_20250729.csv` | 0.38 | .csv | 2a2e4606 | yes |
| `to_identify/solar_pv/PV_Rooftop_RGA/PV_Future_PV_cfs.png` | 1.48 | .png | 060a44ba |  |
| `to_identify/solar_pv/PV_Rooftop_RGA/solar_PV_Rooftop_RGA.csv` | 945.22 | .csv | bf5a6b16 |  |
| `to_identify/solar_pv/PV_Rooftop_RGB/PV_Future_PV_cfs.png` | 1.48 | .png | b2d6091d |  |
| `to_identify/solar_pv/PV_Rooftop_RGB/solar_PV_Rooftop_RGB.csv` | 939.57 | .csv | cea135c9 |  |
| `to_identify/solar_pv/PV_Rooftop_RGC/PV_Future_PV_cfs.png` | 1.49 | .png | 2aebab4a |  |
| `to_identify/solar_pv/PV_Rooftop_RGC/solar_PV_Rooftop_RGC.csv` | 939.57 | .csv | 33341135 |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGA/PV_Future_PV_cfs.png` | 1.47 | .png | 665ebb9c |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGA/solar_PV_no_tracking_RGA.csv` | 939.57 | .csv | e4e42257 |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGB/PV_Future_PV_cfs.png` | 1.47 | .png | 0b23f5a1 |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGB/solar_PV_no_tracking_RGB.csv` | 933.93 | .csv | a110e833 |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGC/PV_Future_PV_cfs.png` | 1.47 | .png | 3ece1db1 |  |
| `to_identify/solar_pv/PV_Utility_scale_no_tracking_RGC/solar_PV_no_tracking_RGC.csv` | 933.93 | .csv | da716164 |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGA/PV_Future_PV_cfs.png` | 1.47 | .png | e8146bb3 |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGA/solar_PV_tracking_RGA.csv` | 939.57 | .csv | 62fed920 |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGB/PV_Future_PV_cfs.png` | 1.46 | .png | c4589d4d |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGB/solar_PV_tracking_RGB.csv` | 933.93 | .csv | e51b1bf4 |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGC/PV_Future_PV_cfs.png` | 1.47 | .png | c2fa1959 |  |
| `to_identify/solar_pv/PV_Utility_scale_tracking_RGC/solar_PV_tracking_RGC.csv` | 933.93 | .csv | b3316752 |  |
| `uci_household_power/household_power_consumption.txt` | 132.96 | .txt | e05286cd | yes |

76 files in total. Full machine-readable profile: `reports/audit/inventory.json`.

## Addendum 2 (2026-10-28): datasets downloaded for the second cleaning pass

| Dataset | Rows / size | Type | Checks | Status |
|---|---|---|---|---|
| Prayas eMARC info (`Household-Deployment basic info.xlsx`) | 273 deployments, 128 households | Metadata | MD5 = Dataverse; 57 deployments have no data; 1 type conflict (H128) | Used |
| Prayas eMARC daily (`eMARC daily consumption.csv`) | 111,280 rows, 216 meters, 2018-01-01 – 2020-06-30 | **Measured household consumption (India)** | MD5 = Dataverse; 7 exact duplicates; 0 negatives; max 44.6 kWh/day | **Used: objective A (India)** |
| Prayas eMARC 15-minute blocks (`eMARC load blocks.csv`) | 12,702,801 rows, 368 MB | Measured load (kW per 15 min) | MD5 = Dataverse; 0 duplicates, 0 negatives, 0 NaN; agrees with the daily file (median ratio 0.997) | Used: load-shape features, solar-hours share |
| NASA POWER Aurangabad, Gonda 2001–2025 | 9,131 days each | Weather (satellite/reanalysis) | NASA header, full calendar | Used |
| NASA POWER Sceaux full (9 variables) 2006-12 – 2010-11 | 1,461 days | Weather | T2M identical to the earlier file | Used (replaces the 3-variable file) |
| NASA POWER 1° grid, 169 cells, May–Jun 2020 | 61 days each | Weather | NASA header, full calendar | Used (plant-location estimate) |
| French school holidays 1990–2028 | 14,245 days | Calendar | Not checkable against the official API before 2013 | Used (zone C) |

None of these is synthetic. eMARC is real measured data published by Prayas (Energy Group) with a DOI;
it is **household consumption, not PV generation**: the only measured PV generation remains the two Kaggle plants.
