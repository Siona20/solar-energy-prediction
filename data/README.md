# Data sources

`data/raw/` is read-only and not committed to git (the UCI file alone is ~133 MB, above
GitHub's 100 MB limit). Anyone cloning the project must download the files below into the
paths shown. Download dates are not recorded in the files themselves; they were collected
before 2026-10-28.

Statuses come from the Phase 0 audit (`reports/data_audit.md`).

| Dataset | Path | Source / link | Licence | Status |
|---|---|---|---|---|
| UCI Individual Household Electric Power Consumption (Sceaux, France, 2006–2010, 1-min) | `raw/uci_household_power/` | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption | CC BY 4.0 | Used |
| Solar Power Generation Data (2 Indian plants, 34 days, 15-min) | `raw/kaggle_solar_plants/archive.zip` | https://www.kaggle.com/datasets/anikannal/solar-power-generation-data (Ani Kannal, version 1; local archive 11,233,523 bytes = Kaggle's) | **"Data files © Original Authors"** (Kaggle metadata, checked 2026-10-28): no open licence; used for academic work with credit, raw files not redistributed | Used |
| NASA POWER daily/hourly point data (Vadodara 2024–25, Delhi 2023–25), original downloads | `raw/nasa_power/` | https://power.larc.nasa.gov | Free/open; NASA requests citation | Superseded by the city files; not used |
| **NASA POWER daily weather, 50 Indian cities, 2001-01-01 – 2025-12-31** (GHI, clear-sky GHI, clearness index, T2M mean/max/min, rain, humidity, wind) | `raw/nasa_power/india_cities/` (+ `_manifest.json` with URL, MD5, time per file) | https://power.larc.nasa.gov/api/temporal/daily/point (downloaded 2026-10-28; re-run `python -m solvia.data.download_nasa_power`; city list in `configs/india_locations.yaml`; Aurangabad and Gonda added 2026-10-28 for eMARC) | Free/open; NASA requests citation | **Primary India weather source** |
| NASA POWER daily temperature (T2M, T2M_MAX, T2M_MIN) at Sceaux, France (48.78 N, 2.29 E), 2006-12-01 – 2010-11-30 | `raw/nasa_power/sceaux_48.78N_2.29E_daily_2006-2010.csv` | https://power.larc.nasa.gov/api/temporal/daily/point (downloaded 2026-10-28, community RE) | Free/open; NASA requests citation | Superseded by the `_full` file (same T2M values) |
| **NASA POWER daily weather, all 9 variables, Sceaux, 2006-12-01 – 2010-11-30** | `raw/nasa_power/sceaux_48.78N_2.29E_daily_2006-2010_full.csv` (+ `_manifest_extra_points.json`) | same API (downloaded 2026-10-28; `extra_points` in `configs/india_locations.yaml`) | Free/open; NASA requests citation | Used (UCI consumption weather features) |
| **NASA POWER 1° grid, 169 cells (10–22 °N, 72–84 °E), 2020-05-01 – 2020-06-30** | `raw/nasa_power/plant_search_grid_2020/` (+ `_manifest.json`) | same API (downloaded 2026-10-28; `search_grid` in `configs/india_locations.yaml`) | Free/open; NASA requests citation | Used (plant-location estimate, figure i08) |
| **Prayas (Energy Group) eMARC smart meters**: 216 meters in 128 households, Pune / Aurangabad (Maharashtra), Kanpur rural / Gonda (Uttar Pradesh), 2018-01-01 – 2020-06-30; daily kWh, 15-minute block load (kW), household info | `raw/prayas_emarc/` (+ `_manifest.json`, MD5s match Dataverse) | Harvard Dataverse doi:10.7910/DVN/YJ5SP1, v1.1 (downloaded 2026-10-28) | CC0 1.0 on Dataverse; terms of use: non-commercial academic/research use, **acknowledge Prayas (Energy Group)** | **Used: Indian household consumption** |
| **French school holidays by zone, 1990–2028** | `raw/french_school_holidays/vacances_scolaires_1990-2028.csv` (+ `_manifest.json`) | github.com/AntoineAugusti/vacances-scolaires (Etalab compilation of the Ministère de l'Éducation nationale calendars; downloaded 2026-10-28) | Licence Ouverte / Open Licence (Etalab) | Used (UCI `is_school_holiday`, zone C = académie de Versailles). Could not be cross-checked: the official open calendar has no mainland records before 2013 |
| NITI Aayog ICED district insolation (Dec 2024) and Kheda monthly averages (2009–2023) | `raw/india_solar_irradiance/` | https://iced.niti.gov.in | Government of India open data (terms on site) | Used (with caveats) |
| Grid-India (NLDC) all-India demand, solar, wind (2021–2025) | `raw/electricity_demand_solar_wind/` | https://grid-india.in | terms on site | Grid-level context only: hourly Jan 2024 – Jun 2025 file used by the `recent` step (dashboard "Latest" tab) |
| Modelled PV capacity factors for India grid cells (1980–2025) | `raw/modelled_pv_capacity_factors/` | **unknown** | unknown | Not used yet (no region → location key) |
| NREL PVDAQ system list | `raw/nrel_pvdaq_metadata/` | https://openei.org/wiki/PVDAQ | CC BY 4.0 (to confirm) | Metadata only, not used |
| NCA solar radiation, Sardar Sarovar Dam | `raw/unusable/` | India-WRIS / Narmada Control Authority (to confirm) | to confirm | **Unusable** (sensor faults) |
| "PV360" Philippines household PV + battery | `raw/unverified/pv360_philippines/` | **unverified**: no DOI/URL | claims CC BY 4.0 | **SYNTHETIC (provisional)**: dashboard testing only |

`raw/_duplicates/` holds byte-identical copies found in the audit. They were kept, not deleted, and can be removed at any time.
