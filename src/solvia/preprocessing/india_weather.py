"""Tidy the NASA POWER India city files into daily, monthly-climatology and per-location tables."""
from __future__ import annotations

import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog

RENAME = {"ALLSKY_SFC_SW_DWN": "ghi_kwh_m2_day", "CLRSKY_SFC_SW_DWN": "clearsky_ghi_kwh_m2_day",
          "ALLSKY_KT": "clearness_index", "T2M": "t2m_c", "T2M_MAX": "t2m_max_c", "T2M_MIN": "t2m_min_c",
          "PRECTOTCORR": "precip_mm_day", "RH2M": "rh2m_pct", "WS2M": "ws2m_m_s"}
DRY_MONTH_MM = 10.0
RAINY_DAY_MM = 1.0


def tidy_india_daily(frames: dict[str, pd.DataFrame], meta: dict[str, list], clog: CleaningLog) -> pd.DataFrame:
    """Long table, one row per location and day. `meta[name] = [state, region, lat, lon]`."""
    out = []
    for name, df in frames.items():
        state, region, _, _ = meta[name]
        d = df.rename(columns=RENAME).rename(columns={"timestamp": "date"})
        d.insert(0, "location", name)
        d.insert(1, "state", state)
        d.insert(2, "region", region)
        out.append(d)
    daily = pd.concat(out, ignore_index=True)
    n = len(daily)
    clog.add("india weather", f"combine {len(frames)} NASA POWER city files; -999 -> NaN", n, n, int(daily[list(RENAME.values())].isna().sum().sum()),
             f"{daily['location'].nunique()} locations, {daily['date'].min().date()} to {daily['date'].max().date()}")
    bad_t = (daily["t2m_min_c"] > daily["t2m_c"]) | (daily["t2m_c"] > daily["t2m_max_c"])
    clog.add("india weather", "check min <= mean <= max temperature", n, n, int(bad_t.sum()))
    above_clear = daily["ghi_kwh_m2_day"] > daily["clearsky_ghi_kwh_m2_day"] * 1.05
    clog.add("india weather", "check all-sky GHI <= 1.05 x clear-sky GHI", n, n, int(above_clear.sum()),
             "kept; satellite clear-sky estimate can be slightly low")
    return daily


def monthly_climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """Per location and calendar month: long-term mean GHI and temperature, mean monthly rain and rainy days."""
    d = daily.assign(year=daily["date"].dt.year, month=daily["date"].dt.month)
    per_month = d.groupby(["location", "year", "month"]).agg(
        ghi=("ghi_kwh_m2_day", "mean"), clear=("clearsky_ghi_kwh_m2_day", "mean"), t2m=("t2m_c", "mean"),
        tmax=("t2m_max_c", "mean"), rain=("precip_mm_day", "sum"),
        rainy_days=("precip_mm_day", lambda s: (s >= RAINY_DAY_MM).sum())).reset_index()
    clim = per_month.groupby(["location", "month"]).agg(
        ghi_kwh_m2_day=("ghi", "mean"), clearsky_ghi_kwh_m2_day=("clear", "mean"), t2m_c=("t2m", "mean"),
        t2m_max_c=("tmax", "mean"), rain_mm_month=("rain", "mean"), rainy_days=("rainy_days", "mean"),
        ghi_p10_kwh_m2_day=("ghi", lambda s: s.quantile(0.10)), ghi_p90_kwh_m2_day=("ghi", lambda s: s.quantile(0.90)),
        n_years=("year", "nunique")).reset_index()
    return clim


def location_climate(daily: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """One row per location: the numbers the energy balance and the maintenance rules need."""
    first = daily.groupby("location")[["state", "region", "latitude", "longitude"]].first()
    g = clim.groupby("location")
    summary = pd.DataFrame({
        "ghi_kwh_m2_day": daily.groupby("location")["ghi_kwh_m2_day"].mean(),
        "best_month": g.apply(lambda c: int(c.loc[c["ghi_kwh_m2_day"].idxmax(), "month"]), include_groups=False),
        "worst_month": g.apply(lambda c: int(c.loc[c["ghi_kwh_m2_day"].idxmin(), "month"]), include_groups=False),
        "t2m_c": daily.groupby("location")["t2m_c"].mean(),
        "rain_mm_year": g["rain_mm_month"].sum(),
        "dry_months": g["rain_mm_month"].apply(lambda s: int((s < DRY_MONTH_MM).sum())),
        "share_rain_jun_sep": g.apply(lambda c: c.loc[c["month"].between(6, 9), "rain_mm_month"].sum() / c["rain_mm_month"].sum(),
                                      include_groups=False),
        "clearness_index": daily.groupby("location")["clearness_index"].mean(),
        "years": daily.groupby("location")["date"].apply(lambda s: f"{s.min().year}-{s.max().year}"),
    })
    return first.join(summary).reset_index().sort_values("ghi_kwh_m2_day", ascending=False).reset_index(drop=True)


def match_plants_to_locations(daily: pd.DataFrame, daily_plant: pd.DataFrame,
                              window: tuple[str, str] = ("2020-05-15", "2020-06-17")) -> pd.DataFrame:
    """Correlate each plant's measured daily irradiation (and temperature) with every city's NASA
    values over the same days. The best-correlated city is an *estimate* of the plant's region."""
    w = daily[daily["date"].between(*window)]
    rows = []
    for pid, g in daily_plant.groupby("plant_id"):
        g = g.set_index("date")
        for loc, c in w.groupby("location"):
            j = g.join(c.set_index("date"))
            rows.append({"plant_id": pid, "location": loc,
                         "r_ghi": j["irradiation_kwh_m2"].corr(j["ghi_kwh_m2_day"]),
                         "r_temp": j["ambient_temp_c"].corr(j["t2m_c"]),
                         "sensor_over_nasa_ghi": j["irradiation_kwh_m2"].mean() / j["ghi_kwh_m2_day"].mean(),
                         "rain_mm_window": j["precip_mm_day"].sum()})
    m = pd.DataFrame(rows)
    m["rank"] = m.groupby("plant_id")["r_ghi"].rank(ascending=False).astype(int)
    return m.sort_values(["plant_id", "rank"]).reset_index(drop=True)


def plant_estimated_weather(daily: pd.DataFrame, match: pd.DataFrame,
                            window: tuple[str, str] = ("2020-05-15", "2020-06-17")) -> pd.DataFrame:
    """NASA daily weather at each plant's best-matching city, labelled as an estimate."""
    best = match[match["rank"] == 1][["plant_id", "location"]]
    w = daily[daily["date"].between(*window)]
    out = best.merge(w, on="location")[["plant_id", "location", "date", "precip_mm_day", "ghi_kwh_m2_day",
                                        "t2m_c", "rh2m_pct", "ws2m_m_s"]]
    out.insert(2, "source", "ESTIMATE: NASA POWER at best-matching city")
    return out


def tidy_grid(frames: dict[str, pd.DataFrame], clog: CleaningLog) -> pd.DataFrame:
    """NASA POWER 1° grid files -> long daily table; each cell is a 'location' named grid_<lat>N_<lon>E."""
    out = []
    for name, df in frames.items():
        d = df.rename(columns=RENAME).rename(columns={"timestamp": "date"})
        d.insert(0, "location", name)
        out.append(d)
    grid = pd.concat(out, ignore_index=True)
    clog.add("plant search grid", "combine NASA POWER 1° grid files; -999 -> NaN", len(grid), len(grid),
             int(grid[[c for c in RENAME.values() if c in grid]].isna().sum().sum()),
             f"{grid['location'].nunique()} cells, {grid['date'].min().date()} to {grid['date'].max().date()}")
    return grid


def match_plants_to_grid(grid: pd.DataFrame, daily_plant: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Same correlation as match_plants_to_locations, on every grid cell; adds the cell's lat/lon and,
    per plant, the r-weighted centre of the top-n cells as a single ESTIMATE of the plant location."""
    m = match_plants_to_locations(grid, daily_plant)
    coords = grid.groupby("location")[["latitude", "longitude"]].first()
    m = m.join(coords, on="location")
    top = m[m["rank"] <= top_n]
    centre = top.groupby("plant_id").apply(
        lambda t: pd.Series({"est_latitude": (t["latitude"] * t["r_ghi"]).sum() / t["r_ghi"].sum(),
                             "est_longitude": (t["longitude"] * t["r_ghi"]).sum() / t["r_ghi"].sum()}),
        include_groups=False)
    return m.join(centre, on="plant_id")
