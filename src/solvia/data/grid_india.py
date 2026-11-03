"""Loader for the Grid-India (NLDC) hourly all-India report, Jan 2024 - Jun 2025.

This is GRID-LEVEL data for the whole country: it is never treated as a household or a plant. Only the hourly
`Report` file is used; the older 5-minute / 10-second SCADA files have a different layout (data_audit.md §2.5).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HOURLY_FILE = "January 2024- June 2025.xlsx"
COLUMNS = {"Demand (MW)": "demand_mw", "Wind (MW)": "wind_mw", "Solar (MW)": "solar_mw",
           "Total Generation (MW)": "total_generation_mw"}
STUCK_RUN_HOURS = 3        # the same value to two decimals for >= 3 hours in a row = a frozen SCADA reading
NIGHT_HOURS = [*range(0, 5), *range(20, 24)]  # 20:00-04:00: after the latest sunset (~19:45) anywhere in India
NIGHT_SOLAR_MAX_MW = 2000  # the night-time median is ~65 MW (SCADA offset); far above that = bad reading
NIGHT_LEVEL_MW = 500       # runs of identical solar values at or below this are night-time, not frozen readings


def _stuck(s: pd.Series, ignore_at_or_below: float | None = None) -> pd.Series:
    run = (s != s.shift()).cumsum()
    stuck = s.groupby(run).transform("size") >= STUCK_RUN_HOURS
    if ignore_at_or_below is not None:
        stuck &= s > ignore_at_or_below
    return stuck


def load_hourly(raw_dir: Path) -> pd.DataFrame:
    """Hourly MW values with quality flags. The 6 summary rows at the end (min/max/average/sum) are dropped."""
    df = pd.read_excel(raw_dir / HOURLY_FILE, sheet_name="Report")
    ts = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    log.info("Grid-India: %d rows read, %d summary rows dropped", len(df), int(ts.isna().sum()))
    df = df[ts.notna()].rename(columns=COLUMNS).assign(timestamp=ts[ts.notna()])
    out = df[["timestamp", *COLUMNS.values()]].astype({c: float for c in COLUMNS.values()}).reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("Grid-India hourly file has duplicate timestamps")
    out["flag_demand_stuck"] = _stuck(out["demand_mw"])
    out["flag_solar_stuck"] = _stuck(out["solar_mw"], NIGHT_LEVEL_MW)
    out["flag_solar_night"] = out["timestamp"].dt.hour.isin(NIGHT_HOURS) & (out["solar_mw"] > NIGHT_SOLAR_MAX_MW)
    out["flag_wind_stuck"] = _stuck(out["wind_mw"])
    out["bad_hour"] = out[["flag_demand_stuck", "flag_solar_stuck", "flag_solar_night"]].any(axis=1)
    log.info("Grid-India: %d hours, %s to %s; %d hours flagged (demand/solar)", len(out), out["timestamp"].min(),
             out["timestamp"].max(), int(out["bad_hour"].sum()))
    return out


def daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily energy (GWh = sum of 24 hourly MW values / 1000) and peaks. A day is `valid` only with 24 hours and
    no flagged demand or solar hour; invalid days keep their numbers but must not be charted as measurements."""
    h = hourly.assign(date=hourly["timestamp"].dt.floor("D"))
    d = h.groupby("date").agg(hours=("demand_mw", "size"), bad_hours=("bad_hour", "sum"),
                              demand_gwh=("demand_mw", "sum"), solar_gwh=("solar_mw", "sum"), wind_gwh=("wind_mw", "sum"),
                              peak_demand_gw=("demand_mw", "max"), peak_solar_gw=("solar_mw", "max")).reset_index()
    d[["demand_gwh", "solar_gwh", "wind_gwh", "peak_demand_gw", "peak_solar_gw"]] /= 1000
    d["solar_share_pct"] = 100 * d["solar_gwh"] / d["demand_gwh"]
    d["valid"] = (d["hours"] == 24) & (d["bad_hours"] == 0)
    log.info("Grid-India: %d days, %d valid", len(d), int(d["valid"].sum()))
    return d
