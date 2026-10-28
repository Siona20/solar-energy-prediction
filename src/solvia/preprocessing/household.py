"""Clean the UCI household data and aggregate it to one row per day."""
from __future__ import annotations

import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog
from solvia.preprocessing.india_weather import RENAME as NASA_RENAME

MINUTES_PER_DAY = 1440
SUB_METERS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def daily_from_minutes(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute readings to daily energy.

    consumption_kwh = mean(Global_active_power [kW]) x 24 h. For a complete day this equals
    sum(kW)/60; for a day with a few missing minutes it avoids under-counting.
    Sub-metering is Wh per minute, so kWh = mean(Wh/min) x 1440 / 1000.
    """
    day = minute.index.floor("D")
    g = minute.groupby(day)
    daily = pd.DataFrame({
        "n_minutes": g["Global_active_power"].count(),
        "consumption_kwh": g["Global_active_power"].mean() * 24,
        "peak_power_kw": g["Global_active_power"].max(),
        "mean_voltage_v": g["Voltage"].mean(),
    })
    for i, col in enumerate(SUB_METERS, start=1):
        daily[f"sub{i}_kwh"] = g[col].mean() * MINUTES_PER_DAY / 1000
    daily["unmetered_kwh"] = daily["consumption_kwh"] - daily[[f"sub{i}_kwh" for i in (1, 2, 3)]].sum(axis=1)
    daily["coverage"] = daily["n_minutes"] / MINUTES_PER_DAY
    daily.index.name = "date"
    return daily


def tidy_household_weather(raw: pd.DataFrame, hdd_base_c: float, clog: CleaningLog) -> pd.DataFrame:
    """NASA POWER daily weather at Sceaux -> date-indexed table with heating degree-days."""
    w = raw.rename(columns=NASA_RENAME)
    w = w.set_index(w["timestamp"].rename("date"))[[c for c in NASA_RENAME.values() if c in w]]
    w["hdd_c"] = (hdd_base_c - w["t2m_c"]).clip(lower=0)
    clog.add("household weather", f"rename {len(w.columns) - 1} NASA variables, -999 -> NaN, add heating degree-days (base {hdd_base_c} °C)",
             len(w), len(w), int(w.isna().sum().sum()), f"{w.index.min().date()} to {w.index.max().date()}")
    return w


def interpolate_short_gaps(s: pd.Series, max_days: int) -> tuple[pd.Series, pd.Series]:
    """Linearly fill runs of NaN no longer than `max_days`. Returns (filled series, was_filled mask)."""
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    fillable = na & (run_len <= max_days)
    filled = s.interpolate(method="linear", limit_area="inside")
    out = s.where(~fillable, filled)
    return out, fillable & out.notna()


def clean_household(minute: pd.DataFrame, min_coverage: float, max_interp_days: int,
                    clog: CleaningLog) -> pd.DataFrame:
    """Daily table on a complete calendar with coverage, missing and interpolation flags."""
    n = len(minute)
    all_missing = minute["Global_active_power"].isna()
    clog.add("household", "minute rows with all measurements missing ('?') kept as NaN", n, n,
             int(all_missing.sum()))

    daily = daily_from_minutes(minute)
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    before = len(daily)
    daily = daily.reindex(calendar)
    daily["n_minutes"] = daily["n_minutes"].fillna(0).astype(int)
    daily["coverage"] = daily["coverage"].fillna(0.0)
    clog.add("household", "aggregate to days and reindex to a complete calendar", before, len(daily),
             len(daily) - before, f"{len(minute):,} minutes -> {len(daily):,} days")

    energy_cols = ["consumption_kwh", "peak_power_kw", "mean_voltage_v", "sub1_kwh", "sub2_kwh", "sub3_kwh", "unmetered_kwh"]
    low = daily["coverage"] < min_coverage
    daily.loc[low, energy_cols] = float("nan")
    clog.add("household", f"days below {min_coverage:.0%} minute coverage set to missing", len(daily), len(daily),
             int(low.sum()))

    # Partial first/last days cannot be interpolated; drop them so the series starts and ends on real days.
    valid = daily["consumption_kwh"].notna()
    before = len(daily)
    daily = daily.loc[valid.idxmax(): valid[::-1].idxmax()].copy()
    clog.add("household", "trim leading/trailing incomplete days", before, len(daily),
             note=f"series now {daily.index.min().date()} to {daily.index.max().date()}")

    filled_any = pd.Series(False, index=daily.index)
    for col in energy_cols:
        daily[col], was_filled = interpolate_short_gaps(daily[col], max_interp_days)
        filled_any |= was_filled
    daily["is_interpolated"] = filled_any
    daily["is_missing"] = daily["consumption_kwh"].isna()
    clog.add("household", f"interpolate gaps of <= {max_interp_days} days", len(daily), len(daily),
             int(filled_any.sum()), f"{int(daily['is_missing'].sum())} days remain missing (longer gaps)")
    clog.fact("household_interpolated_days", [str(d.date()) for d in daily.index[filled_any]])
    clog.fact("household_missing_days", [str(d.date()) for d in daily.index[daily["is_missing"]]])
    clog.fact("household_daily_kwh_summary", daily["consumption_kwh"].describe().round(2).to_dict())
    return daily
