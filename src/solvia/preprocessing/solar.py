"""Clean the Kaggle solar plants: join inverters to plant weather, add flags, build daily tables.

Nothing is rescaled or silently dropped. Suspicious rows get boolean flags so later phases can
decide (and log) what to exclude. Plant 1 DC_POWER is ~10x AC_POWER (unit inconsistency, see
reports/data_audit.md); it is kept only as `dc_power_raw` and never used as a feature.
"""
from __future__ import annotations

import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog

GEN_RENAME = {"DATE_TIME": "timestamp", "PLANT_ID": "plant_id", "SOURCE_KEY": "inverter_id",
              "DC_POWER": "dc_power_raw", "AC_POWER": "ac_power_kw",
              "DAILY_YIELD": "daily_yield_kwh", "TOTAL_YIELD": "total_yield_kwh"}
WX_RENAME = {"DATE_TIME": "timestamp", "PLANT_ID": "plant_id", "SOURCE_KEY": "sensor_id",
             "AMBIENT_TEMPERATURE": "ambient_temp_c", "MODULE_TEMPERATURE": "module_temp_c",
             "IRRADIATION": "irradiation_kw_m2"}


def _dedupe(df: pd.DataFrame, keys: list[str], name: str, clog: CleaningLog) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    clog.add(name, "drop exact duplicate rows", before, len(df))
    conflicting = df.duplicated(keys, keep=False)
    before = len(df)
    df = df[~conflicting]
    clog.add(name, f"drop rows with conflicting duplicate {'+'.join(keys)}", before, len(df))
    return df


def clean_plant(gen: pd.DataFrame, wx: pd.DataFrame, plant: int, interval_h: float,
                off_irr: float, clog: CleaningLog) -> pd.DataFrame:
    """15-minute inverter rows joined to plant weather, with interval energy and quality flags."""
    name = f"plant{plant}"
    gen = _dedupe(gen.rename(columns=GEN_RENAME), ["timestamp", "inverter_id"], f"{name} generation", clog)
    wx = _dedupe(wx.rename(columns=WX_RENAME).drop(columns="sensor_id"), ["timestamp"], f"{name} weather", clog)

    for col in ["ac_power_kw", "dc_power_raw", "irradiation_kw_m2"]:
        frame = gen if col in gen else wx
        neg = int((frame[col] < 0).sum())
        if neg:
            frame.loc[frame[col] < 0, col] = 0.0
        clog.add(name, f"negative {col} set to 0", len(frame), len(frame), neg)

    before = len(gen)
    df = gen.merge(wx, on=["timestamp", "plant_id"], how="inner", validate="many_to_one")
    clog.add(name, "join generation to weather on timestamp + plant_id (inner)", before, len(df),
             note="rows without a weather reading cannot be modelled")

    df["energy_kwh"] = df["ac_power_kw"] * interval_h
    df["is_daylight"] = df["irradiation_kw_m2"] > 0
    df["flag_power_at_night"] = (df["ac_power_kw"] > 0) & ~df["is_daylight"]
    df["flag_off_in_daylight"] = (df["ac_power_kw"] == 0) & (df["irradiation_kw_m2"] > off_irr)
    clog.add(name, "flag AC power > 0 with zero irradiation", len(df), len(df), int(df["flag_power_at_night"].sum()))
    clog.add(name, f"flag inverter off (AC = 0) while irradiation > {off_irr} kW/m2", len(df), len(df),
             int(df["flag_off_in_daylight"].sum()), "kept; excluded only when training the expected-output model")

    on = df[df["ac_power_kw"] > 0]
    clog.fact(f"{name}_dc_ac_ratio_median", round(float((on["dc_power_raw"] / on["ac_power_kw"]).median()), 3))
    return df.sort_values(["timestamp", "inverter_id"]).reset_index(drop=True)


def daily_per_inverter(df: pd.DataFrame, interval_h: float) -> pd.DataFrame:
    """One row per plant, inverter and day.

    The logger's DAILY_YIELD counter sometimes still holds the previous day's total after
    midnight (seen in Plant 2), so the logger's daily total is read from afternoon rows only.
    """
    df = df.assign(date=df["timestamp"].dt.floor("D"),
                   yield_after_reset=df["daily_yield_kwh"].where(df["timestamp"].dt.hour >= 12))
    g = df.groupby(["plant_id", "inverter_id", "date"])
    daily = g.agg(
        energy_kwh=("energy_kwh", "sum"),
        daily_yield_kwh=("yield_after_reset", "max"),
        n_intervals=("energy_kwh", "size"),
        n_daylight_intervals=("is_daylight", "sum"),
        n_off_in_daylight=("flag_off_in_daylight", "sum"),
        peak_ac_power_kw=("ac_power_kw", "max"),
        irradiation_kwh_m2=("irradiation_kw_m2", lambda s: s.sum() * interval_h),
        ambient_temp_c=("ambient_temp_c", "mean"),
        module_temp_max_c=("module_temp_c", "max"),
    ).reset_index()
    daily["coverage"] = daily["n_intervals"] / round(24 / interval_h)
    return daily


def daily_per_plant(daily_inv: pd.DataFrame) -> pd.DataFrame:
    """Plant totals per day (sum of inverters; weather is the same for every inverter)."""
    g = daily_inv.groupby(["plant_id", "date"])
    return g.agg(
        energy_kwh=("energy_kwh", "sum"),
        daily_yield_kwh=("daily_yield_kwh", "sum"),
        n_inverters=("inverter_id", "nunique"),
        n_off_in_daylight=("n_off_in_daylight", "sum"),
        irradiation_kwh_m2=("irradiation_kwh_m2", "first"),
        ambient_temp_c=("ambient_temp_c", "first"),
        module_temp_max_c=("module_temp_max_c", "first"),
        min_coverage=("coverage", "min"),
    ).reset_index()


def yield_crosscheck(daily_inv: pd.DataFrame, plant: int, clog: CleaningLog, min_coverage: float = 0.95) -> None:
    """Compare sum(AC_POWER) x 0.25 with the logger's own DAILY_YIELD on well-covered inverter-days of one plant."""
    d = daily_inv[(daily_inv["coverage"] >= min_coverage) & (daily_inv["daily_yield_kwh"] > 100)]
    ratio = d["energy_kwh"] / d["daily_yield_kwh"]
    clog.fact(f"plant{plant}_ac_energy_vs_daily_yield", {
        "inverter_days_compared": int(len(d)),
        "ratio_median": round(float(ratio.median()), 4),
        "ratio_p05": round(float(ratio.quantile(0.05)), 4),
        "ratio_p95": round(float(ratio.quantile(0.95)), 4),
        "share_within_5pct": round(float(((ratio - 1).abs() <= 0.05).mean()), 4),
    })
