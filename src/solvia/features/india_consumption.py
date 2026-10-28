"""Next-day consumption features for the eMARC Indian households (one row per meter and target day).

Same rules as the UCI features: every history and weather feature uses days up to d-1 only,
calendar features describe d. Each meter's features are built from that meter alone; the
meter id is kept so that models can be evaluated per household and split without mixing.
Weather is NASA POWER at the city mapped to the meter's region (configs/config.yaml: emarc.region_weather).
"""
from __future__ import annotations

import pandas as pd

from solvia.features.calendar import india_holidays
from solvia.features.consumption import (BASELINES, HISTORY_FEATURES, TARGET, add_calendar_features, assign_split,
                                         history_features)
from solvia.preprocessing.cleaning_log import CleaningLog

ID_COLUMNS = ["deployment_id", "household_id", "date"]
CATEGORICAL_FEATURES = ["household_type", "region"]
LOAD_SHAPE_FEATURES = ["peak_kw_lag_1", "base_kw_lag_1", "peak_kw_mean_7d", "evening_share_mean_7d"]
CALENDAR_FEATURES = ["day_of_week", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos"]
WEATHER_FEATURES = ["t2m_lag_1", "t2m_max_lag_1", "t2m_mean_3d", "cdd_lag_1", "cdd_sum_7d", "rh2m_lag_1", "precip_lag_1"]
NUMERIC_FEATURES = HISTORY_FEATURES + LOAD_SHAPE_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FORECAST_PROXY_FEATURES = ["t2m_max_target_day", "cdd_target_day", "rh2m_target_day"]


def join_weather(mains: pd.DataFrame, india_daily: pd.DataFrame, region_weather: dict[str, str],
                 cdd_base_c: float, clog: CleaningLog) -> pd.DataFrame:
    unknown = set(mains["region"]) - set(region_weather)
    if unknown:
        raise KeyError(f"no weather city configured for eMARC regions {sorted(unknown)}")
    w = india_daily[["location", "date", "t2m_c", "t2m_max_c", "rh2m_pct", "precip_mm_day"]].copy()
    w["cdd_c"] = (w["t2m_c"] - cdd_base_c).clip(lower=0)
    out = mains.assign(weather_location=mains["region"].map(region_weather)).merge(
        w.rename(columns={"location": "weather_location"}), on=["weather_location", "date"], how="left")
    clog.add("india consumption features", f"join NASA POWER weather of the mapped city; cooling degree-days base {cdd_base_c} °C",
             len(mains), len(out), int(out["t2m_c"].isna().sum()),
             ", ".join(f"{r} -> {c}" for r, c in region_weather.items()))
    return out


def _features_one(g: pd.DataFrame, min_periods_frac: float) -> pd.DataFrame:
    g = g.set_index("date").sort_index()
    f = history_features(g["kwh"], min_periods_frac)
    f[TARGET] = g["kwh"]
    f["target_is_interpolated"] = g["is_interpolated"]
    f["target_is_outlier"] = g["is_outlier"]
    f["peak_kw_lag_1"] = g["peak_kw"].shift(1)
    f["base_kw_lag_1"] = g["base_kw"].shift(1)
    f["peak_kw_mean_7d"] = g["peak_kw"].shift(1).rolling(7, min_periods=4).mean()
    f["evening_share_mean_7d"] = g["evening_share"].shift(1).rolling(7, min_periods=4).mean()
    f["solar_hours_share_mean_7d"] = g["solar_hours_share"].shift(1).rolling(7, min_periods=4).mean()
    f["appliance_kwh_lag_1"] = g["appliance_kwh"].shift(1)       # only homes with an appliance meter
    f["t2m_lag_1"] = g["t2m_c"].shift(1)
    f["t2m_max_lag_1"] = g["t2m_max_c"].shift(1)
    f["t2m_mean_3d"] = g["t2m_c"].shift(1).rolling(3).mean()
    f["cdd_lag_1"] = g["cdd_c"].shift(1)
    f["cdd_sum_7d"] = g["cdd_c"].shift(1).rolling(7).sum()
    f["rh2m_lag_1"] = g["rh2m_pct"].shift(1)
    f["precip_lag_1"] = g["precip_mm_day"].shift(1)
    f["t2m_max_target_day"] = g["t2m_max_c"]
    f["cdd_target_day"] = g["cdd_c"]
    f["rh2m_target_day"] = g["rh2m_pct"]
    for col in ["household_id", "household_type", "region"]:
        f[col] = g[col].iloc[0]
    return f


def build_india_consumption_features(mains: pd.DataFrame, min_periods_frac: float = 0.8) -> pd.DataFrame:
    """Long feature table for model-eligible mains meters."""
    m = mains[mains["model_eligible"]]
    parts = [_features_one(g, min_periods_frac).assign(deployment_id=dep) for dep, g in m.groupby("deployment_id")]
    f = pd.concat(parts).reset_index()
    years = range(f["date"].dt.year.min(), f["date"].dt.year.max() + 1)
    add_calendar_features(f, india_holidays(years))
    for name, col in BASELINES.items():
        f[name] = f[col]
    return f


def finalize(features: pd.DataFrame, cfg: dict, clog: CleaningLog) -> pd.DataFrame:
    """Drop rows with a missing or interpolated target or missing features; attach split and test period."""
    before = len(features)
    out = features[features[TARGET].notna() & ~features["target_is_interpolated"]]
    clog.add("india consumption features", "drop rows whose target is missing or interpolated (targets must be measured)",
             before, len(out))
    n = len(out)
    out = out.dropna(subset=NUMERIC_FEATURES).copy()
    clog.add("india consumption features", "drop rows with missing features", n, len(out),
             note="first 30 days of each meter, days around gaps, days without a usable 15-minute load shape")
    out["split"] = assign_split(pd.DatetimeIndex(out["date"]), cfg["split"]["train_end"], cfg["split"]["valid_end"]).values
    out["period"] = (out["date"] >= pd.Timestamp(cfg["lockdown_start"])).map({True: "lockdown", False: "normal"})
    counts = out["split"].value_counts()
    clog.fact("india_consumption_split_rows", {k: int(counts.get(k, 0)) for k in ("train", "valid", "test")})
    clog.fact("india_consumption_split_meters", {k: int(g["deployment_id"].nunique()) for k, g in out.groupby("split")})
    clog.fact("india_consumption_test_periods", out.loc[out["split"] == "test", "period"].value_counts().to_dict())
    lead = ID_COLUMNS + ["split", "period", TARGET]
    return out[lead + [c for c in out.columns if c not in lead]].sort_values(["deployment_id", "date"]).reset_index(drop=True)
