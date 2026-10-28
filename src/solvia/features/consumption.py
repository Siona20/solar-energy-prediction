"""Next-day consumption features. One row per *target day* d; the target is consumption on d.

Every feature uses only days up to d-1 (shift >= 1), so a row could be built at the end of
day d-1 to forecast day d. Calendar features describe d itself, which is known in advance.

Weather (NASA POWER, Sceaux): the main features use past temperatures only. The observed
temperature of day d is kept separately in FORECAST_PROXY_FEATURES; it stands in for a
next-day weather forecast and is only used in a clearly labelled experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from solvia.features.calendar import french_holidays
from solvia.preprocessing.cleaning_log import CleaningLog

TARGET = "target_kwh"
LAGS = (1, 2, 7, 14)
WINDOWS = (7, 30)
BASELINES = {"baseline_yesterday": "lag_1", "baseline_last_week": "lag_7", "baseline_mean_7d": "roll_mean_7"}

LAG_FEATURES = [f"lag_{k}" for k in LAGS]
ROLL_FEATURES = [f"roll_{s}_{w}" for w in WINDOWS for s in ("mean", "std")]
HISTORY_FEATURES = LAG_FEATURES + ROLL_FEATURES + ["same_weekday_mean_4w", "trend_7_vs_30"]
EXTRA_FEATURES = ["sub3_kwh_lag_1", "peak_power_kw_lag_1"]
CALENDAR_FEATURES = ["day_of_week", "month", "is_weekend", "is_holiday", "is_school_holiday", "doy_sin", "doy_cos"]
WEATHER_FEATURES = ["t2m_lag_1", "t2m_min_lag_1", "t2m_mean_3d", "hdd_lag_1", "hdd_sum_7d", "ghi_lag_1"]
FEATURES = HISTORY_FEATURES + EXTRA_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES
FORECAST_PROXY_FEATURES = ["t2m_target_day", "hdd_target_day"]


def add_weather_features(f: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Past-temperature features (shift >= 1) plus the target-day forecast proxies."""
    t, tmin, hdd = daily["t2m_c"], daily["t2m_min_c"], daily["hdd_c"]
    f["t2m_lag_1"] = t.shift(1)
    f["t2m_min_lag_1"] = tmin.shift(1)
    f["t2m_mean_3d"] = t.shift(1).rolling(3).mean()
    f["hdd_lag_1"] = hdd.shift(1)
    f["hdd_sum_7d"] = hdd.shift(1).rolling(7).sum()
    f["ghi_lag_1"] = daily["ghi_kwh_m2_day"].shift(1)       # dull days -> more lighting
    f["t2m_target_day"] = t
    f["hdd_target_day"] = hdd


def history_features(s: pd.Series, min_periods_frac: float = 0.8) -> pd.DataFrame:
    """Past-consumption features for one series on a complete daily calendar (shift >= 1 throughout)."""
    past = s.shift(1)                                  # value of day d-1 at row d
    f = pd.DataFrame(index=s.index)
    for k in LAGS:
        f[f"lag_{k}"] = s.shift(k)
    for w in WINDOWS:
        roll = past.rolling(w, min_periods=int(w * min_periods_frac))
        f[f"roll_mean_{w}"] = roll.mean()
        f[f"roll_std_{w}"] = roll.std()
    f["same_weekday_mean_4w"] = pd.concat([s.shift(7 * k) for k in range(1, 5)], axis=1).mean(axis=1, skipna=True)
    f["trend_7_vs_30"] = f["roll_mean_7"] - f["roll_mean_30"]
    return f


def add_calendar_features(f: pd.DataFrame, holidays: set, school_holidays: set | None = None) -> None:
    idx = pd.DatetimeIndex(f.index if isinstance(f.index, pd.DatetimeIndex) else f["date"])
    f["day_of_week"] = idx.day_of_week
    f["month"] = idx.month
    f["is_weekend"] = (idx.day_of_week >= 5).astype(int)
    f["is_holiday"] = np.isin(idx.date, list(holidays)).astype(int)
    if school_holidays is not None:
        f["is_school_holiday"] = np.isin(idx.date, list(school_holidays)).astype(int)
    doy = 2 * np.pi * (idx.day_of_year - 1) / 365.25
    f["doy_sin"], f["doy_cos"] = np.sin(doy), np.cos(doy)


def build_consumption_features(daily: pd.DataFrame, school_holidays: set, min_periods_frac: float = 0.8) -> pd.DataFrame:
    """Feature table indexed by target date. Rolling windows tolerate a few missing days.
    `daily` must contain the household columns and the joined weather columns (t2m_c, t2m_min_c, hdd_c, ghi)."""
    s = daily["consumption_kwh"]
    f = pd.DataFrame(index=daily.index)
    f[TARGET] = s
    f = f.join(history_features(s, min_periods_frac))
    f["sub3_kwh_lag_1"] = daily["sub3_kwh"].shift(1)
    f["peak_power_kw_lag_1"] = daily["peak_power_kw"].shift(1)
    add_weather_features(f, daily)
    add_calendar_features(f, french_holidays(range(f.index.year.min(), f.index.year.max() + 1)), school_holidays)
    for name, col in BASELINES.items():
        f[name] = f[col]
    f.index.name = "date"
    return f


def assign_split(index: pd.DatetimeIndex, train_end: str, valid_end: str) -> pd.Series:
    """Chronological labels: train <= train_end < valid <= valid_end < test."""
    split = np.where(index <= pd.Timestamp(train_end), "train",
                     np.where(index <= pd.Timestamp(valid_end), "valid", "test"))
    return pd.Series(split, index=index, name="split")


def finalize(features: pd.DataFrame, split_cfg: dict, clog: CleaningLog) -> pd.DataFrame:
    """Drop rows whose target or any feature is missing, then attach split labels."""
    before = len(features)
    out = features.dropna(subset=[TARGET, *FEATURES]).copy()
    clog.add("consumption features", "drop rows with missing target or features", before, len(out),
             note="first 30 days (rolling windows) and days around the missing-data gaps")
    out["split"] = assign_split(out.index, split_cfg["train_end"], split_cfg["valid_end"])
    counts = out["split"].value_counts()
    clog.fact("consumption_split_rows", {k: int(counts.get(k, 0)) for k in ("train", "valid", "test")})
    clog.fact("consumption_split_ranges", {k: f"{g.index.min().date()} to {g.index.max().date()}" for k, g in out.groupby("split")})
    return out
