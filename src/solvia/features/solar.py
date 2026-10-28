"""Expected-solar-output features. One row per 15-minute interval per inverter.

Only weather and time of day are features. DAILY_YIELD, TOTAL_YIELD and DC_POWER are computed
from the output being predicted, so they can never be features (enforced by `check_no_leakage`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog

TARGET = "ac_power_kw"
FEATURES = ["irradiation_kw_m2", "module_temp_c", "ambient_temp_c", "module_minus_ambient_c", "hour_sin", "hour_cos"]
FORBIDDEN = {"daily_yield_kwh", "total_yield_kwh", "dc_power_raw", "energy_kwh", TARGET}
KEYS = ["timestamp", "date", "plant_id", "inverter_id"]
FLAGS = ["is_daylight", "flag_off_in_daylight", "flag_power_at_night"]


def check_no_leakage(features: list[str]) -> None:
    """Raise if any output-derived column is used as a feature."""
    bad = FORBIDDEN.intersection(features)
    if bad:
        raise ValueError(f"Leakage: output-derived columns used as features: {sorted(bad)}")


def build_solar_features(df15: pd.DataFrame) -> pd.DataFrame:
    check_no_leakage(FEATURES)
    hour = df15["timestamp"].dt.hour + df15["timestamp"].dt.minute / 60
    out = df15[["timestamp", "plant_id", "inverter_id", "irradiation_kw_m2", "module_temp_c",
                "ambient_temp_c", TARGET, *FLAGS]].copy()
    out.insert(1, "date", df15["timestamp"].dt.floor("D"))
    out["module_minus_ambient_c"] = out["module_temp_c"] - out["ambient_temp_c"]
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["train_eligible"] = ~out["flag_off_in_daylight"]          # decision D5
    return out[KEYS + FEATURES + [TARGET] + FLAGS + ["train_eligible"]]


def assign_day_split(dates: pd.Series, fractions: dict[str, float]) -> pd.Series:
    """Chronological split by calendar day: first ~70% of days train, next ~15% valid, rest test."""
    days = np.sort(dates.unique())
    n_train = int(round(len(days) * fractions["train"]))
    n_valid = int(round(len(days) * fractions["valid"]))
    label = {d: "train" if i < n_train else "valid" if i < n_train + n_valid else "test" for i, d in enumerate(days)}
    return dates.map(label).rename("split")


def finalize(features: pd.DataFrame, fractions: dict[str, float], clog: CleaningLog) -> pd.DataFrame:
    before = len(features)
    out = features.dropna(subset=FEATURES + [TARGET]).copy()
    clog.add("solar features", "drop rows with missing features or target", before, len(out))
    out["split"] = assign_day_split(out["date"], fractions)
    for name, g in out.groupby("split"):
        clog.fact(f"solar_split_{name}", {"days": int(g["date"].nunique()), "rows": int(len(g)),
                                         "train_eligible_rows": int(g["train_eligible"].sum()),
                                         "range": f"{g['date'].min().date()} to {g['date'].max().date()}"})
    return out
