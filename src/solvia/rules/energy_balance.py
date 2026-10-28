"""Energy balance for a user scenario: rooftop solar estimate vs household consumption.

Solar estimate (labelled "estimate" everywhere):
    daily kWh = installed kWp x GHI (kWh/m²/day) x performance ratio
GHI in kWh/m²/day equals "peak sun hours" because panels are rated at 1 kW/m². GHI is on a
horizontal surface; a tilted roof usually receives a little more, so this is slightly conservative.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

NOTES = [
    "Solar output is an ESTIMATE: kWp × long-term NASA POWER irradiance × performance ratio, not a measurement.",
    "Real savings depend on when you use electricity (solar is produced in daytime only), net-metering rules and tariffs.",
    "Range shows a typical poor month (10th percentile) to a typical good month (90th percentile) of 2001–2025.",
    "Direct use (solar used at the moment it is produced) combines the measured 15-minute shape of the Kaggle "
    "plants' output (southern India, May–June) with the measured eMARC load shape of the household type; "
    "it is an estimate for an average day, not a simulation of each day.",
]


def daily_solar_kwh(kwp: float, ghi_kwh_m2_day: float, performance_ratio: float) -> float:
    if kwp < 0 or ghi_kwh_m2_day < 0 or not 0 < performance_ratio <= 1:
        raise ValueError("kWp and GHI must be >= 0 and performance ratio in (0, 1]")
    return kwp * ghi_kwh_m2_day * performance_ratio


def direct_use_kwh(solar_kwh: float, consumption_kwh: float, pv_shape: np.ndarray, load_shape: np.ndarray) -> float:
    """Energy used at the moment it is produced: sum over the day's blocks of min(solar, load).
    Both shapes are the share of the daily total in each block (each sums to 1)."""
    return float(np.minimum(solar_kwh * np.asarray(pv_shape), consumption_kwh * np.asarray(load_shape)).sum())


def balance(solar_kwh: float, consumption_kwh: float) -> dict:
    """Surplus (> 0) or shortfall (< 0) and how much of consumption solar could cover."""
    diff = solar_kwh - consumption_kwh
    return {"solar_kwh": solar_kwh, "consumption_kwh": consumption_kwh, "balance_kwh": diff,
            "status": "surplus" if diff > 0 else "shortfall" if diff < 0 else "balanced",
            "coverage_pct": 100 * solar_kwh / consumption_kwh if consumption_kwh > 0 else float("nan")}


@dataclass
class Scenario:
    location: str
    kwp: float
    daily_consumption_kwh: float
    performance_ratio: float = 0.75


def monthly_balance(s: Scenario, climatology: pd.DataFrame, shapes: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Twelve rows: estimated solar, consumption and balance per month for one location.
    With `shapes` = (pv_shape, load_shape) it also splits each day into direct use, export and grid import."""
    c = climatology[climatology["location"] == s.location].sort_values("month")
    if len(c) != 12:
        raise KeyError(f"no 12-month climatology for location '{s.location}'")
    days = c["month"].map(lambda m: 28.25 if m == 2 else calendar.monthrange(2023, m)[1])
    out = pd.DataFrame({"month": c["month"].values, "month_name": [calendar.month_abbr[m] for m in c["month"]],
                        "days": days.values, "ghi_kwh_m2_day": c["ghi_kwh_m2_day"].values})
    for col, ghi in [("solar_kwh_day", "ghi_kwh_m2_day"), ("solar_low_kwh_day", "ghi_p10_kwh_m2_day"),
                     ("solar_high_kwh_day", "ghi_p90_kwh_m2_day")]:
        out[col] = [daily_solar_kwh(s.kwp, g, s.performance_ratio) for g in c[ghi].values]
    out["consumption_kwh_day"] = s.daily_consumption_kwh
    out["balance_kwh_day"] = out["solar_kwh_day"] - out["consumption_kwh_day"]
    out["solar_kwh_month"] = out["solar_kwh_day"] * out["days"]
    out["consumption_kwh_month"] = out["consumption_kwh_day"] * out["days"]
    out["balance_kwh_month"] = out["solar_kwh_month"] - out["consumption_kwh_month"]
    out["status"] = out["balance_kwh_day"].map(lambda b: "surplus" if b > 0 else "shortfall" if b < 0 else "balanced")
    if shapes is not None:
        pv, load = shapes
        out["direct_use_kwh_day"] = [direct_use_kwh(x, s.daily_consumption_kwh, pv, load) for x in out["solar_kwh_day"]]
        out["export_kwh_day"] = out["solar_kwh_day"] - out["direct_use_kwh_day"]
        out["grid_import_kwh_day"] = out["consumption_kwh_day"] - out["direct_use_kwh_day"]
        out["self_sufficiency_pct"] = 100 * out["direct_use_kwh_day"] / out["consumption_kwh_day"]
        out["self_consumption_pct"] = 100 * out["direct_use_kwh_day"] / out["solar_kwh_day"].replace(0, np.nan)
    return out


def annual_summary(s: Scenario, climatology: pd.DataFrame, shapes: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    m = monthly_balance(s, climatology, shapes)
    year = balance(float(m["solar_kwh_month"].sum()), float(m["consumption_kwh_month"].sum()))
    if shapes is not None:
        direct = float((m["direct_use_kwh_day"] * m["days"]).sum())
        year |= {"direct_use_kwh": direct, "export_kwh": year["solar_kwh"] - direct, "grid_import_kwh": year["consumption_kwh"] - direct,
                 "self_sufficiency_pct": 100 * direct / year["consumption_kwh"],
                 "self_consumption_pct": 100 * direct / year["solar_kwh"] if year["solar_kwh"] > 0 else float("nan")}
    return {**year, "location": s.location, "kwp": s.kwp, "performance_ratio": s.performance_ratio,
            "surplus_months": int((m["status"] == "surplus").sum()),
            "worst_month": m.loc[m["balance_kwh_day"].idxmin(), "month_name"],
            "best_month": m.loc[m["balance_kwh_day"].idxmax(), "month_name"],
            "kwp_to_cover_consumption": s.daily_consumption_kwh * 365.25 / (m["ghi_kwh_m2_day"].mul(m["days"]).sum() * s.performance_ratio),
            "label": "estimate", "notes": NOTES}


def pv_shape_from_plants(df15: pd.DataFrame, healthy: pd.DataFrame) -> np.ndarray:
    """Mean output per 15-minute block of healthy inverter-days, normalised to sum 1 (96 blocks)."""
    d = df15.merge(healthy[healthy["healthy"]][["plant_id", "inverter_id", "date"]], on=["plant_id", "inverter_id", "date"])
    block = d["timestamp"].dt.hour * 4 + d["timestamp"].dt.minute // 15
    shape = d.groupby(block)["ac_power_kw"].mean().reindex(range(96), fill_value=0.0).clip(lower=0).to_numpy()
    return shape / shape.sum()


def load_shapes_from_profiles(profiles: pd.DataFrame) -> dict[str, np.ndarray]:
    """eMARC mean load per block by household type (meter-weighted over regions), each normalised to sum 1."""
    out = {}
    for t, g in profiles.groupby("household_type"):
        p = g.groupby("block").apply(lambda x: np.average(x["load_kw"], weights=x["n_meters"]), include_groups=False)
        p = p.reindex(range(96)).interpolate().to_numpy()
        out[t] = p / p.sum()
    return out
