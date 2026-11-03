"""Data layer for the dashboard (Phase 6).

Reads the files written by the pipeline and applies the saved models; it never trains a model and never
writes a file. Every table it returns says what kind of number it holds (measured, model output,
estimate or synthetic) so the pages can label it.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml

from solvia.config import get_path, load_config, project_path
from solvia.models import consumption as cons
from solvia.rules import energy_balance as eb

log = logging.getLogger(__name__)

# What a number is. Pages show these labels next to every chart and table.
KINDS = {
    "MEASURED": "Measured by a meter or an inverter.",
    "MODEL": "Output of a saved model (trained in Phase 4, tested once in Phase 5).",
    "ESTIMATE": "Calculated from long-term weather and assumptions, not measured.",
    "SYNTHETIC": "Unverified source treated as synthetic: dashboard testing only, never in results.",
}


def metrics_json(name: str) -> dict:
    return json.loads((get_path("metrics") / f"{name}.json").read_text(encoding="utf-8"))


def metrics_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(get_path("metrics") / f"{name}.csv")


def interim(name: str) -> pd.DataFrame:
    return pd.read_parquet(get_path("interim") / f"{name}.parquet")


def saved_model(name: str):
    return joblib.load(get_path("models") / f"{name}.joblib")


def model_metadata(name: str) -> dict:
    return json.loads((get_path("models") / f"{name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- consumption
def consumption_replay(track: str) -> pd.DataFrame:
    """Validation and test days with the saved model's next-day forecast, its 80 % interval and the best baseline.
    Test is always out-of-sample; validation is in-sample when the saved model was refitted on train + validation
    (listed in attrs["in_sample_splits"])."""
    tr = cons.TRACKS[track]
    df = cons.load_track(tr)
    df = df[df["split"].isin(["valid", "test"])].reset_index(drop=True)
    cm = saved_model(f"consumption_{track}")
    baseline = metrics_json(f"consumption_{track}_summary")["best_baseline"]
    lo, hi = cm.predict_interval(df)
    out = pd.DataFrame({"date": df["date"], "split": df["split"], "actual_kwh": df[cons.TARGET],
                        "forecast_kwh": cm.predict(df), "low_kwh": lo, "high_kwh": hi, "baseline_kwh": df[baseline]})
    for col in ("period", tr.meter_col, "household_type", "region"):
        if col and col in df.columns:
            out[col] = df[col].to_numpy()
    fit_on = model_metadata(f"consumption_{track}").get("fit_on", "train")
    out.attrs |= {"model": cm.name, "baseline": baseline,
                  "in_sample_splits": ["valid"] if fit_on == "train_valid" else []}
    return out


def india_meters(replay: pd.DataFrame) -> pd.DataFrame:
    """One row per eMARC meter in the replay, with whether the saved model was fitted on any of its days."""
    df = cons.load_track(cons.TRACKS["india"])
    fit_splits = ["train", "valid"] if model_metadata("consumption_india").get("fit_on") == "train_valid" else ["train"]
    seen = set(df.loc[df["split"].isin(fit_splits), "deployment_id"])     # meters the saved model was fitted on
    m = (replay.groupby("deployment_id")
         .agg(region=("region", "first"), household_type=("household_type", "first"), days=("date", "size"),
              first_day=("date", "min"), last_day=("date", "max"), mean_kwh=("actual_kwh", "mean")).reset_index())
    m["seen_in_training"] = m["deployment_id"].isin(seen)
    per_split = replay.groupby(["deployment_id", "split"]).size().unstack(fill_value=0)
    m["valid_days"] = m["deployment_id"].map(per_split.get("valid", {})).fillna(0).astype(int)
    m["test_days"] = m["deployment_id"].map(per_split.get("test", {})).fillna(0).astype(int)
    return m.sort_values(["region", "household_type", "deployment_id"]).reset_index(drop=True)


def error_summary(r: pd.DataFrame) -> dict:
    """MAE of the forecast and the baseline, and interval coverage, on the rows given."""
    y = r["actual_kwh"]
    return {"days": int(len(r)), "mae_model": float((y - r["forecast_kwh"]).abs().mean()),
            "mae_baseline": float((y - r["baseline_kwh"]).abs().mean()),
            "coverage": float(((y >= r["low_kwh"]) & (y <= r["high_kwh"])).mean()), "mean_kwh": float(y.mean())}


# --------------------------------------------------------------------------- solar
def solar_replay() -> pd.DataFrame:
    """15-minute measured AC power and the saved model's expected output for every inverter."""
    f = pd.read_parquet(get_path("processed") / "solar_features_15min.parquet")
    model = saved_model("solar_expected_output")
    f["expected_kw"] = model.predict_kw(f)
    return f[["timestamp", "date", "plant_id", "inverter_id", "split", "irradiation_kw_m2", "module_temp_c",
              "ac_power_kw", "expected_kw", "train_eligible"]]


def solar_view(replay: pd.DataFrame, plant_id: int, inverter: str | None, days: list) -> pd.DataFrame:
    """Plant total (sum of inverters) or one inverter, 15-minute kW, for the chosen days."""
    d = replay[(replay["plant_id"] == plant_id) & replay["date"].isin(days)]
    if inverter:
        d = d[d["inverter_id"] == inverter]
    return (d.groupby("timestamp").agg(actual_kw=("ac_power_kw", "sum"), expected_kw=("expected_kw", "sum"),
                                       irradiation_kw_m2=("irradiation_kw_m2", "mean"), split=("split", "first"))
            .reset_index())


def solar_daily(replay: pd.DataFrame, plant_id: int, inverter: str | None = None) -> pd.DataFrame:
    hours = load_config()["solar"]["interval_hours"]
    d = replay[replay["plant_id"] == plant_id]
    if inverter:
        d = d[d["inverter_id"] == inverter]
    g = d.groupby("date").agg(actual_kwh=("ac_power_kw", "sum"), expected_kwh=("expected_kw", "sum"),
                              split=("split", "first")).reset_index()
    g[["actual_kwh", "expected_kwh"]] *= hours
    g["ratio"] = g["actual_kwh"] / g["expected_kwh"]
    return g


# --------------------------------------------------------------------------- monitoring and maintenance
def inverter_labels() -> dict[str, str]:
    """Short display names (P1-01 … P2-22, sorted by source id within each plant) for the long inverter ids."""
    inv = interim("solar_daily_inverter")[["plant_id", "inverter_id"]].drop_duplicates().sort_values(["plant_id", "inverter_id"])
    plant_no = {p: k + 1 for k, p in enumerate(sorted(inv["plant_id"].unique()))}
    inv["n"] = inv.groupby("plant_id").cumcount() + 1
    return {r.inverter_id: f"P{plant_no[r.plant_id]}-{r.n:02d}" for r in inv.itertuples()}


def maintenance() -> pd.DataFrame:
    r = interim("maintenance_recommendations")
    r["evidence"] = r["evidence"].map(json.loads)
    return r


# --------------------------------------------------------------------------- energy balance
def locations() -> pd.DataFrame:
    """The 50 NASA POWER cities: name, state, region, coordinates, label for menus."""
    y = yaml.safe_load(project_path("configs/india_locations.yaml").read_text(encoding="utf-8"))["locations"]
    out = pd.DataFrame([{"location": k, "state": v[0], "region": v[1], "lat": v[2], "lon": v[3]} for k, v in y.items()])
    out["label"] = out["location"].str.replace("_", " ").str.title() + " (" + out["state"] + ")"
    clim = interim("india_monthly_climatology")
    return out[out["location"].isin(clim["location"])].sort_values("label").reset_index(drop=True)


def energy_shapes() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    s = interim("energy_balance_shapes")
    loads = {c.removeprefix("load_"): s[c].to_numpy() for c in s.columns if c.startswith("load_")}
    return s["pv_shape"].to_numpy(), loads


def energy_scenario(location: str, kwp: float, daily_kwh: float, performance_ratio: float,
                    household_type: str) -> tuple[pd.DataFrame, dict]:
    """Monthly and annual ESTIMATE for one home (scenario, not a real household)."""
    clim = interim("india_monthly_climatology")
    pv, loads = energy_shapes()
    s = eb.Scenario(location, kwp, daily_kwh, performance_ratio)
    return eb.monthly_balance(s, clim, (pv, loads[household_type])), eb.annual_summary(s, clim, (pv, loads[household_type]))


def typical_use_by_type() -> dict[str, float]:
    """Median daily use of eMARC homes by type (measured days only), from the Phase 4E step."""
    return {r["household_type"]: r["daily_use_kwh"] for r in metrics_json("energy_balance_examples")["scenarios_pune"]}


# --------------------------------------------------------------------------- synthetic demo data
def pv360_daily() -> pd.DataFrame:
    """PV360 daily totals, read from the raw zip without changing it. SYNTHETIC: never use in results."""
    path = project_path(load_config()["datasets"]["pv360_unverified"])
    with zipfile.ZipFile(path) as z:
        d = pd.read_csv(io.BytesIO(z.read("PV360_data/data/daily_with_weather.csv")), parse_dates=["date"])
    out = d.rename(columns={"prod_e": "pv_kwh", "cons_e": "consumption_kwh", "buy_e": "grid_import_kwh",
                            "self_e": "self_use_kwh", "ghi": "ghi_kwh_m2"})
    out = out[["date", "pv_kwh", "consumption_kwh", "grid_import_kwh", "self_use_kwh", "ghi_kwh_m2", "coverage"]]
    out.attrs["kind"] = "SYNTHETIC"
    return out


# --------------------------------------------------------------------------- latest data (2024-2025)
def grid_daily() -> pd.DataFrame:
    """Grid-India all-India daily demand and solar (MEASURED, grid level). Only days with 24 clean hours."""
    d = interim("grid_india_daily")
    d = d[d["valid"]].reset_index(drop=True)
    d.attrs["kind"] = "MEASURED"
    return d


def grid_profile(year: int, month: int) -> pd.DataFrame:
    """Average all-India demand and solar (GW) by hour of the day for one month, clean hours only."""
    h = interim("grid_india_hourly")
    h = h[~h["bad_hour"] & (h["timestamp"].dt.year == year) & (h["timestamp"].dt.month == month)]
    return (h.groupby(h["timestamp"].dt.hour)[["demand_mw", "solar_mw"]].mean().div(1000)
            .rename(columns={"demand_mw": "demand_gw", "solar_mw": "solar_gw"}).rename_axis("hour").reset_index())


def city_recent_solar(location: str, performance_ratio: float, years=(2024, 2025), typical_until: int = 2023) -> pd.DataFrame:
    """ESTIMATE: rooftop output per 1 kWp, month by month, from NASA POWER daily sunlight of the given years,
    next to the typical value for that month (mean of 2001..typical_until). Not a measurement of any panel."""
    w = interim("india_weather_daily")
    w = w[w["location"] == location].assign(year=lambda x: x["date"].dt.year, month=lambda x: x["date"].dt.month)
    kwh = lambda g: g * performance_ratio                       # eb.daily_solar_kwh with kwp = 1
    typical = kwh(w[w["year"] <= typical_until].groupby("month")["ghi_kwh_m2_day"].mean())
    rec = w[w["year"].isin(years)].groupby(["year", "month"])["ghi_kwh_m2_day"].mean().map(kwh).rename("kwh_per_kwp_day")
    out = rec.reset_index()
    out["typical_kwh_per_kwp_day"] = out["month"].map(typical)
    out["vs_typical_pct"] = 100 * (out["kwh_per_kwp_day"] / out["typical_kwh_per_kwp_day"] - 1)
    out["month_start"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    out.attrs |= {"kind": "ESTIMATE", "last_date": w["date"].max()}
    return out
