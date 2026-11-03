"""Tomorrow and the next 7 days for one home: weather, solar units and electricity use.

Chains the three saved models: weather (models/weather_nextdays.joblib) -> solar units (models/pv_home.json) and
home use (models/home_use_simple.joblib). Weather comes from NASA POWER up to its last published day; the days
between that day and tomorrow are forecast too, so the "horizon" of tomorrow is usually 8-10 days.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from solvia.config import get_path, load_config
from solvia.data import live_weather
from solvia.models import pv_home
from solvia.models.weather import MAX_H, VARS

log = logging.getLogger(__name__)

TEMPS = ["tmax", "tmin", "tmean"]
DAYS = 7


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    location: str | None = None       # one of the 50 stored cities, else None


def load_models() -> dict:
    m = get_path("models")
    return {"weather": joblib.load(m / "weather_nextdays.joblib"), "pv": pv_home.load(),
            "home": joblib.load(m / "home_use_simple.joblib")}


def weather_forecast(models: dict, w: pd.DataFrame, place: Place, today: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Daily weather from the day after the last published day to today + DAYS: measured where NASA has it,
    forecast otherwise. Temperatures and sunlight are forecast from their own last published day."""
    wf = models["weather"]
    end = today + pd.Timedelta(days=DAYS)
    o_t = live_weather.last_complete(w, [VARS[k] for k in TEMPS])
    o_g = live_weather.last_complete(w, [VARS["ghi"]])
    if (end - min(o_t, o_g)).days > MAX_H:
        raise ValueError(f"NASA POWER data for this place ends {min(o_t, o_g):%d %b %Y}: too old to forecast to {end:%d %b}")
    ft = wf.forecast(w[w.index <= o_t], place.lat, place.lon, o_t, range(1, (end - o_t).days + 1))
    fg = wf.forecast(w[w.index <= o_g], place.lat, place.lon, o_g, range(1, (end - o_g).days + 1))
    cols_t = [c for c in ft.columns if c.split("_")[0] in TEMPS]
    cols_g = [c for c in fg.columns if c.startswith("ghi")]
    out = ft.set_index("target_date")[cols_t].join(fg.set_index("target_date")[cols_g], how="outer")
    out = out.rename_axis("date")
    out["h_temp"] = (out.index - o_t).days
    out["h_sun"] = (out.index - o_g).days
    return out, {"temperature_data_end": o_t, "sunlight_data_end": o_g}


def sunny_reference(w: pd.DataFrame) -> dict:
    """A clear summer day for calibration: 95th percentile of March-May sunlight, with the mean max/min of those days."""
    spring = w[w.index.month.isin([3, 4, 5])].dropna(subset=[VARS["ghi"], VARS["tmax"], VARS["tmin"]])
    q = spring[VARS["ghi"]].quantile(0.95)
    top = spring[spring[VARS["ghi"]] >= q]
    return {"sunlight": float(q), "tmax": float(top[VARS["tmax"]].mean()), "tmin": float(top[VARS["tmin"]].mean())}


def plan(models: dict, place: Place, kwp: float, last7: list[float], household_type: str,
         today: pd.Timestamp, sunny_day_units: float | None = None, w: pd.DataFrame | None = None) -> dict:
    """Everything the forecast page shows. `last7` = units used on the 7 days up to and including today."""
    today = pd.Timestamp(today).normalize()
    if len(last7) != 7 or min(last7) < 0:
        raise ValueError("give 7 daily values (kWh), none negative")
    w = live_weather.history(place.lat, place.lon, today, place.location) if w is None else w
    wx, ends = weather_forecast(models, w, place, today)
    pv: pv_home.PVHomeModel = models["pv"]
    ref = sunny_reference(w)
    scale = pv.calibration_scale(kwp, sunny_day_units, ref["sunlight"], ref["tmax"], ref["tmin"]) if sunny_day_units else 1.0

    days = pd.date_range(today + pd.Timedelta(days=1), periods=DAYS)
    cdd_base = load_config()["emarc"]["cdd_base_c"]
    tmean = pd.concat([w.loc[w.index <= ends["temperature_data_end"], VARS["tmean"]], wx["tmean"]]).sort_index()
    tmean = tmean[~tmean.index.duplicated(keep="first")]
    home_in = pd.DataFrame({"date": days, "t2m_max_target_day": wx.loc[days, "tmax"].to_numpy(),
                            "cdd_target_day": (wx.loc[days, "tmean"] - cdd_base).clip(lower=0).to_numpy(),
                            "t2m_mean_3d": [tmean.loc[d - pd.Timedelta(days=3): d - pd.Timedelta(days=1)].mean() for d in days]})
    use = models["home"].forecast(last7, household_type, home_in).set_index("date")

    t = wx.loc[days].copy()
    t.index.name = "date"
    t["solar_kwh"] = pv.daily_kwh(kwp, t["ghi"], t["tmax"], t["tmin"], scale)
    t["solar_low_kwh"] = pv.daily_kwh(kwp, t["ghi_low"], t["tmax"], t["tmin"], scale)
    t["solar_high_kwh"] = pv.daily_kwh(kwp, t["ghi_high"], t["tmax"], t["tmin"], scale)
    t["use_kwh"], t["use_low_kwh"], t["use_high_kwh"] = use["kwh"], use["low"], use["high"]
    t["net_kwh"] = t["solar_kwh"] - t["use_kwh"]
    return {"days": t.reset_index(), "data_end": ends, "calibration_scale": scale, "sunny_reference": ref,
            "sunny_day_model_units": float(pv.daily_kwh(kwp, ref["sunlight"], ref["tmax"], ref["tmin"], scale)),
            "history": w}
