"""Forecast tests: solar units behave physically and calibrate to the user's sunny-day units; the weather and home
models give finite next-days values; the whole chain runs offline on stored history (no network)."""
import numpy as np
import pandas as pd
import pytest

from solvia.config import get_path
from solvia.models import home_simple as hs
from solvia.models import pv_home
from solvia.models.weather import climatology

PV = pv_home.PVHomeModel(k_c_per_kw_m2=27.0, a_sunny_air=0.72, b_sunny_irradiance=0.113)


def test_solar_units_grow_with_sunlight_and_fall_with_heat():
    assert PV.daily_kwh(12, 6.5, 35, 22) > PV.daily_kwh(12, 4.0, 35, 22)
    assert PV.daily_kwh(12, 6.5, 42, 28) < PV.daily_kwh(12, 6.5, 30, 18)
    assert PV.daily_kwh(12, 0.0, 35, 22) == 0
    assert PV.daily_kwh(24, 6.0, 35, 22) == pytest.approx(2 * PV.daily_kwh(12, 6.0, 35, 22))


def test_calibration_reproduces_the_users_sunny_day():
    scale = PV.calibration_scale(12, 60, 6.8, 38, 24)            # the owner's example: 12 kW gives 60 units
    assert PV.daily_kwh(12, 6.8, 38, 24, scale) == pytest.approx(60)


def test_climatology_is_smooth_and_complete():
    idx = pd.date_range("2001-01-01", "2010-12-31")
    doy = idx.day_of_year.to_numpy()
    w = pd.DataFrame({c: 30 + 5 * np.sin(2 * np.pi * doy / 365.25) for c in
                      ["t2m_max_c", "t2m_min_c", "t2m_c", "ghi_kwh_m2_day", "precip_mm_day", "rh2m_pct",
                       "clearsky_ghi_kwh_m2_day"]}, index=idx)
    c = climatology(w, "2010-12-31")
    assert len(c) == 366 and c.notna().all().all()
    assert abs(c["t2m_c"].iloc[0] - c["t2m_c"].iloc[-1]) < 0.5      # wraps around the new year


def test_history_row_uses_the_last_seven_days():
    r = hs.history_row([1, 2, 3, 4, 5, 6, 7])
    assert (r["lag_1"], r["lag_2"], r["lag_7"], r["roll_mean_7"]) == (7, 6, 1, 4)


needs_models = pytest.mark.skipif(
    not all((get_path("models") / f).exists() for f in ["weather_nextdays.joblib", "pv_home.json", "home_use_simple.joblib"])
    or not (get_path("interim") / "india_weather_daily.parquet").exists(),
    reason="forecast models or weather data not present (data not pushed)",
)


@needs_models
def test_full_chain_offline_on_stored_history():
    from solvia import forecast as F
    w = pd.read_parquet(get_path("interim") / "india_weather_daily.parquet")
    w = w[w["location"] == "pune"].set_index("date")
    w = w[w.index <= "2025-12-20"]                                 # pretend today is 24 Dec 2025
    r = F.plan(F.load_models(), F.Place("Pune", 18.52, 73.86, "pune"), 12.0, [6.0] * 7, "Basic",
               pd.Timestamp("2025-12-24"), sunny_day_units=60, w=w)
    t = r["days"]
    assert len(t) == 7 and t["date"].iloc[0] == pd.Timestamp("2025-12-25")
    for c in ["tmax", "tmin", "ghi", "solar_kwh", "use_kwh"]:
        assert np.isfinite(t[c]).all() and (t[c] >= 0).all(), c
    assert (t["tmax"] > t["tmin"]).all()
    assert (t["solar_low_kwh"] <= t["solar_kwh"]).all() and (t["solar_kwh"] <= t["solar_high_kwh"]).all()
    assert r["sunny_day_model_units"] == pytest.approx(60)
