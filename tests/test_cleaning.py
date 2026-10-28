"""Phase 1 tests: datetime parsing, daily kWh maths, gap filling, solar interval energy and flags."""
import numpy as np
import pandas as pd
import pytest

from solvia.data.kaggle_solar import DATETIME_FORMATS
from solvia.data.uci import parse_uci_datetime
from solvia.preprocessing.cleaning_log import CleaningLog
from solvia.preprocessing.household import daily_from_minutes, interpolate_short_gaps
from solvia.preprocessing.solar import clean_plant, daily_per_inverter


def test_uci_datetime_is_day_first():
    ts = parse_uci_datetime(pd.Series(["01/02/2007"]), pd.Series(["13:05:00"]))
    assert ts.iloc[0] == pd.Timestamp("2007-02-01 13:05:00")  # 1 Feb, not 2 Jan


def test_kaggle_plant1_generation_format_is_day_first():
    ts = pd.to_datetime(pd.Series(["03-06-2020 14:15"]), format=DATETIME_FORMATS["Plant_1_Generation_Data.csv"])
    assert ts.iloc[0] == pd.Timestamp("2020-06-03 14:15")


def _minutes(kw: float, wh_sub: float, n: int = 1440) -> pd.DataFrame:
    idx = pd.date_range("2008-01-01", periods=n, freq="min")
    return pd.DataFrame({"Global_active_power": kw, "Voltage": 240.0, "Sub_metering_1": wh_sub,
                         "Sub_metering_2": 0.0, "Sub_metering_3": 0.0}, index=idx)


def test_daily_kwh_constant_load():
    daily = daily_from_minutes(_minutes(kw=1.5, wh_sub=10.0))
    row = daily.iloc[0]
    assert row["consumption_kwh"] == pytest.approx(36.0)        # 1.5 kW x 24 h
    assert row["sub1_kwh"] == pytest.approx(14.4)               # 10 Wh/min x 1440 / 1000
    assert row["unmetered_kwh"] == pytest.approx(36.0 - 14.4)
    assert row["coverage"] == 1.0


def test_daily_kwh_is_coverage_adjusted():
    m = _minutes(kw=2.0, wh_sub=0.0)
    m.iloc[:144, 0] = np.nan                                     # 10% of the day missing
    row = daily_from_minutes(m).iloc[0]
    assert row["coverage"] == pytest.approx(0.9)
    assert row["consumption_kwh"] == pytest.approx(48.0)        # not 43.2


def test_interpolation_fills_only_short_gaps():
    s = pd.Series([10, np.nan, np.nan, 40, np.nan, np.nan, np.nan, 80], dtype=float)
    filled, mask = interpolate_short_gaps(s, max_days=2)
    assert filled.iloc[1:3].tolist() == [20.0, 30.0]
    assert filled.iloc[4:7].isna().all()                         # 3-day gap left missing
    assert mask.sum() == 2


def _plant_frames():
    ts = pd.date_range("2020-05-15", periods=96, freq="15min")
    irr = np.where((ts.hour >= 6) & (ts.hour < 18), 0.5, 0.0)
    gen = pd.DataFrame({"DATE_TIME": ts, "PLANT_ID": 1, "SOURCE_KEY": "inv", "DC_POWER": 0.0,
                        "AC_POWER": np.where(irr > 0, 100.0, 0.0), "DAILY_YIELD": 0.0, "TOTAL_YIELD": 0.0})
    gen.loc[40, "AC_POWER"] = 0.0                                # switched off at 10:00 in sunlight
    wx = pd.DataFrame({"DATE_TIME": ts, "PLANT_ID": 1, "SOURCE_KEY": "s", "AMBIENT_TEMPERATURE": 25.0,
                       "MODULE_TEMPERATURE": 35.0, "IRRADIATION": irr})
    return gen, wx


def test_solar_interval_energy_and_off_flag():
    gen, wx = _plant_frames()
    df = clean_plant(gen, wx, 1, interval_h=0.25, off_irr=0.1, clog=CleaningLog())
    assert len(df) == 96
    assert df["flag_off_in_daylight"].sum() == 1
    daily = daily_per_inverter(df, 0.25)
    # 48 daylight intervals, one off -> 47 x 100 kW x 0.25 h
    assert daily["energy_kwh"].iloc[0] == pytest.approx(47 * 25.0)
    assert daily["irradiation_kwh_m2"].iloc[0] == pytest.approx(48 * 0.5 * 0.25)


def test_daily_yield_ignores_stale_counter_after_midnight():
    gen, wx = _plant_frames()
    gen["DAILY_YIELD"] = np.linspace(0, 1175, 96)
    gen.loc[0, "DAILY_YIELD"] = 9425.0                           # yesterday's total before reset
    df = clean_plant(gen, wx, 1, interval_h=0.25, off_irr=0.1, clog=CleaningLog())
    assert daily_per_inverter(df, 0.25)["daily_yield_kwh"].iloc[0] == pytest.approx(1175.0)
