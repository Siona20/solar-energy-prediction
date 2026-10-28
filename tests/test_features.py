"""Phase 3 tests: no future data in consumption features, solar leakage guard, holidays, splits."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from solvia.features import consumption as cf
from solvia.features import solar as sf
from solvia.features.calendar import easter_sunday, french_holidays


SCHOOL = {date(2008, 2, 20), date(2008, 2, 21)}


def _daily(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", periods=n, freq="D", name="date")
    t = rng.uniform(-5, 25, n)
    return pd.DataFrame({"consumption_kwh": rng.uniform(10, 40, n), "sub3_kwh": rng.uniform(2, 10, n),
                         "peak_power_kw": rng.uniform(2, 8, n), "t2m_c": t, "t2m_min_c": t - 4,
                         "hdd_c": np.clip(18 - t, 0, None), "ghi_kwh_m2_day": rng.uniform(0.5, 7, n)}, index=idx)


def test_consumption_features_never_use_target_day_or_future():
    daily = _daily()
    base = cf.build_consumption_features(daily, SCHOOL)
    d = daily.index[80]
    changed = daily.copy()
    changed.loc[d:, :] *= 100                                                # change day d and everything after
    after = cf.build_consumption_features(changed, SCHOOL)
    pd.testing.assert_frame_equal(base.loc[d, cf.FEATURES].to_frame(), after.loc[d, cf.FEATURES].to_frame())
    assert after.loc[d, cf.TARGET] != base.loc[d, cf.TARGET]                 # only the target moved
    assert after.loc[d, "t2m_target_day"] != base.loc[d, "t2m_target_day"]   # proxy is target-day info by design
    assert not set(cf.FORECAST_PROXY_FEATURES) & set(cf.FEATURES)


def test_lag_definitions():
    daily = _daily()
    f = cf.build_consumption_features(daily, SCHOOL)
    d = daily.index[50]
    assert f.loc[d, "lag_1"] == daily["consumption_kwh"].iloc[49]
    assert f.loc[d, "lag_7"] == daily["consumption_kwh"].iloc[43]
    assert f.loc[d, "roll_mean_7"] == pytest.approx(daily["consumption_kwh"].iloc[43:50].mean())
    assert f.loc[d, "baseline_last_week"] == f.loc[d, "lag_7"]


def test_solar_features_exclude_output_derived_columns():
    assert not sf.FORBIDDEN.intersection(sf.FEATURES)
    with pytest.raises(ValueError):
        sf.check_no_leakage(sf.FEATURES + ["daily_yield_kwh"])


def test_easter_and_french_holidays():
    assert easter_sunday(2008) == date(2008, 3, 23)
    assert easter_sunday(2010) == date(2010, 4, 4)
    h = french_holidays([2008])
    assert date(2008, 3, 24) in h          # Easter Monday
    assert date(2008, 5, 1) in h           # Labour Day
    assert date(2008, 5, 12) in h          # Whit Monday (Easter + 50)
    assert len(h) == 10                    # 2008: Ascension (Easter + 39) fell on 1 May, Labour Day
    assert len(french_holidays([2009])) == 11


def test_splits_are_chronological_and_disjoint():
    idx = pd.date_range("2008-12-30", periods=5, freq="D")
    s = cf.assign_split(idx, "2008-12-31", "2009-01-01")
    assert s.tolist() == ["train", "train", "valid", "test", "test"]
    dates = pd.Series(pd.date_range("2020-05-15", periods=34, freq="D"))
    split = sf.assign_day_split(dates, {"train": 0.70, "valid": 0.15, "test": 0.15})
    assert split.value_counts().to_dict() == {"train": 24, "valid": 5, "test": 5}
    assert dates[split == "train"].max() < dates[split == "valid"].min() < dates[split == "test"].min()


def test_school_holiday_flag():
    f = cf.build_consumption_features(_daily(), SCHOOL)
    assert f.loc["2008-02-20", "is_school_holiday"] == 1
    assert f.loc["2008-02-22", "is_school_holiday"] == 0
