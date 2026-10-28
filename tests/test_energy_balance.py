"""Energy-balance arithmetic on a hand-made climatology."""
import numpy as np
import pandas as pd
import pytest

from solvia.rules.energy_balance import (Scenario, annual_summary, balance, daily_solar_kwh, direct_use_kwh,
                                         monthly_balance)


def _clim(ghi: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame({"location": "testville", "month": range(1, 13), "ghi_kwh_m2_day": ghi,
                         "ghi_p10_kwh_m2_day": ghi - 1, "ghi_p90_kwh_m2_day": ghi + 1})


def test_daily_solar_formula():
    assert daily_solar_kwh(3.0, 5.0, 0.75) == pytest.approx(11.25)      # 3 kWp x 5 sun-hours x 0.75
    with pytest.raises(ValueError):
        daily_solar_kwh(3.0, 5.0, 1.5)


def test_balance_signs():
    assert balance(12.0, 10.0)["status"] == "surplus"
    assert balance(8.0, 10.0)["balance_kwh"] == pytest.approx(-2.0)
    assert balance(8.0, 10.0)["coverage_pct"] == pytest.approx(80.0)


def test_monthly_and_annual():
    s = Scenario("testville", kwp=2.0, daily_consumption_kwh=7.5, performance_ratio=0.75)
    m = monthly_balance(s, _clim())
    assert len(m) == 12
    assert m["solar_kwh_day"].iloc[0] == pytest.approx(7.5)              # exactly balanced
    assert m["solar_low_kwh_day"].iloc[0] == pytest.approx(6.0)
    a = annual_summary(s, _clim())
    assert a["balance_kwh"] == pytest.approx(0.0, abs=1e-9)
    assert a["kwp_to_cover_consumption"] == pytest.approx(2.0)
    assert a["label"] == "estimate"


def test_unknown_location_raises():
    with pytest.raises(KeyError):
        monthly_balance(Scenario("nowhere", 1.0, 5.0), _clim())


def test_direct_use_same_shape_is_min_and_disjoint_shapes_is_zero():
    day = np.zeros(96); day[40:60] = 1 / 20                                # all energy 10:00-15:00
    night = np.zeros(96); night[76:96] = 1 / 20                            # all use 19:00-24:00
    assert direct_use_kwh(8.0, 5.0, day, day) == pytest.approx(5.0)       # same shape: min(solar, load)
    assert direct_use_kwh(8.0, 5.0, day, night) == pytest.approx(0.0)     # no overlap: nothing used directly


def test_monthly_direct_use_columns_add_up():
    flat = np.full(96, 1 / 96)
    day = np.zeros(96); day[40:60] = 1 / 20
    s = Scenario("testville", kwp=2.0, daily_consumption_kwh=7.5, performance_ratio=0.75)
    m = monthly_balance(s, _clim(), (day, flat))
    r = m.iloc[0]
    assert r["direct_use_kwh_day"] == pytest.approx(7.5 * 20 / 96)        # load in the 20 sunny blocks
    assert r["direct_use_kwh_day"] + r["export_kwh_day"] == pytest.approx(r["solar_kwh_day"])
    assert r["direct_use_kwh_day"] + r["grid_import_kwh_day"] == pytest.approx(r["consumption_kwh_day"])
    a = annual_summary(s, _clim(), (day, flat))
    assert a["self_sufficiency_pct"] == pytest.approx(100 * 20 / 96)
