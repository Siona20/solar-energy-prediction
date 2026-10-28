"""Phase 4B tests: baseline maths, healthy reference, night rule, training rows, daily evaluation."""
import numpy as np
import pandas as pd
import pytest

from solvia.features.solar import FEATURES
from solvia.features.preprocess import make_pipeline
from solvia.models.common import regression_metrics
from solvia.models.solar import (ExpectedSolarModel, IrradiationBaseline, evaluate, healthy_inverter_days,
                                 model_rows, training_mask)


def test_baseline_recovers_k():
    X = pd.DataFrame({"irradiation_kw_m2": [0.2, 0.5, 1.0]})
    assert IrradiationBaseline().fit(X, 1000 * X["irradiation_kw_m2"]).k_ == pytest.approx(1000)


def test_healthy_reference_compares_with_same_day_p90():
    daily = pd.DataFrame({"plant_id": 1, "date": pd.Timestamp("2020-05-15"), "irradiation_kwh_m2": 5.0,
                          "inverter_id": list("abcdefghij"), "energy_kwh": [1000.0] * 9 + [500.0]})
    h = healthy_inverter_days(daily, 0.9).set_index("inverter_id")
    assert h.loc["a", "healthy"] and not h.loc["j", "healthy"]
    assert h.loc["j", "specific_yield_ratio"] == pytest.approx(0.5)


def _frame() -> pd.DataFrame:
    n = 8
    f = pd.DataFrame({"timestamp": pd.date_range("2020-05-15 10:00", periods=n, freq="15min"), "plant_id": 1,
                      "inverter_id": "a", "irradiation_kw_m2": [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
                      "module_temp_c": 40.0, "ambient_temp_c": 30.0, "module_minus_ambient_c": 10.0,
                      "hour_sin": 0.0, "hour_cos": 1.0, "ac_power_kw": [0, 500, 500, 500, 500, 500, 500, 0],
                      "train_eligible": True, "split": ["train"] * 4 + ["valid"] * 4})
    f["date"] = f["timestamp"].dt.floor("D") + pd.to_timedelta(f.index // 4, unit="D")
    return f


def test_night_rule_and_no_negative_predictions():
    f = _frame()
    pipe = make_pipeline(IrradiationBaseline(), FEATURES, linear=False).fit(f[FEATURES], f["ac_power_kw"])
    m = ExpectedSolarModel("baseline", "pooled", "all_operating", {"all": pipe})
    pred = m.predict_kw(f)
    assert pred[0] == 0 and pred[-1] == 0 and (pred >= 0).all()
    assert pred[1] == pytest.approx(500)


def test_training_rows_are_train_daytime_eligible_and_healthy():
    f = _frame()
    healthy = pd.DataFrame({"plant_id": 1, "inverter_id": "a", "date": sorted(f["date"].unique()),
                            "specific_yield_ratio": [1.0, 0.5], "healthy": [True, False]})
    f = model_rows(f, healthy)
    f.loc[2, "train_eligible"] = False
    m_all, m_h = training_mask(f, "all_operating"), training_mask(f, "healthy_reference")
    assert (f.loc[m_all, "split"] == "train").all() and f.loc[m_all, "daytime"].all()
    assert m_all.sum() == 2                                  # rows 1, 3 (row 0 night, row 2 not eligible)
    assert m_h.sum() == 2                                    # first day is healthy
    assert not training_mask(f.assign(healthy=False), "healthy_reference").any()


def test_daily_evaluation_sums_intervals_to_kwh():
    f = _frame()
    f["daytime"] = f["irradiation_kw_m2"] > 0
    pred = f["ac_power_kw"].to_numpy() * 0.9
    res = evaluate(f, pred, pd.Series(True, index=f.index), 0.25)
    # day 1: 3 x 500 kW x 0.25 h = 375 kWh actual, 337.5 predicted; same on day 2
    assert res["daily"]["mae"] == pytest.approx(37.5)
    assert res["daily"]["mape_pct"] == pytest.approx(10.0)
    assert res["interval"]["n"] == 6


def test_regression_metrics():
    r = regression_metrics([1, 2, 3], [1, 2, 4], mape_min=0)
    assert r["mae"] == pytest.approx(1 / 3) and r["rmse"] == pytest.approx(np.sqrt(1 / 3))
