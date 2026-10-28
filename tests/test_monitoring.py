"""Phase 4C tests: out-of-sample expected output, run counting, PR / peer flags, injected faults."""
import numpy as np
import pandas as pd
import pytest

from solvia.features.preprocess import make_pipeline
from solvia.features.solar import FEATURES
from solvia.models.monitoring import (consecutive_low, expanding_expected, injected_fault_check, inverter_day_table,
                                      robust_z)
from solvia.models.solar import IrradiationBaseline

MC = {"ratio_threshold": 0.85, "peer_z_threshold": -3.0, "peer_max_pr": 0.95, "min_consecutive_days": 3}


def test_consecutive_low_resets_on_ok_day_and_on_calendar_gap():
    dates = pd.Series(pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-05-06", "2020-05-07"]))
    low = pd.Series([True, True, False, True, True, True])
    assert consecutive_low(low, dates).tolist() == [1, 2, 0, 1, 1, 2]     # 05-05 missing -> run restarts


def test_robust_z_flags_the_outlier():
    z = robust_z(pd.Series([1.0, 0.99, 1.01, 1.0, 0.98, 0.5]))
    assert z.iloc[-1] < -3 and abs(z.iloc[0]) < 1


def _rows(days: int = 4) -> pd.DataFrame:
    parts = []
    for i in range(days):
        ts = pd.date_range(pd.Timestamp("2020-05-15") + pd.Timedelta(days=i) + pd.Timedelta(hours=10), periods=4, freq="15min")
        for inv, scale in (("a", 1.0), ("b", 1.0 if i < 2 else 0.5)):
            parts.append(pd.DataFrame({"timestamp": ts, "plant_id": 1, "inverter_id": inv, "irradiation_kw_m2": 0.5,
                                       "module_temp_c": 40.0, "ambient_temp_c": 30.0, "module_minus_ambient_c": 10.0,
                                       "hour_sin": 0.0, "hour_cos": 1.0, "ac_power_kw": 500.0 * scale,
                                       "train_eligible": True, "flag_off_in_daylight": False,
                                       "healthy": scale == 1.0, "daytime": True}))
    f = pd.concat(parts, ignore_index=True)
    f["date"] = f["timestamp"].dt.floor("D")
    return f


def test_expected_output_uses_only_earlier_days():
    f = _rows()
    tmpl = {1: make_pipeline(IrradiationBaseline(), FEATURES, linear=False)}
    exp = expanding_expected(f, tmpl, min_train_days=2)
    assert exp[f["date"] < "2020-05-17"].isna().all()                     # warm-up days are not monitored
    later = f["date"] >= "2020-05-17"
    # trained only on healthy earlier rows (500 kW at 0.5 kW/m2) -> k = 1000; the faulty future is never used
    assert exp[later].to_numpy() == pytest.approx(500.0)


def test_inverter_day_table_flags_low_pr():
    f = _rows()
    exp = pd.Series(500.0, index=f.index)
    d = inverter_day_table(f, exp, MC, 0.25)
    b = d[d["inverter_id"] == "b"].set_index("date")
    assert b["pr"].iloc[-1] == pytest.approx(0.5)
    assert b["low"].tolist() == [False, False, True, True]
    assert b["low_run_days"].iloc[-1] == 2 and b["status"].iloc[-1] == "watch"


def test_injected_fault_check_false_alarm_and_detection():
    d = pd.DataFrame({"plant_id": 4135001, "healthy_reference": True, "actual_kwh": [100.0, 100.0],
                      "expected_kwh": [100.0, 100.0]})
    t = injected_fault_check(d, 0.85).set_index("output_kept")["flagged_share"]
    assert t[1.0] == 0 and t[0.9] == 0 and t[0.8] == 1 and t[0.5] == 1
