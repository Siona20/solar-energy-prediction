"""Phase 4D tests: each maintenance rule fires on the situation it describes, with evidence."""
import pandas as pd

from solvia.rules.maintenance import days_since_rain, dry_months_for, recommend

CFG = {"monitoring": {"min_consecutive_days": 3, "ratio_threshold": 0.85},
       "maintenance": {"gradual_decline_share": 0.6, "recent_rain_mm": 5.0, "offline_hours": 1.0, "days_since_cleaning": 30,
                       "decline_window_days": 5, "decline_slope_per_day": -0.005, "sensor_high_pr": 1.15}}
DAYS = pd.date_range("2020-06-01", periods=5, freq="D")


def _inv(pr_b: list[float], off_b: list[int] | None = None, n_ok: int = 4) -> pd.DataFrame:
    rows = []
    for i, day in enumerate(DAYS):
        for k in range(n_ok):
            rows.append({"plant_id": 1, "inverter_id": f"ok{k}", "date": day, "pr": 1.0, "off_intervals": 0, "peer_z": 0.0,
                         "low_run_days": 0})
        low_run = sum(1 for _ in range(i + 1)) if all(p < 0.85 for p in pr_b[:i + 1]) else 0
        rows.append({"plant_id": 1, "inverter_id": "b", "date": day, "pr": pr_b[i], "off_intervals": (off_b or [0] * 5)[i],
                     "peer_z": -5.0, "low_run_days": low_run})
    return pd.DataFrame(rows)


def _plant(share_low: list[float], rain: list[float] | None = None, running: list[float] | None = None,
           pr: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"plant_id": 1, "date": DAYS, "share_low": share_low, "plant_pr": pr or [0.98] * 5,
                         "plant_running_pr": running or [0.99] * 5, "est_rain_mm": rain or [0.0] * 5})


def _rules(recs) -> dict:
    return {r.rule.split()[0]: r for r in recs}


def test_r2_shutdowns_point_to_the_inverter_and_steady_loss_to_strings():
    shut = _rules(recommend(_inv([0.6] * 5, off_b=[12] * 5), _plant([0.2] * 5), 1, DAYS[-1], CFG))
    assert "event log" in shut["R2"].action and shut["R2"].evidence["mean_off_hours_per_day"] == 3.0
    assert "R1" not in shut                                   # not repeated as R1 for the same inverter
    steady = _rules(recommend(_inv([0.6] * 5), _plant([0.2] * 5), 1, DAYS[-1], CFG))
    assert "strings" in steady["R2"].action


def test_r1_single_day_outage():
    recs = _rules(recommend(_inv([1.0] * 4 + [0.7], off_b=[0, 0, 0, 0, 8]), _plant([0.0] * 5), 1, DAYS[-1], CFG))
    assert recs["R1"].target == "b" and recs["R1"].evidence["off_hours"] == 2.0


def test_r3_plant_wide_loss_dust_vs_rain():
    dry = _rules(recommend(_inv([1.0] * 5), _plant([0.8] * 5), 1, DAYS[-1], CFG))
    assert "cleaning" in dry["R3"].action
    wet = _rules(recommend(_inv([1.0] * 5), _plant([0.8] * 5, rain=[0, 0, 0, 10, 0]), 1, DAYS[-1], CFG))
    assert "Not dust" in wet["R3"].action


def test_r4_uses_running_pr_so_shutdowns_do_not_trigger_cleaning():
    falling_running = _plant([0.0] * 5, running=[1.0, 0.98, 0.96, 0.94, 0.92])
    assert "R4" in _rules(recommend(_inv([1.0] * 5), falling_running, 1, DAYS[-1], CFG))
    only_pr_falls = _plant([0.0] * 5, pr=[1.0, 0.9, 0.8, 0.7, 0.6], running=[0.99] * 5)
    assert "R4" not in _rules(recommend(_inv([1.0] * 5), only_pr_falls, 1, DAYS[-1], CFG))


def test_r5_r6_r7_and_r0():
    assert "R5" in _rules(recommend(_inv([1.0] * 5), _plant([0.0] * 5, pr=[1.2] * 5), 1, DAYS[-1], CFG))
    assert "R6" in _rules(recommend(_inv([1.0] * 5), _plant([0.0] * 5), 1, DAYS[-1], CFG, dry_months=[6]))
    inv = _inv([1.0] * 5)
    silent = inv[~((inv["inverter_id"] == "b") & (inv["date"] == DAYS[-1]))]     # b sends nothing on the last day
    reports = silent[["plant_id", "inverter_id", "date"]]
    r7 = _rules(recommend(silent, _plant([0.0] * 5), 1, DAYS[-1], CFG, reports=reports))["R7"]
    assert r7.target == "b" and r7.evidence["days_without_data"] == 1
    assert list(_rules(recommend(inv, _plant([0.0] * 5), 1, DAYS[-1], CFG))) == ["R0"]


def test_helpers():
    rain = pd.DataFrame({"date": DAYS, "est_rain_mm": [6, 0, 0, 0, 0]})
    assert days_since_rain(rain, DAYS[-1], 5.0) == 4
    clim = pd.DataFrame({"location": "x", "month": range(1, 13), "rain_mm_month": [4, 3, 11] + [50] * 9})
    assert dry_months_for(clim, "x") == [1, 2]
