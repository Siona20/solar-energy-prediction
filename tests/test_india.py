"""eMARC cleaning, 15-minute block maths, India features (no leakage, per-meter isolation), downloads."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from solvia.data.download_nasa_power import file_name
from solvia.features import india_consumption as ic
from solvia.features.calendar import india_holidays
from solvia.preprocessing import emarc as pe
from solvia.preprocessing.cleaning_log import CleaningLog

CFG = {"zero_run_days": 3, "outlier_iqr_k": 6.0, "max_interpolate_days": 2, "min_valid_days": 5,
       "min_blocks_per_day": 90, "solar_hours": [9, 16]}


def _raw(dep: str, values: list[float | None], start: str = "2019-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    rows = [(dep, d, v) for d, v in zip(dates, values) if v is not None]       # None = day not reported
    return pd.DataFrame(rows, columns=["deployment_id", "date", "kwh"])


def _meta(deps: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"deployment_id": deps, "household_id": [f"H{d}" for d in deps], "region": "Pune city",
                         "household_type": "Basic", "deployment_type": "mainline", "meta_conflict": False})


def test_zero_runs_become_missing_but_single_zero_days_stay():
    raw = _raw("D1", [4, 0, 4, 0, 0, 0, 4, 5])
    out = pe.clean_emarc_daily(raw, _meta(["D1"]), CFG, CleaningLog()).set_index("date")["kwh"]
    assert out.iloc[1] == 0                                   # one zero day: kept (e.g. a full-day power cut)
    assert out.iloc[3:6].isna().all()                         # 3 zeros in a row: meter/supply off -> missing


def test_short_unreported_gap_interpolated_long_gap_not():
    raw = _raw("D1", [2, None, 4, 5, None, None, None, 6])
    out = pe.clean_emarc_daily(raw, _meta(["D1"]), CFG, CleaningLog()).set_index("date")
    assert out["kwh"].iloc[1] == pytest.approx(3.0) and out["is_interpolated"].iloc[1]
    assert out["kwh"].iloc[4:7].isna().all()
    assert len(out) == 8                                      # complete calendar


def test_duplicate_rows_dropped_conflicting_rows_rejected():
    raw = pd.concat([_raw("D1", [1, 2, 3]), _raw("D1", [1])])
    assert len(pe.clean_emarc_daily(raw, _meta(["D1"]), CFG, CleaningLog())) == 3
    bad = pd.concat([_raw("D1", [1, 2, 3]), _raw("D1", [9])])
    with pytest.raises(ValueError):
        pe.clean_emarc_daily(bad, _meta(["D1"]), CFG, CleaningLog())


def test_block_daily_energy_and_solar_share():
    blocks = pd.DataFrame({"deployment_id": "D1", "block": np.arange(96), "date": pd.Timestamp("2019-05-01"),
                           "load_kw": np.where((np.arange(96) >= 36) & (np.arange(96) < 64), 2.0, 1.0)})
    d = pe.daily_from_blocks(blocks, (9, 16)).iloc[0]
    assert d["kwh_blocks"] == pytest.approx((28 * 2.0 + 68 * 1.0) * 0.25)     # 7 h at 2 kW + 17 h at 1 kW
    assert d["solar_hours_share"] == pytest.approx(14 / 31)
    assert d["peak_kw"] == 2.0 and d["base_kw"] == 1.0 and d["n_blocks"] == 96


def _mains(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for dep in ("D1", "D2"):
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
        parts.append(pd.DataFrame({
            "deployment_id": dep, "household_id": f"H{dep}", "household_type": "Basic", "region": "Pune city",
            "date": dates, "kwh": rng.uniform(2, 8, n), "is_interpolated": False, "is_outlier": False,
            "peak_kw": rng.uniform(0.5, 2, n), "base_kw": rng.uniform(0, 0.1, n), "evening_share": rng.uniform(0.1, 0.3, n),
            "solar_hours_share": rng.uniform(0.2, 0.3, n), "appliance_kwh": np.nan, "t2m_c": rng.uniform(20, 35, n),
            "t2m_max_c": rng.uniform(28, 42, n), "cdd_c": rng.uniform(0, 8, n), "rh2m_pct": rng.uniform(20, 90, n),
            "precip_mm_day": rng.uniform(0, 5, n), "model_eligible": True}))
    return pd.concat(parts, ignore_index=True)


def test_india_features_use_only_the_past_and_only_the_same_meter():
    mains = _mains()
    base = ic.build_india_consumption_features(mains).set_index(["deployment_id", "date"])
    d = pd.Timestamp("2019-02-20")
    changed = mains.copy()
    future = changed["date"] >= d
    changed.loc[future & (changed["deployment_id"] == "D1"), ["kwh", "peak_kw", "t2m_c", "t2m_max_c", "cdd_c"]] *= 10
    changed.loc[changed["deployment_id"] == "D2", "kwh"] *= 10                        # another household entirely
    after = ic.build_india_consumption_features(changed).set_index(["deployment_id", "date"])
    num = ic.NUMERIC_FEATURES
    pd.testing.assert_series_equal(base.loc[("D1", d), num], after.loc[("D1", d), num])   # D1 untouched by D1's future or D2
    assert after.loc[("D1", d), ic.TARGET] != base.loc[("D1", d), ic.TARGET]
    assert not set(ic.FORECAST_PROXY_FEATURES) & set(ic.FEATURES)


def test_india_finalize_drops_interpolated_targets_and_labels_lockdown():
    mains = _mains(n=500)
    mains.loc[100, "is_interpolated"] = True
    f = ic.build_india_consumption_features(mains)
    cfg = {"split": {"train_end": "2019-06-30", "valid_end": "2019-12-31"}, "lockdown_start": "2020-03-25"}
    out = ic.finalize(f, cfg, CleaningLog())
    assert not ((out["deployment_id"] == "D1") & (out["date"] == mains.loc[100, "date"])).any()
    assert set(out["split"]) == {"train", "valid", "test"}
    assert (out.loc[out["date"] >= "2020-03-25", "period"] == "lockdown").all()


def test_india_holidays():
    h = india_holidays([2019])
    assert {date(2019, 1, 26), date(2019, 8, 15), date(2019, 10, 2), date(2019, 10, 27)} <= h
    assert date(2018, 11, 7) not in h                        # other years' festivals are not mixed in


def test_download_file_name_tag_keeps_old_raw_files_distinct():
    assert file_name("sceaux", 48.78, 2.29, "20061201", "20101130") == "sceaux_48.78N_2.29E_daily_2006-2010.csv"
    assert file_name("sceaux", 48.78, 2.29, "20061201", "20101130", "full") == "sceaux_48.78N_2.29E_daily_2006-2010_full.csv"
