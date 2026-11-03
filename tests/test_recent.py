"""Latest-data tests: the Grid-India loader drops the summary rows, flags frozen and night-time solar readings, and
only fully clean days are valid; the dashboard helpers return labelled 2024-2025 data."""
import numpy as np
import pandas as pd
import pytest

from solvia.config import get_path
from solvia.data import grid_india as g


def _hourly_file(tmp_path, solar):
    ts = pd.date_range("2024-01-01", periods=48, freq="h")
    rows = pd.DataFrame({"Timestamp": ts.strftime("%d-%m-%Y %H:%M:%S"),
                         "Demand (MW)": 150000 + np.arange(48) * 10.0, "Wind (MW)": 8000 + np.arange(48) * 1.0,
                         "Solar (MW)": solar, "Total Generation (MW)": 151000 + np.arange(48) * 10.0})
    footer = pd.DataFrame({"Timestamp": ["Minimum", "Maximum", "Average", "Sum"], "Demand (MW)": [0, 1, 2, 3],
                           "Wind (MW)": [0, 1, 2, 3], "Solar (MW)": [0, 1, 2, 3], "Total Generation (MW)": [0, 1, 2, 3]})
    pd.concat([rows, footer]).to_excel(tmp_path / g.HOURLY_FILE, sheet_name="Report", index=False)
    return tmp_path


def _solar_shape():
    hour = np.arange(48) % 24
    return np.where((hour >= 6) & (hour <= 18), 40000 * np.sin(np.pi * (hour - 6) / 12) + np.arange(48), 60 + np.arange(48) * 0.1)


def test_clean_file_gives_two_valid_days(tmp_path):
    h = g.load_hourly(_hourly_file(tmp_path, _solar_shape()))
    assert len(h) == 48 and not h["bad_hour"].any()            # summary rows dropped, nothing flagged
    d = g.daily(h)
    assert d["valid"].all() and len(d) == 2
    assert d["solar_gwh"].iloc[0] == pytest.approx(h["solar_mw"].iloc[:24].sum() / 1000)


def test_frozen_and_night_solar_make_the_day_invalid(tmp_path):
    solar = _solar_shape()
    solar[10:13] = 30000.0                                      # frozen for 3 daylight hours on day 1
    solar[24 + 21] = 25000.0                                    # 21:00 on day 2: impossible after sunset
    h = g.load_hourly(_hourly_file(tmp_path, solar))
    assert h["flag_solar_stuck"].sum() == 3 and h["flag_solar_night"].sum() == 1
    assert not g.daily(h)["valid"].any()


needs_recent = pytest.mark.skipif(not (get_path("interim") / "grid_india_daily.parquet").exists(),
                                  reason="recent step not run (python -m solvia.pipeline --steps recent)")


@needs_recent
def test_dashboard_latest_data_is_recent_and_labelled():
    from solvia.dashboard import data as D
    grid = D.grid_daily()
    assert grid["valid"].all() and grid["date"].max() >= pd.Timestamp("2025-06-01") and grid.attrs["kind"] == "MEASURED"
    city = D.city_recent_solar("pune", 0.75)
    assert set(city["year"]) == {2024, 2025} and city.attrs["kind"] == "ESTIMATE" and len(city) == 24
