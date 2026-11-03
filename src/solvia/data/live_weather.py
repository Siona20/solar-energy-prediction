"""Recent NASA POWER daily weather for one point, fetched on demand for the forecast page.

NASA POWER publishes new days with a lag (about a week for temperature, ~10 days for sunlight); missing days come
back as -999 and are turned into NaN. Responses are cached in data/cache/nasa_power/ (one file per point and date
range; data/raw/ is never written). Column names match data/interim/india_weather_daily.parquet.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from solvia.config import get_path, project_path

log = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {"ALLSKY_SFC_SW_DWN": "ghi_kwh_m2_day", "CLRSKY_SFC_SW_DWN": "clearsky_ghi_kwh_m2_day", "T2M": "t2m_c",
          "T2M_MAX": "t2m_max_c", "T2M_MIN": "t2m_min_c", "PRECTOTCORR": "precip_mm_day", "RH2M": "rh2m_pct"}
CACHE_DIR = "data/cache/nasa_power"


def fetch_daily(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, timeout: int = 120) -> pd.DataFrame:
    """Daily values for [start, end] (index = date); days NASA has not published yet are NaN."""
    s, e = pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d")
    cache = project_path(CACHE_DIR) / f"{lat:.2f}_{lon:.2f}_{s}_{e}.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        query = {"parameters": ",".join(PARAMS), "community": "RE", "latitude": f"{lat:.2f}", "longitude": f"{lon:.2f}",
                 "start": s, "end": e, "format": "JSON"}
        url = f"{BASE_URL}?{urllib.parse.urlencode(query, safe=',')}"
        log.info("NASA POWER: fetching %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if "properties" not in payload:
            raise ValueError(f"NASA POWER returned no data: {str(payload)[:300]}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
    p = payload["properties"]["parameter"]
    df = pd.DataFrame({col: pd.Series(p[name]) for name, col in PARAMS.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.replace(-999.0, np.nan).sort_index().rename_axis("date")


def history(lat: float, lon: float, today: pd.Timestamp, location: str | None = None) -> pd.DataFrame:
    """Daily history up to `today` for a point: the stored 2001-2025 file for one of the 50 cities plus a fresh
    fetch after it; any other point is fetched whole from 2001 (one request, cached for the day)."""
    today = pd.Timestamp(today).normalize()
    if location:
        w = pd.read_parquet(get_path("interim") / "india_weather_daily.parquet")
        w = w[w["location"] == location].set_index("date")[list(PARAMS.values())].sort_index()
        start = w.index.max() + pd.Timedelta(days=1)
        if start <= today:
            w = pd.concat([w, fetch_daily(lat, lon, start, today)])
        return w
    return fetch_daily(lat, lon, pd.Timestamp("2001-01-01"), today)


def last_complete(w: pd.DataFrame, cols: list[str]) -> pd.Timestamp:
    """Last day on which all `cols` are published."""
    ok = w[cols].notna().all(axis=1)
    return ok[ok].index.max()
