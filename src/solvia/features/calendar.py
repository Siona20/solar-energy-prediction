"""Calendar helpers: French public and school holidays, Indian national holidays and major festivals."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

FIXED_HOLIDAYS = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]


def easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def french_holidays(years: range | list[int]) -> set[date]:
    """National public holidays in metropolitan France (Easter Monday, Ascension, Whit Monday + fixed days)."""
    out: set[date] = set()
    for y in years:
        out.update(date(y, m, d) for m, d in FIXED_HOLIDAYS)
        easter = easter_sunday(y)
        out.update(easter + timedelta(days=n) for n in (1, 39, 50))
    return out


# India: national days (fixed) plus the major festivals that move each year, for the eMARC period.
# Festival dates checked on 2026-10-28 against the DoPT holiday lists for central government offices
# (Delhi/New Delhi, Annexure I gazetted; Ganesh Chaturthi from Annexure II restricted), 2018-2020: all match.
# Ganesh Chaturthi is also a Maharashtra state holiday (most eMARC homes are in Pune/Aurangabad). Treat the
# holiday effect as indicative: regional calendars differ and the list is deliberately short.
INDIA_FIXED_HOLIDAYS = [(1, 26), (8, 15), (10, 2), (12, 25)]
INDIA_MOVING_HOLIDAYS = {
    "holi": ["2018-03-02", "2019-03-21", "2020-03-10"],
    "eid_ul_fitr": ["2018-06-16", "2019-06-05", "2020-05-25"],
    "ganesh_chaturthi": ["2018-09-13", "2019-09-02"],
    "dussehra": ["2018-10-19", "2019-10-08"],
    "diwali": ["2018-11-07", "2019-10-27"],
}


def india_holidays(years: range | list[int]) -> set[date]:
    out: set[date] = set()
    for y in years:
        out.update(date(y, m, d) for m, d in INDIA_FIXED_HOLIDAYS)
    for days in INDIA_MOVING_HOLIDAYS.values():
        out.update(date.fromisoformat(d) for d in days if int(d[:4]) in years)
    return out


def french_school_holidays(path: Path, zone: str = "c") -> set[date]:
    """Days on which the given zone is on school holiday (Etalab compilation CSV)."""
    v = pd.read_csv(path, usecols=["date", f"vacances_zone_{zone}"], parse_dates=["date"])
    return set(v.loc[v[f"vacances_zone_{zone}"].astype(bool), "date"].dt.date)
