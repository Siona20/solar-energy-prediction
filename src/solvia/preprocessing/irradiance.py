"""Tidy the NITI Aayog ICED insolation workbooks. NASA POWER India data: preprocessing/india_weather.py."""
from __future__ import annotations

import calendar

import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog

MONTHS = {m: i for i, m in enumerate(calendar.month_abbr) if m}


def tidy_niti_districts(raw: pd.DataFrame, clog: CleaningLog) -> pd.DataFrame:
    """NITI ICED district insolation for a single month (Dec 2024). Monthly kWh/m2 and daily mean."""
    df = raw.rename(columns={"State": "state", "District": "district", "Year": "year", "Month": "month",
                             raw.columns[-1]: "insolation_kwh_m2_month"})
    df["year"] = df["year"].astype(int)
    before = len(df)
    df = df.drop_duplicates()
    clog.add("niti districts", "drop exact duplicate rows", before, len(df))
    # Some districts appear twice with different values; keep one row with the mean and the spread.
    keys = ["state", "district", "year", "month"]
    conflict = df.duplicated(keys, keep=False)
    before = len(df)
    df = (df.groupby(keys, as_index=False)["insolation_kwh_m2_month"]
            .agg(insolation_kwh_m2_month="mean", n_source_rows="size",
                 source_spread_kwh_m2=lambda s: s.max() - s.min()))
    clog.add("niti districts", "average districts listed twice with different values", before, len(df),
             int(conflict.sum()), f"{int((df['n_source_rows'] > 1).sum())} districts; spread kept in source_spread_kwh_m2")
    days = df.apply(lambda r: calendar.monthrange(r["year"], list(calendar.month_name).index(r["month"]))[1], axis=1)
    df["insolation_kwh_m2_day"] = df["insolation_kwh_m2_month"] / days
    df["period"] = df["month"].str[:3] + " " + df["year"].astype(str)
    return df.reset_index(drop=True)


def tidy_niti_monthly(raw: pd.DataFrame, clog: CleaningLog) -> pd.DataFrame:
    """Kheda long-term monthly averages (2009-2023)."""
    df = raw.rename(columns={"Year": "years", "Month": "month_name", "State": "state", "District": "district",
                             raw.columns[-1]: "insolation_kwh_m2_month"})
    df["month"] = df["month_name"].map(MONTHS)
    df["days_in_month"] = df["month"].map(lambda m: 28.25 if m == 2 else calendar.monthrange(2023, m)[1])
    df["insolation_kwh_m2_day"] = df["insolation_kwh_m2_month"] / df["days_in_month"]
    clog.add("niti kheda monthly", "add month number and daily mean", len(df), len(df))
    return df.sort_values("month").reset_index(drop=True)
