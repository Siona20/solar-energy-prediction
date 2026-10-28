"""Loader for the UCI Individual Household Electric Power Consumption dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from solvia.config import load_config, project_path

log = logging.getLogger(__name__)

DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"
MEASUREMENT_COLUMNS = [
    "Global_active_power",    # kW (minute-averaged)
    "Global_reactive_power",  # kW
    "Voltage",                # V
    "Global_intensity",       # A
    "Sub_metering_1",         # Wh in the minute (kitchen)
    "Sub_metering_2",         # Wh in the minute (laundry)
    "Sub_metering_3",         # Wh in the minute (water heater + air conditioner)
]


def parse_uci_datetime(date: pd.Series, time: pd.Series) -> pd.Series:
    """Combine Date (dd/mm/yyyy) and Time columns with an explicit day-first format."""
    return pd.to_datetime(date + " " + time, format=DATETIME_FORMAT)


def load_uci(path: Path | None = None) -> pd.DataFrame:
    """Read the raw 1-minute file. Returns a DataFrame indexed by `timestamp`, '?' as NaN."""
    path = path or project_path(load_config()["datasets"]["uci_household"])
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        dtype={"Date": str, "Time": str, **{c: "float64" for c in MEASUREMENT_COLUMNS}},
    )
    df["timestamp"] = parse_uci_datetime(df.pop("Date"), df.pop("Time"))
    log.info("UCI: read %d rows from %s", len(df), path.name)
    return df.set_index("timestamp").sort_index()
