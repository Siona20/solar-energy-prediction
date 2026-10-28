"""Loader for the Kaggle 'Solar Power Generation Data' (two Indian plants, 15-minute data).

The four CSVs are read straight from the original zip so data/raw/ stays untouched.
The two plants store DATE_TIME in different formats, so each file has its own format string.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from solvia.config import load_config, project_path

log = logging.getLogger(__name__)

PLANTS = (1, 2)
DATETIME_FORMATS = {
    "Plant_1_Generation_Data.csv": "%d-%m-%Y %H:%M",
    "Plant_1_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S",
    "Plant_2_Generation_Data.csv": "%Y-%m-%d %H:%M:%S",
    "Plant_2_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S",
}


def _zip_path() -> Path:
    return project_path(load_config()["datasets"]["kaggle_solar_zip"])


def read_member(name: str, zip_path: Path | None = None) -> pd.DataFrame:
    """Read one CSV from the archive and parse DATE_TIME with that file's explicit format."""
    with zipfile.ZipFile(zip_path or _zip_path()) as zf, zf.open(name) as fh:
        df = pd.read_csv(fh, dtype={"DATE_TIME": str, "SOURCE_KEY": str})
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format=DATETIME_FORMATS[name])
    log.info("Kaggle: read %d rows from %s", len(df), name)
    return df


def load_generation(plant: int, zip_path: Path | None = None) -> pd.DataFrame:
    """Inverter-level generation: DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD."""
    return read_member(f"Plant_{plant}_Generation_Data.csv", zip_path)


def load_weather(plant: int, zip_path: Path | None = None) -> pd.DataFrame:
    """Plant-level sensor: DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION."""
    return read_member(f"Plant_{plant}_Weather_Sensor_Data.csv", zip_path)
