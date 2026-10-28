"""Loaders for the Prayas (Energy Group) eMARC smart-meter data (Harvard Dataverse doi:10.7910/DVN/YJ5SP1).

Three files: household/deployment info, daily kWh per deployment (meter), and 15-minute block
average load in kW (blocks 0..95, block b covers b*15 .. b*15+15 minutes after midnight).
A household has one "Mainline" meter (whole home) and, often, one "appliance" meter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from solvia.config import load_config, project_path

log = logging.getLogger(__name__)

DATE_FORMAT = "%m/%d/%Y"
INFO_FILE = "Household-Deployment basic info.xlsx"
DAILY_FILE = "eMARC daily consumption.csv"
BLOCKS_FILE = "eMARC load blocks.csv"


def emarc_dir() -> Path:
    return project_path(load_config()["datasets"]["emarc_dir"])


def load_info(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_excel(path or emarc_dir() / INFO_FILE)
    return df.rename(columns={"Deployment type": "deployment_type", "Region": "region", "Household type": "household_type"})


def load_daily(path: Path | None = None) -> pd.DataFrame:
    path = path or emarc_dir() / DAILY_FILE
    df = pd.read_csv(path, dtype=str)
    df.columns = ["region", "household_type", "household_id", "deployment_type", "deployment_id", "date", "kwh"]
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    df["kwh"] = df["kwh"].astype("float64")
    log.info("eMARC daily: read %d rows from %s", len(df), path.name)
    return df


def load_blocks(path: Path | None = None) -> pd.DataFrame:
    """~10 M rows: deployment_id, block (0..95), date, load_kw (block-average active power)."""
    path = path or emarc_dir() / BLOCKS_FILE
    df = pd.read_csv(path, encoding="utf-8-sig", engine="pyarrow",
                     dtype={"deployment_id": "category", "block": "int16", "date": str, "Load (kW)": "float64"})
    df = df.rename(columns={"Load (kW)": "load_kw"})
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    log.info("eMARC blocks: read %d rows from %s", len(df), path.name)
    return df
