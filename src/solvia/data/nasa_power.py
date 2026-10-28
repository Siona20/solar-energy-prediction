"""Loader for NASA POWER point CSV exports (header block between -BEGIN HEADER- and -END HEADER-)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MISSING = -999


def read_header(path: Path) -> tuple[list[str], int]:
    """Return the header lines and the line index of '-END HEADER-'."""
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("-END HEADER-")
    return lines[1:end], end


def load_nasa_power(path: Path) -> pd.DataFrame:
    """Read a NASA POWER daily or hourly file. Adds `timestamp`, `latitude`, `longitude`; -999 -> NaN."""
    header, end = read_header(path)
    loc = re.search(r"[Ll]atitude\s+([-\d.]+)\s+[Ll]ongitude\s+([-\d.]+)", " ".join(header))
    df = pd.read_csv(path, skiprows=end + 1).replace(MISSING, np.nan)
    df["timestamp"] = pd.to_datetime(dict(year=df.pop("YEAR"), month=df.pop("MO"), day=df.pop("DY"),
                                          hour=df.pop("HR") if "HR" in df else 0))
    df["latitude"], df["longitude"] = float(loc.group(1)), float(loc.group(2))
    log.info("NASA POWER: read %d rows from %s", len(df), path.name)
    return df
