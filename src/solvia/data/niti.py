"""Loader for NITI Aayog ICED insolation workbooks (data rows followed by a copyright footer)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_niti_sheet(path: Path) -> pd.DataFrame:
    """Read the first sheet and drop the footer rows (copyright, URL, disclaimer)."""
    df = pd.read_excel(path, sheet_name=0)
    footer = df.iloc[:, 0].astype(str).str.startswith("Copyright")
    if footer.any():
        df = df.loc[: footer.idxmax() - 1]
    df = df.dropna(how="all")
    log.info("NITI: read %d data rows from %s", len(df), path.name)
    return df
