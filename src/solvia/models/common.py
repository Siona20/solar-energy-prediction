"""Shared modelling helpers: metrics, time-ordered CV over whole days, saving models with metadata."""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from solvia.config import get_path

log = logging.getLogger(__name__)


def regression_metrics(y: pd.Series | np.ndarray, p: pd.Series | np.ndarray, mape_min: float | None = None) -> dict:
    """MAE, RMSE, R², normalised MAE (MAE / mean actual) and, if `mape_min` is given, MAPE on |y| >= mape_min."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
           "r2": float(r2_score(y, p)), "nmae": float(mean_absolute_error(y, p) / np.mean(y))}
    if mape_min is not None:
        m = np.abs(y) >= mape_min
        out["mape_pct"] = float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)
    return out


def day_cv_splits(dates: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV on whole calendar days: every fold trains on earlier days and validates on the
    next block of days, so no day is split and no fold sees the future."""
    days = np.sort(dates.unique())
    pos = np.asarray(dates.reset_index(drop=True))
    splits = []
    for tr_days, va_days in TimeSeriesSplit(n_splits=n_splits).split(days):
        tr = np.flatnonzero(np.isin(pos, days[tr_days]))
        va = np.flatnonzero(np.isin(pos, days[va_days]))
        splits.append((tr, va))
    return splits


def save_model(model, name: str, metadata: dict) -> Path:
    """models/<name>.joblib + models/<name>.json (metadata, library versions, time)."""
    out = get_path("models")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.joblib"
    joblib.dump(model, path)
    meta = {"name": name, "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, **metadata}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("model saved: %s", path.name)
    return path
