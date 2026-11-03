"""Next-days weather for an Indian location: daily max / min / mean temperature and sunlight, 1-21 days ahead.

One model per variable, shared by all 50 NASA POWER cities (they are places, not households). It forecasts the
*anomaly* (difference from the normal for that place and day of year) from the recent anomalies, the season and the
horizon. NASA POWER data reaches the present with a lag of about a week, so "tomorrow" is often 8-10 days after
the last available day: the horizon is a feature, and accuracy is reported for every horizon.

Split by target date: train <= 2018, validation 2019-2021 (model choice), test 2022-2025 (scored once, after the
choice, with the final model refitted on 2001-2021).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solvia.config import get_path, load_config, project_path

log = logging.getLogger(__name__)

VARS = {"tmax": "t2m_max_c", "tmin": "t2m_min_c", "tmean": "t2m_c", "ghi": "ghi_kwh_m2_day",
        "rain": "precip_mm_day", "rh": "rh2m_pct"}
TARGETS = ["tmax", "tmin", "tmean", "ghi"]
UNITS = {"tmax": "°C", "tmin": "°C", "tmean": "°C", "ghi": "kWh/m²/day"}
MAX_H = 21
CLIM_WINDOW = 15            # days, centred, circular: smooths the day-of-year normals
TRAIN_END, VALID_END = "2018-12-31", "2021-12-31"
ORIGIN_STEP = {"train": 4, "valid": 2, "test": 2}   # every k-th origin day (neighbouring days are near-duplicates)


def climatology(w: pd.DataFrame, until: str) -> pd.DataFrame:
    """Normals by day of year (1..366) for one location from the years up to `until`, incl. clear-sky sunlight."""
    h = w[w.index <= pd.Timestamp(until)]
    cols = list(VARS.values()) + ["clearsky_ghi_kwh_m2_day"]
    raw = h[cols].groupby(h.index.day_of_year).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    pad = CLIM_WINDOW // 2
    ext = pd.concat([raw.iloc[-pad:], raw, raw.iloc[:pad]])
    return ext.rolling(CLIM_WINDOW, center=True).mean().iloc[pad:-pad].set_axis(raw.index)


def _doy_lookup(clim: pd.DataFrame, dates: pd.DatetimeIndex, col: str) -> np.ndarray:
    return clim[col].to_numpy()[dates.day_of_year.to_numpy() - 1]


def origin_features(w: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Per origin day: anomalies today and the 2 days before, and their 7- and 30-day means (all known at the origin)."""
    w = w.asfreq("D")
    f = pd.DataFrame(index=w.index)
    for k, col in VARS.items():
        a = w[col] - _doy_lookup(clim, w.index, col)
        for lag in (0, 1, 2):
            f[f"{k}_anom_lag{lag}"] = a.shift(lag)
        f[f"{k}_anom_7d"] = a.rolling(7, min_periods=5).mean()
        f[f"{k}_anom_30d"] = a.rolling(30, min_periods=20).mean()
    return f


def _rows(f: pd.DataFrame, clim: pd.DataFrame, h: int, lat: float, lon: float) -> pd.DataFrame:
    target_dates = f.index + pd.Timedelta(days=h)
    r = f.copy()
    r["h"] = h
    r["lat"], r["lon"] = lat, lon
    doy = target_dates.day_of_year.to_numpy()
    r["doy_sin"], r["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for k in TARGETS:
        r[f"clim_{k}"] = _doy_lookup(clim, target_dates, VARS[k])
    r["clim_clearsky"] = _doy_lookup(clim, target_dates, "clearsky_ghi_kwh_m2_day")
    r["origin"], r["target_date"] = f.index, target_dates
    return r.reset_index(drop=True)


FEATURES = ([f"{k}_anom_{s}" for k in VARS for s in ("lag0", "lag1", "lag2", "7d", "30d")]
            + ["h", "lat", "lon", "doy_sin", "doy_cos", *[f"clim_{k}" for k in TARGETS], "clim_clearsky"])


def _origin_split(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.where(dates <= pd.Timestamp(TRAIN_END), "train", np.where(dates <= pd.Timestamp(VALID_END), "valid", "test"))


def build_rows(weather: pd.DataFrame, clim_until: str, horizons=range(1, MAX_H + 1)) -> pd.DataFrame:
    """Origin x horizon rows for every city, with the target anomaly `y_<var>` (target value minus its normal).
    Only every k-th origin day is kept (ORIGIN_STEP), because neighbouring origins give near-duplicate rows."""
    out = []
    for loc, g in weather.groupby("location"):
        w = g.set_index("date").sort_index().asfreq("D")
        clim = climatology(w, clim_until)
        f = origin_features(w, clim)
        step = pd.Series(_origin_split(f.index)).map(ORIGIN_STEP).to_numpy()
        f = f[((f.index - pd.Timestamp("2001-01-01")).days % step) == 0].astype("float32")
        for h in horizons:
            r = _rows(f, clim, h, float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0]))
            for k in TARGETS:
                r[f"y_{k}"] = w[VARS[k]].reindex(r["target_date"]).to_numpy() - r[f"clim_{k}"]
            r["location"] = loc
            out.append(r)
    return pd.concat(out, ignore_index=True)


def _split(rows: pd.DataFrame) -> pd.Series:
    """Split by target date; rows whose origin is in an earlier split than their target are dropped ("gap"), so no
    validation or test day is ever a training target, and no training row looks across a split boundary."""
    s = pd.Series(_origin_split(pd.DatetimeIndex(rows["target_date"])), index=rows.index)
    return s.where(s.to_numpy() == _origin_split(pd.DatetimeIndex(rows["origin"])), "gap")


def _thin(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    return rows[rows["split"] == split]


CANDIDATES = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
        learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200, random_state=42),
}


@dataclass
class WeatherForecaster:
    """Saved weather model: one regressor per target, the normals period and 80 % half-widths per target/horizon."""
    models: dict
    features: list
    clim_until: str
    half_width_80: dict = field(default_factory=dict)     # target -> {h: half-width}
    selected: str = ""

    def forecast(self, w: pd.DataFrame, lat: float, lon: float, origin: pd.Timestamp, horizons) -> pd.DataFrame:
        """Forecast from one location's daily history `w` (index date, NASA column names) made on day `origin`."""
        w = w.sort_index().asfreq("D")
        clim = climatology(w, self.clim_until)
        f = origin_features(w[w.index <= origin], clim).loc[[origin]]
        rows = pd.concat([_rows(f, clim, h, lat, lon) for h in horizons], ignore_index=True)
        out = rows[["target_date", "h"]].copy()
        for k, m in self.models.items():
            normal = rows[f"clim_{k}"].to_numpy()
            pred = normal + m.predict(rows[self.features])
            if k == "ghi":
                pred = np.clip(pred, 0, rows["clim_clearsky"].to_numpy() * 1.05)
            hw = np.array([self.half_width_80[k][int(h)] for h in rows["h"]])
            out[k], out[f"{k}_low"], out[f"{k}_high"], out[f"{k}_normal"] = pred, pred - hw, pred + hw, normal
        out["ghi_low"] = out["ghi_low"].clip(lower=0)
        return out


def _scores(rows: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """MAE per target and horizon for the model and two baselines (normal for the day, persistence of today)."""
    out = []
    for k in TARGETS:
        y = rows[f"y_{k}"].to_numpy()
        for name, p in {"model": preds[k], "baseline_normal": np.zeros(len(rows)),
                        "baseline_persistence": rows[f"{k}_anom_lag0"].fillna(0).to_numpy()}.items():
            e = pd.Series(np.abs(y - p)).groupby(rows["h"].to_numpy()).mean()
            out += [{"target": k, "model": name, "h": int(h), "mae": float(v)} for h, v in e.items()]
    return pd.DataFrame(out)


def run_weather_model() -> dict:
    weather = pd.read_parquet(get_path("interim") / "india_weather_daily.parquet")
    log.info("weather model: %d city-days, %d cities", len(weather), weather["location"].nunique())

    # --- model choice: normals and training from <= 2018, scored on 2019-2021
    rows = build_rows(weather, TRAIN_END)
    rows["split"] = _split(rows)
    train, valid = _thin(rows, "train"), _thin(rows, "valid")
    ok = lambda r, k: r[r[f"y_{k}"].notna()]
    log.info("weather model: %d train rows, %d validation rows", len(train), len(valid))
    valid_mae, fitted = {}, {}
    for name, make in CANDIDATES.items():
        for k in TARGETS:
            t = ok(train, k)
            fitted[(name, k)] = make().fit(t[FEATURES].fillna(0) if name == "ridge" else t[FEATURES], t[f"y_{k}"])
            v = ok(valid, k)
            p = fitted[(name, k)].predict(v[FEATURES].fillna(0) if name == "ridge" else v[FEATURES])
            valid_mae[(name, k)] = float(np.mean(np.abs(v[f"y_{k}"] - p)))
            log.info("  %s %s: validation MAE %.3f", name, k, valid_mae[(name, k)])
    choice = {k: min(CANDIDATES, key=lambda n: valid_mae[(n, k)]) for k in TARGETS}
    selected = "hist_gradient_boosting" if all(c == "hist_gradient_boosting" for c in choice.values()) else json.dumps(choice)

    # validation scores by horizon and 80 % half-widths from validation residuals (train-only fit)
    vpred, half = {}, {}
    vv = valid.dropna(subset=[f"y_{k}" for k in TARGETS])
    for k in TARGETS:
        m = fitted[(choice[k], k)]
        vpred[k] = m.predict(vv[FEATURES].fillna(0) if choice[k] == "ridge" else vv[FEATURES])
        res = pd.Series(np.abs(vv[f"y_{k}"].to_numpy() - vpred[k])).groupby(vv["h"].to_numpy())
        half[k] = {int(h): float(q) for h, q in res.quantile(0.8).items()}
    valid_scores = _scores(vv, vpred).assign(split="valid")

    # --- final model: normals and training from <= 2021, scored once on 2022-2025
    rows = build_rows(weather, VALID_END)
    rows["split"] = _split(rows)
    fit = pd.concat([_thin(rows, "train"), _thin(rows, "valid")])
    test = _thin(rows, "test").dropna(subset=[f"y_{k}" for k in TARGETS])
    final = {}
    for k in TARGETS:
        t = ok(fit, k)
        final[k] = CANDIDATES[choice[k]]().fit(t[FEATURES].fillna(0) if choice[k] == "ridge" else t[FEATURES], t[f"y_{k}"])
    tpred = {k: final[k].predict(test[FEATURES].fillna(0) if choice[k] == "ridge" else test[FEATURES]) for k in TARGETS}
    test_scores = _scores(test, tpred).assign(split="test")
    scores = pd.concat([valid_scores, test_scores], ignore_index=True)
    scores.to_csv(get_path("metrics") / "weather_model_scores.csv", index=False)

    if any(c == "ridge" for c in choice.values()):
        final = {k: (_RidgeNaN(m) if choice[k] == "ridge" else m) for k, m in final.items()}
    model = WeatherForecaster(final, FEATURES, VALID_END, half, selected)
    joblib.dump(model, get_path("models") / "weather_nextdays.joblib")
    summary = {"selected": choice, "validation_mae_all_h": {f"{n}/{k}": v for (n, k), v in valid_mae.items()},
               "train_rows": len(train), "valid_rows": len(vv), "test_rows": len(test),
               "cities": int(weather["location"].nunique()), "data_end": str(weather["date"].max().date())}
    (get_path("models") / "weather_nextdays.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _write_report(scores, summary)
    return summary


class _RidgeNaN:
    """Ridge pipeline that treats missing anomalies as 0 (= normal), as in training."""
    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.predict(X.fillna(0))


def _write_report(scores: pd.DataFrame, summary: dict) -> None:
    pick = lambda split, k, name, hs: scores[(scores["split"] == split) & (scores["target"] == k) & (scores["model"] == name)
                                             & scores["h"].isin(hs)]["mae"].mean()
    lines = ["# Next-days weather model", "",
             "*Generated by `python -m solvia.pipeline --steps model_weather`. Data: NASA POWER daily, "
             f"{summary['cities']} Indian cities, 2001 – {summary['data_end']}.*", "",
             "The model forecasts the difference from the normal for that place and day of the year. Baselines: "
             "**normal** (forecast = the 2001–2018 or 2001–2021 normal for that day) and **persistence** (today's "
             "difference from normal continues).", "",
             f"Chosen on validation (2019–2021): {summary['selected']}. Test = 2022–2025, scored once with the final "
             "model (refitted on 2001–2021).", "",
             "## Mean absolute error, test 2022–2025", "",
             "| Variable | Horizon (days) | Model | Normal | Persistence |", "|---|---|---|---|---|"]
    for k in TARGETS:
        for label, hs in [("1", [1]), ("2–3", [2, 3]), ("4–7", range(4, 8)), ("8–14", range(8, 15)), ("15–21", range(15, 22))]:
            lines.append(f"| {k} ({UNITS[k]}) | {label} | {pick('test', k, 'model', hs):.2f} | "
                         f"{pick('test', k, 'baseline_normal', hs):.2f} | {pick('test', k, 'baseline_persistence', hs):.2f} |")
    lines += ["", "Reading: close to the present the recent weather helps; 8+ days ahead the model is only slightly better "
              "than the normal for the date, which is expected for a statistical model without a weather simulation.", "",
              "NASA POWER publishes new days with a lag of about a week (temperature) to 10 days (sunlight), so the "
              "dashboard's 'tomorrow' is usually 8–11 days after the last available day. Full table: "
              "`reports/metrics/weather_model_scores.csv`.", ""]
    project_path("reports/weather_model_report.md").write_text("\n".join(lines), encoding="utf-8")
