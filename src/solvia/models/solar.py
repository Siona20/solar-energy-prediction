"""Phase 4B: expected solar output (AC kW per inverter, 15-minute) from weather and time of day.

"Expected" means what a *healthy* inverter of that plant would produce in these conditions. The model
only sees weather and time features (features.solar.FEATURES); output-derived columns are blocked.

Design (decisions in docs/PROJECT_LOG.md):
- Model domain: daytime rows (irradiation > 0). At night the expected output is 0 by rule.
- Training rows: train split only, `train_eligible` (no inverter-off intervals, decision D5), and either
  all operating inverter-days or only the healthy reference (decision D7).
- Scope: one pooled model for both plants, or one model per plant.
- Tuning: grid search with expanding-window CV over whole training days (never valid/test days).
- Selection: lowest validation daily-kWh MAE on healthy inverter-days, both plants together.
  The test split is not used here (Phase 5, once).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from solvia.config import get_path, load_config, project_path
from solvia.features.preprocess import make_pipeline
from solvia.features.solar import FEATURES, TARGET, check_no_leakage
from solvia.models.common import day_cv_splits, regression_metrics, save_model
from solvia.viz import PLANT_NAMES

log = logging.getLogger(__name__)

TRAINING_SETS = ("all_operating", "healthy_reference")
SCOPES = ("pooled", "per_plant")
EVAL_SETS = {"valid_healthy": True, "valid_all_operating": False}
DAILY_MAPE_MIN_KWH = 100.0


class IrradiationBaseline(BaseEstimator, RegressorMixin):
    """AC = k × irradiation, k by least squares through the origin. The simplest physical model."""

    def fit(self, X, y):
        x, y = np.asarray(X["irradiation_kw_m2"], dtype=float), np.asarray(y, dtype=float)
        self.k_ = float(x @ y / (x @ x))
        return self

    def predict(self, X):
        return self.k_ * np.asarray(X["irradiation_kw_m2"], dtype=float)


def candidates(random_state: int) -> dict[str, tuple]:
    """name -> (estimator, is_linear, parameter grid)."""
    return {
        "baseline_k_irradiation": (IrradiationBaseline(), False, {}),
        "linear_regression": (LinearRegression(), True, {}),
        "random_forest": (RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1), False,
                          {"model__min_samples_leaf": [5, 20], "model__max_features": [1.0, 0.5]}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=300, random_state=random_state), False,
                                   {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
                                    "model__min_samples_leaf": [20, 100]}),
    }


@dataclass
class ExpectedSolarModel:
    """Fitted pipelines keyed by plant id (per-plant scope) or 'all' (pooled). Night -> 0, never negative."""
    name: str
    scope: str
    training_set: str
    pipelines: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURES))

    def predict_kw(self, df: pd.DataFrame) -> np.ndarray:
        pred = np.zeros(len(df))
        day = (df["irradiation_kw_m2"] > 0).to_numpy()
        if self.scope == "pooled":
            if day.any():
                pred[day] = self.pipelines["all"].predict(df.loc[day, self.features])
        else:
            plant = df["plant_id"].to_numpy()
            for pid, pipe in self.pipelines.items():
                m = day & (plant == pid)
                if m.any():
                    pred[m] = pipe.predict(df.loc[m, self.features])
        return np.clip(pred, 0, None)


def healthy_inverter_days(daily_inv: pd.DataFrame, min_ratio: float) -> pd.DataFrame:
    """Specific yield (kWh per kWh/m²) of each inverter-day relative to the plant's 90th-percentile
    inverter on the same day. Same-day comparison removes the weather; p90 is robust to plant-wide faults.
    Used only to choose training rows and an evaluation subset, never as a feature."""
    d = daily_inv[["plant_id", "inverter_id", "date"]].copy()
    sy = daily_inv["energy_kwh"] / daily_inv["irradiation_kwh_m2"]
    p90 = sy.groupby([daily_inv["plant_id"], daily_inv["date"]]).transform(lambda s: s.quantile(0.9))
    d["specific_yield_ratio"] = sy / p90
    d["healthy"] = d["specific_yield_ratio"] >= min_ratio
    return d


def model_rows(features: pd.DataFrame, healthy: pd.DataFrame) -> pd.DataFrame:
    f = features.merge(healthy, on=["plant_id", "inverter_id", "date"], how="left")
    f["healthy"] = f["healthy"].astype("boolean").fillna(False).astype(bool)
    f["daytime"] = f["irradiation_kw_m2"] > 0
    return f


def training_mask(f: pd.DataFrame, training_set: str) -> pd.Series:
    m = (f["split"] == "train") & f["daytime"] & f["train_eligible"]
    return m & f["healthy"] if training_set == "healthy_reference" else m


def eval_mask(f: pd.DataFrame, healthy_only: bool, split: str = "valid") -> pd.Series:
    m = (f["split"] == split) & f["train_eligible"]
    return m & f["healthy"] if healthy_only else m


def tune(estimator, linear: bool, grid: dict, train: pd.DataFrame, n_splits: int) -> tuple:
    """Fit one pipeline; if a grid is given, choose it by expanding-window day CV on the training rows."""
    pipe = make_pipeline(clone(estimator), FEATURES, linear=linear)
    train = train.reset_index(drop=True)
    if not grid:
        return pipe.fit(train[FEATURES], train[TARGET]), {}
    search = GridSearchCV(pipe, grid, cv=day_cv_splits(train["date"], n_splits), scoring="neg_mean_absolute_error",
                          refit=True, n_jobs=1)
    search.fit(train[FEATURES], train[TARGET])
    info = {"best_params": search.best_params_, "cv_mae_kw": float(-search.best_score_),
            "cv_results": [{"params": p, "mae_kw": float(-s)} for p, s in
                           zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])]}
    return search.best_estimator_, info


def evaluate(f: pd.DataFrame, pred: np.ndarray, mask: pd.Series, interval_hours: float) -> dict:
    """Interval metrics on daytime rows (kW) and daily metrics per inverter-day (kWh)."""
    day = mask & f["daytime"]
    interval = regression_metrics(f.loc[day, TARGET], pred[day.to_numpy()])
    g = (f.loc[mask, ["plant_id", "inverter_id", "date", TARGET]].assign(pred=pred[mask.to_numpy()])
         .groupby(["plant_id", "inverter_id", "date"])[[TARGET, "pred"]].sum() * interval_hours)
    daily = regression_metrics(g[TARGET], g["pred"], mape_min=DAILY_MAPE_MIN_KWH)
    return {"interval": interval, "daily": daily}


def _metric_rows(run: dict, f: pd.DataFrame, pred: np.ndarray, interval_hours: float) -> list[dict]:
    rows = []
    for eval_set, healthy_only in EVAL_SETS.items():
        base = eval_mask(f, healthy_only)
        for plant in ("both", *PLANT_NAMES):
            m = base if plant == "both" else base & (f["plant_id"] == plant)
            res = evaluate(f, pred, m, interval_hours)
            for level, met in res.items():
                rows.append({**run, "eval_set": eval_set, "plant": PLANT_NAMES.get(plant, plant), "level": level, **met})
    return rows


def run_solar_models() -> dict:
    cfg = load_config()
    sc, rs = cfg["solar"], cfg["random_state"]
    check_no_leakage(FEATURES)
    features = pd.read_parquet(get_path("processed") / "solar_features_15min.parquet")
    daily_inv = pd.read_parquet(get_path("interim") / "solar_daily_inverter.parquet")
    healthy = healthy_inverter_days(daily_inv, sc["healthy_min_ratio"])
    f = model_rows(features, healthy)
    log_healthy = {PLANT_NAMES[p]: {s: round(float(g.loc[g["split"] == s, "healthy"].mean()), 3) for s in ("train", "valid")}
                   for p, g in f[f["daytime"]].groupby("plant_id")}
    log.info("healthy share of daytime rows: %s", log_healthy)

    metrics, tuning, cross, fitted = [], {}, [], {}
    for training_set in TRAINING_SETS:
        tmask = training_mask(f, training_set)
        for name, (est, linear, grid) in candidates(rs).items():
            per_plant = {}
            for unit in ("all", *PLANT_NAMES):
                rows = f[tmask] if unit == "all" else f[tmask & (f["plant_id"] == unit)]
                pipe, info = tune(est, linear, grid, rows, sc["cv_splits"])
                tuning[f"{training_set}/{name}/{PLANT_NAMES.get(unit, 'pooled')}"] = {**info, "train_rows": int(len(rows))}
                per_plant[unit] = pipe
                log.info("fitted %s / %s / %s on %d rows %s", training_set, name, unit, len(rows),
                         info.get("best_params", ""))
            for scope in SCOPES:
                pipes = {"all": per_plant["all"]} if scope == "pooled" else {p: per_plant[p] for p in PLANT_NAMES}
                model = ExpectedSolarModel(name=name, scope=scope, training_set=training_set, pipelines=pipes)
                pred = model.predict_kw(f)
                run = {"training_set": training_set, "scope": scope, "model": name}
                metrics += _metric_rows(run, f, pred, sc["interval_hours"])
                fitted[(training_set, scope, name)] = model
            # cross-plant: a model fitted on one plant predicts the other plant's healthy validation days
            for src, dst in [(4135001, 4136001), (4136001, 4135001)]:
                m = eval_mask(f, True) & (f["plant_id"] == dst)
                pred = np.zeros(len(f))
                day = (m & f["daytime"]).to_numpy()
                pred[day] = np.clip(per_plant[src].predict(f.loc[day, FEATURES]), 0, None)
                own = fitted[(training_set, "per_plant", name)].predict_kw(f)
                for label, p_ in [("trained on other plant", pred), ("trained on same plant", own)]:
                    res = evaluate(f, p_, m, sc["interval_hours"])
                    cross.append({"training_set": training_set, "model": name, "test_plant": PLANT_NAMES[dst],
                                  "trained_on": PLANT_NAMES[src] if label.startswith("trained on other") else PLANT_NAMES[dst],
                                  "daily_mae_kwh": res["daily"]["mae"], "daily_r2": res["daily"]["r2"],
                                  "daily_mape_pct": res["daily"]["mape_pct"], "interval_mae_kw": res["interval"]["mae"]})

    flat = pd.DataFrame(metrics)
    sel =flat[(flat["eval_set"] == "valid_healthy") & (flat["plant"] == "both") & (flat["level"] == "daily")]
    best = sel.loc[sel["mae"].idxmin()]
    chosen = fitted[(best["training_set"], best["scope"], best["model"])]
    log.info("selected: %s / %s / %s (validation daily MAE %.1f kWh)", best["training_set"], best["scope"], best["model"], best["mae"])

    metrics_dir = get_path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    flat.to_csv(metrics_dir / "solar_validation_metrics.csv", index=False)
    pd.DataFrame(cross).to_csv(metrics_dir / "solar_cross_plant.csv", index=False)
    (metrics_dir / "solar_tuning.json").write_text(json.dumps(tuning, indent=1, default=str), encoding="utf-8")

    best_rows = flat[(flat["training_set"] == best["training_set"]) & (flat["scope"] == best["scope"]) & (flat["model"] == best["model"])]
    save_model(chosen, "solar_expected_output", {
        "objective": "B: expected AC power (kW) of a healthy inverter, 15-minute, from weather and time of day",
        "model": best["model"], "scope": best["scope"], "training_set": best["training_set"],
        "features": FEATURES, "target": TARGET, "night_rule": "irradiation == 0 -> 0 kW; predictions clipped at 0",
        "trained_on": "train split, daytime, train_eligible" + (", healthy inverter-days" if best["training_set"] == "healthy_reference" else ""),
        "train_days": f"{f.loc[f['split'] == 'train', 'date'].min().date()} to {f.loc[f['split'] == 'train', 'date'].max().date()}",
        "healthy_min_ratio": sc["healthy_min_ratio"],
        "hyperparameters": {k: v.get("best_params", {}) for k, v in tuning.items()
                            if k.startswith(f"{best['training_set']}/{best['model']}/")},
        "selection": "lowest validation daily-kWh MAE on healthy inverter-days, both plants",
        "validation_metrics": best_rows.to_dict(orient="records"), "test_metrics": "not computed (Phase 5)"})
    summary = {"selected": {k: best[k] for k in ("training_set", "scope", "model")}, "healthy_share": log_healthy}
    write_report(flat, pd.DataFrame(cross), tuning, summary, sc)
    return summary


def _fmt(df: pd.DataFrame, cols: list[str], digits: dict) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(f"{r[c]:,.{digits[c]}f}" if c in digits and pd.notna(r[c]) else str(r[c]) for c in cols) + " |\n"
                   for _, r in df.iterrows())
    return head + body


def write_report(flat: pd.DataFrame, cross: pd.DataFrame, tuning: dict, summary: dict, sc: dict) -> None:
    s = summary["selected"]
    d = flat[(flat["eval_set"] == "valid_healthy") & (flat["plant"] == "both")]
    daily = d[d["level"] == "daily"].sort_values("mae")
    interval = d[d["level"] == "interval"].set_index(["training_set", "scope", "model"])
    daily = daily.join(interval[["mae", "r2"]].rename(columns={"mae": "interval_mae_kw", "r2": "interval_r2"}),
                       on=["training_set", "scope", "model"])
    sel = flat[(flat["training_set"] == s["training_set"]) & (flat["scope"] == s["scope"]) & (flat["model"] == s["model"])]
    lines = [
        "# Expected-solar model (Phase 4B)", "",
        "*Generated by `python -m solvia.pipeline --steps model_solar`. Validation split only; the test split is untouched until Phase 5.*", "",
        "**Target:** AC power (kW) of one inverter per 15 minutes. **Features:** " + ", ".join(f"`{c}`" for c in FEATURES) + ".",
        f"**Healthy reference:** inverter-days whose kWh per kWh/m² is ≥ {sc['healthy_min_ratio']} × the plant's 90th-percentile inverter that day. "
        f"Share of daytime rows that are healthy: {summary['healthy_share']}.", "",
        f"## Selected model: `{s['model']}`, scope `{s['scope']}`, trained on `{s['training_set']}`", "",
        "Chosen by the lowest validation daily-kWh MAE on healthy inverter-days (both plants).", "",
        _fmt(sel.sort_values(["eval_set", "plant", "level"]), ["eval_set", "plant", "level", "n", "mae", "rmse", "r2", "nmae", "mape_pct"],
             {"mae": 2, "rmse": 2, "r2": 3, "nmae": 3, "mape_pct": 2}),
        "Units: interval rows in kW (daytime only), daily rows in kWh per inverter-day. `nmae` = MAE / mean actual.", "",
        "## All candidates (validation, healthy inverter-days, both plants)", "",
        _fmt(daily, ["training_set", "scope", "model", "mae", "rmse", "r2", "mape_pct", "interval_mae_kw", "interval_r2"],
             {"mae": 1, "rmse": 1, "r2": 3, "mape_pct": 2, "interval_mae_kw": 2, "interval_r2": 3}),
        "`mae`, `rmse` in kWh per inverter-day; `interval_*` on 15-minute daytime rows (kW).", "",
        "## Validation on all operating inverter-days (includes under-performing inverters)", "",
        "Here a large error is expected and is *not* model error: it is the gap the monitoring step (4C) has to detect.", "",
        _fmt(flat[(flat["eval_set"] == "valid_all_operating") & (flat["level"] == "daily") & (flat["plant"] != "both")
                  & (flat["model"] == s["model"]) & (flat["scope"] == s["scope"])].sort_values(["training_set", "plant"]),
             ["training_set", "plant", "n", "mae", "r2", "mape_pct"], {"mae": 1, "r2": 3, "mape_pct": 2}),
        "## Cross-plant check (healthy validation days)", "",
        "A per-plant model trained on one plant predicts the other plant, compared with the model trained on that plant itself.", "",
        _fmt(cross.sort_values(["training_set", "model", "test_plant", "trained_on"]),
             ["training_set", "model", "test_plant", "trained_on", "daily_mae_kwh", "daily_r2", "daily_mape_pct", "interval_mae_kw"],
             {"daily_mae_kwh": 1, "daily_r2": 3, "daily_mape_pct": 2, "interval_mae_kw": 2}),
        "## Tuning (expanding-window CV over whole training days)", "",
        "| run | best parameters | CV MAE (kW) | training rows |", "|---|---|---|---|",
        *[f"| {k} | {v.get('best_params', '—')} | {v['cv_mae_kw']:.2f} | {v['train_rows']:,} |" if "cv_mae_kw" in v
          else f"| {k} | — (no hyperparameters) | — | {v['train_rows']:,} |" for k, v in tuning.items()],
        "", "Full metrics: `reports/metrics/solar_validation_metrics.csv`, `solar_cross_plant.csv`, `solar_tuning.json`. "
        "Saved model: `models/solar_expected_output.joblib` + `.json`.", ""]
    project_path("reports/solar_model_report.md").write_text("\n".join(lines), encoding="utf-8")
