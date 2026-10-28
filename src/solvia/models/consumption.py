"""Phase 4A: next-day household electricity consumption (kWh/day). India (eMARC) first, then France (UCI).

Same procedure for both tracks:
- 3 baselines: yesterday, same day last week, mean of the last 7 days.
- Ridge, Random Forest, HistGradientBoosting (+ for India a "relative" HistGradientBoosting that predicts the
  ratio to the home's own 30-day mean, so one model can serve homes using 0.4 or 20 kWh/day).
- Tuning: grid search with expanding-window CV over whole training days (never valid/test days).
- Selection: lowest validation MAE. The test split is not used here (Phase 5, once).
- Optional `final_model` in the config (Phase 8, India): an equal-weight ensemble of fixed members, scored on
  validation with members fitted on train, then refitted on train + validation for the saved model.
- Extras: 10th-90th percentile intervals (quantile HistGradientBoosting, widened by a conformal adjustment
  computed on validation so that 80 % of validation days fall inside) and the labelled forecast-proxy
  experiment (target-day weather standing in for a weather forecast).
India: one model across meters, each row is one meter-day built from that meter only; metrics per meter,
for meters seen in training vs new meters, by household type and region.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from solvia.config import get_path, load_config, project_path
from solvia.features import consumption as fr
from solvia.features import india_consumption as ind
from solvia.features.preprocess import make_pipeline
from solvia.models.common import day_cv_splits, regression_metrics, save_model

log = logging.getLogger(__name__)

TARGET = fr.TARGET
MAPE_MIN_KWH = 1.0
BASELINE_COLS = list(fr.BASELINES)


@dataclass
class Track:
    name: str
    title: str
    features_file: str
    features: list
    categorical: list
    proxy: list
    meter_col: str | None
    config_key: str
    relative_candidates: bool


TRACKS = {
    "india": Track("india", "Indian households (Prayas eMARC)", "india_consumption_features.parquet", ind.FEATURES,
                   ind.CATEGORICAL_FEATURES, ind.FORECAST_PROXY_FEATURES, "deployment_id", "emarc", True),
    "france": Track("france", "French household (UCI, Sceaux)", "consumption_features.parquet", fr.FEATURES, [],
                    fr.FORECAST_PROXY_FEATURES, None, "consumption", False),
}


class RelativeTarget(BaseEstimator, RegressorMixin):
    """Fit the wrapped estimator on y / scale and multiply back, scale = the row's own 30-day mean (a past value)."""

    def __init__(self, estimator=None, scale_col: str = "roll_mean_30", floor: float = 0.1):
        self.estimator = estimator
        self.scale_col = scale_col
        self.floor = floor

    def _scale(self, X) -> np.ndarray:
        return np.asarray(X[self.scale_col], dtype=float).clip(min=self.floor)

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype=float) / self._scale(X))
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * self._scale(X)


@dataclass
class ConsumptionModel:
    """Selected point model plus 10th / 90th percentile models. Predictions are clipped at 0 kWh.
    `interval_adjust_kwh` widens both ends (conformal adjustment from validation; 0 = raw quantiles)."""
    track: str
    name: str
    features: list
    point: object
    p10: object | None = None
    p90: object | None = None
    info: dict = field(default_factory=dict)
    interval_adjust_kwh: float = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.point.predict(X[self.features]), 0, None)

    def predict_interval(self, X: pd.DataFrame, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.p10.predict(X[self.features]), self.p90.predict(X[self.features])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        q = self.interval_adjust_kwh if calibrated else 0.0
        return np.clip(lo - q, 0, None), np.clip(hi + q, 0, None)


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Equal-weight mean of already fitted regressors (members are fixed in advance, not fitted on validation)."""

    def __init__(self, members=None, names=None):
        self.members = members
        self.names = names

    def fit(self, X, y):
        """Refit every member on (X, y); not used by the pipeline, which builds the ensemble from fitted members."""
        self.members = [clone(m).fit(X, y) for m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def conformal_adjustment(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, coverage: float = 0.8) -> float:
    """Conformalized quantile regression: the amount q such that [lo - q, hi + q] contains `coverage` of the
    calibration days (finite-sample quantile of max(lo - y, y - hi)). q < 0 narrows an interval that is too wide."""
    scores = np.maximum(lo - y, y - hi)
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def candidates(track: Track, rs: int) -> dict[str, tuple]:
    """name -> (pipeline factory(features), parameter grid)."""
    cat = track.categorical
    hgb_grid = {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31], "model__min_samples_leaf": [20, 100]}
    out = {
        "ridge": (lambda f: make_pipeline(Ridge(), f, linear=True, categorical=cat), {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
        "random_forest": (lambda f: make_pipeline(RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1), f,
                                                  linear=False, categorical=cat),
                          {"model__min_samples_leaf": [5, 20], "model__max_features": [1.0, 0.33]}),
        "hist_gradient_boosting": (lambda f: make_pipeline(HistGradientBoostingRegressor(max_iter=400, random_state=rs), f,
                                                           linear=False, categorical=cat), hgb_grid),
    }
    if track.relative_candidates:
        out["hist_gradient_boosting_relative"] = (
            lambda f: RelativeTarget(make_pipeline(HistGradientBoostingRegressor(max_iter=400, random_state=rs), f,
                                                   linear=False, categorical=cat)),
            {f"estimator__{k}": v for k, v in hgb_grid.items()})
        # Phase 8: absolute-error loss matches the MAE metric (ensemble member)
        out["hist_gradient_boosting_relative_absolute"] = (
            lambda f: RelativeTarget(make_pipeline(HistGradientBoostingRegressor(max_iter=400, loss="absolute_error",
                                                                                 random_state=rs), f,
                                                   linear=False, categorical=cat)),
            {f"estimator__{k}": v for k, v in hgb_grid.items()})
    return out


def tune(factory, grid: dict, features: list, train: pd.DataFrame, n_splits: int) -> tuple:
    train = train.reset_index(drop=True)
    search = GridSearchCV(factory(features), grid, cv=day_cv_splits(train["date"], n_splits),
                          scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
    search.fit(train[features], train[TARGET])
    folds = [round(float(-search.cv_results_[f"split{k}_test_score"][search.best_index_]), 3) for k in range(n_splits)]
    return search.best_estimator_, {
        "best_params": search.best_params_, "cv_mae_kwh": float(-search.best_score_), "cv_mae_by_fold_kwh": folds,
        "cv_results": [{"params": p, "mae_kwh": float(-s)} for p, s in
                       zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])]}


def load_track(track: Track) -> pd.DataFrame:
    df = pd.read_parquet(get_path("processed") / track.features_file)
    if "date" not in df.columns:
        df = df.reset_index()
    return df.sort_values(["date"] + ([track.meter_col] if track.meter_col else [])).reset_index(drop=True)


def breakdown(df: pd.DataFrame, pred: np.ndarray, track: Track, seen: set) -> dict:
    """Per-meter and group metrics on one evaluation frame (India only)."""
    e = df.assign(pred=pred, abs_err=np.abs(df[TARGET] - pred))
    per_meter = e.groupby(track.meter_col)["abs_err"].mean()
    out = {"meters": int(per_meter.size), "per_meter_mae_median": float(per_meter.median()),
           "per_meter_mae_p90": float(per_meter.quantile(0.9))}
    e["meter_group"] = np.where(e[track.meter_col].isin(seen), "seen in training", "new meter")
    for col in ("meter_group", "household_type", "region"):
        out[f"by_{col}"] = {k: {"rows": int(len(g)), "meters": int(g[track.meter_col].nunique()),
                                "mae": float(g["abs_err"].mean()), "mean_kwh": float(g[TARGET].mean())}
                            for k, g in e.groupby(col)}
    return out


def run_consumption(track_name: str) -> dict:
    track = TRACKS[track_name]
    cfg = load_config()
    rs, n_splits = cfg["random_state"], cfg[track.config_key]["cv_splits"]
    df = load_track(track)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    log.info("%s: train %d rows, valid %d rows (test untouched)", track.name, len(train), len(valid))

    results, tuning, fitted, preds = [], {}, {}, {}
    for b in BASELINE_COLS:
        preds[b] = valid[b].to_numpy()
        results.append({"model": b, "kind": "baseline", **regression_metrics(valid[TARGET], preds[b], MAPE_MIN_KWH)})
    for name, (factory, grid) in candidates(track, rs).items():
        model, info = tune(factory, grid, track.features, train, n_splits)
        fitted[name], tuning[name] = model, info
        preds[name] = np.clip(model.predict(valid[track.features]), 0, None)
        results.append({"model": name, "kind": "model", **regression_metrics(valid[TARGET], preds[name], MAPE_MIN_KWH)})
        log.info("%s / %s: CV MAE %.3f, valid MAE %.3f %s", track.name, name, info["cv_mae_kwh"], results[-1]["mae"], info["best_params"])
    final = cfg[track.config_key].get("final_model")
    if final:
        # same definition as the saved model: mean of the members' raw predictions, then clipped at 0
        preds["ensemble"] = np.clip(np.mean([fitted[m].predict(valid[track.features]) for m in final["ensemble"]], axis=0), 0, None)
        results.append({"model": "ensemble", "kind": "ensemble", **regression_metrics(valid[TARGET], preds["ensemble"], MAPE_MIN_KWH)})
    table = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    best_baseline = table[table["kind"] == "baseline"].iloc[0]
    best = table[table["kind"] == "model"].iloc[0]                  # best single model (quantile settings, proxy)
    selected = "ensemble" if final else best["model"]
    table["skill_vs_best_baseline_pct"] = (1 - table["mae"] / best_baseline["mae"]) * 100
    fit_data = pd.concat([train, valid]) if final and final["fit_on"] == "train_valid" else train

    # 10th / 90th percentile models with the tuned HistGradientBoosting settings of the selected target form
    hgb_name = best["model"] if best["model"].startswith("hist_gradient_boosting") else "hist_gradient_boosting"
    rel = "relative" in hgb_name
    params = {k.replace("estimator__", ""): v for k, v in tuning[hgb_name]["best_params"].items()}
    quant = {}
    for q in (0.1, 0.9):
        pipe = make_pipeline(HistGradientBoostingRegressor(max_iter=400, random_state=rs, loss="quantile", quantile=q),
                             track.features, linear=False, categorical=track.categorical).set_params(**params)
        quant[q] = (RelativeTarget(pipe) if rel else pipe).fit(train[track.features], train[TARGET])
    point = fitted[best["model"]]
    cm = ConsumptionModel(track.name, selected, list(track.features), point, quant[0.1], quant[0.9])
    y = valid[TARGET].to_numpy()
    lo, hi = cm.predict_interval(valid, calibrated=False)
    cm.interval_adjust_kwh = conformal_adjustment(y, lo, hi, 0.8)
    clo, chi = cm.predict_interval(valid)
    interval = {"coverage_p10_p90": float(((y >= lo) & (y <= hi)).mean()), "target_coverage": 0.8,
                "mean_width_kwh": float(np.mean(hi - lo)), "quantile_model": hgb_name,
                "conformal_adjust_kwh": cm.interval_adjust_kwh,
                "calibrated_coverage_valid": float(((y >= clo) & (y <= chi)).mean()),
                "calibrated_mean_width_kwh": float(np.mean(chi - clo))}
    if final:
        # the saved model: members refitted (same tuned settings) on fit_data; the conformal adjustment from the
        # train-only quantile models on validation is kept, which is conservative after refitting on more data
        cand = candidates(track, rs)
        cm.point = EnsembleRegressor([cand[m][0](track.features).set_params(**tuning[m]["best_params"])
                                      .fit(fit_data[track.features], fit_data[TARGET]) for m in final["ensemble"]],
                                     list(final["ensemble"]))
        if fit_data is not train:
            cm.p10, cm.p90 = (clone(m).fit(fit_data[track.features], fit_data[TARGET]) for m in (cm.p10, cm.p90))

    # forecast-proxy experiment (labelled): same selected configuration + target-day weather
    feats_proxy = list(track.features) + list(track.proxy)
    factory, grid = candidates(track, rs)[best["model"]]
    proxy_model = factory(feats_proxy).set_params(**tuning[best["model"]]["best_params"]).fit(train[feats_proxy], train[TARGET])
    proxy_pred = np.clip(proxy_model.predict(valid[feats_proxy]), 0, None)
    proxy = {"features_added": track.proxy, "valid_mae_without": float(best["mae"]),
             "valid_mae_with_forecast_proxy": float(np.mean(np.abs(y - proxy_pred))),
             "note": "EXPERIMENT ONLY: uses the observed weather of the target day as a stand-in for a weather forecast; "
                     "a real forecast is less accurate, so this is an upper bound on what forecasts could add."}

    extra = {}
    if track.meter_col:
        seen = set(train[track.meter_col])
        extra = {"breakdown_selected": breakdown(valid, preds[selected], track, seen),
                 "breakdown_best_baseline": breakdown(valid, preds[best_baseline["model"]], track, seen)}
        e = valid.assign(m=np.abs(y - preds[selected]), b=np.abs(y - preds[best_baseline["model"]]))
        per = e.groupby(track.meter_col)[["m", "b"]].mean()
        extra["share_of_meters_where_model_beats_baseline"] = float((per["m"] < per["b"]).mean())

    members = final["ensemble"] if final else [best["model"]]
    fit_on = final["fit_on"] if final else "train"
    cm.info = {"valid_mae": float(table.loc[table["model"] == selected, "mae"].iloc[0]), "fit_on": fit_on,
               "best_params": {m: tuning[m]["best_params"] for m in members}}
    save_model(cm, f"consumption_{track.name}", {
        "objective": f"A: next-day consumption (kWh/day), {track.title}",
        "model": f"equal-weight ensemble of {', '.join(members)}" if final else best["model"],
        "features": track.features, "categorical": track.categorical, "target": TARGET, "fit_on": fit_on,
        "train_dates": f"{fit_data['date'].min().date()} to {fit_data['date'].max().date()}", "train_rows": int(len(fit_data)),
        "hyperparameters": cm.info["best_params"] if final else tuning[best["model"]]["best_params"],
        "selection": ("ensemble with members fixed in advance (Phase 8), validation MAE with members fitted on train only; "
                      "saved model refitted on train + validation" if final else "lowest validation MAE"),
        "validation_metrics": table[table["model"] == selected].iloc[0].to_dict(),
        "interval": interval, "test_metrics": "not computed (Phase 5)"})

    metrics = get_path("metrics")
    table.to_csv(metrics / f"consumption_{track.name}_validation.csv", index=False)
    summary = {"track": track.name, "selected": selected, "best_single_model": best["model"],
               "final_model": {"members": members, "fit_on": fit_on} if final else None,
               "best_baseline": best_baseline["model"],
               "rows": {"train": int(len(train)), "valid": int(len(valid))}, "validation": table.to_dict(orient="records"),
               "tuning": {k: {kk: vv for kk, vv in v.items() if kk != "cv_results"} for k, v in tuning.items()},
               "interval": interval, "forecast_proxy_experiment": proxy, **extra}
    (metrics / f"consumption_{track.name}_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (metrics / f"consumption_{track.name}_tuning.json").write_text(json.dumps(tuning, indent=1, default=str), encoding="utf-8")
    write_report(track, summary, table)
    return summary


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and pd.notna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(track: Track, s: dict, table: pd.DataFrame) -> None:
    sel = table[table["model"] == s["selected"]].iloc[0]
    iv, px = s["interval"], s["forecast_proxy_experiment"]
    lines = [f"# Next-day consumption: {track.title} (Phase 4A)", "",
             f"*Generated by `python -m solvia.pipeline --steps model_consumption_{track.name}`. Validation split only; "
             "the test split is untouched until Phase 5.*", "",
             f"Training rows {s['rows']['train']:,}, validation rows {s['rows']['valid']:,}. Target: kWh on the next day. "
             f"{len(track.features)} features (all from the past or the calendar).", "",
             f"## Selected: `{s['selected']}`", "",
             f"Validation MAE **{sel['mae']:.3f} kWh/day** (RMSE {sel['rmse']:.3f}, R² {sel['r2']:.3f}, MAPE {sel['mape_pct']:.1f} % on days "
             f"≥ {MAPE_MIN_KWH:g} kWh), **{sel['skill_vs_best_baseline_pct']:.1f} % lower MAE than the best baseline** (`{s['best_baseline']}`).", "",
             *([f"Final model (Phase 8): equal-weight ensemble of {', '.join(f'`{m}`' for m in s['final_model']['members'])}, "
                f"members fixed in advance. Its validation score above uses members fitted on the training split only; the "
                f"saved model is then refitted on {s['final_model']['fit_on'].replace('_', ' + ')}. Best single model: "
                f"`{s['best_single_model']}`.", ""] if s.get("final_model") else []),
             "## All baselines and models (validation)", "",
             _md_table(table, ["model", "kind", "mae", "rmse", "r2", "nmae", "mape_pct", "skill_vs_best_baseline_pct"],
                       {"mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}", "nmae": "{:.3f}", "mape_pct": "{:.1f}",
                        "skill_vs_best_baseline_pct": "{:+.1f} %"}), "",
             "`mae`, `rmse` in kWh/day; `nmae` = MAE / mean actual; skill = MAE reduction vs the best baseline.", ""]
    if track.meter_col:
        m, b = s["breakdown_selected"], s["breakdown_best_baseline"]
        lines += ["## Per household (validation)", "",
                  f"Median per-meter MAE {m['per_meter_mae_median']:.3f} kWh/day over {m['meters']} meters (best baseline "
                  f"{b['per_meter_mae_median']:.3f}); the model beats the best baseline for "
                  f"{s['share_of_meters_where_model_beats_baseline']:.0%} of meters.", ""]
        for col, label in [("meter_group", "Meters seen in training vs new meters"), ("household_type", "By household type"),
                           ("region", "By region")]:
            rows = pd.DataFrame([{"group": k, "rows": v["rows"], "meters": v["meters"], "mean_kwh": v["mean_kwh"], "mae": v["mae"],
                                  "baseline_mae": b[f"by_{col}"][k]["mae"]} for k, v in m[f"by_{col}"].items()])
            lines += [f"**{label}**", "", _md_table(rows, ["group", "rows", "meters", "mean_kwh", "mae", "baseline_mae"],
                                                    {"mean_kwh": "{:.2f}", "mae": "{:.3f}", "baseline_mae": "{:.3f}"}), ""]
    lines += ["## Prediction interval (10th–90th percentile)", "",
              f"Quantile HistGradientBoosting (`{iv['quantile_model']}` settings): {iv['coverage_p10_p90']:.1%} of validation days fall "
              f"inside the interval (target 80 %), mean width {iv['mean_width_kwh']:.2f} kWh.", "",
              f"**Conformal adjustment** (computed on validation): each end is moved out by {iv['conformal_adjust_kwh']:+.2f} kWh, "
              f"giving {iv['calibrated_coverage_valid']:.1%} coverage on validation (80 % by construction, so not evidence) and a mean "
              f"width of {iv['calibrated_mean_width_kwh']:.2f} kWh. The honest check is the test coverage in Phase 5.", "",
              "## Experiment: target-day weather as a stand-in for a forecast (labelled)", "",
              f"Adding {', '.join(f'`{c}`' for c in px['features_added'])} changes validation MAE from {px['valid_mae_without']:.3f} to "
              f"{px['valid_mae_with_forecast_proxy']:.3f} kWh/day. {px['note']}", "",
              "## Tuning (expanding-window CV over whole training days)", "",
              "| model | best parameters | CV MAE (kWh) | MAE per fold, earliest first |", "|---|---|---|---|",
              *[f"| {k} | {v['best_params']} | {v['cv_mae_kwh']:.3f} | {v['cv_mae_by_fold_kwh']} |" for k, v in s["tuning"].items()], "",
              "The first fold trains on the shortest history; a model that is much worse there than on later folds "
              "is sensitive to how much history it has seen (linear models extrapolate when the season changes).", "",
              f"Files: `reports/metrics/consumption_{track.name}_*.{{csv,json}}`, `models/consumption_{track.name}.joblib` + `.json`.", ""]
    project_path(f"reports/consumption_{track.name}_report.md").write_text("\n".join(lines), encoding="utf-8")
