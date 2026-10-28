"""Phase 8: can the models be made more accurate? Experiments on train / validation only.

The test split was scored once in Phase 5 and is not touched here. Every candidate is tuned with the same
expanding-window day CV on the training split and compared on validation; ensembles have fixed members and
equal weights (chosen before looking at results) so validation is not used to fit them.

Experiments
- Consumption (India, France): gradient boosting with absolute-error loss (the metric is MAE), Ridge on the
  ratio to the home's own 30-day mean, a wider gradient-boosting grid with L2 regularisation, equal-weight
  ensembles, and a "more recent data" backtest (fit on train vs train + first half of validation, both scored
  on the second half of validation).
- Solar: wider grid, irradiation-monotonic gradient boosting, gradient boosting + random forest ensemble.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from solvia.config import get_path, load_config, project_path
from solvia.features.preprocess import make_pipeline
from solvia.features.solar import FEATURES as SOLAR_FEATURES
from solvia.features.solar import TARGET as SOLAR_TARGET
from solvia.models import consumption as cons
from solvia.models import solar
from solvia.models.common import regression_metrics
from solvia.viz import PLANT_NAMES

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- consumption
def _new_candidates(track: cons.Track, rs: int) -> dict[str, tuple]:
    cat = track.categorical
    hgb_grid = {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31], "model__min_samples_leaf": [20, 100]}
    wide = {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_leaf_nodes": [7, 15, 31, 63],
            "model__min_samples_leaf": [20, 100], "model__l2_regularization": [0.0, 1.0]}

    def hgb(loss="squared_error", it=400):
        return lambda f: make_pipeline(HistGradientBoostingRegressor(max_iter=it, loss=loss, random_state=rs), f,
                                       linear=False, categorical=cat)
    return {
        "hgb_absolute_loss": (hgb("absolute_error"), hgb_grid),
        "hgb_relative_absolute_loss": (lambda f: cons.RelativeTarget(hgb("absolute_error")(f)),
                                       {f"estimator__{k}": v for k, v in hgb_grid.items()}),
        "ridge_relative": (lambda f: cons.RelativeTarget(make_pipeline(Ridge(), f, linear=True, categorical=cat)),
                           {"estimator__model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
        "hgb_wide_grid": (hgb(), wide),
    }


def _fit(factory, params: dict, features: list, d: pd.DataFrame):
    return factory(features).set_params(**params).fit(d[features], d[cons.TARGET])


def consumption_experiments(track_name: str, cfg: dict) -> dict:
    track = cons.TRACKS[track_name]
    rs, n_splits = cfg["random_state"], cfg[track.config_key]["cv_splits"]
    df = cons.load_track(track)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    y = valid[cons.TARGET].to_numpy()
    summary = json.loads((get_path("metrics") / f"consumption_{track.name}_summary.json").read_text(encoding="utf-8"))
    old = {**cons.candidates(track, rs)}
    new = _new_candidates(track, rs)

    preds, rows, params = {}, [], {}
    for name, (factory, _) in old.items():
        p = summary["tuning"][name]
        params[name] = p["best_params"]
        preds[name] = np.clip(_fit(factory, p["best_params"], track.features, train).predict(valid[track.features]), 0, None)
        rows.append({"model": name, "group": "Phase 4", "cv_mae": p["cv_mae_kwh"], **regression_metrics(y, preds[name], cons.MAPE_MIN_KWH)})
    for name, (factory, grid) in new.items():
        model, info = cons.tune(factory, grid, track.features, train, n_splits)
        params[name] = info["best_params"]
        preds[name] = np.clip(model.predict(valid[track.features]), 0, None)
        rows.append({"model": name, "group": "new", "cv_mae": info["cv_mae_kwh"], **regression_metrics(y, preds[name], cons.MAPE_MIN_KWH)})
        log.info("%s / %s: CV MAE %.3f, valid MAE %.3f %s", track.name, name, info["cv_mae_kwh"], rows[-1]["mae"], info["best_params"])

    # ensembles: members fixed in advance, equal weights
    hgb_member = "hgb_relative_absolute_loss" if track.relative_candidates else "hgb_absolute_loss"
    ensembles = {"ensemble_ridge_rf": ["ridge", "random_forest"],
                 f"ensemble_ridge_rf_{hgb_member}": ["ridge", "random_forest", hgb_member]}
    for name, members in ensembles.items():
        preds[name] = np.mean([preds[m] for m in members], axis=0)
        rows.append({"model": name, "group": "ensemble", "cv_mae": np.nan, **regression_metrics(y, preds[name], cons.MAPE_MIN_KWH)})
    for b in cons.BASELINE_COLS:
        rows.append({"model": b, "group": "baseline", "cv_mae": np.nan, **regression_metrics(y, valid[b], cons.MAPE_MIN_KWH)})

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    ref = table.loc[table["model"] == summary["selected"], "mae"].iloc[0]
    table["change_vs_selected_pct"] = (table["mae"] / ref - 1) * 100

    # more recent data: train vs train + first half of validation, both scored on the second half
    days = np.sort(valid["date"].unique())
    cut = days[len(days) // 2]
    h1, h2 = valid[valid["date"] < cut], valid[valid["date"] >= cut]
    recency = []
    best_new = table[table["group"] == "new"].iloc[0]["model"]
    for name in dict.fromkeys([summary["selected"], "random_forest", best_new]):
        factory = (old | new)[name][0]
        for label, fit_on in (("train only", train), ("train + first half of validation", pd.concat([train, h1]))):
            p = np.clip(_fit(factory, params[name], track.features, fit_on).predict(h2[track.features]), 0, None)
            recency.append({"model": name, "fitted_on": label, "rows": len(fit_on),
                            "mae_second_half_valid": float(np.mean(np.abs(h2[cons.TARGET].to_numpy() - p)))})
    rec = pd.DataFrame(recency)
    if track.meter_col:
        seen_tr, seen_h1 = set(train[track.meter_col]), set(pd.concat([train, h1])[track.meter_col])
        rec.attrs["meters_in_h2_unseen"] = {"train only": int((~h2[track.meter_col].drop_duplicates().isin(seen_tr)).sum()),
                                            "train + h1": int((~h2[track.meter_col].drop_duplicates().isin(seen_h1)).sum())}
    return {"table": table, "params": params, "recency": rec, "selected": summary["selected"], "split_day": str(pd.Timestamp(cut).date())}


# --------------------------------------------------------------------------- solar
def solar_experiments(cfg: dict) -> dict:
    sc, rs = cfg["solar"], cfg["random_state"]
    f = solar.model_rows(pd.read_parquet(get_path("processed") / "solar_features_15min.parquet"),
                         solar.healthy_inverter_days(pd.read_parquet(get_path("interim") / "solar_daily_inverter.parquet"),
                                                     sc["healthy_min_ratio"]))
    tuning = json.loads((get_path("metrics") / "solar_tuning.json").read_text(encoding="utf-8"))
    tmask = solar.training_mask(f, "healthy_reference")
    mono = [1 if c == "irradiation_kw_m2" else 0 for c in SOLAR_FEATURES]
    wide = {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [20, 100], "model__l2_regularization": [0.0, 1.0]}
    configs = {
        "hist_gradient_boosting (Phase 4)": ("tuned", HistGradientBoostingRegressor(max_iter=300, random_state=rs), "hist_gradient_boosting"),
        "random_forest (Phase 4)": ("tuned", RandomForestRegressor(n_estimators=200, random_state=rs, n_jobs=-1), "random_forest"),
        "hgb_wide_grid": ("grid", HistGradientBoostingRegressor(max_iter=600, random_state=rs), wide),
        "hgb_monotonic_irradiation": ("grid", HistGradientBoostingRegressor(max_iter=300, random_state=rs, monotonic_cst=mono),
                                      {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
                                       "model__min_samples_leaf": [20, 100]}),
    }
    preds, info = {}, {}
    for name, (kind, est, spec) in configs.items():
        pipes = {}
        for pid in PLANT_NAMES:
            rows = f[tmask & (f["plant_id"] == pid)]
            if kind == "tuned":
                p = tuning[f"healthy_reference/{spec}/{PLANT_NAMES[pid]}"]["best_params"]
                pipes[pid] = make_pipeline(clone(est), SOLAR_FEATURES, linear=False).set_params(**p).fit(rows[SOLAR_FEATURES], rows[SOLAR_TARGET])
                info[f"{name}/{PLANT_NAMES[pid]}"] = p
            else:
                pipes[pid], i = solar.tune(est, False, spec, rows, sc["cv_splits"])
                info[f"{name}/{PLANT_NAMES[pid]}"] = {**i["best_params"], "cv_mae_kw": round(i["cv_mae_kw"], 2)}
        preds[name] = solar.ExpectedSolarModel(name, "per_plant", "healthy_reference", pipes).predict_kw(f)
        log.info("solar / %s done", name)
    preds["ensemble_hgb_rf"] = (preds["hist_gradient_boosting (Phase 4)"] + preds["random_forest (Phase 4)"]) / 2
    rows = []
    for name, p in preds.items():
        for plant in ("both", *PLANT_NAMES):
            m = solar.eval_mask(f, True, "valid")
            m = m if plant == "both" else m & (f["plant_id"] == plant)
            r = solar.evaluate(f, p, m, sc["interval_hours"])
            rows.append({"model": name, "plant": PLANT_NAMES.get(plant, plant), "daily_mae_kwh": r["daily"]["mae"],
                         "daily_mape_pct": r["daily"]["mape_pct"], "interval_mae_kw": r["interval"]["mae"]})
    t = pd.DataFrame(rows)
    ref = t[(t["model"] == "hist_gradient_boosting (Phase 4)")].set_index("plant")["daily_mae_kwh"]
    t["change_vs_selected_pct"] = (t["daily_mae_kwh"] / t["plant"].map(ref) - 1) * 100
    return {"table": t.sort_values(["plant", "daily_mae_kwh"]).reset_index(drop=True), "params": info}


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and pd.notna(r[c]) else ("—" if pd.isna(r[c]) else str(r[c]))
                                      for c in cols) + " |")
    return "\n".join(out)


def run_improvement_experiments() -> dict:
    cfg = load_config()
    res = {t: consumption_experiments(t, cfg) for t in ("india", "france")}
    res["solar"] = solar_experiments(cfg)
    metrics = get_path("metrics")
    for t in ("india", "france"):
        res[t]["table"].to_csv(metrics / f"improve_consumption_{t}.csv", index=False)
        res[t]["recency"].to_csv(metrics / f"improve_recency_{t}.csv", index=False)
    res["solar"]["table"].to_csv(metrics / "improve_solar.csv", index=False)
    (metrics / "improve_params.json").write_text(json.dumps({k: v["params"] for k, v in res.items()}, indent=1, default=str),
                                                 encoding="utf-8")
    fmt = {"cv_mae": "{:.3f}", "mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}", "mape_pct": "{:.1f}",
           "change_vs_selected_pct": "{:+.1f} %", "mae_second_half_valid": "{:.3f}"}
    lines = ["# Can the models be more accurate? (experiments on validation; test untouched)", "",
             "*Generated by `python -m solvia.pipeline --steps improve`. New candidates are tuned with the same day-based CV on the "
             "training split; ensembles use fixed members and equal weights. The test split is not used.*", ""]
    for t in ("india", "france"):
        r = res[t]
        keep = ["model", "group", "cv_mae", "mae", "rmse", "r2", "mape_pct", "change_vs_selected_pct"]
        lines += [f"## Consumption, {t} (validation, kWh/day; reference = Phase 4 selection `{r['selected']}`)", "",
                  _md(r["table"][keep], fmt), "",
                  f"**More recent data** (scored on validation days from {r['split_day']} on):", "",
                  _md(r["recency"], fmt), ""]
        if r["recency"].attrs:
            lines += [f"Meters in the scored half never seen in fitting: {r['recency'].attrs['meters_in_h2_unseen']}.", ""]
    lines += ["## Solar expected output (validation, healthy inverter-days)", "",
              _md(res["solar"]["table"], {"daily_mae_kwh": "{:.1f}", "daily_mape_pct": "{:.2f}", "interval_mae_kw": "{:.2f}",
                                          "change_vs_selected_pct": "{:+.1f} %"}), ""]
    project_path("reports/improvement_experiments.md").write_text("\n".join(lines), encoding="utf-8")
    return res
