"""Next-day and next-7-days electricity use of a home from inputs a person can type in.

Inputs: the units (kWh) used on each of the last 7 days, the type of home (eMARC categories), the date, and the
temperature of the target day and of the 3 days before (forecast or measured). Unlike the Phase 4/8 model it needs
no 15-minute load shape and no 30-day history, so it can be used for any home. Trained on the eMARC train split;
model choice and the 7-day error are measured on the validation split (Jul-Dec 2019) with models fitted on train
only. The eMARC test split is not used (it has already been scored twice for the Phase 4/8 model).

Days 2-7 are forecast recursively: each forecast becomes the "yesterday" of the next day.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from solvia.config import get_path, load_config, project_path
from solvia.features.preprocess import make_pipeline
from solvia.models.consumption import EnsembleRegressor, RelativeTarget

log = logging.getLogger(__name__)

TARGET = "target_kwh"
HISTORY = ["lag_1", "lag_2", "lag_7", "roll_mean_7", "roll_std_7"]
CALENDAR = ["day_of_week", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos"]
WEATHER = ["t2m_max_target_day", "cdd_target_day", "t2m_mean_3d"]
CATEGORICAL = ["household_type"]
FEATURES = HISTORY + CALENDAR + WEATHER + CATEGORICAL
HORIZONS = range(1, 8)


def _candidates(rs: int) -> dict:
    hgb = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=100, loss="absolute_error", random_state=rs)
    return {
        "ridge": lambda: make_pipeline(Ridge(alpha=100.0), FEATURES, linear=True, categorical=CATEGORICAL),
        "hist_gradient_boosting_relative": lambda: RelativeTarget(
            make_pipeline(hgb(), FEATURES, linear=False, categorical=CATEGORICAL), scale_col="roll_mean_7"),
    }


def history_row(last7: list[float]) -> dict:
    """History features for the day after `last7` (oldest first, yesterday last)."""
    s = np.asarray(last7, dtype=float)
    return {"lag_1": s[-1], "lag_2": s[-2], "lag_7": s[-7], "roll_mean_7": s.mean(), "roll_std_7": s.std(ddof=1)}


@dataclass
class HomeUseModel:
    model: object
    features: list
    name: str
    half_width_80: dict = field(default_factory=dict)   # horizon -> half-width as a share of the 7-day mean
    info: dict = field(default_factory=dict)
    blend_weight: dict = field(default_factory=dict)    # horizon -> weight of the model vs the last-7-days mean

    def forecast(self, last7: list[float], household_type: str, days: pd.DataFrame) -> pd.DataFrame:
        """`days`: one row per future day (date, t2m_max_target_day, cdd_target_day, t2m_mean_3d), in order."""
        hist, rows = list(last7), []
        for h, d in enumerate(days.itertuples(index=False), start=1):
            date = pd.Timestamp(d.date)
            x = {**history_row(hist[-7:]), **calendar_row(date), "household_type": household_type,
                 "t2m_max_target_day": d.t2m_max_target_day, "cdd_target_day": d.cdd_target_day,
                 "t2m_mean_3d": d.t2m_mean_3d}
            p = float(max(0.0, self.model.predict(pd.DataFrame([x])[self.features])[0]))
            w, base = self.blend_weight.get(h, 1.0), float(np.mean(last7))
            out = w * p + (1 - w) * base
            hw = self.half_width_80[h] * base
            rows.append({"date": date, "h": h, "kwh": out, "low": max(0.0, out - hw), "high": out + hw})
            hist.append(p)                                    # the recursion uses the model, as in validation
        return pd.DataFrame(rows)


def calendar_row(date: pd.Timestamp) -> dict:
    from solvia.features.calendar import india_holidays
    doy = date.day_of_year          # festival dates are listed for 2018-2020 only; later years get the fixed holidays
    return {"day_of_week": date.day_of_week, "month": date.month, "is_weekend": int(date.day_of_week >= 5),
            "is_holiday": int(date.date() in india_holidays([date.year])), "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25)}


def _recursive_valid(model, df: pd.DataFrame) -> pd.DataFrame:
    """7-day recursive forecasts from every validation origin with 7 measured days before it and 7 after.
    Weather of the target days is the measured one (the dashboard uses forecasts)."""
    out = []
    for mid, g in df.groupby("deployment_id"):
        g = g.set_index("date").sort_index()
        y = g[TARGET].asfreq("D")
        for origin in g.index[g["split"] == "valid"][::3]:
            past = y.loc[origin - pd.Timedelta(days=6): origin]
            fut = g.reindex(pd.date_range(origin + pd.Timedelta(days=1), periods=7))
            if past.isna().any() or len(past) < 7 or fut[TARGET].isna().any() or (fut["split"] != "valid").any():
                continue
            hist = list(past.to_numpy())
            for h, (date, r) in enumerate(fut.iterrows(), start=1):
                x = r[FEATURES].to_frame().T.copy()
                for k, v in history_row(hist[-7:]).items():
                    x[k] = v
                p = float(max(0.0, model.predict(x.astype({c: float for c in FEATURES if c not in CATEGORICAL}))[0]))
                out.append({"deployment_id": mid, "origin": origin, "h": h, "actual": r[TARGET], "pred": p,
                            "baseline": float(past.mean()), "scale": float(past.mean())})
                hist.append(p)
    return pd.DataFrame(out)


def run_home_simple() -> dict:
    rs = load_config()["random_state"]
    df = pd.read_parquet(get_path("processed") / "india_consumption_features.parquet")
    df = df.dropna(subset=[c for c in FEATURES if c not in CATEGORICAL])
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    fitted, mae = {}, {}
    for name, make in _candidates(rs).items():
        fitted[name] = make().fit(train[FEATURES], train[TARGET])
        mae[name] = float(np.mean(np.abs(valid[TARGET] - np.clip(fitted[name].predict(valid[FEATURES]), 0, None))))
    fitted["ensemble"] = EnsembleRegressor([fitted["ridge"], fitted["hist_gradient_boosting_relative"]], ["ridge", "hgb"])
    mae["ensemble"] = float(np.mean(np.abs(valid[TARGET] - np.clip(fitted["ensemble"].predict(valid[FEATURES]), 0, None))))
    mae["baseline_mean_7d"] = float(np.mean(np.abs(valid[TARGET] - valid["roll_mean_7"])))
    selected = min(["ridge", "hist_gradient_boosting_relative", "ensemble"], key=mae.get)
    log.info("home_simple: validation MAE (1 day) %s -> %s", {k: round(v, 3) for k, v in mae.items()}, selected)

    # weather sensitivity: target-day temperature replaced by the normal for the date (a forecast with no skill)
    cols = ["t2m_max_target_day", "cdd_target_day"]
    normal = train.groupby(["region", "month"])[cols].mean()
    v2 = valid.copy()
    by_month = train.groupby("month")[cols].mean()         # fallback: a region with no train rows in that month
    v2[cols] = (normal.reindex(pd.MultiIndex.from_frame(valid[["region", "month"]])).set_index(valid.index)
                .fillna(by_month.reindex(valid["month"]).set_index(valid.index))).to_numpy()
    mae_normal_weather = float(np.mean(np.abs(v2[TARGET] - np.clip(fitted[selected].predict(v2[FEATURES]), 0, None))))

    rec = _recursive_valid(fitted[selected], df)
    # per day ahead: model, last-7-days mean, or half of each, whichever has the lower validation MAE
    blend = lambda w: w * rec["pred"] + (1 - w) * rec["baseline"]
    weights = {}
    for h, g in rec.groupby("h"):
        weights[int(h)] = min((1.0, 0.5, 0.0), key=lambda w: float((g["actual"] - blend(w).loc[g.index]).abs().mean()))
    rec["final"] = blend(rec["h"].map(weights))
    rec["err"], rec["base_err"] = (rec["actual"] - rec["pred"]).abs(), (rec["actual"] - rec["baseline"]).abs()
    rec["final_err"] = (rec["actual"] - rec["final"]).abs()
    by_h = rec.groupby("h").agg(mae=("err", "mean"), baseline_mae=("base_err", "mean"), final_mae=("final_err", "mean")).reset_index()
    by_h["model_weight"] = by_h["h"].map(weights)
    rel = rec["final_err"] / rec["scale"].clip(lower=0.1)
    half = {int(h): float(q) for h, q in rel.groupby(rec["h"]).quantile(0.8).items()}
    cover = float((rec["final_err"] <= rec["h"].map(half) * rec["scale"]).mean())

    # final: same choice, refitted on train + validation
    fit_data = pd.concat([train, valid])
    if selected == "ensemble":
        final = EnsembleRegressor([clone(m).fit(fit_data[FEATURES], fit_data[TARGET]) for m in fitted["ensemble"].members],
                                  fitted["ensemble"].names)
    else:
        final = clone(fitted[selected]).fit(fit_data[FEATURES], fit_data[TARGET])
    info = {"selected": selected, "valid_mae_1day": mae, "valid_mae_1day_normal_weather": mae_normal_weather,
            "valid_7day": by_h.round(4).to_dict("records"), "valid_7day_origins": int(rec["origin"].nunique()),
            "valid_7day_meters": int(rec["deployment_id"].nunique()), "interval_80_coverage_valid": cover,
            "fit_on": "train_valid", "train_meters": int(train["deployment_id"].nunique()),
            "household_types": sorted(df["household_type"].unique())}
    model = HomeUseModel(final, FEATURES, selected, half, info, weights)
    joblib.dump(model, get_path("models") / "home_use_simple.joblib")
    (get_path("models") / "home_use_simple.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    _report(info)
    return info


def _report(info: dict) -> None:
    m = info["valid_mae_1day"]
    lines = ["# Home use from simple inputs (next day and next 7 days)", "",
             "*Generated by `python -m solvia.pipeline --steps model_home_simple`. Prayas eMARC homes; validation "
             "Jul–Dec 2019; models fitted on train only for every number below; the eMARC test split is not used.*", "",
             "Inputs: last 7 days of units, type of home, date, temperature of the target day and the 3 days before.", "",
             "## Next day (validation MAE, kWh/day)", "", "| Model | MAE |", "|---|---|",
             *[f"| {k}{' **(selected)**' if k == info['selected'] else ''} | {v:.3f} |" for k, v in m.items()], "",
             f"With the target-day temperature replaced by the normal for the month: {info['valid_mae_1day_normal_weather']:.3f} "
             "kWh/day, so a poor temperature forecast costs little.", "",
             f"## Next 7 days (recursive; {info['valid_7day_origins']} start days, {info['valid_7day_meters']} homes)", "",
             "| Day ahead | Model MAE | 'Last 7 days' average' MAE | Weight of the model | Used MAE |", "|---|---|---|---|---|",
             *[f"| {r['h']} | {r['mae']:.3f} | {r['baseline_mae']:.3f} | {r['model_weight']:.1f} | {r['final_mae']:.3f} |"
               for r in info["valid_7day"]], "",
             "The weight per day ahead (1 = model, 0.5 = half each, 0 = last 7 days' average) is chosen on these "
             "validation days, so 'Used MAE' is slightly optimistic.", "",
             f"80 % range: half-width = validation quantile of the error relative to the 7-day mean, per day ahead; it "
             f"contained {info['interval_80_coverage_valid']:.0%} of the validation days (in-sample, by construction ~80 %).", ""]
    project_path("reports/home_use_simple_report.md").write_text("\n".join(lines), encoding="utf-8")
