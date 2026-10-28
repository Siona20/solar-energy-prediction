"""Phase 5: the single evaluation on the test split, plus diagnostics and model cards.

Rules (decided before the test split was opened; docs/PROJECT_LOG.md):
- The saved models (trained on the training split, chosen on validation in Phase 4) are scored as they are.
  The other candidates are refitted with their stored tuned settings only so every row of the table has a
  test value; test numbers are never used to choose or change anything.
- Guard: the first run records the SHA-256 of every model file it scored. A later run with different model
  files stops with an error, because re-scoring changed models on the test split makes the test optimistic.
  Re-running with the same files reproduces the same numbers and is allowed.
- Permutation importance is computed on validation (it explains the model, it is not a performance claim).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from solvia.config import get_path, load_config, project_path
from solvia.features.preprocess import make_pipeline
from solvia.features.solar import FEATURES as SOLAR_FEATURES
from solvia.features.solar import TARGET as SOLAR_TARGET
from solvia.models import consumption as cons
from solvia.models import solar
from solvia.models.common import regression_metrics
from solvia.viz import INK, INK_2, MUTED, PLANT_COLORS, PLANT_NAMES, SERIES, apply_style, save_figure

log = logging.getLogger(__name__)

MODEL_FILES = ("solar_expected_output", "consumption_india", "consumption_france")
LOG_FILE = "test_run_log.json"
FIG = "reports/figures"


# --------------------------------------------------------------------------- guard
def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_test_guard() -> dict:
    """Return the model hashes; raise if the test split was already scored with different model files."""
    models = get_path("models")
    hashes = {n: _sha256(models / f"{n}.joblib") for n in MODEL_FILES}
    path = get_path("metrics") / LOG_FILE
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        changed = [n for n in MODEL_FILES if prev["model_sha256"].get(n) != hashes[n]]
        if changed:
            raise RuntimeError(
                f"The test split was already scored on {prev['first_run_utc']} with other versions of {changed}. "
                "Scoring changed models on the test split would make the test result optimistic. Restore the "
                f"scored models, or ask the project owner before deleting reports/metrics/{LOG_FILE}.")
        log.info("test split already scored on %s with these exact models: re-running reproduces the same numbers",
                 prev["first_run_utc"])
    return hashes


def _write_test_log(hashes: dict) -> None:
    path = get_path("metrics") / LOG_FILE
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"first_run_utc": now, "runs": 0}
    out = {"first_run_utc": prev["first_run_utc"], "last_run_utc": now, "runs": prev["runs"] + 1, "model_sha256": hashes,
           "note": "Models were chosen on validation before this run. Re-runs use identical model files."}
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- solar
def _solar_daily(f: pd.DataFrame, pred: np.ndarray, mask: pd.Series, hours: float) -> pd.DataFrame:
    keys = ["plant_id", "inverter_id", "date"]
    g = (f.loc[mask, keys + [SOLAR_TARGET, "healthy"]].assign(pred=pred[mask.to_numpy()])
         .groupby(keys).agg(actual=(SOLAR_TARGET, "sum"), pred=("pred", "sum"), healthy=("healthy", "first")))
    g[["actual", "pred"]] *= hours
    return g.reset_index()


def evaluate_solar(cfg: dict) -> dict:
    sc, rs = cfg["solar"], cfg["random_state"]
    f = solar.model_rows(pd.read_parquet(get_path("processed") / "solar_features_15min.parquet"),
                         solar.healthy_inverter_days(pd.read_parquet(get_path("interim") / "solar_daily_inverter.parquet"),
                                                     sc["healthy_min_ratio"]))
    selected = joblib.load(get_path("models") / "solar_expected_output.joblib")
    tuning = json.loads((get_path("metrics") / "solar_tuning.json").read_text(encoding="utf-8"))
    tmask = solar.training_mask(f, selected.training_set)
    units = ["all"] if selected.scope == "pooled" else list(PLANT_NAMES)

    models = {}
    for name, (est, linear, _) in solar.candidates(rs).items():
        if name == selected.name:
            models[name] = selected
            continue
        pipes = {}
        for unit in units:
            rows = f[tmask] if unit == "all" else f[tmask & (f["plant_id"] == unit)]
            params = tuning[f"{selected.training_set}/{name}/{PLANT_NAMES.get(unit, 'pooled')}"].get("best_params", {})
            pipes[unit] = make_pipeline(clone(est), SOLAR_FEATURES, linear=linear).set_params(**params).fit(
                rows[SOLAR_FEATURES], rows[SOLAR_TARGET])
        models[name] = solar.ExpectedSolarModel(name, selected.scope, selected.training_set, pipes)

    rows, preds = [], {}
    for name, m in models.items():
        preds[name] = m.predict_kw(f)
        for split in ("valid", "test"):
            for eval_set, healthy_only in solar.EVAL_SETS.items():
                base = solar.eval_mask(f, healthy_only, split)
                for plant in ("both", *PLANT_NAMES):
                    mask = base if plant == "both" else base & (f["plant_id"] == plant)
                    for level, met in solar.evaluate(f, preds[name], mask, sc["interval_hours"]).items():
                        rows.append({"model": name, "selected": name == selected.name, "split": split,
                                     "eval_set": eval_set.replace("valid_", ""), "plant": PLANT_NAMES.get(plant, plant),
                                     "level": level, **met})
    table = pd.DataFrame(rows)

    # reproducibility: the selected model's validation numbers must equal Phase 4's
    p4 = pd.read_csv(get_path("metrics") / "solar_validation_metrics.csv")
    p4 = p4[(p4["training_set"] == selected.training_set) & (p4["scope"] == selected.scope) & (p4["model"] == selected.name)
            & (p4["eval_set"] == "valid_healthy") & (p4["plant"] == "both") & (p4["level"] == "daily")]["mae"].iloc[0]
    now = table[(table["selected"]) & (table["split"] == "valid") & (table["eval_set"] == "healthy")
                & (table["plant"] == "both") & (table["level"] == "daily")]["mae"].iloc[0]
    if not np.isclose(p4, now):
        raise RuntimeError(f"selected solar model does not reproduce Phase 4 validation MAE ({now} vs {p4})")

    pred = preds[selected.name]
    daily = {s: _solar_daily(f, pred, solar.eval_mask(f, False, s), sc["interval_hours"]) for s in ("valid", "test")}
    imp = {}
    for pid in PLANT_NAMES:
        m = solar.eval_mask(f, True, "valid") & f["daytime"] & (f["plant_id"] == pid)
        pipe = selected.pipelines[pid] if selected.scope == "per_plant" else selected.pipelines["all"]
        r = permutation_importance(pipe, f.loc[m, SOLAR_FEATURES], f.loc[m, SOLAR_TARGET], scoring="neg_mean_absolute_error",
                                   n_repeats=5, random_state=rs)
        imp[PLANT_NAMES[pid]] = pd.Series(r.importances_mean, index=SOLAR_FEATURES).sort_values(ascending=False)
    return {"selected": selected, "table": table, "f": f, "pred": pred, "daily": daily, "importance": imp,
            "test_days": sorted(f.loc[f["split"] == "test", "date"].dt.date.unique())}


# --------------------------------------------------------------------------- consumption
def _cons_rows(model: str, kind: str, subset: str, y, p) -> dict:
    return {"model": model, "kind": kind, "subset": subset, **regression_metrics(y, p, cons.MAPE_MIN_KWH)}


def _coverage(y, lo, hi) -> dict:
    return {"coverage": float(((y >= lo) & (y <= hi)).mean()), "mean_width_kwh": float(np.mean(hi - lo)), "n": int(len(y))}


def evaluate_consumption(track_name: str, cfg: dict) -> dict:
    track = cons.TRACKS[track_name]
    rs = cfg["random_state"]
    df = cons.load_track(track)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    cm = joblib.load(get_path("models") / f"consumption_{track.name}.joblib")
    summary = json.loads((get_path("metrics") / f"consumption_{track.name}_summary.json").read_text(encoding="utf-8"))
    best_baseline = summary["best_baseline"]

    models = {}
    for name, (factory, _) in cons.candidates(track, rs).items():
        models[name] = cm.point if name == cm.name else factory(track.features).set_params(
            **summary["tuning"][name]["best_params"]).fit(train[track.features], train[cons.TARGET])
    # Phase 8 final model (ensemble refitted on train + validation): its validation score must come from the members
    # fitted on train only; the train-only ensemble is also scored on test to show what refitting changed
    final = summary.get("final_model")
    valid_model, fit_data = cm.point, train
    if final:
        valid_model = cons.EnsembleRegressor([models[m] for m in final["members"]], final["members"])
        if final["fit_on"] == "train_valid":
            models[f"{cm.name} (members fitted on train only)"] = valid_model
            fit_data = pd.concat([train, valid])

    subsets = {"valid": valid, "test": test}
    if track.meter_col:
        subsets |= {"test_normal": test[test["period"] == "normal"], "test_lockdown": test[test["period"] == "lockdown"]}
    preds = {}
    rows = []
    for sub, d in subsets.items():
        for b in cons.BASELINE_COLS:
            rows.append(_cons_rows(b, "baseline", sub, d[cons.TARGET], d[b]))
        for name, m in models.items():
            p = np.clip(m.predict(d[track.features]), 0, None)
            preds[(name, sub)] = p
            rows.append(_cons_rows(name, "model", sub, d[cons.TARGET], p))
        if final:
            p = np.clip((valid_model if sub == "valid" else cm.point).predict(d[track.features]), 0, None)
            preds[(cm.name, sub)] = p
            rows.append(_cons_rows(cm.name, "ensemble", sub, d[cons.TARGET], p))
    table = pd.DataFrame(rows)
    bb = table[table["model"] == best_baseline].set_index("subset")["mae"]
    table["skill_vs_best_baseline_pct"] = (1 - table["mae"] / table["subset"].map(bb)) * 100
    table["selected"] = table["model"] == cm.name

    p4 = next(r for r in summary["validation"] if r["model"] == summary["selected"])
    now = table[(table["model"] == cm.name) & (table["subset"] == "valid")]["mae"].iloc[0]
    if summary["selected"] != cm.name or not np.isclose(p4["mae"], now):
        raise RuntimeError(f"{track.name}: saved model does not reproduce Phase 4 validation MAE ({now} vs {p4['mae']})")

    y = test[cons.TARGET].to_numpy()
    raw, cal = cm.predict_interval(test, calibrated=False), cm.predict_interval(test)
    interval = {"conformal_adjust_kwh": cm.interval_adjust_kwh, "test_raw": _coverage(y, *raw), "test_calibrated": _coverage(y, *cal)}
    groups = {}
    if track.meter_col:
        for col in ("period", "household_type"):
            groups[col] = {k: _coverage(y[idx], cal[0][idx], cal[1][idx])
                           for k, idx in ((k, np.flatnonzero(test[col].to_numpy() == k)) for k in sorted(test[col].unique()))}
    interval["test_calibrated_by"] = groups

    out = {"track": track, "cm": cm, "table": table, "interval": interval, "best_baseline": best_baseline,
           "valid": valid, "test": test, "train_rows": len(train), "final": final,
           "pred": {s: preds[(cm.name, s)] for s in ("valid", "test")}}
    if track.meter_col:
        seen = set(fit_data[track.meter_col])                       # meters the saved model was fitted on
        out["breakdown"] = {"selected": cons.breakdown(test, out["pred"]["test"], track, seen),
                            "best_baseline": cons.breakdown(test, test[best_baseline].to_numpy(), track, seen)}
        e = test.assign(m=np.abs(y - out["pred"]["test"]), b=np.abs(y - test[best_baseline].to_numpy()))
        per = e.groupby([track.meter_col, "period"])[["m", "b"]].mean().reset_index()
        out["share_meters_beating_baseline"] = {"test": float((e.groupby(track.meter_col)["m"].mean()
                                                                < e.groupby(track.meter_col)["b"].mean()).mean()),
                                                **{p: float((g["m"] < g["b"]).mean()) for p, g in per.groupby("period")}}
    r = permutation_importance(valid_model, valid[track.features], valid[cons.TARGET], scoring="neg_mean_absolute_error",
                               n_repeats=5, random_state=rs)
    out["importance"] = pd.Series(r.importances_mean, index=track.features).sort_values(ascending=False)
    return out


# --------------------------------------------------------------------------- figures
def fig_solar(s: dict) -> list[str]:
    names = []
    d = s["daily"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    hi = max(d["actual"].max(), d["pred"].max()) * 1.05
    for ax, (pid, g) in zip(axes, d.groupby("plant_id")):
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, zorder=1, label="actual = expected")
        ax.scatter(g.loc[g["healthy"], "pred"], g.loc[g["healthy"], "actual"], s=22, color=PLANT_COLORS[pid],
                   edgecolor="white", linewidth=0.6, label="healthy inverter-day", zorder=3)
        ax.scatter(g.loc[~g["healthy"], "pred"], g.loc[~g["healthy"], "actual"], s=22, facecolor="none",
                   edgecolor=PLANT_COLORS[pid], linewidth=1.2, label="not healthy (below peers)", zorder=2)
        ax.set_title(f"{PLANT_NAMES[pid]}: test days")
        ax.set_xlabel("Expected output (kWh per inverter-day)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Actual output (kWh per inverter-day)")
    fig.suptitle("Healthy inverters sit on the line; points below it are lost energy, not model error",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    names.append(save_figure(fig, project_path(f"{FIG}/p01_solar_test_expected_vs_actual.png")).name)

    f, pred = s["f"], s["pred"]
    t = f[(f["split"] == "test") & f["train_eligible"]].assign(pred=pred[((f["split"] == "test") & f["train_eligible"]).to_numpy()])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for ax, (pid, g) in zip(axes, t.groupby("plant_id")):
        m = g.groupby("timestamp")[[SOLAR_TARGET, "pred"]].mean()
        ax.plot(m.index, m["pred"], color=MUTED, lw=2, label="expected (healthy inverter)")
        ax.plot(m.index, m[SOLAR_TARGET], color=PLANT_COLORS[pid], lw=1.4, label="actual (mean of its inverters)")
        ax.set_ylabel("kW per inverter")
        ax.set_title(f"{PLANT_NAMES[pid]}")
        ax.legend(loc="upper right", ncols=2)
    axes[-1].set_xlabel("Test days (15-minute readings)")
    fig.suptitle("Expected vs actual output on the 5 test days", x=0.01, ha="left", fontsize=10, color=INK_2)
    names.append(save_figure(fig, project_path(f"{FIG}/p02_solar_test_day_curves.png")).name)

    h = f[solar.eval_mask(f, True, "valid") | solar.eval_mask(f, True, "test")]
    h = h[h["daytime"]].assign(err=np.abs(h[SOLAR_TARGET] - pred[h.index]), hour=h["timestamp"].dt.hour)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    h = h[h["hour"].between(5, 19)]             # a few night readings have irradiation > 0 (sensor noise)
    for pid, g in h.groupby("plant_id"):
        e = g.groupby("hour")["err"].mean()
        ax.plot(e.index, e.values, marker="o", ms=4, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (kW per inverter)")
    ax.set_title("Error by hour of day (healthy inverter-days, validation + test, 05–19 h)")
    ax.legend()
    names.append(save_figure(fig, project_path(f"{FIG}/p03_solar_error_by_hour.png")).name)
    return names


def _bars_model_vs_baseline(ax, model_err: pd.Series, base_err: pd.Series, labels: list[str], title: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x - 0.2, model_err.values, 0.38, color=SERIES[0], label="selected model")
    ax.bar(x + 0.2, base_err.values, 0.38, color=MUTED, label="best baseline")
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel("MAE (kWh/day)")


def fig_india(c: dict) -> list[str]:
    names = []
    tr, bb = c["track"], c["best_baseline"]
    both = pd.concat([c["valid"].assign(pred=c["pred"]["valid"]), c["test"].assign(pred=c["pred"]["test"])])
    both = both.assign(m=np.abs(both[cons.TARGET] - both["pred"]), b=np.abs(both[cons.TARGET] - both[bb]))
    daily = both.groupby("date")[["m", "b"]].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily["b"], color=MUTED, lw=1.2, label=f"best baseline ({bb})")
    ax.plot(daily.index, daily["m"], color=SERIES[0], lw=1.4, label=f"selected model ({c['cm'].name})")
    for x, txt in [(c["test"]["date"].min(), "test starts"), (pd.Timestamp(load_config()["emarc"]["lockdown_start"]), "lockdown")]:
        ax.axvline(x, color=INK_2, lw=0.8, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.97, f" {txt}", color=INK_2, fontsize=8, va="top")
    ax.set_ylabel("MAE across meters (kWh/day)")
    ax.set_title("India: daily error, each meter scored on its own (validation, then test)")
    ax.legend(loc="upper left")
    names.append(save_figure(fig, project_path(f"{FIG}/p04_india_daily_error.png")).name)

    t = c["test"].assign(pred=c["pred"]["test"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    lim = float(np.quantile(t[cons.TARGET], 0.995))
    for ax, per in zip(axes, ["normal", "lockdown"]):
        g = t[t["period"] == per]
        hb = ax.hexbin(g["pred"], g[cons.TARGET], gridsize=40, extent=(0, lim, 0, lim), mincnt=1, cmap="Blues", bins="log")
        ax.plot([0, lim], [0, lim], color=INK_2, lw=0.8)
        ax.set_title(f"{per.capitalize()} days")
        ax.text(0.02, 0.97, f"{len(g):,} meter-days", transform=ax.transAxes, va="top", fontsize=9, color=INK_2)
        ax.set_xlabel("Predicted (kWh/day)")
    axes[0].set_ylabel("Actual (kWh/day)")
    fig.colorbar(hb, ax=axes, label="meter-days (log scale)")
    fig.suptitle("India test (2020): predicted vs actual", x=0.01, ha="left", fontsize=10, color=INK_2)
    names.append(save_figure(fig, project_path(f"{FIG}/p05_india_test_predicted_vs_actual.png")).name)

    e = t.assign(m=np.abs(t[cons.TARGET] - t["pred"]), b=np.abs(t[cons.TARGET] - t[bb]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    mo = e.groupby(e["date"].dt.month)[["m", "b"]].mean()
    _bars_model_vs_baseline(axes[0], mo["m"], mo["b"], [pd.Timestamp(2020, k, 1).strftime("%b") for k in mo.index],
                            "By month (lockdown from 25 Mar)")
    wd = e.groupby(e["date"].dt.dayofweek)[["m", "b"]].mean()
    _bars_model_vs_baseline(axes[1], wd["m"], wd["b"], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "By weekday")
    axes[1].legend()
    fig.suptitle("India test error (2020)", x=0.01, ha="left", fontsize=10, color=INK_2)
    names.append(save_figure(fig, project_path(f"{FIG}/p06_india_error_by_month_weekday.png")).name)
    return names


def fig_france(c: dict) -> list[str]:
    names = []
    lo, hi = c["cm"].predict_interval(c["test"])
    t = c["test"].assign(pred=c["pred"]["test"], lo=lo, hi=hi)
    g = t.set_index("date")[[cons.TARGET, "pred", "lo", "hi"]].asfreq("D").reset_index()   # missing days break the lines
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(g["date"], g["lo"], g["hi"], color=SERIES[0], alpha=0.15, lw=0, label="80 % interval (calibrated)")
    ax.plot(g["date"], g[cons.TARGET], color=INK, lw=1.1, label="actual")
    ax.plot(g["date"], g["pred"], color=SERIES[0], lw=1.4, label=f"predicted ({c['cm'].name})")
    ax.set_ylabel("kWh/day")
    ax.set_title("France: next-day consumption on the test year (2010; gaps = days without data)")
    ax.legend(loc="upper right", ncols=3)
    names.append(save_figure(fig, project_path(f"{FIG}/p07_france_test_timeseries.png")).name)

    both = pd.concat([c["valid"].assign(pred=c["pred"]["valid"]), t])
    both["res"] = both[cons.TARGET] - both["pred"]
    both = both.set_index("date")[["res"]].asfreq("D").reset_index()
    e = t.assign(m=np.abs(t[cons.TARGET] - t["pred"]), b=np.abs(t[cons.TARGET] - t[c["best_baseline"]]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.6, 1, 1]}, layout="constrained")
    axes[0].axhline(0, color=INK_2, lw=0.8)
    axes[0].plot(both["date"], both["res"], color=SERIES[0], lw=0.8)
    axes[0].axvline(t["date"].min(), color=INK_2, lw=0.8, ls="--")
    axes[0].set_title("Residuals (actual − predicted)")
    axes[0].text(t["date"].min(), axes[0].get_ylim()[1], " test →", color=INK_2, fontsize=8, va="top")
    axes[0].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    axes[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[0].set_ylabel("kWh/day")
    mo = e.groupby(e["date"].dt.month)[["m", "b"]].mean()
    _bars_model_vs_baseline(axes[1], mo["m"], mo["b"], [pd.Timestamp(2010, k, 1).strftime("%b")[:1] for k in mo.index],
                            "Test MAE by month")
    wd = e.groupby(e["date"].dt.dayofweek)[["m", "b"]].mean()
    _bars_model_vs_baseline(axes[2], wd["m"], wd["b"], ["M", "T", "W", "T", "F", "S", "S"], "Test MAE by weekday")
    axes[2].legend()
    names.append(save_figure(fig, project_path(f"{FIG}/p08_france_residuals_and_error.png")).name)
    return names


def fig_importance(s: dict, india: dict, france: dict) -> str:
    panels = [(f"Solar, {k}", v) for k, v in s["importance"].items()] + [("India consumption", india["importance"]),
                                                                         ("France consumption", france["importance"])]
    units = ["kW", "kW", "kWh/day", "kWh/day"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for ax, (title, imp), unit in zip(axes.ravel(), panels, units):
        top = imp.head(8)[::-1]
        ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
        ax.set_title(title)
        ax.set_xlabel(f"MAE increase when shuffled ({unit})")
    fig.suptitle("Permutation importance on validation (top 8; correlated features share credit)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    return save_figure(fig, project_path(f"{FIG}/p09_permutation_importance.png")).name


# --------------------------------------------------------------------------- report and cards
def _md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and pd.notna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


F_CONS = {"mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}", "mape_pct": "{:.1f}", "skill_vs_best_baseline_pct": "{:+.1f} %"}
F_SOLAR = {"mae": "{:.1f}", "rmse": "{:.1f}", "r2": "{:.3f}", "mape_pct": "{:.2f}", "nmae": "{:.3f}"}


def _solar_pick(t: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        m &= t[k].to_numpy() == v
    return t[m].iloc[0]


def _phase5_line(c: dict, sel: pd.DataFrame) -> list[str]:
    """First (Phase 5) test result of this track, if the model has changed since and the archive exists."""
    path = project_path(f"reports/archive_phase5/metrics/evaluation_consumption_{c['track'].name}.csv")
    if not path.exists():
        return []
    old = pd.read_csv(path)
    old = old[old["selected"]].set_index("subset")
    if old["model"].iloc[0] == c["cm"].name:
        return []
    parts = [f"{s.replace('_', ' ')} {old.loc[s, 'mae']:.3f} → **{sel.loc[s, 'mae']:.3f}**" for s in old.index if s in sel.index]
    return [f"**Change since the Phase 5 test run** (`{old['model'].iloc[0]}`, trained on train only): MAE " + ", ".join(parts) + ".", ""]


def _cons_section(c: dict, title: str) -> list[str]:
    t, iv = c["table"], c["interval"]
    sel = t[t["selected"]].set_index("subset")
    piv = t.pivot_table(index=["model", "kind"], columns="subset", values="mae").reset_index()
    order = ["valid", "test"] + [s for s in ("test_normal", "test_lockdown") if s in piv.columns]
    piv = piv[["model", "kind", *order]].sort_values("valid")
    lines = [f"## {title}", "",
             (f"Final model (Phase 8): equal-weight ensemble of {', '.join(f'`{m}`' for m in c['final']['members'])}, "
              f"refitted on {c['final']['fit_on'].replace('_', ' + ')}. " if c["final"] else f"Selected in Phase 4: `{c['cm'].name}`. ") +
             f"Test MAE **{sel.loc['test', 'mae']:.3f} kWh/day** (validation "
             f"{sel.loc['valid', 'mae']:.3f}), R² {sel.loc['test', 'r2']:.3f}, MAPE {sel.loc['test', 'mape_pct']:.1f} %; "
             f"{sel.loc['test', 'skill_vs_best_baseline_pct']:+.1f} % vs the best validation baseline (`{c['best_baseline']}`).", "",
             *_phase5_line(c, sel), "MAE (kWh/day) of every baseline and model:", "",
             _md(piv, {k: "{:.3f}" for k in order}), ""]
    if "test_lockdown" in sel.index:
        lines += [f"Lockdown (from {load_config()['emarc']['lockdown_start']}): selected-model MAE "
                  f"{sel.loc['test_normal', 'mae']:.3f} on normal days vs {sel.loc['test_lockdown', 'mae']:.3f} in lockdown "
                  f"({sel.loc['test_normal', 'skill_vs_best_baseline_pct']:+.1f} % / {sel.loc['test_lockdown', 'skill_vs_best_baseline_pct']:+.1f} % vs baseline). "
                  "Lockdown is a change in behaviour the training data never contained.", ""]
        b, bb = c["breakdown"]["selected"], c["breakdown"]["best_baseline"]
        sh = c["share_meters_beating_baseline"]
        lines += [f"Per meter (test): median MAE {b['per_meter_mae_median']:.3f} (baseline {bb['per_meter_mae_median']:.3f}) over "
                  f"{b['meters']} meters; the model beats the baseline on {sh['test']:.0%} of meters "
                  f"({sh['normal']:.0%} on normal days, {sh['lockdown']:.0%} in lockdown).", ""]
        for col, label in [("meter_group", "Meters seen in training vs new"), ("household_type", "By household type"),
                           ("region", "By region")]:
            rows = pd.DataFrame([{"group": k, "rows": v["rows"], "meters": v["meters"], "mean_kwh": v["mean_kwh"],
                                  "mae": v["mae"], "baseline_mae": bb[f"by_{col}"][k]["mae"]} for k, v in b[f"by_{col}"].items()])
            lines += [f"**{label}** (test)", "", _md(rows, {"mean_kwh": "{:.2f}", "mae": "{:.3f}", "baseline_mae": "{:.3f}"}), ""]
    r, k = iv["test_raw"], iv["test_calibrated"]
    lines += ["**80 % prediction interval (test):** raw quantile models cover "
              f"{r['coverage']:.1%} (width {r['mean_width_kwh']:.2f} kWh); after the conformal adjustment from validation "
              f"({iv['conformal_adjust_kwh']:+.2f} kWh each side) {k['coverage']:.1%} (width {k['mean_width_kwh']:.2f} kWh).", ""]
    for col, g in iv["test_calibrated_by"].items():
        lines += [f"Calibrated coverage by {col}: " + ", ".join(f"{a} {v['coverage']:.1%}" for a, v in g.items()) + ".", ""]
    return lines


def write_report(s: dict, india: dict, france: dict, figs: list[str], log_info: dict) -> None:
    t = s["table"]
    sel = t[t["selected"]]
    keep = ["model", "split", "eval_set", "plant", "n", "mae", "rmse", "r2", "mape_pct"]
    solar_daily = t[(t["level"] == "daily") & (t["eval_set"] == "healthy") & (t["plant"] == "both")][keep].sort_values(["split", "mae"])
    solar_sel = sel[sel["level"] == "daily"][keep].sort_values(["split", "eval_set", "plant"])
    v = _solar_pick(sel, split="valid", eval_set="healthy", plant="both", level="daily")
    te = _solar_pick(sel, split="test", eval_set="healthy", plant="both", level="daily")
    ti = _solar_pick(sel, split="test", eval_set="healthy", plant="both", level="interval")
    archive = project_path("reports/archive_phase5")
    lines = ["# Test-set evaluation", "",
             *(["> **Second test run (Phase 8), approved by the project owner.** The India consumption model was changed on "
                "validation evidence only (ensemble + refit on train + validation; `reports/improvement_experiments.md`) and "
                "the test split was scored again. The first run (Phase 5) is kept unchanged in `reports/archive_phase5/`; "
                "both are reported. The test split had been seen once before this decision, so the second India number "
                "may be slightly optimistic. Solar and France models are unchanged (same file hashes).", ""]
               if archive.exists() else []),
             "*Generated by `python -m solvia.pipeline --steps evaluate`. The models were chosen on validation (Phase 4; India's "
             "final model in Phase 8); "
             f"the test split was scored with them on {log_info['first_run_utc']} (UTC). Model file hashes are in "
             f"`reports/metrics/{LOG_FILE}`; a run with different model files is refused.*", "",
             "## B. Expected solar output (Kaggle plants, test days " + f"{s['test_days'][0]} … {s['test_days'][-1]})", "",
             f"Selected: `{s['selected'].name}`, {s['selected'].scope}, trained on `{s['selected'].training_set}`. "
             f"Healthy inverter-days, both plants: test daily MAE **{te['mae']:.1f} kWh** (MAPE {te['mape_pct']:.2f} %, R² {te['r2']:.3f}) "
             f"vs validation {v['mae']:.1f} kWh; 15-minute daytime MAE {ti['mae']:.1f} kW (R² {ti['r2']:.3f}).", "",
             "Daily kWh per inverter-day, healthy inverter-days, both plants:", "", _md(solar_daily, F_SOLAR), "",
             "Selected model by plant and evaluation set (on all operating inverter-days the gap is mostly lost energy that "
             "monitoring must catch, not model error):", "", _md(solar_sel, F_SOLAR), "",
             "Only 5 test days (the plants' data covers 34 days), so the test numbers are a check, not a precise estimate.", ""]
    lines += _cons_section(india, "A-India. Next-day consumption, Prayas eMARC (test 2020-01-01 … 06-30)")
    lines += _cons_section(france, "A-France. Next-day consumption, UCI household (test 2010-01-01 … 11-25)")
    lines += ["## Monitoring and maintenance (C, D)", "",
              "These are rules on top of the expected-output model, not trained predictors, so they have no test split of their own. "
              "They were checked in Phase 4 with simulated faults (0 % false alarms; 99.8 % of 20 % losses caught in one day) "
              "and against the plants' own history; see `reports/monitoring_report.md`.", "",
              "## Figures", "", *[f"- `{FIG}/{n}`" for n in figs], "",
              "Model cards: `reports/model_card_solar_expected_output.md`, `reports/model_card_consumption_india.md`, "
              "`reports/model_card_consumption_france.md`.", ""]
    project_path("reports/evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _card(name: str, meta: dict, body: list[str]) -> None:
    hp = meta["hyperparameters"]
    if meta.get("scope") == "per_plant":
        hp = {k: v for k, v in hp.items() if not k.endswith("/pooled")}
    head = [f"# Model card: {name}", "",
            f"*Generated by `python -m solvia.pipeline --steps evaluate` from `models/{name}.json` and the Phase 5 test run.*", "",
            f"- **File:** `models/{name}.joblib` (+ `.json` metadata); scikit-learn {meta['scikit_learn']}, Python {meta['python']}",
            f"- **Purpose:** {meta['objective']}",
            f"- **Model:** `{meta['model']}`" + (f", one per plant, trained on `{meta['training_set']}`" if "scope" in meta else "")
            + f"; hyperparameters {hp}",
            f"- **Selected by:** {meta['selection']}" + ("; test split scored a second time for this model (Phase 8, approved "
            "by the owner; first result in `reports/archive_phase5/`)" if meta.get("fit_on") == "train_valid"
            else " (Phase 4); test split used once, afterwards (Phase 5)"), ""]
    project_path(f"reports/model_card_{name}.md").write_text("\n".join(head + body), encoding="utf-8")


def write_cards(s: dict, india: dict, france: dict) -> None:
    models = get_path("models")
    meta = {n: json.loads((models / f"{n}.json").read_text(encoding="utf-8")) for n in MODEL_FILES}
    sel = s["table"][s["table"]["selected"]]
    v = _solar_pick(sel, split="valid", eval_set="healthy", plant="both", level="daily")
    te = _solar_pick(sel, split="test", eval_set="healthy", plant="both", level="daily")
    _card("solar_expected_output", meta["solar_expected_output"], [
        "## Data", "",
        "Kaggle 'Solar Power Generation Data': two utility plants in India (estimated by sunlight matching to be in the "
        "southern Deccan, near Anantapur), 22 inverters each, 15-minute readings, 15 May – 17 Jun 2020. "
        f"Split by whole days: train {meta['solar_expected_output']['train_days']}, validation 5 days, test 5 days.", "",
        "## Inputs and output", "",
        f"Inputs: {', '.join(meta['solar_expected_output']['features'])} (weather and time of day only; output-derived "
        "columns are blocked in code). Output: AC power (kW) that a *healthy* inverter of that plant would produce; 0 at night.", "",
        "## Performance (healthy inverter-days, both plants, daily kWh per inverter-day)", "",
        f"| split | MAE (kWh) | MAPE | R² |\n|---|---|---|---|\n| validation | {v['mae']:.1f} | {v['mape_pct']:.2f} % | {v['r2']:.3f} |\n"
        f"| test | {te['mae']:.1f} | {te['mape_pct']:.2f} % | {te['r2']:.3f} |", "",
        "## Intended use", "",
        "The reference for performance monitoring at these two plants: actual ÷ expected = performance ratio.", "",
        "## Limits", "",
        "- 34 days of one season (May–June, pre-monsoon to early monsoon); other seasons are untested.",
        "- One model per plant; on the other plant it is worse (Phase 4 cross-plant check), so a new plant needs its own healthy days.",
        "- 'Healthy' is defined relative to the plant's best inverters on the same day; a plant-wide slow loss "
        "(e.g. uniform soiling) would be learned as normal.",
        "- No module-level or string-level data; the model cannot locate a fault inside an inverter.", ""])
    for c, key in ((india, "consumption_india"), (france, "consumption_france")):
        m, t = meta[key], c["table"]
        selr = t[t["selected"]].set_index("subset")
        bl = t[t["model"] == c["best_baseline"]].set_index("subset")
        perf = [f"| {sub} | {selr.loc[sub, 'mae']:.3f} | {bl.loc[sub, 'mae']:.3f} | {selr.loc[sub, 'r2']:.3f} | "
                f"{selr.loc[sub, 'mape_pct']:.1f} % |" for sub in selr.index]
        k = c["interval"]["test_calibrated"]
        if key == "consumption_india":
            data = ["Prayas (Energy Group) eMARC smart-meter data: 15-minute readings from Indian homes in Maharashtra and "
                    "Uttar Pradesh, 2018–2020, used for non-commercial academic purposes with credit to Prayas (Energy Group). "
                    "One row per meter-day; features come only from that meter's own past, its household type and region, the "
                    "calendar and NASA POWER weather. Homes are never added together."]
            tun = json.loads((get_path("metrics") / "consumption_india_summary.json").read_text(encoding="utf-8"))["tuning"]
            f0 = {k: v["cv_mae_by_fold_kwh"][0] for k, v in tun.items()}
            limits = [f"- Ridge (an ensemble member) on the shortest-history CV fold: MAE {f0['ridge']:.2f} kWh vs "
                      f"{min(v for k, v in f0.items() if k != 'ridge'):.2f} for the best other model; a linear model extrapolates "
                      "when the season changes, so check a new deployment for a few weeks before relying on it.",
                      *(["- The saved model is refitted on train + validation (Phase 8); its validation row comes from the "
                         "members fitted on train only. The test split was scored a second time for this model, with the "
                         "owner's approval; the first (Phase 5) result is kept in `reports/archive_phase5/`."]
                        if c["final"] and c["final"]["fit_on"] == "train_valid" else []),
                      "- The 2020 lockdown changed behaviour; the model had never seen it (see test_lockdown row).",
                      f"- {c['test'][track_col].nunique() if (track_col := c['track'].meter_col) else 0} meters in the test split, "
                      "from 5 regions of two states; not representative of all Indian households.",
                      "- Uses yesterday's observed weather, not a forecast."]
        else:
            data = ["UCI 'Individual household electric power consumption': one home in Sceaux (France), 2006–2010, "
                    "aggregated to daily kWh; NASA POWER weather for Sceaux; French public and school holidays."]
            limits = ["- One household: the numbers describe this home, not homes in general.",
                      "- Heating-driven winter peaks dominate the error.",
                      "- Kept as a secondary track; the project's priority is Indian homes."]
        _card(key, m, ["## Data", "", *data, "", f"Training rows: {m['train_rows']:,} ({m['train_dates']}).", "",
                       "## Inputs and output", "", f"Inputs: {', '.join(m['features'])}.", "",
                       "Output: kWh used on the next day (clipped at 0) and an 80 % interval.", "",
                       "## Performance (MAE in kWh/day)", "",
                       "| subset | selected MAE | best baseline MAE | R² | MAPE |", "|---|---|---|---|---|", *perf, "",
                       f"80 % interval on test: {k['coverage']:.1%} coverage, mean width {k['mean_width_kwh']:.2f} kWh.", "",
                       "## Intended use", "", "Day-ahead planning of a home's electricity use and its solar balance in the dashboard.", "",
                       "## Limits", "", *limits, ""])


def _update_metadata(s: dict, india: dict, france: dict, when: str) -> None:
    models = get_path("models")
    sel = s["table"][s["table"]["selected"]]
    solar_test = sel[(sel["split"] == "test")].drop(columns=["selected", "split"]).to_dict(orient="records")
    updates = {"solar_expected_output": {"test": solar_test}}
    for c, key in ((india, "consumption_india"), (france, "consumption_france")):
        t = c["table"]
        updates[key] = {"test": t[t["selected"] & t["subset"].str.startswith("test")].drop(columns=["selected"]).to_dict(orient="records"),
                        "interval_test": {k: v for k, v in c["interval"].items() if k != "test_calibrated_by"}}
    for name, upd in updates.items():
        path = models / f"{name}.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["test_metrics"] = {**upd, "scored_utc": when}
        path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- entry point
def run_evaluation() -> dict:
    cfg = load_config()
    hashes = check_test_guard()
    apply_style()
    s = evaluate_solar(cfg)
    india = evaluate_consumption("india", cfg)
    france = evaluate_consumption("france", cfg)
    _write_test_log(hashes)
    log_info = json.loads((get_path("metrics") / LOG_FILE).read_text(encoding="utf-8"))

    metrics = get_path("metrics")
    s["table"].to_csv(metrics / "evaluation_solar.csv", index=False)
    for c in (india, france):
        c["table"].to_csv(metrics / f"evaluation_consumption_{c['track'].name}.csv", index=False)
    summary = {"solar_selected": s["selected"].name,
               **{f"{c['track'].name}_interval": c["interval"] for c in (india, france)},
               "india_breakdown_test": india["breakdown"], "india_share_meters_beating_baseline": india["share_meters_beating_baseline"],
               "importance_top": {**{f"solar_{k}": v.head(5).round(3).to_dict() for k, v in s["importance"].items()},
                                  "india": india["importance"].head(5).round(3).to_dict(),
                                  "france": france["importance"].head(5).round(3).to_dict()}}
    (metrics / "evaluation_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    figs = fig_solar(s) + fig_india(india) + fig_france(france) + [fig_importance(s, india, france)]
    _update_metadata(s, india, france, log_info["last_run_utc"])
    write_report(s, india, france, figs, log_info)
    write_cards(s, india, france)
    return summary
