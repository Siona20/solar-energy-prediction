"""Phase 4C: performance monitoring of every inverter-day against out-of-sample expected output.

Expected output comes from the Phase 4B configuration (per-plant HistGradientBoosting, healthy reference,
same hyperparameters) refitted with an expanding window: the expected output of day d uses a model trained
only on days before d (leakage check 10). Days before `min_train_days` are not monitored.

Signals per inverter-day:
- performance ratio PR = actual kWh / expected kWh (low if < monitoring.ratio_threshold)
- robust peer z-score of PR among the plant's inverters that day (low if < monitoring.peer_z_threshold
  and PR < monitoring.peer_max_pr, so tiny differences on a very uniform plant are not flagged)
- alert when PR stays low for >= monitoring.min_consecutive_days consecutive days
Comparison: Isolation Forest on the same signals. Check: detection rate on injected (simulated) derates.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

from solvia.config import get_path, load_config, project_path
from solvia.features.solar import FEATURES, TARGET
from solvia.models.solar import healthy_inverter_days, model_rows
from solvia.viz import PLANT_NAMES

log = logging.getLogger(__name__)

KEYS = ["plant_id", "inverter_id", "date"]
STATUS_ORDER = ["ok", "watch", "alert"]
DERATES = (0.95, 0.90, 0.85, 0.80, 0.70, 0.50)


def expanding_expected(f: pd.DataFrame, template: dict, min_train_days: int) -> pd.Series:
    """Out-of-sample expected kW for every row of days after the warm-up (NaN during warm-up).
    `template[plant_id]` is an unfitted pipeline with the selected hyperparameters."""
    expected = pd.Series(np.nan, index=f.index)
    days = np.sort(f["date"].unique())
    for i, day in enumerate(days):
        if i < min_train_days:
            continue
        past = f["date"] < day
        for pid, pipe in template.items():
            train = f[past & (f["plant_id"] == pid) & f["daytime"] & f["train_eligible"] & f["healthy"]]
            model = clone(pipe).fit(train[FEATURES], train[TARGET])
            rows = (f["date"] == day) & (f["plant_id"] == pid)
            pred = np.zeros(int(rows.sum()))
            day_rows = (f.loc[rows, "irradiation_kw_m2"] > 0).to_numpy()
            if day_rows.any():
                pred[day_rows] = model.predict(f.loc[rows, FEATURES][day_rows])
            expected.loc[rows] = np.clip(pred, 0, None)
        log.info("expected output for %s (trained on %d earlier days)", pd.Timestamp(day).date(), i)
    return expected


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / mad if mad > 0 else pd.Series(0.0, index=s.index)


def consecutive_low(low: pd.Series, dates: pd.Series) -> pd.Series:
    """Length of the current run of low days (calendar-consecutive) for one inverter, in date order."""
    run, prev_day, out = 0, None, []
    for is_low, d in zip(low, dates):
        consecutive = prev_day is not None and (d - prev_day).days == 1
        run = (run + 1 if consecutive else 1) if is_low else 0
        out.append(run)
        prev_day = d
    return pd.Series(out, index=low.index)


def inverter_day_table(f: pd.DataFrame, expected_kw: pd.Series, mc: dict, interval_hours: float) -> pd.DataFrame:
    g = f.assign(expected_kw=expected_kw, off=f["flag_off_in_daylight"],
                 expected_on_kw=expected_kw.where(~f["flag_off_in_daylight"], 0.0)).groupby(KEYS)
    d = pd.DataFrame({"actual_kwh": g[TARGET].sum() * interval_hours,
                      "expected_kwh": g["expected_kw"].sum(min_count=1) * interval_hours,
                      "expected_running_kwh": g["expected_on_kw"].sum(min_count=1) * interval_hours,
                      "off_intervals": g["off"].sum(), "healthy_reference": g["healthy"].first(),
                      "irradiation_kwh_m2": g["irradiation_kw_m2"].sum() * interval_hours}).reset_index()
    d = d[d["expected_kwh"].notna() & (d["expected_kwh"] > 0)].copy()
    d["pr"] = d["actual_kwh"] / d["expected_kwh"]
    # PR while the inverter was running: shutdowns lower `pr` but not `running_pr`; dust lowers both
    d["running_pr"] = d["actual_kwh"] / d["expected_running_kwh"].replace(0, np.nan)
    d["peer_z"] = d.groupby(["plant_id", "date"])["pr"].transform(robust_z)
    d["low"] = d["pr"] < mc["ratio_threshold"]
    d["peer_low"] = (d["peer_z"] < mc["peer_z_threshold"]) & (d["pr"] < mc["peer_max_pr"])
    d = d.sort_values(KEYS).reset_index(drop=True)
    d["low_run_days"] = d.groupby(["plant_id", "inverter_id"], group_keys=False).apply(
        lambda x: consecutive_low(x["low"], x["date"]), include_groups=False)
    d["status"] = np.select([d["low_run_days"] >= mc["min_consecutive_days"], d["low"] | d["peer_low"]],
                            ["alert", "watch"], "ok")
    return d


def plant_day_table(d: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["plant_id", "date"])
    p = pd.DataFrame({"plant_pr": g["actual_kwh"].sum() / g["expected_kwh"].sum(), "median_pr": g["pr"].median(),
                      "plant_running_pr": g["actual_kwh"].sum() / g["expected_running_kwh"].sum(),
                      "off_hours": g["off_intervals"].sum() / 4,
                      "irradiation_kwh_m2": g["irradiation_kwh_m2"].median(),
                      "share_low": g["low"].mean(), "inverters": g.size(),
                      "alerts": g["status"].apply(lambda s: int((s == "alert").sum()))}).reset_index()
    r = rain[["plant_id", "date", "precip_mm_day"]].rename(columns={"precip_mm_day": "est_rain_mm"})
    return p.merge(r, on=["plant_id", "date"], how="left")


def isolation_forest_comparison(d: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, dict]:
    """Unsupervised flags on the same inverter-day signals, with the same flag rate as the PR rule."""
    X = pd.DataFrame({"pr": d["pr"], "peer_z": d["peer_z"].clip(-20, 20),
                      "off_share": d["off_intervals"] / 40.0})
    rate = float(d["low"].mean())
    iso = IsolationForest(n_estimators=300, contamination=min(max(rate, 0.001), 0.5), random_state=random_state).fit(X)
    d = d.assign(iforest_flag=iso.predict(X) == -1)
    rule, iso_f = d["low"], d["iforest_flag"]
    inter, union = int((rule & iso_f).sum()), int((rule | iso_f).sum())
    stats = {"flag_rate": round(rate, 4), "rule_flags": int(rule.sum()), "iforest_flags": int(iso_f.sum()),
             "both": inter, "rule_only": int((rule & ~iso_f).sum()), "iforest_only": int((iso_f & ~rule).sum()),
             "jaccard": round(inter / union, 3) if union else None,
             "iforest_only_median_pr": round(float(d.loc[iso_f & ~rule, "pr"].median()), 3) if (iso_f & ~rule).any() else None,
             "rule_only_median_pr": round(float(d.loc[rule & ~iso_f, "pr"].median()), 3) if (rule & ~iso_f).any() else None}
    return d, stats


def injected_fault_check(d: pd.DataFrame, threshold: float, plant_id: int = 4135001) -> pd.DataFrame:
    """Simulated derates on healthy-reference inverter-days of one plant: how often does the PR rule fire?
    derate 1.0 = no fault, so its 'detection rate' is the false-alarm rate."""
    h = d[(d["plant_id"] == plant_id) & d["healthy_reference"]]
    rows = []
    for k in (1.0, *DERATES):
        pr = h["actual_kwh"] * k / h["expected_kwh"]
        rows.append({"plant": PLANT_NAMES[plant_id], "output_kept": k, "loss_pct": round((1 - k) * 100),
                     "inverter_days": int(len(h)), "flagged_share": float((pr < threshold).mean())})
    return pd.DataFrame(rows)


def trip_profile(f: pd.DataFrame, d: pd.DataFrame, often_low_share: float = 0.4) -> pd.DataFrame:
    """Share of daylight intervals (irradiation > 0.1) with the inverter off, by irradiation band, for
    'often-low' inverters (low on >= 40 % of monitored days) and the rest. Shows whether losses are
    shutdowns in bright conditions (trips) or a steady shortfall (dust, strings)."""
    low_share = d.groupby(["plant_id", "inverter_id"])["low"].mean().rename("low_share")
    x = f[(f["irradiation_kw_m2"] > 0.1) & (f["date"] >= d["date"].min())].join(low_share, on=["plant_id", "inverter_id"])
    x["group"] = np.where(x["low_share"] >= often_low_share, "often-low inverters", "other inverters")
    x["irradiation_band_kw_m2"] = pd.cut(x["irradiation_kw_m2"], [0.1, 0.4, 0.6, 0.8, 1.0, 1.4]).astype(str)
    out = (x.groupby(["plant_id", "group", "irradiation_band_kw_m2"])
           .agg(intervals=("flag_off_in_daylight", "size"), off_share=("flag_off_in_daylight", "mean"),
                inverters=("inverter_id", "nunique")).reset_index())
    out["plant"] = out.pop("plant_id").map(PLANT_NAMES)
    return out


def rain_effect(p: pd.DataFrame, min_rain_mm: float) -> pd.DataFrame:
    """Plant PR on days after estimated rain >= min_rain_mm vs other days, and PR change day over day.
    Rain is an ESTIMATE (NASA POWER at the best-matching city)."""
    p = p.sort_values(["plant_id", "date"]).copy()
    p["rain_prev_day_mm"] = p.groupby("plant_id")["est_rain_mm"].shift(1)
    p["pr_change"] = p.groupby("plant_id")["plant_pr"].diff()
    p["after_rain"] = p["rain_prev_day_mm"] >= min_rain_mm
    rows = []
    for pid, g in p.dropna(subset=["rain_prev_day_mm"]).groupby("plant_id"):
        for flag, gg in g.groupby("after_rain"):
            rows.append({"plant": PLANT_NAMES[pid], "group": "after rain" if flag else "other days", "days": int(len(gg)),
                         "mean_plant_pr": float(gg["plant_pr"].mean()), "mean_pr_change": float(gg["pr_change"].mean()),
                         "mean_running_pr": float(gg["plant_running_pr"].mean()),
                         "mean_irradiation_kwh_m2": float(gg["irradiation_kwh_m2"].mean()),
                         "corr_pr_vs_irradiation_all_days": float(g["plant_pr"].corr(g["irradiation_kwh_m2"]))})
    return pd.DataFrame(rows)


def run_monitoring() -> dict:
    cfg = load_config()
    sc, mc, rs = cfg["solar"], cfg["monitoring"], cfg["random_state"]
    selected = joblib.load(get_path("models") / "solar_expected_output.joblib")
    if selected.scope != "per_plant":
        raise ValueError("monitoring expects the per-plant configuration selected in Phase 4B")
    template = {pid: clone(pipe) for pid, pipe in selected.pipelines.items()}

    features = pd.read_parquet(get_path("processed") / "solar_features_15min.parquet")
    daily_inv = pd.read_parquet(get_path("interim") / "solar_daily_inverter.parquet")
    f = model_rows(features, healthy_inverter_days(daily_inv, sc["healthy_min_ratio"]))
    expected = expanding_expected(f, template, mc["min_train_days"])
    d = inverter_day_table(f, expected, mc, sc["interval_hours"])
    rain = pd.read_parquet(get_path("interim") / "plant_estimated_weather.parquet")
    p = plant_day_table(d, rain)
    d, iso_stats = isolation_forest_comparison(d, rs)
    inj = injected_fault_check(d, mc["ratio_threshold"])
    rain_tab = rain_effect(p, cfg["maintenance"]["recent_rain_mm"])
    trips = trip_profile(f, d)

    interim, metrics = get_path("interim"), get_path("metrics")
    d.to_parquet(interim / "monitoring_inverter_day.parquet")
    p.to_parquet(interim / "monitoring_plant_day.parquet")
    inj.to_csv(metrics / "monitoring_injected_faults.csv", index=False)
    rain_tab.to_csv(metrics / "monitoring_rain_effect.csv", index=False)
    trips.to_csv(metrics / "monitoring_trips_by_irradiance.csv", index=False)

    status = d.groupby(["plant_id", "status"]).size().unstack(fill_value=0).reindex(columns=STATUS_ORDER, fill_value=0)
    h = d[d["healthy_reference"]]
    summary = {
        "monitored_days": f"{d['date'].min().date()} to {d['date'].max().date()} ({d['date'].nunique()} days)",
        "inverter_days": int(len(d)),
        "status_counts": {PLANT_NAMES[k]: v for k, v in status.to_dict(orient="index").items()},
        "healthy_reference_pr": {PLANT_NAMES[k]: {"median": round(float(g["pr"].median()), 3),
                                                  "p05": round(float(g["pr"].quantile(0.05)), 3),
                                                  "p95": round(float(g["pr"].quantile(0.95)), 3)}
                                 for k, g in h.groupby("plant_id")},
        "shutdown_share_of_lost_energy": {
            PLANT_NAMES[k]: round(float((g["expected_kwh"] - g["expected_running_kwh"]).sum() / (g["expected_kwh"] - g["actual_kwh"]).sum()), 3)
            for k, g in d.groupby("plant_id") if (g["expected_kwh"] - g["actual_kwh"]).sum() > 0},
        "mean_plant_pr": {PLANT_NAMES[k]: round(float(g["plant_pr"].mean()), 3) for k, g in p.groupby("plant_id")},
        "mean_plant_running_pr": {PLANT_NAMES[k]: round(float(g["plant_running_pr"].mean()), 3) for k, g in p.groupby("plant_id")},
        "corr_running_pr_vs_est_rain": {PLANT_NAMES[k]: round(float(g["plant_running_pr"].corr(g["est_rain_mm"])), 3) for k, g in p.groupby("plant_id")},
        "out_of_sample_daily_mae_kwh_healthy": {PLANT_NAMES[k]: round(float((g["actual_kwh"] - g["expected_kwh"]).abs().mean()), 1)
                                               for k, g in h.groupby("plant_id")},
        "inverters_with_alert": {PLANT_NAMES[k]: int(g.loc[g["status"] == "alert", "inverter_id"].nunique())
                                 for k, g in d.groupby("plant_id")},
        "plant_days_share_low_ge_0.6": {PLANT_NAMES[k]: int((g["share_low"] >= cfg["maintenance"]["gradual_decline_share"]).sum())
                                        for k, g in p.groupby("plant_id")},
        "isolation_forest": iso_stats,
        "injected_faults": inj.to_dict(orient="records"),
        "rain_effect_estimate": rain_tab.to_dict(orient="records"),
        "trips_by_irradiance": trips.to_dict(orient="records"),
    }
    (metrics / "monitoring_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log.info("monitoring: %s", {k: summary[k] for k in ("monitored_days", "status_counts", "inverters_with_alert")})
    return summary
