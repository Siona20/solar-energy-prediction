"""Phase 4D: transparent maintenance recommendations from the monitoring tables.

Every recommendation states the action, the reason in plain words and the evidence (numbers) behind it.
Rules are checked in this order for one plant on one day (`as_of`):

R2 inverter_underperforming an inverter's PR stayed below the threshold for >= N days while most peers were fine;
                           shutdowns -> inverter/trip check, steady shortfall -> string/module check
R1 inverter_offline        an inverter was off for >= `offline_hours` while the sun was up (if not already R2)
R3 plant_wide_loss         >= `gradual_decline_share` of inverters were low for >= N consecutive days;
                           rain decides the likely cause (rain should have washed off dust)
R4 gradual_decline         plant *running* PR (inverters on) fell steadily over `decline_window_days`, no washing rain
R5 sensor_check            the plant produced far more than expected (irradiation sensor may read low or be dirty)
R6 seasonal_cleaning       informational: the estimated location is in a dry month (no natural washing)
R7 not_reporting           an inverter that reported before sends no data today (logger / communication)
R0 no_action               nothing above fired
Rain is an ESTIMATE (NASA POWER at the best-matching city); the plant has no cleaning log, so
"days since washing rain" stands in for "days since cleaning".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Recommendation:
    plant_id: int
    date: pd.Timestamp
    target: str               # inverter id or "plant"
    rule: str
    severity: str
    action: str
    reason: str
    evidence: dict = field(default_factory=dict)


def days_since_rain(rain: pd.DataFrame, as_of: pd.Timestamp, min_mm: float) -> int | None:
    wet = rain.loc[(rain["date"] <= as_of) & (rain["est_rain_mm"] >= min_mm), "date"]
    return int((as_of - wet.max()).days) if len(wet) else None


def _run_length(flags: pd.Series) -> int:
    """Number of consecutive True values at the end of a date-ordered series."""
    n = 0
    for v in flags[::-1]:
        if not v:
            break
        n += 1
    return n


def recommend(inv_day: pd.DataFrame, plant_day: pd.DataFrame, plant_id: int, as_of: pd.Timestamp, cfg: dict,
              dry_months: list[int] | None = None, reports: pd.DataFrame | None = None) -> list[Recommendation]:
    """`reports` (plant_id, inverter_id, date) lists every day each inverter sent data, including days before
    monitoring started; it defines which inverters should be reporting (rule R7)."""
    mc, mt = cfg["monitoring"], cfg["maintenance"]
    n_days, thr = mc["min_consecutive_days"], mc["ratio_threshold"]
    d = inv_day[(inv_day["plant_id"] == plant_id) & (inv_day["date"] <= as_of)]
    p = plant_day[(plant_day["plant_id"] == plant_id) & (plant_day["date"] <= as_of)].sort_values("date")
    today = d[d["date"] == as_of]
    if today.empty:
        return []
    p_today = p[p["date"] == as_of].iloc[0]
    recs: list[Recommendation] = []

    # R2 single inverter under-performing while the plant as a whole is fine. If the lost energy comes from
    # repeated shutdowns (off in daylight), the inverter itself is the suspect, not dust or strings.
    min_intervals = int(mt["offline_hours"] * 4)
    plant_ok = p_today["share_low"] < mt["gradual_decline_share"]
    covered = set()
    for _, r in today[(today["low_run_days"] >= n_days) & plant_ok].iterrows():
        hist = d[(d["inverter_id"] == r["inverter_id"])].tail(int(r["low_run_days"]))
        off_h = float(hist["off_intervals"].mean()) / 4
        sev = "high" if r["pr"] < 0.5 else "medium"
        ev = {"days_low": int(r["low_run_days"]), "mean_pr": round(float(hist["pr"].mean()), 3),
              "mean_off_hours_per_day": round(off_h, 2), "peer_z_today": round(float(r["peer_z"]), 2),
              "plant_share_low": round(float(p_today["share_low"]), 3)}
        base = (f"Output was below {thr:.0%} of expected for {int(r['low_run_days'])} days in a row "
                f"(mean {hist['pr'].mean():.0%}); {p_today['share_low']:.0%} of the plant's inverters were low today, below the "
                f"{mt['gradual_decline_share']:.0%} that would point to a plant-wide cause")
        if off_h >= mt["offline_hours"]:
            covered.add(r["inverter_id"])
            recs.append(Recommendation(
                plant_id, as_of, r["inverter_id"], "R2 inverter_underperforming", sev,
                "Read the inverter's event log for trips (grid over-voltage, over-temperature, overload) and check "
                "its trip settings and AC voltage at the terminals.",
                base + f"; it switched off for {off_h:.1f} h a day on average while the sun was up, so the loss is shutdowns, not dust.", ev))
        else:
            recs.append(Recommendation(
                plant_id, as_of, r["inverter_id"], "R2 inverter_underperforming", sev,
                "Inspect this inverter's strings: blown fuses, loose connectors, damaged or shaded modules.",
                base + "; it kept running, so part of its array is probably not contributing.", ev))

    # R1 inverter offline in daylight today (not repeated for inverters already covered by R2)
    for _, r in today[(today["off_intervals"] >= min_intervals) & ~today["inverter_id"].isin(covered)].iterrows():
        recs.append(Recommendation(plant_id, as_of, r["inverter_id"], "R1 inverter_offline", "high",
                                   "Check the inverter (trip, fault code, AC/DC isolator).",
                                   f"It produced nothing for {r['off_intervals'] / 4:.1f} h while the sun was up.",
                                   {"off_hours": r["off_intervals"] / 4, "pr": round(float(r["pr"]), 3)}))

    # R3 plant-wide loss for N days
    wide = p["share_low"] >= mt["gradual_decline_share"]
    run = _run_length(wide)
    since = days_since_rain(p, as_of, mt["recent_rain_mm"])
    recent_rain = p.loc[p["date"] > as_of - pd.Timedelta(days=n_days), "est_rain_mm"].sum()
    if run >= n_days:
        ev = {"days": run, "share_low_today": round(float(p_today["share_low"]), 3), "plant_pr_today": round(float(p_today["plant_pr"]), 3),
              "est_rain_last_days_mm": round(float(recent_rain), 1), "days_since_washing_rain": since}
        if recent_rain >= mt["recent_rain_mm"]:
            recs.append(Recommendation(plant_id, as_of, "plant", "R3 plant_wide_loss", "high",
                                       "Not dust: check grid export limits/curtailment, plant-level breakers and the irradiation sensor.",
                                       f"{p_today['share_low']:.0%} of inverters have been low for {run} days even though about "
                                       f"{recent_rain:.0f} mm of rain (estimate) should have washed the panels.", ev))
        else:
            recs.append(Recommendation(plant_id, as_of, "plant", "R3 plant_wide_loss", "high",
                                       "Schedule panel cleaning, then re-check; if output does not recover, inspect plant-level equipment.",
                                       f"{p_today['share_low']:.0%} of inverters have been low for {run} days with no washing rain "
                                       f"(estimate), which fits dust/soiling across the whole plant.", ev))

    # R4 gradual decline of the plant's *running* PR (output while inverters were on). Dust lowers it;
    # shutdowns do not, so a falling plant PR that is only caused by trips does not trigger a cleaning advice.
    w = p.tail(mt["decline_window_days"])
    if len(w) >= mt["decline_window_days"] and recent_rain < mt["recent_rain_mm"]:
        slope = float(np.polyfit(np.arange(len(w)), w["plant_running_pr"], 1)[0])
        if slope <= mt["decline_slope_per_day"]:
            recs.append(Recommendation(plant_id, as_of, "plant", "R4 gradual_decline", "medium",
                                       "Plan a cleaning; compare output before and after to confirm soiling.",
                                       f"Output while the inverters were running has fallen by about {-slope:.1%} per day over the "
                                       f"last {len(w)} days without rain, the typical pattern of dust building up.",
                                       {"slope_per_day": round(slope, 4), "window_days": len(w),
                                        "running_pr_start": round(float(w['plant_running_pr'].iloc[0]), 3),
                                        "running_pr_end": round(float(w['plant_running_pr'].iloc[-1]), 3)}))

    # R5 sensor check
    if p_today["plant_pr"] >= mt["sensor_high_pr"]:
        recs.append(Recommendation(plant_id, as_of, "plant", "R5 sensor_check", "low",
                                   "Clean and check the irradiation sensor; expected output is probably too low.",
                                   f"The whole plant produced {p_today['plant_pr']:.0%} of expected, which a healthy plant cannot do.",
                                   {"plant_pr": round(float(p_today["plant_pr"]), 3)}))

    # R6 seasonal advice (climate of the estimated location)
    if dry_months and as_of.month in dry_months:
        recs.append(Recommendation(plant_id, as_of, "plant", "R6 seasonal_cleaning", "info",
                                   f"Dry season: clean at least every {mt['days_since_cleaning']} days.",
                                   "This month normally gets almost no rain at the estimated plant location, so dust is not washed off.",
                                   {"dry_months": dry_months}))
    # R7 inverter not reporting: it has reported on earlier days but sent nothing today
    rep = reports if reports is not None else inv_day
    rep = rep[(rep["plant_id"] == plant_id) & (rep["date"] < as_of)]
    reporting_today = set(today["inverter_id"]) if reports is None else set(
        reports.loc[(reports["plant_id"] == plant_id) & (reports["date"] == as_of), "inverter_id"]) | set(today["inverter_id"])
    for inv in sorted(set(rep["inverter_id"]) - reporting_today):
        last = rep.loc[rep["inverter_id"] == inv, "date"].max()
        recs.append(Recommendation(plant_id, as_of, inv, "R7 not_reporting", "medium",
                                   "Check the data logger and the communication link of this inverter.",
                                   f"No readings received today; last data on {last.date()}. Its output cannot be monitored.",
                                   {"last_data": str(last.date()), "days_without_data": int((as_of - last).days)}))
    if not recs:
        recs.append(Recommendation(plant_id, as_of, "plant", "R0 no_action", "info", "No action needed.",
                                   "All inverters are within the expected range today.",
                                   {"plant_pr": round(float(p_today["plant_pr"]), 3), "days_since_washing_rain": since}))
    return sorted(recs, key=lambda r: SEVERITY_ORDER[r.severity])


def recommendation_history(inv_day: pd.DataFrame, plant_day: pd.DataFrame, cfg: dict,
                           dry_months: list[int] | None = None, reports: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for (pid, day), _ in plant_day.groupby(["plant_id", "date"]):
        rows += [asdict(r) for r in recommend(inv_day, plant_day, pid, day, cfg, dry_months, reports)]
    return pd.DataFrame(rows)


def dry_months_for(climatology: pd.DataFrame, location: str, max_rain_mm: float = 10.0) -> list[int]:
    c = climatology[climatology["location"] == location]
    return sorted(int(m) for m in c.loc[c["rain_mm_month"] < max_rain_mm, "month"])


def run_maintenance() -> pd.DataFrame:
    """Apply the rules to every monitored plant-day; dry months come from the estimated plant location."""
    import json

    from solvia.config import get_path, load_config
    cfg = load_config()
    interim = get_path("interim")
    inv_day = pd.read_parquet(interim / "monitoring_inverter_day.parquet")
    plant_day = pd.read_parquet(interim / "monitoring_plant_day.parquet")
    match = pd.read_parquet(interim / "plant_location_match.parquet")
    clim = pd.read_parquet(interim / "india_monthly_climatology.parquet")
    reports = pd.read_parquet(interim / "solar_daily_inverter.parquet")[["plant_id", "inverter_id", "date"]]
    parts = []
    for pid, loc in match[match["rank"] == 1].set_index("plant_id")["location"].items():
        dry = dry_months_for(clim, loc)
        h = recommendation_history(inv_day[inv_day["plant_id"] == pid], plant_day[plant_day["plant_id"] == pid], cfg, dry, reports)
        parts.append(h.assign(estimated_location=loc, dry_months=str(dry)))
    hist = pd.concat(parts, ignore_index=True)
    out = hist.assign(evidence=hist["evidence"].map(json.dumps))
    out.to_parquet(interim / "maintenance_recommendations.parquet")
    counts = hist.groupby(["plant_id", "rule"]).size().unstack(fill_value=0)
    latest = hist[hist["date"] == hist["date"].max()]
    summary = {"rule_counts_plant_days": {int(k): v for k, v in counts.to_dict(orient="index").items()},
               "inverters_with_R2": hist[hist["rule"].str.startswith("R2")].groupby("plant_id")["target"].nunique().to_dict(),
               "latest_day": str(hist["date"].max().date()),
               "latest": latest.drop(columns=["evidence"]).astype(str).to_dict(orient="records")}
    (get_path("metrics") / "maintenance_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return hist
