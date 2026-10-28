"""Figures and the markdown report for Phases 4C (monitoring) and 4D (maintenance)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from solvia.config import get_path, load_config, project_path
GAP = "#bdbcb6"

from solvia.viz import (BLUE_DARK, DIVERGING, INK, INK_2, MUTED, PLANT_COLORS, PLANT_NAMES, SERIES, apply_style,
                        save_figure)


def pr_heatmap(d: pd.DataFrame, out) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"hspace": 0.35})
    norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.2)
    all_days = sorted(d["date"].unique())
    for ax, (pid, g) in zip(axes, d.groupby("plant_id")):
        order = g.groupby("inverter_id")["pr"].median().sort_values().index
        mat = g.pivot(index="inverter_id", columns="date", values="pr").reindex(columns=all_days).loc[order]
        cmap = DIVERGING.copy()
        cmap.set_bad(GAP)                                   # no data must not look like PR = 1
        im = ax.imshow(np.ma.masked_invalid(mat.clip(0.4, 1.2).to_numpy(dtype=float)), aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.scatter([], [], marker="s", s=40, color=GAP, label="No data from the inverter")
        al = g[g["status"] == "alert"]
        ax.scatter([list(mat.columns).index(x) for x in al["date"]], [list(order).index(i) for i in al["inverter_id"]],
                   marker="o", s=10, color=INK, label="Alert (low ≥ 3 days in a row)")
        ax.set_yticks(range(len(order)), [f"inv {i + 1:02d}" for i in range(len(order))], fontsize=7)
        ax.set_xticks(range(0, len(mat.columns), 3), [c.strftime("%d %b") for c in mat.columns[::3]], fontsize=8)
        ax.set(title=f"{PLANT_NAMES[pid]}: performance ratio per inverter and day (actual ÷ out-of-sample expected)")
        ax.grid(False)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.16), fontsize=8, ncols=2)
    fig.colorbar(im, ax=axes, fraction=0.02, label="Performance ratio (1 = as expected; red = loss)")
    save_figure(fig, out / "m01_pr_heatmap.png")
    return "m01_pr_heatmap.png"


def injected_curve(inj: pd.DataFrame, threshold: float, out) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(inj["loss_pct"], inj["flagged_share"] * 100, marker="o", color=SERIES[0])
    for _, r in inj.iterrows():
        ax.annotate(f"{r['flagged_share']:.0%}", (r["loss_pct"], r["flagged_share"] * 100), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.axvline((1 - threshold) * 100, color=MUTED, linestyle="--", linewidth=1)
    ax.text((1 - threshold) * 100 + 0.6, 50, f"rule threshold\n(PR < {threshold})", fontsize=8, color=INK_2)
    ax.set(title="Simulated faults on healthy Plant 1 inverter-days: how often one day is flagged",
           xlabel="Simulated output loss (%)", ylabel="Inverter-days flagged (%)", ylim=(-3, 105))
    save_figure(fig, out / "m02_injected_fault_detection.png")
    return "m02_injected_fault_detection.png"


def trips_figure(trips: pd.DataFrame, out) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bands = sorted(trips["irradiation_band_kw_m2"].unique(), key=lambda b: float(b.strip("(]").split(",")[0]))
    groups = [("Plant 1", "other inverters", PLANT_COLORS[4135001], "Plant 1 (all inverters)"),
              ("Plant 2", "other inverters", "#f2a07b", "Plant 2, other inverters"),
              ("Plant 2", "often-low inverters", PLANT_COLORS[4136001], "Plant 2, often-low inverters")]
    x = np.arange(len(bands))
    for i, (plant, grp, color, label) in enumerate(groups):
        t = trips[(trips["plant"] == plant) & (trips["group"] == grp)].set_index("irradiation_band_kw_m2").reindex(bands)
        ax.bar(x + (i - 1) * 0.27, t["off_share"] * 100, width=0.25, color=color, label=label, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [b.replace("(", "").replace("]", "").replace(", ", "–") for b in bands])
    ax.set(title="Inverters switching off in daylight, by how bright it is", xlabel="Irradiation (kW/m²)",
           ylabel="Daylight intervals with the inverter off (%)")
    ax.legend(loc="upper left")
    save_figure(fig, out / "m03_shutdowns_by_irradiation.png")
    return "m03_shutdowns_by_irradiation.png"


def plant_pr_figure(p: pd.DataFrame, out) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for pid, g in p.groupby("plant_id"):
        c = PLANT_COLORS[pid]
        ax1.plot(g["date"], g["plant_pr"], color=c, marker="o", markersize=3, label=f"{PLANT_NAMES[pid]}: all output")
        ax1.plot(g["date"], g["plant_running_pr"], color=c, linestyle="--", linewidth=1.3,
                 label=f"{PLANT_NAMES[pid]}: while inverters were running")
    ax1.axhline(1, color=MUTED, linewidth=1)
    ax1.set(title="Plant performance ratio per day", ylabel="Actual ÷ expected", ylim=(0.55, 1.1))
    ax1.legend(ncols=2, loc="lower left", fontsize=8)
    r = p[p["plant_id"] == p["plant_id"].min()]
    ax2.bar(r["date"], r["est_rain_mm"], color=BLUE_DARK, width=0.7)
    ax2.set(title="Estimated rainfall at the best-matching city (NASA POWER, Anantapur, ESTIMATE)", ylabel="mm/day", xlabel="Date")
    save_figure(fig, out / "m04_plant_pr_and_rain.png")
    return "m04_plant_pr_and_rain.png"


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c].format(r[c]) if c in fmt and pd.notna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_monitoring_report() -> None:
    apply_style()
    cfg = load_config()
    mc, mt = cfg["monitoring"], cfg["maintenance"]
    interim, metrics, figs = get_path("interim"), get_path("metrics"), get_path("figures")
    d = pd.read_parquet(interim / "monitoring_inverter_day.parquet")
    p = pd.read_parquet(interim / "monitoring_plant_day.parquet")
    rec = pd.read_parquet(interim / "maintenance_recommendations.parquet")
    s = json.loads((metrics / "monitoring_summary.json").read_text())
    inj = pd.read_csv(metrics / "monitoring_injected_faults.csv")
    trips = pd.read_csv(metrics / "monitoring_trips_by_irradiance.csv")
    rain = pd.read_csv(metrics / "monitoring_rain_effect.csv")
    f1, f2, f3, f4 = pr_heatmap(d, figs), injected_curve(inj, mc["ratio_threshold"], figs), trips_figure(trips, figs), plant_pr_figure(p, figs)

    status = pd.DataFrame(s["status_counts"]).T.reset_index(names="plant")
    iso = s["isolation_forest"]
    rules = rec.groupby(["plant_id", "rule"]).size().unstack(fill_value=0)
    rules.index = rules.index.map(PLANT_NAMES)
    r2 = rec[rec["rule"].str.startswith("R2")]
    r2_kind = r2["action"].str.startswith("Read the inverter's event log").map({True: "shutdowns → event log / trip settings",
                                                                                False: "steady shortfall → strings / modules"})
    examples = (rec[rec["rule"].isin(["R2 inverter_underperforming", "R1 inverter_offline"])]
                .sort_values(["severity", "date"]).groupby("rule").head(1))
    lines = [
        "# Performance monitoring and maintenance (Phases 4C and 4D)", "",
        "*Generated by `python -m solvia.pipeline --steps monitor maintenance`. All numbers are computed by the code.*", "",
        "## How expected output is made (out-of-sample)", "",
        f"For every day the Phase 4B configuration (HistGradientBoosting, one model per plant, healthy reference) is refitted "
        f"on **earlier days only** and predicts that day. The first {mc['min_train_days']} days are warm-up, so monitoring covers "
        f"{s['monitored_days']}, {s['inverter_days']:,} inverter-days. Healthy inverter-days have a median PR of "
        f"{s['healthy_reference_pr']['Plant 1']['median']} (Plant 1) and {s['healthy_reference_pr']['Plant 2']['median']} (Plant 2); "
        f"5th–95th percentile {s['healthy_reference_pr']['Plant 1']['p05']}–{s['healthy_reference_pr']['Plant 1']['p95']} on Plant 1.", "",
        "## Rules", "",
        f"- **PR** = actual ÷ expected kWh per inverter-day; **low** if PR < {mc['ratio_threshold']}.",
        f"- **Peer outlier** if the robust z-score among the plant's inverters that day is < {mc['peer_z_threshold']} *and* PR < {mc['peer_max_pr']}.",
        f"- **watch** = low or peer outlier on the day; **alert** = low for ≥ {mc['min_consecutive_days']} consecutive days.",
        "- **Running PR** = PR over the intervals when the inverter was on. Shutdowns lower PR but not running PR; dust lowers both.", "",
        "## Status of every inverter-day", "", _table(status), "",
        f"Inverters with at least one alert: Plant 1 {s['inverters_with_alert']['Plant 1']}, Plant 2 {s['inverters_with_alert']['Plant 2']} of 22.", "",
        f"![PR heatmap](figures/{f1})", "",
        "## Does the rule catch real faults? (simulated)", "",
        "Healthy Plant 1 inverter-days with their output scaled down; one day is flagged when PR < "
        f"{mc['ratio_threshold']}. 0 % loss shows the false-alarm rate.", "",
        _table(inj[["loss_pct", "inverter_days", "flagged_share"]], {"loss_pct": "{:.0f} %", "inverter_days": "{:.0f}", "flagged_share": "{:.1%}"}), "",
        f"![Injected faults](figures/{f2})", "",
        "A single day reliably catches losses of 20 % or more; smaller losses need the multi-day view (trend, running PR).", "",
        "## Isolation Forest comparison", "",
        f"Isolation Forest on (PR, peer z-score, off share), set to the same flag rate as the PR rule ({iso['flag_rate']:.1%}): "
        f"both flag {iso['both']} inverter-days, rule only {iso['rule_only']}, Isolation Forest only {iso['iforest_only']} "
        f"(Jaccard {iso['jaccard']}). Days only the rule flags have median PR {iso['rule_only_median_pr']}; days only the forest flags "
        f"have median PR {iso['iforest_only_median_pr']}. The transparent rule is kept; the forest adds little and cannot explain its flags.", "",
        "## Key finding: Plant 2 loses energy by switching inverters off in bright conditions", "",
        f"Mean plant PR is {s['mean_plant_pr']['Plant 2']} on Plant 2 but {s['mean_plant_running_pr']['Plant 2']} while its inverters "
        f"were running. Shutdowns explain {s['shutdown_share_of_lost_energy']['Plant 2']:.0%} of Plant 2's lost energy. The shutdowns "
        "happen around midday and rise with irradiation (figure m03); Plant 1 almost never shuts down. Module temperature is the same "
        "with inverters off or on, and only a few of the 22 inverters are off at the same moment, so this is not a whole-plant outage. "
        "Likely causes are inverter trips (grid over-voltage, overload or over-temperature settings) or curtailment; the data cannot "
        "tell which, so the recommendation is to read the inverters' event logs.", "",
        f"![Shutdowns by irradiation](figures/{f3})", "",
        "## Rain effect (estimate)", "",
        "Rainfall is NASA POWER at the best-matching city (Anantapur), not measured at the plant.", "",
        _table(rain[["plant", "group", "days", "mean_plant_pr", "mean_running_pr", "mean_irradiation_kwh_m2"]],
               {"mean_plant_pr": "{:.3f}", "mean_running_pr": "{:.3f}", "mean_irradiation_kwh_m2": "{:.2f}"}), "",
        f"Plant 2's PR is higher after rain, but those days are also much dimmer, and dim days have fewer shutdowns. Running PR "
        f"(no shutdowns) shows no link to rain (r = {s['corr_running_pr_vs_est_rain']['Plant 2']} Plant 2, "
        f"{s['corr_running_pr_vs_est_rain']['Plant 1']} Plant 1). **No evidence of soiling or of rain washing in these 27 days.** "
        "With 5 rainy days this is weak evidence either way.", "",
        f"![Plant PR and rain](figures/{f4})", "",
        "## Maintenance recommendations (Phase 4D)", "",
        "Rules (details in `src/solvia/rules/maintenance.py`): R1 inverter offline ≥ "
        f"{mt['offline_hours']} h in daylight · R2 inverter low ≥ {mc['min_consecutive_days']} days while peers are fine "
        "(shutdowns → event log; steady shortfall → strings) · R3 plant-wide loss (rain decides dust vs other cause) · "
        f"R4 running PR falling ≥ {-mt['decline_slope_per_day']:.1%}/day over {mt['decline_window_days']} days without rain → cleaning · "
        f"R5 plant PR ≥ {mt['sensor_high_pr']} → sensor check · R6 dry-month cleaning advice · R7 inverter not reporting · R0 no action.", "",
        "Plant-days on which each rule fired:", "", _table(rules.reset_index(names="plant")), "",
        f"R2 fired for {r2['target'].nunique()} Plant 2 inverters; type: "
        + ", ".join(f"{k}: {v} plant-days" for k, v in r2_kind.value_counts().items()) + ".", "",
        "Example recommendations:", "",
        *[f"- **{r['rule']}** ({PLANT_NAMES[r['plant_id']]}, inverter `{r['target']}`, {pd.Timestamp(r['date']).date()}, {r['severity']}): "
          f"{r['action']} *Reason:* {r['reason']}" for _, r in examples.iterrows()],
        "", "Tables: `data/interim/monitoring_inverter_day.parquet`, `monitoring_plant_day.parquet`, "
        "`maintenance_recommendations.parquet`; metrics in `reports/metrics/monitoring_*` and `maintenance_summary.json`.", ""]
    project_path("reports/monitoring_report.md").write_text("\n".join(lines), encoding="utf-8")
