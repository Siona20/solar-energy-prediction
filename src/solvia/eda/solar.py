"""EDA figures for the two Kaggle solar plants (34 days, 15-minute, 22 inverters each)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solvia.eda.consumption import _result
from solvia.viz import CRITICAL, DIVERGING, INK_2, MUTED, PLANT_COLORS, PLANT_NAMES, SURFACE, save_figure


def _plants(df: pd.DataFrame):
    for pid in sorted(df["plant_id"].unique()):
        yield pid, df[df["plant_id"] == pid]


def peer_ratio(daily_inv: pd.DataFrame) -> pd.Series:
    """Each inverter's daily kWh divided by the median inverter of the same plant on the same day."""
    med = daily_inv.groupby(["plant_id", "date"])["energy_kwh"].transform("median")
    return daily_inv["energy_kwh"] / med


def daily_power_curves(df15: pd.DataFrame, out: Path) -> dict:
    tod = df15["timestamp"].dt.hour + df15["timestamp"].dt.minute / 60
    d = df15.assign(tod=tod)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    peaks = {}
    for pid, g in _plants(d):
        irr = g.drop_duplicates("timestamp").groupby("tod")["irradiation_kw_m2"].mean()
        ac = g.groupby("tod")["ac_power_kw"].mean()
        ax1.plot(irr.index, irr.values * 1000, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
        ax2.plot(ac.index, ac.values, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
        peaks[PLANT_NAMES[pid]] = {"peak_hour": round(float(ac.idxmax()), 2), "peak_mean_ac_kw": round(float(ac.max()), 1)}
    ax1.set(title="Average day: irradiation (top) and AC power per inverter (bottom)", ylabel="Irradiation (W/m²)")
    ax2.set(xlabel="Time of day (hour)", ylabel="Mean AC power per inverter (kW)")
    ax1.legend(loc="upper right")
    ax2.set_xticks(range(0, 25, 3))
    path = save_figure(fig, out / "s01_daily_power_curves.png")
    p1, p2 = peaks["Plant 1"], peaks["Plant 2"]
    return _result("daily_power_curves", "Daily power curves", (
        f"Output follows the sun: zero at night and peaking around midday (Plant 1 mean peak {p1['peak_mean_ac_kw']:.0f} kW "
        f"per inverter, Plant 2 {p2['peak_mean_ac_kw']:.0f} kW); Plant 2's lower midday output despite similar sunlight hints at outages."),
        path, **peaks)


def ac_vs_irradiation(df15: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    stats = {}
    for ax, (pid, g) in zip(axes, _plants(df15)):
        g = g[g["irradiation_kw_m2"] > 0]
        ok, off = g[~g["flag_off_in_daylight"]], g[g["flag_off_in_daylight"]]
        sample = ok.sample(min(len(ok), 6000), random_state=42)
        ax.scatter(sample["irradiation_kw_m2"] * 1000, sample["ac_power_kw"], s=5, alpha=0.35,
                   color=PLANT_COLORS[pid], label="Operating", linewidths=0)
        ax.scatter(off["irradiation_kw_m2"] * 1000, off["ac_power_kw"], s=9, color=CRITICAL, marker="x",
                   linewidths=0.8, label=f"Off in daylight ({len(off):,})")
        ax.set(title=PLANT_NAMES[pid], xlabel="Irradiation (W/m²)")
        ax.legend(loc="upper left", markerscale=2)
        stats[PLANT_NAMES[pid]] = {"corr_all_daytime": round(float(g[["ac_power_kw", "irradiation_kw_m2"]].corr().iloc[0, 1]), 4),
                                   "corr_operating": round(float(ok[["ac_power_kw", "irradiation_kw_m2"]].corr().iloc[0, 1]), 4),
                                   "off_rows": int(len(off))}
    axes[0].set_ylabel("AC power per inverter (kW)")
    fig.suptitle("AC power vs irradiation (daytime, 15-minute readings)", x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "s02_ac_vs_irradiation.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("ac_vs_irradiation", "AC power vs irradiation", (
        f"Power is almost a straight line in irradiation (operating rows: r = {s1['corr_operating']:.3f} Plant 1, "
        f"{s2['corr_operating']:.3f} Plant 2); Plant 2's {s2['off_rows']:,} 'off in daylight' readings drag its overall r down to {s2['corr_all_daytime']:.2f}, "
        f"so they must be excluded when learning expected output."), path, **stats)


def module_temperature_effect(df15: pd.DataFrame, out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    stats = {}
    for pid, g in _plants(df15):
        g = g[(g["irradiation_kw_m2"] > 0.3) & ~g["flag_off_in_daylight"] & (g["ac_power_kw"] > 0)]
        eff = g["ac_power_kw"] / g["irradiation_kw_m2"]          # kW per kW/m2
        bins = pd.cut(g["module_temp_c"], np.arange(20, 72, 4))
        m = eff.groupby(bins, observed=True).agg(["mean", "size"])
        m = m[m["size"] >= 50]
        mids = [b.mid for b in m.index]
        ax.plot(mids, m["mean"], marker="o", markersize=5, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
        slope, intercept = np.polyfit(g["module_temp_c"], eff, 1)
        stats[PLANT_NAMES[pid]] = {"pct_change_per_degC": round(float(slope / eff.mean() * 100), 3), "rows": int(len(g))}
    ax.set(title="Module temperature effect: output per unit of sunlight",
           xlabel="Module temperature (°C)", ylabel="AC power ÷ irradiation (kW per kW/m²)")
    ax.legend()
    path = save_figure(fig, out / "s03_module_temperature_effect.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("module_temperature_effect", "Module temperature effect", (
        f"Hotter panels convert sunlight less efficiently: output per unit irradiation changes by {s1['pct_change_per_degC']:+.2f}%/°C "
        f"(Plant 1) and {s2['pct_change_per_degC']:+.2f}%/°C (Plant 2), so module temperature is a useful second feature."),
        path, **stats)


def inverter_comparison(daily_inv: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    stats = {}
    for ax, (pid, g) in zip(axes, _plants(daily_inv)):
        order = g.groupby("inverter_id")["energy_kwh"].median().sort_values()
        data = [g.loc[g["inverter_id"] == i, "energy_kwh"].values for i in order.index]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=True,
                        medianprops={"color": SURFACE, "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": PLANT_COLORS[pid], "markeredgewidth": 0})
        for box in bp["boxes"]:
            box.set(facecolor=PLANT_COLORS[pid], edgecolor=PLANT_COLORS[pid])
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=INK_2, linewidth=0.8)
        ax.set_yticks(range(1, len(order) + 1), [i[:6] for i in order.index], fontsize=7)
        ax.set(title=PLANT_NAMES[pid], xlabel="Daily energy per inverter (kWh)")
        peak = g.groupby("inverter_id")["peak_ac_power_kw"].max()
        stats[PLANT_NAMES[pid]] = {"lowest_median_kwh": round(order.iloc[0], 0), "highest_median_kwh": round(order.iloc[-1], 0),
                                   "highest_over_lowest": round(order.iloc[-1] / order.iloc[0], 3),
                                   "max_ac_kw_range": [round(peak.min(), 0), round(peak.max(), 0)],
                                   "corr_median_vs_off_intervals": round(float(order.corr(g.groupby("inverter_id")["n_off_in_daylight"].sum())), 3)}
    axes[0].set_ylabel("Inverter (first 6 characters of ID)")
    fig.suptitle("Inverter-to-inverter comparison of daily energy", x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "s04_inverter_comparison.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("inverter_comparison", "Inverter comparison", (
        f"Plant 1's inverters are tightly grouped (best median only {(s1['highest_over_lowest'] - 1) * 100:.0f}% above the worst); "
        f"Plant 2 is far more uneven ({(s2['highest_over_lowest'] - 1) * 100:.0f}% gap). Its inverters are the same size (peak {s2['max_ac_kw_range'][0]:.0f}–{s2['max_ac_kw_range'][1]:.0f} kW) "
        f"and the low ones have the most 'off in daylight' intervals (r = {s2['corr_median_vs_off_intervals']:.2f}), so the gap is lost output, not smaller hardware."),
        path, **stats)


def plant_comparison(daily_plant: pd.DataFrame, out: Path) -> dict:
    d = daily_plant.assign(kwh_per_inverter=daily_plant["energy_kwh"] / daily_plant["n_inverters"],
                           yield_per_irr=daily_plant["energy_kwh"] / daily_plant["n_inverters"] / daily_plant["irradiation_kwh_m2"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    stats = {}
    for pid, g in _plants(d):
        ax1.plot(g["date"], g["kwh_per_inverter"], marker="o", markersize=3, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
        ax2.plot(g["date"], g["irradiation_kwh_m2"], marker="o", markersize=3, color=PLANT_COLORS[pid], label=PLANT_NAMES[pid])
        stats[PLANT_NAMES[pid]] = {"mean_kwh_per_inverter_day": round(g["kwh_per_inverter"].mean(), 0),
                                   "mean_irradiation_kwh_m2_day": round(g["irradiation_kwh_m2"].mean(), 2),
                                   "kwh_per_inverter_per_kwh_m2": round(g["yield_per_irr"].mean(), 0)}
    ax1.set(title="Plant 1 vs Plant 2: daily energy (top) and daily irradiation (bottom)", ylabel="Energy per inverter (kWh/day)")
    ax2.set(xlabel="Date (2020)", ylabel="Irradiation (kWh/m²/day)")
    ax1.legend(loc="lower left")
    fig.autofmt_xdate()
    path = save_figure(fig, out / "s05_plant_comparison.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("plant_comparison", "Plant 1 vs Plant 2", (
        f"Both plants get similar sunlight ({s1['mean_irradiation_kwh_m2_day']:.2f} vs {s2['mean_irradiation_kwh_m2_day']:.2f} kWh/m²/day), "
        f"but Plant 2 produces less per inverter ({s2['mean_kwh_per_inverter_day']:.0f} vs {s1['mean_kwh_per_inverter_day']:.0f} kWh/day); "
        f"the gap is about performance, not weather."), path, **stats)


def inverter_heatmap(daily_inv: pd.DataFrame, out: Path) -> dict:
    d = daily_inv.assign(peer_ratio=peer_ratio(daily_inv))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    stats = {}
    for ax, (pid, g) in zip(axes, _plants(d)):
        mat = g.pivot(index="inverter_id", columns="date", values="peer_ratio")
        mat = mat.loc[mat.mean(axis=1).sort_values().index]
        cmap = DIVERGING.copy()
        cmap.set_bad(MUTED)
        im = ax.imshow(np.ma.masked_invalid(mat.values), aspect="auto", cmap=cmap, vmin=0.5, vmax=1.5, interpolation="nearest")
        ax.set_yticks(range(len(mat)), [i[:6] for i in mat.index], fontsize=7)
        ticks = list(range(0, mat.shape[1], 5))
        ax.set_xticks(ticks, [mat.columns[t].strftime("%d %b") for t in ticks], fontsize=8, rotation=45)
        ax.grid(False)
        ax.set(title=PLANT_NAMES[pid], xlabel="Date (2020)")
        share_low = g.assign(low=g["peer_ratio"] < 0.85).groupby("date")["low"].mean()
        stats[PLANT_NAMES[pid]] = {"inverter_days_below_0.85": int((g["peer_ratio"] < 0.85).sum()), "inverter_days": int(len(g)),
                                   "days_with_over_30pct_inverters_low": int((share_low > 0.3).sum()),
                                   "missing_inverter_days": int(mat.isna().sum().sum())}
    fig.colorbar(im, ax=axes, label="Daily kWh ÷ plant median that day", fraction=0.02)
    fig.suptitle("Daily energy of each inverter relative to its peers (red = below peers, gray = no data)",
                 x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "s06_inverter_daily_heatmap.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("inverter_heatmap", "Inverter × day heatmap", (
        f"Under-performance is concentrated: {s2['inverter_days_below_0.85']} of Plant 2's {s2['inverter_days']} inverter-days are below 85% of the "
        f"plant median vs {s1['inverter_days_below_0.85']} in Plant 1. They come in runs on the same inverters (persistent issues), but on "
        f"{s2['days_with_over_30pct_inverters_low']} days many Plant 2 inverters drop together, which pulls the median down and makes healthy ones look high, "
        f"so monitoring must also compare each inverter with model-expected output, not only with its peers."),
        path, **stats)


def peer_ratio_distribution(daily_inv: pd.DataFrame, out: Path, threshold: float = 0.85) -> dict:
    d = daily_inv.assign(peer_ratio=peer_ratio(daily_inv))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1.3, 53)
    stats = {}
    for pid, g in _plants(d):
        ax.hist(g["peer_ratio"].clip(upper=1.3), bins=bins, color=PLANT_COLORS[pid], alpha=0.6, label=PLANT_NAMES[pid])
        r = g["peer_ratio"]
        stats[PLANT_NAMES[pid]] = {"p05": round(r.quantile(0.05), 3), "p50": round(r.median(), 3),
                                   "share_below_threshold": round(float((r < threshold).mean()), 4)}
    ax.axvline(threshold, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(threshold, ax.get_ylim()[1] * 0.9, f"  threshold {threshold}", color=INK_2, fontsize=8)
    ax.set_yscale("log")
    ax.set(title="How far below its peers each inverter runs (daily)", xlabel="Daily kWh ÷ plant median that day",
           ylabel="Inverter-days (log scale)")
    ax.legend(loc="upper left")
    path = save_figure(fig, out / "s07_peer_ratio_distribution.png")
    s1, s2 = stats["Plant 1"], stats["Plant 2"]
    return _result("peer_ratio_distribution", "Threshold check", (
        f"Normal days sit tightly around 1.0 (Plant 1 5th percentile {s1['p05']:.2f}), so 0.85 is well outside normal variation: "
        f"it flags {s1['share_below_threshold'] * 100:.1f}% of Plant 1 and {s2['share_below_threshold'] * 100:.1f}% of Plant 2 inverter-days. Keeping 0.85."),
        path, **stats)


ALL_15MIN = [daily_power_curves, ac_vs_irradiation, module_temperature_effect]
ALL_DAILY_INV = [inverter_comparison, inverter_heatmap, peer_ratio_distribution]
ALL_DAILY_PLANT = [plant_comparison]
