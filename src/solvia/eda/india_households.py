"""EDA for the Indian households (Prayas eMARC smart meters, Maharashtra + Uttar Pradesh, 2018-2020).

Every statistic is computed per meter first and then summarised across meters, so a household that
reported for longer does not count more. Households are never added together into one series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solvia.eda.consumption import MONTHS, _result
from solvia.viz import INK_2, MUTED, NEUTRAL, SERIES, save_figure

TYPE_ORDER = ["Basic", "Water Heaters but no AC", "With Air Conditioners"]
TYPE_COLORS = dict(zip(TYPE_ORDER, SERIES[:3]))
REGION_ORDER = ["Pune city", "Pune", "Aurangabad", "Kanpur rural", "Gonda"]
REGION_COLORS = dict(zip(REGION_ORDER, SERIES[:5]))


def _measured(mains: pd.DataFrame) -> pd.DataFrame:
    return mains[mains["kwh"].notna() & ~mains["is_interpolated"]]


def households_overview(mains: pd.DataFrame, out: Path) -> dict:
    m = _measured(mains)
    per = m.groupby("deployment_id").agg(region=("region", "first"), household_type=("household_type", "first"),
                                         days=("kwh", "size"), mean_kwh=("kwh", "mean"))
    counts = per.groupby(["region", "household_type"]).size().unstack(fill_value=0).reindex(index=REGION_ORDER, columns=TYPE_ORDER, fill_value=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3})
    left = np.zeros(len(counts))
    for t in TYPE_ORDER:
        ax1.barh(range(len(counts)), counts[t], left=left, color=TYPE_COLORS[t], height=0.6, label=t,
                 edgecolor="white", linewidth=2)
        left += counts[t].values
    ax1.set_yticks(range(len(counts)), counts.index)
    ax1.invert_yaxis()
    ax1.set(title="Whole-home meters by region and type", xlabel="Meters")
    ax1.legend(loc="lower right")
    ax2.hist(per["days"], bins=30, color=SERIES[0], edgecolor="white", linewidth=1)
    ax2.set(title="How long each meter reported", xlabel="Measured days per meter", ylabel="Meters")
    path = save_figure(fig, out / "e01_households_overview.png")
    by_state = per.assign(state=per["region"].map(lambda r: "Uttar Pradesh" if r in ("Kanpur rural", "Gonda") else "Maharashtra"))
    ac_regions = ", ".join(counts.index[counts["With Air Conditioners"] > 0])
    return _result("households_overview", "Who is in the eMARC sample", (
        f"{len(per)} whole-home meters with measured data ({(by_state['state'] == 'Maharashtra').sum()} in Maharashtra, "
        f"{(by_state['state'] == 'Uttar Pradesh').sum()} in Uttar Pradesh), {m['household_id'].nunique()} households. "
        f"{int(counts['Basic'].sum())} are 'Basic' homes, {int(counts['Water Heaters but no AC'].sum())} have water heaters and "
        f"{int(counts['With Air Conditioners'].sum())} have air conditioners (all in {ac_regions}). "
        f"Half the meters reported for at least {per['days'].median():.0f} days ({(per['days'] >= 365).sum()} for a year or more)."),
        path, meters=int(len(per)), households=int(m["household_id"].nunique()), counts=counts.to_dict(),
        median_days=float(per["days"].median()))


def consumption_by_type(mains: pd.DataFrame, out: Path) -> dict:
    m = _measured(mains)
    per = m.groupby("deployment_id").agg(household_type=("household_type", "first"), mean_kwh=("kwh", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [per.loc[per["household_type"] == t, "mean_kwh"] for t in TYPE_ORDER]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": "white", "linewidth": 2}, flierprops={"markeredgecolor": MUTED, "markersize": 4})
    for patch, t in zip(bp["boxes"], TYPE_ORDER):
        patch.set(facecolor=TYPE_COLORS[t], edgecolor=TYPE_COLORS[t])
    rng = np.random.default_rng(42)
    for i, d in enumerate(data, start=1):
        ax.scatter(d, i + rng.uniform(-0.12, 0.12, len(d)), s=10, color=INK_2, alpha=0.5, zorder=3)
    ax.set_yticks(range(1, 4), [f"{t}\n(n = {len(d)})" for t, d in zip(TYPE_ORDER, data)])
    ax.set(title="Average daily use per household, by household type", xlabel="Mean consumption (kWh/day), one dot per meter")
    path = save_figure(fig, out / "e02_consumption_by_type.png")
    med = per.groupby("household_type")["mean_kwh"].median().reindex(TYPE_ORDER)
    return _result("consumption_by_type", "Daily use by household type", (
        f"The typical (median) household uses {med['Basic']:.1f} kWh/day in a Basic home, {med['Water Heaters but no AC']:.1f} with a water heater "
        f"and {med['With Air Conditioners']:.1f} with AC. Spread within each group is wide ({per['mean_kwh'].min():.1f}–{per['mean_kwh'].max():.1f} kWh/day "
        f"overall), so the household's own history matters more than its type label."),
        path, median_by_type=med.round(2).to_dict(), range=[round(float(per["mean_kwh"].min()), 2), round(float(per["mean_kwh"].max()), 2)])


def seasonal_by_region(mains: pd.DataFrame, india_daily: pd.DataFrame, region_weather: dict, out: Path) -> dict:
    """Month-of-year profile: each meter is scaled by its own annual mean (needs >= 300 measured days)."""
    m = _measured(mains).assign(month=lambda d: d["date"].dt.month)
    days = m.groupby("deployment_id")["kwh"].transform("size")
    m = m[days >= 300]
    per = m.groupby(["deployment_id", "region", "month"])["kwh"].mean().reset_index()
    per["index"] = per["kwh"] / per.groupby("deployment_id")["kwh"].transform("mean")
    prof = per.groupby(["region", "month"])["index"].mean().unstack(0)
    temp = (india_daily[india_daily["date"].between("2018-01-01", "2020-06-30")]
            .assign(month=lambda d: d["date"].dt.month).groupby(["location", "month"])["t2m_max_c"].mean().unstack(0))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for r in [r for r in REGION_ORDER if r in prof]:
        ax1.plot(prof.index, prof[r], marker="o", markersize=4, color=REGION_COLORS[r], label=r)
    ax1.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    ax1.set(title="Seasonal pattern: monthly use relative to each home's own average (1 = average)", ylabel="Relative use",
            ylim=(0.3, prof.max().max() + 0.25))
    ax1.legend(ncols=5, loc="upper right")
    for loc, c in [("pune", SERIES[0]), ("aurangabad", SERIES[2]), ("kanpur", SERIES[3]), ("gonda", SERIES[4])]:
        ax2.plot(temp.index, temp[loc], color=c, marker="o", markersize=3, label=loc.title())
    ax2.set(title="Daily maximum temperature (NASA POWER, 2018–2020 mean)", xlabel="Month", ylabel="°C")
    ax2.set_xticks(range(1, 13), MONTHS)
    ax2.legend(ncols=4, loc="lower center")
    path = save_figure(fig, out / "e03_seasonal_by_region.png")
    peak = prof.idxmax()
    low = prof.idxmin()
    return _result("seasonal_by_region", "Seasonal pattern by region", (
        "Use peaks in the hot months, not in winter as in France: peak month is "
        + ", ".join(f"{r} {MONTHS[int(peak[r]) - 1]} ({prof[r].max():.2f}×)" for r in prof)
        + "; the lowest month is " + ", ".join(f"{r} {MONTHS[int(low[r]) - 1]}" for r in prof)
        + f". Only meters with ≥ 300 measured days ({m['deployment_id'].nunique()}) are used. Cooling (fans, coolers, AC) drives the seasonal cycle, "
        "so temperature features should matter more here than for the French household."),
        path, peak_month={r: int(v) for r, v in peak.items()}, peak_index={r: round(float(prof[r].max()), 3) for r in prof},
        meters=int(m["deployment_id"].nunique()))


def temperature_response(mains: pd.DataFrame, india_daily: pd.DataFrame, region_weather: dict, out: Path) -> dict:
    """Relative daily use vs that day's maximum temperature, binned, per household type."""
    m = _measured(mains).assign(loc=lambda d: d["region"].map(region_weather))
    m = m.merge(india_daily[["location", "date", "t2m_max_c"]].rename(columns={"location": "loc"}), on=["loc", "date"])
    m["rel"] = m["kwh"] / m.groupby("deployment_id")["kwh"].transform("mean")
    bins = np.arange(18, 48, 2)
    m["tbin"] = pd.cut(m["t2m_max_c"], bins, labels=bins[:-1] + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    slopes = {}
    for t in TYPE_ORDER:
        g = m[m["household_type"] == t]
        b = g.groupby("tbin", observed=True)["rel"].agg(["mean", "size"])
        b = b[b["size"] >= 100]
        ax.plot(b.index.astype(float), b["mean"], marker="o", markersize=5, color=TYPE_COLORS[t], label=t)
        hot = g[g["t2m_max_c"] >= 30]
        slopes[t] = float(np.polyfit(hot["t2m_max_c"], hot["rel"], 1)[0]) if len(hot) > 100 else float("nan")
    ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.set(title="Hotter days, more electricity", xlabel="Daily maximum temperature (°C, NASA POWER at the mapped city)",
           ylabel="Use relative to the home's own average")
    ax.legend(loc="upper left")
    path = save_figure(fig, out / "e04_temperature_response.png")
    corr = m.groupby("deployment_id").apply(lambda g: g["kwh"].corr(g["t2m_max_c"]), include_groups=False)
    return _result("temperature_response", "Temperature response", (
        f"Above 30 °C each extra degree of daily maximum adds about {slopes['Basic'] * 100:.1f}% of a Basic home's average use, "
        f"{slopes['Water Heaters but no AC'] * 100:.1f}% for water-heater homes and {slopes['With Air Conditioners'] * 100:.1f}% for AC homes. "
        f"Per meter, the median correlation between daily use and maximum temperature is {corr.median():.2f} "
        f"({(corr > 0).mean():.0%} of meters positive). Cooling degree-days are therefore included as features."),
        path, slope_per_degC_above_30={k: round(v, 4) for k, v in slopes.items()}, median_corr=round(float(corr.median()), 3),
        share_positive=round(float((corr > 0).mean()), 3))


def load_profile(profiles: pd.DataFrame, solar_hours: tuple[int, int], out: Path) -> dict:
    """Average 15-minute load shape by household type, with the rooftop-solar window shaded."""
    p = profiles.groupby(["household_type", "block"]).apply(
        lambda g: np.average(g["load_kw"], weights=g["n_meters"]), include_groups=False).unstack(0)
    hours = p.index / 4
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(*solar_hours, color=NEUTRAL, zorder=0, label=f"Solar hours ({solar_hours[0]}:00–{solar_hours[1]}:00)")
    for t in TYPE_ORDER:
        ax.plot(hours, p[t], color=TYPE_COLORS[t], label=t)
    ax.set(title="Average daily load shape of a household (15-minute blocks)", xlabel="Hour of day", ylabel="Average load (kW)",
           xlim=(0, 24), ylim=(0, None))
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="upper left")
    path = save_figure(fig, out / "e05_load_profile.png")
    in_sun = (hours >= solar_hours[0]) & (hours < solar_hours[1])
    share = (p[in_sun].sum() / p.sum()).reindex(TYPE_ORDER)
    peak_hour = p.idxmax().reindex(TYPE_ORDER) / 4
    return _result("load_profile", "Daily load shape and the solar window", (
        f"Only {share['Basic']:.0%} (Basic), {share['Water Heaters but no AC']:.0%} (water heater) and {share['With Air Conditioners']:.0%} (AC) "
        f"of daily use falls in {solar_hours[0]}:00–{solar_hours[1]}:00 when rooftop solar produces most. The peak is at "
        + ", ".join(f"{int(h)}:{int(round(h % 1 * 60)):02d}" for h in peak_hour)
        + " respectively. Without a battery, most solar output in these homes would be exported, not used directly; "
        "the energy balance therefore reports this share alongside the daily totals."),
        path, solar_hours_share=share.round(3).to_dict(), peak_hour=peak_hour.round(2).to_dict())


def lockdown_effect(mains: pd.DataFrame, lockdown_start: str, out: Path) -> dict:
    """Same meters, same calendar window: 25 Mar-30 Jun 2020 vs 25 Mar-30 Jun 2019 (meters with >= 60 days in both)."""
    m = _measured(mains)
    start = pd.Timestamp(lockdown_start)
    win = {2019: (start - pd.DateOffset(years=1), pd.Timestamp("2019-06-30")), 2020: (start, pd.Timestamp("2020-06-30"))}
    per = {y: m[m["date"].between(*w)].groupby("deployment_id")["kwh"].agg(["mean", "size"]) for y, w in win.items()}
    j = per[2019].join(per[2020], lsuffix="_2019", rsuffix="_2020", how="inner")
    j = j[(j["size_2019"] >= 60) & (j["size_2020"] >= 60)]
    j["change"] = j["mean_2020"] / j["mean_2019"] - 1
    j = j.join(m.groupby("deployment_id")["household_type"].first())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    jj = j.sort_values("change")
    ax.bar(range(len(jj)), jj["change"] * 100, color=[SERIES[0] if c >= 0 else SERIES[1] for c in jj["change"]], width=0.8)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.axhline(j["change"].median() * 100, color=INK_2, linewidth=1, linestyle="--",
               label=f"Median {j['change'].median():+.0%}")
    ax.set(title="COVID-19 lockdown: change in daily use, 25 Mar–30 Jun 2020 vs same days 2019",
           xlabel=f"Meters (n = {len(j)}), sorted", ylabel="Change (%)")
    ax.set_xticks([])
    ax.legend(loc="upper left")
    path = save_figure(fig, out / "e06_lockdown_effect.png")
    return _result("lockdown_effect", "Lockdown shift (test-set warning)", (
        f"For the {len(j)} meters with ≥ 60 measured days in both windows, median use changed by {j['change'].median():+.0%} "
        f"({(j['change'] > 0).mean():.0%} of homes used more; not adjusted for the two years' different weather). The model never sees a lockdown in training, so test results are "
        "reported separately for 'normal' days (Jan–24 Mar 2020) and 'lockdown' days."),
        path, meters=int(len(j)), median_change=round(float(j["change"].median()), 4),
        share_increase=round(float((j["change"] > 0).mean()), 3))

