"""EDA figures for the daily household consumption series (UCI, Sceaux, France)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solvia.viz import BLUE_DARK, INK_2, MUTED, SEQUENTIAL, SERIES, save_figure

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _result(name: str, title: str, takeaway: str, path: Path, **stats) -> dict:
    return {"name": name, "title": title, "takeaway": takeaway, "file": path.name, "stats": stats}


def daily_trend(daily: pd.DataFrame, out: Path) -> dict:
    s = daily["consumption_kwh"]
    roll = s.rolling(30, center=True, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s, color=SERIES[0], linewidth=0.7, alpha=0.55, label="Daily")
    ax.plot(roll.index, roll, color=BLUE_DARK, linewidth=2, label="30-day average")
    miss = daily.index[daily["is_missing"]]
    ax.plot(miss, np.zeros(len(miss)), "|", color=INK_2, markersize=10, label="Missing day")
    ax.set(title="Daily household electricity consumption, 2006–2010",
           xlabel="Date", ylabel="Consumption (kWh/day)", ylim=(0, None))
    ax.legend(loc="upper right", ncols=3)
    path = save_figure(fig, out / "c01_daily_trend.png")
    winter = s[s.index.month.isin([12, 1, 2])].mean()
    summer = s[s.index.month.isin([6, 7, 8])].mean()
    return _result("daily_trend", "Daily trend", (
        f"Consumption follows a clear yearly cycle: winter days (Dec–Feb) average {winter:.1f} kWh "
        f"vs {summer:.1f} kWh in summer (Jun–Aug), {winter / summer:.1f}× higher; there is no strong long-term trend."),
        path, winter_mean=round(winter, 2), summer_mean=round(summer, 2),
        yearly_mean={int(y): round(v, 2) for y, v in s.groupby(s.index.year).mean().items()})


def weekday_vs_weekend(daily: pd.DataFrame, out: Path) -> dict:
    s = daily["consumption_kwh"].dropna()
    by_dow = s.groupby(s.index.day_of_week).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [SERIES[0]] * 5 + [SERIES[1]] * 2
    bars = ax.bar(WEEKDAYS, by_dow.values, color=colors, width=0.62)
    for b, v in zip(bars, by_dow.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    ax.bar([0], [0], color=SERIES[0], label="Weekday")
    ax.bar([0], [0], color=SERIES[1], label="Weekend")
    ax.set(title="Average consumption by day of week", xlabel="Day of week", ylabel="Mean consumption (kWh/day)")
    ax.legend(loc="lower right")
    path = save_figure(fig, out / "c02_weekday_vs_weekend.png")
    wk, we = s[s.index.day_of_week < 5].mean(), s[s.index.day_of_week >= 5].mean()
    return _result("weekday_vs_weekend", "Weekday vs weekend", (
        f"Weekend days use {we:.1f} kWh on average vs {wk:.1f} kWh on weekdays ({(we / wk - 1) * 100:+.0f}%), "
        f"so day of week is a useful feature; the highest day is {WEEKDAYS[int(by_dow.idxmax())]}."),
        path, weekday_mean=round(wk, 2), weekend_mean=round(we, 2),
        by_weekday={WEEKDAYS[i]: round(v, 2) for i, v in by_dow.items()})


def monthly_pattern(daily: pd.DataFrame, out: Path) -> dict:
    s = daily["consumption_kwh"]
    table = s.groupby([s.index.year, s.index.month]).mean().unstack(0)
    years = [y for y in table.columns if s[s.index.year == y].notna().sum() > 300]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, y in enumerate(years):
        ax.plot(range(1, 13), table[y], marker="o", markersize=4, color=SERIES[i], label=str(y))
        ax.annotate(str(y), (12, table[y].iloc[-1]) if pd.notna(table[y].iloc[-1]) else (11, table[y].iloc[-2]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set(title="Monthly seasonality (mean daily consumption per month)", xlabel="Month",
           ylabel="Mean consumption (kWh/day)", ylim=(0, None))
    ax.legend(loc="lower left", ncols=len(years))
    path = save_figure(fig, out / "c03_monthly_pattern.png")
    by_month = s.groupby(s.index.month).mean()
    low_year = int(table.loc[8, years].idxmin())
    return _result("monthly_pattern", "Monthly seasonality", (
        f"All years share a winter-high, summer-low shape: highest in {MONTHS[int(by_month.idxmax()) - 1]} ({by_month.max():.1f} kWh/day), "
        f"lowest in {MONTHS[int(by_month.idxmin()) - 1]} ({by_month.min():.1f} kWh/day). August varies most between years "
        f"(only {table.loc[8, low_year]:.1f} kWh/day in {low_year}, likely the household away on holiday), a pattern lag features alone cannot foresee."),
        path, by_month={MONTHS[m - 1]: round(v, 2) for m, v in by_month.items()}, years=[int(y) for y in years],
        august_by_year={int(y): round(table.loc[8, y], 2) for y in years})


def submetering_shares(daily: pd.DataFrame, out: Path) -> dict:
    cols = {"sub1_kwh": "Kitchen (sub-meter 1)", "sub2_kwh": "Laundry (sub-meter 2)",
            "sub3_kwh": "Water heater + AC (sub-meter 3)", "unmetered_kwh": "Rest of house (unmetered)"}
    d = daily[list(cols)].dropna()
    monthly = d.groupby(d.index.month).sum()
    shares = monthly.div(monthly.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(12)
    for i, (c, label) in enumerate(cols.items()):
        ax.bar(range(1, 13), shares[c], bottom=bottom, color=SERIES[i], label=label, width=0.7,
               edgecolor="#fcfcfb", linewidth=1.5)
        bottom += shares[c].values
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set(title="Where the energy goes: share of daily consumption by sub-meter", xlabel="Month",
           ylabel="Share of consumption (%)", ylim=(0, 100))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
    path = save_figure(fig, out / "c04_submetering_shares.png")
    overall = (d.sum() / d.sum().sum() * 100).round(1)
    return _result("submetering_shares", "Sub-metering shares", (
        f"The unmetered rest of the house is the largest share ({overall['unmetered_kwh']:.0f}%); the water heater + AC "
        f"meter accounts for {overall['sub3_kwh']:.0f}%, kitchen {overall['sub1_kwh']:.0f}% and laundry {overall['sub2_kwh']:.0f}%."),
        path, overall_share_pct=overall.to_dict())


def missing_data_map(daily: pd.DataFrame, out: Path) -> dict:
    miss = 1 - daily["coverage"]
    years = sorted(set(daily.index.year))
    grid = np.full((len(years), 366), np.nan)
    for ts, v in miss.items():
        grid[years.index(ts.year), ts.day_of_year - 1] = v
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(grid, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(years)), years)
    starts = pd.date_range("2008-01-01", periods=12, freq="MS").day_of_year - 1
    ax.set_xticks(starts, MONTHS)
    ax.grid(False)
    ax.set(title="Missing-data map: share of minutes missing per day", xlabel="Month", ylabel="Year")
    fig.colorbar(im, ax=ax, label="Minutes missing (share)", fraction=0.025)
    path = save_figure(fig, out / "c05_missing_data_map.png")
    n_bad = int((daily["coverage"] < 0.9).sum())
    return _result("missing_data_map", "Missing data", (
        f"Missing data is rare and clustered: {n_bad} of {len(daily)} days fall below 90% coverage, mostly in a few "
        f"multi-day outages (2007, 2009 and several in 2010), so only short gaps were interpolated."),
        path, days_below_90pct=n_bad, total_days=len(daily))


def distribution(daily: pd.DataFrame, out: Path) -> dict:
    s = daily["consumption_kwh"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=45, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.8)
    for q, ls in [(s.median(), "-"), (s.mean(), "--")]:
        ax.axvline(q, color=INK_2, linestyle=ls, linewidth=1.2)
    ax.text(s.median(), ax.get_ylim()[1] * 0.95, f" median {s.median():.1f}", color=INK_2, fontsize=8)
    ax.text(s.mean(), ax.get_ylim()[1] * 0.85, f" mean {s.mean():.1f}", color=INK_2, fontsize=8)
    ax.set(title="Distribution of daily consumption", xlabel="Consumption (kWh/day)", ylabel="Number of days")
    path = save_figure(fig, out / "c06_distribution.png")
    return _result("distribution", "Distribution", (
        f"Daily consumption is right-skewed (skewness {s.skew():.2f}): most days use {s.quantile(0.25):.0f}–{s.quantile(0.75):.0f} kWh, "
        f"with a tail of heavy days up to {s.max():.0f} kWh, so MAE is a more robust headline metric than RMSE."),
        path, skew=round(float(s.skew()), 3), q25=round(s.quantile(0.25), 2), q75=round(s.quantile(0.75), 2),
        max=round(s.max(), 2))


def autocorrelation(daily: pd.DataFrame, out: Path, max_lag: int = 60) -> dict:
    s = daily["consumption_kwh"]
    acf = pd.Series({k: s.autocorr(k) for k in range(1, max_lag + 1)})
    bound = 1.96 / np.sqrt(s.notna().sum())
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [SERIES[1] if k % 7 == 0 else SERIES[0] for k in acf.index]
    ax.bar(acf.index, acf.values, color=colors, width=0.7)
    ax.axhspan(-bound, bound, color=MUTED, alpha=0.25, linewidth=0)
    ax.bar([0], [0], color=SERIES[1], label="Multiples of 7 days")
    ax.set(title="Autocorrelation of daily consumption", xlabel="Lag (days)", ylabel="Correlation with value k days earlier")
    ax.legend(loc="upper right")
    path = save_figure(fig, out / "c07_autocorrelation.png")
    return _result("autocorrelation", "Autocorrelation", (
        f"Yesterday is the strongest single predictor (r = {acf[1]:.2f}), with bumps every 7 days (lag 7 r = {acf[7]:.2f}, "
        f"lag 14 r = {acf[14]:.2f}); this supports lag 1, 2, 7 and 14 features and weekly rolling averages."),
        path, acf={int(k): round(v, 3) for k, v in acf.items() if k in (1, 2, 3, 7, 14, 21, 28, 30, 60)})


def temperature_effect(daily: pd.DataFrame, out: Path, hdd_base_c: float = 18.0) -> dict:
    """Needs the Sceaux weather columns joined onto the daily table."""
    d = daily[["consumption_kwh", "t2m_c"]].dropna()
    wk = d.index.day_of_week < 5
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for mask, color, label in [(wk, SERIES[0], "Weekday"), (~wk, SERIES[1], "Weekend")]:
        ax.scatter(d.loc[mask, "t2m_c"], d.loc[mask, "consumption_kwh"], s=10, alpha=0.5, color=color, label=label, linewidths=0)
    bins = pd.cut(d["t2m_c"], np.arange(-8, 28, 2))
    m = d.groupby(bins, observed=True)["consumption_kwh"].mean()
    ax.plot([b.mid for b in m.index], m.values, color=INK_2, linewidth=2, label="Mean per 2 °C bin")
    ax.axvline(hdd_base_c, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(hdd_base_c, ax.get_ylim()[1] * 0.95, f" heating base {hdd_base_c:.0f} °C", color=INK_2, fontsize=8)
    ax.set(title="Colder days use more electricity", xlabel="Daily mean temperature at Sceaux (°C, NASA POWER)",
           ylabel="Consumption (kWh/day)")
    ax.legend(loc="upper left")
    path = save_figure(fig, out / "c08_temperature_effect.png")
    cold, warm = d.loc[d["t2m_c"] < 5, "consumption_kwh"].mean(), d.loc[d["t2m_c"] > 18, "consumption_kwh"].mean()
    slope = np.polyfit(d.loc[d["t2m_c"] < hdd_base_c, "t2m_c"], d.loc[d["t2m_c"] < hdd_base_c, "consumption_kwh"], 1)[0]
    r_lag = daily["consumption_kwh"].corr(daily["t2m_c"].shift(1))
    return _result("temperature_effect", "Temperature effect", (
        f"Days below 5 °C average {cold:.1f} kWh vs {warm:.1f} kWh above 18 °C; below the heating base each 1 °C colder adds about "
        f"{-slope:.2f} kWh. Yesterday's temperature correlates r = {r_lag:.2f} with consumption, which is why past temperature is now a feature."),
        path, cold_mean=round(cold, 2), warm_mean=round(warm, 2), kwh_per_degC_below_base=round(-slope, 3),
        corr_with_yesterday_temp=round(r_lag, 3))


ALL = [daily_trend, weekday_vs_weekend, monthly_pattern, submetering_shares, missing_data_map, distribution, autocorrelation,
       temperature_effect]
