"""India-first EDA: solar potential of 50 cities (NASA POWER 2001-2025), validation against NITI,
and an estimate of where the Kaggle plants are, by matching their daily sunlight to each city."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solvia.eda.consumption import MONTHS, _result
from solvia.viz import INK_2, MUTED, PLANT_COLORS, PLANT_NAMES, SEQUENTIAL, SERIES, save_figure

def _label(loc: str) -> str:
    return loc.replace("_", " ").title()


def city_ranking(climate: pd.DataFrame, out: Path) -> dict:
    c = climate.sort_values("ghi_kwh_m2_day")
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(range(len(c)), c["ghi_kwh_m2_day"], color=SERIES[0], height=0.7)
    for i, (_, r) in enumerate(c.iterrows()):
        ax.text(r["ghi_kwh_m2_day"] + 0.03, i, f"{r['ghi_kwh_m2_day']:.2f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(c)), [f"{_label(l)} ({s})" for l, s in zip(c["location"], c["state"])], fontsize=7)
    ax.set(title=f"Solar potential of {len(c)} Indian cities (NASA POWER, 2001–2025 average)",
           xlabel="Mean daily global horizontal irradiance (kWh/m²/day)", xlim=(0, c["ghi_kwh_m2_day"].max() * 1.12))
    path = save_figure(fig, out / "i03_city_ranking_annual.png")
    top, bottom = climate.iloc[0], climate.iloc[-1]
    by_region = climate.groupby("region")["ghi_kwh_m2_day"].mean().sort_values(ascending=False).round(2)
    return _result("city_ranking", "City ranking (25-year average)", (
        f"Over 25 years the sunniest city is {_label(top['location'])} ({top['ghi_kwh_m2_day']:.2f} kWh/m²/day) and the least sunny "
        f"{_label(bottom['location'])} ({bottom['ghi_kwh_m2_day']:.2f}); the best region on average is {by_region.index[0]} "
        f"({by_region.iloc[0]:.2f}) and the weakest {by_region.index[-1]} ({by_region.iloc[-1]:.2f}). A 1 kWp system in the top city "
        f"gets {top['ghi_kwh_m2_day'] / bottom['ghi_kwh_m2_day']:.2f}× the sunlight of one in the bottom city."),
        path, top=top["location"], bottom=bottom["location"], by_region=by_region.to_dict())


def monthly_heatmap(clim: pd.DataFrame, climate: pd.DataFrame, out: Path) -> dict:
    order = climate.sort_values("ghi_kwh_m2_day", ascending=False)["location"]
    mat = clim.pivot(index="location", columns="month", values="ghi_kwh_m2_day").loc[order]
    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(mat.values, aspect="auto", cmap=SEQUENTIAL, interpolation="nearest")
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(mat)), [_label(l) for l in mat.index], fontsize=7)
    ax.grid(False)
    ax.set(title="Monthly solar resource by city (sorted by annual average)", xlabel="Month")
    fig.colorbar(im, ax=ax, label="Mean GHI (kWh/m²/day)", fraction=0.03)
    path = save_figure(fig, out / "i04_monthly_ghi_heatmap.png")
    swing = (mat.max(axis=1) - mat.min(axis=1)).sort_values()
    c = climate.set_index("location")
    spring_best = int(c["best_month"].between(3, 5).sum())
    monsoon_dark = int(c["worst_month"].between(6, 9).sum())
    winter_dark = int(c["worst_month"].isin([11, 12, 1]).sum())
    return _result("monthly_heatmap", "Monthly heatmap", (
        f"{spring_best} of {len(c)} cities have their sunniest month in March–May. {monsoon_dark} are darkest during the monsoon (Jun–Sep) and "
        f"{winter_dark} in Nov–Jan. The seasonal swing ranges from {swing.iloc[0]:.1f} kWh/m²/day ({_label(swing.index[0])}) to "
        f"{swing.iloc[-1]:.1f} ({_label(swing.index[-1])}), so a monthly estimate is far more honest than one annual number."),
        path, spring_best=spring_best, monsoon_darkest=monsoon_dark, winter_darkest=winter_dark,
        swing_min=swing.index[0], swing_max=swing.index[-1])


def rain_and_dry_months(climate: pd.DataFrame, out: Path) -> dict:
    c = climate.sort_values("rain_mm_year")
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(range(len(c)), c["rain_mm_year"], color=SERIES[0], height=0.7)
    for i, (_, r) in enumerate(c.iterrows()):
        ax.text(r["rain_mm_year"] + 30, i, f"{int(r['dry_months'])} dry", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(c)), [_label(l) for l in c["location"]], fontsize=7)
    ax.set(title="Annual rainfall and number of dry months (< 10 mm)", xlabel="Mean annual rainfall (mm/year, NASA POWER 2001–2025)",
           xlim=(0, c["rain_mm_year"].max() * 1.15))
    path = save_figure(fig, out / "i05_rain_and_dry_months.png")
    dry = climate.sort_values("dry_months", ascending=False).iloc[0]
    return _result("rain_and_dry_months", "Rain and dry months", (
        f"Rainfall ranges from {c['rain_mm_year'].iloc[0]:.0f} mm/year ({_label(c['location'].iloc[0])}) to {c['rain_mm_year'].iloc[-1]:.0f} "
        f"({_label(c['location'].iloc[-1])}); a median of {climate['share_rain_jun_sep'].median() * 100:.0f}% falls in Jun–Sep. Cities with many dry months "
        f"(e.g. {_label(dry['location'])}: {int(dry['dry_months'])}) get no natural panel washing for most of the year, which is when the cleaning rule matters."),
        path, median_share_jun_sep=round(float(climate["share_rain_jun_sep"].median()), 3))


def nasa_vs_niti(daily: pd.DataFrame, districts: pd.DataFrame, kheda: pd.DataFrame, out: Path) -> dict:
    """Validate NASA POWER against NITI ICED where both report the same place and period."""
    dec = daily[(daily["date"].dt.year == 2024) & (daily["date"].dt.month == 12)]
    nasa_dec = dec.groupby("location")["ghi_kwh_m2_day"].mean()
    dist = districts.assign(key=districts["district"].str.lower().str.replace(" ", "_"))
    alias = {"bengaluru": "bengaluru_urban", "kheda": "kheda", "port_blair": "south_andaman"}
    pairs = []
    for loc, v in nasa_dec.items():
        hit = dist[dist["key"] == alias.get(loc, loc)]
        if len(hit) == 1:
            pairs.append({"location": loc, "nasa": v, "niti": float(hit["insolation_kwh_m2_day"].iloc[0])})
    pairs = pd.DataFrame(pairs)
    k = daily[(daily["location"] == "kheda") & daily["date"].dt.year.between(2009, 2023)]
    nasa_kheda = k.groupby(k["date"].dt.month)["ghi_kwh_m2_day"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.scatter(pairs["nasa"], pairs["niti"], s=30, color=SERIES[0])
    lim = (0, max(pairs["nasa"].max(), pairs["niti"].max()) * 1.1)
    ax1.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(lim[1] * 0.97, lim[1] * 0.93, "equal", color=INK_2, fontsize=8, ha="right")
    ax1.set(title=f"December 2024, {len(pairs)} matching districts", xlabel="NASA POWER (kWh/m²/day)",
            ylabel="NITI ICED (kWh/m²/day)", xlim=lim, ylim=lim)
    ax2.plot(range(1, 13), nasa_kheda.values, marker="o", markersize=4, color=SERIES[0], label="NASA POWER")
    ax2.plot(kheda["month"], kheda["insolation_kwh_m2_day"], marker="s", markersize=4, color=SERIES[1], label="NITI ICED")
    ax2.set_xticks(range(1, 13), MONTHS)
    ax2.set(title="Kheda, Gujarat: monthly average 2009–2023", xlabel="Month", ylabel="GHI (kWh/m²/day)", ylim=(0, None))
    ax2.legend(loc="lower left")
    fig.suptitle("Do NASA POWER and NITI Aayog agree?", x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "i06_nasa_vs_niti.png")
    dec_ratio = float((pairs["niti"] / pairs["nasa"]).median())
    kheda_r = float(np.corrcoef(nasa_kheda.values, kheda.sort_values("month")["insolation_kwh_m2_day"].values)[0, 1])
    kheda_bias = float((kheda.sort_values("month")["insolation_kwh_m2_day"].values / nasa_kheda.values).mean())
    rank_r = float(pairs["nasa"].rank().corr(pairs["niti"].rank()))
    return _result("nasa_vs_niti", "NASA vs NITI", (
        f"For the long-term Kheda averages the two sources agree on the seasonal shape (r = {kheda_r:.2f}) and NITI is {kheda_bias:.2f}× NASA on average. "
        f"For December 2024 across {len(pairs)} districts NITI is only {dec_ratio:.2f}× NASA (rank agreement {rank_r:.2f}), confirming the audit finding "
        f"that the single-month NITI file reads low; NASA POWER is used for the energy balance."),
        path, matched_districts=len(pairs), dec2024_niti_over_nasa_median=round(dec_ratio, 3),
        dec2024_rank_corr=round(rank_r, 3), kheda_monthly_corr=round(kheda_r, 3), kheda_niti_over_nasa=round(kheda_bias, 3))


def plant_location_match(match: pd.DataFrame, climate: pd.DataFrame, out: Path) -> dict:
    """Map of how well each city's daily sunlight matches each plant's sensor (match from preprocessing)."""
    match = match.merge(climate[["location", "latitude", "longitude", "state"]], on="location")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, (pid, g) in zip(axes, match.groupby("plant_id")):
        sc = ax.scatter(g["longitude"], g["latitude"], c=g["r_ghi"].clip(lower=0), cmap=SEQUENTIAL, vmin=0, vmax=0.8,
                        s=70, edgecolors=MUTED, linewidths=0.5)
        best = g.nlargest(3, "r_ghi")
        for _, r in best.iterrows():
            ax.annotate(f"{_label(r['location'])} r={r['r_ghi']:.2f}", (r["longitude"], r["latitude"]),
                        xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK_2)
        ax.set(title=PLANT_NAMES[pid], xlabel="Longitude (°E)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(sc, ax=axes, label="Correlation of daily sunlight with the plant (15 May–17 Jun 2020)", fraction=0.025)
    fig.suptitle("Where are the Kaggle plants? Darker = city whose daily sunlight matches the plant's sensor (estimate)",
                 x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "i07_plant_location_match.png")
    best = match.loc[match.groupby("plant_id")["r_ghi"].idxmax()].set_index("plant_id")
    med = match.groupby("plant_id")["r_ghi"].median()
    top5 = {PLANT_NAMES[p]: g.nlargest(5, "r_ghi")["location"].tolist() for p, g in match.groupby("plant_id")}
    b1, b2 = best.loc[4135001], best.loc[4136001]
    result = _result("plant_location_match", "Estimated plant location", (
        f"Both plants match the southern Deccan plateau best: Plant 1 → {_label(b1['location'])} (r = {b1['r_ghi']:.2f}, temperature r = {b1['r_temp']:.2f}), "
        f"Plant 2 → {_label(b2['location'])} (r = {b2['r_ghi']:.2f}, temperature r = {b2['r_temp']:.2f}), against a median of {med.loc[4135001]:.2f} / "
        f"{med.loc[4136001]:.2f} over all {match['location'].nunique()} cities. This is an estimate at city resolution, not a confirmed site; NASA rainfall at the best match "
        f"({b2['rain_mm_window']:.0f} mm over the 34 days) is used only as labelled 'estimated rainfall'."),
        path, best={PLANT_NAMES[p]: r["location"] for p, r in best.iterrows()}, top5=top5,
        best_r={PLANT_NAMES[p]: round(float(r["r_ghi"]), 3) for p, r in best.iterrows()},
        median_r={PLANT_NAMES[p]: round(float(v), 3) for p, v in med.items()})
    return result


def plant_grid_map(grid_match: pd.DataFrame, out: Path) -> dict:
    """1° grid version of i07: correlation surface and the r-weighted centre of the 5 best cells."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    est = {}
    for ax, (pid, g) in zip(axes, grid_match.groupby("plant_id")):
        mat = g.pivot(index="latitude", columns="longitude", values="r_ghi")
        mesh = ax.pcolormesh(mat.columns, mat.index, mat.clip(lower=0).values, cmap=SEQUENTIAL, vmin=0, vmax=0.8, shading="nearest")
        top = g.nsmallest(5, "rank")
        ax.scatter(top["longitude"], top["latitude"], s=40, facecolors="none", edgecolors="white", linewidths=1.5, label="5 best cells")
        lat, lon = g["est_latitude"].iloc[0], g["est_longitude"].iloc[0]
        ax.scatter([lon], [lat], marker="X", s=120, color=PLANT_COLORS[pid], edgecolors="white", linewidths=1.2, zorder=4,
                   label=f"Estimate {lat:.1f}°N {lon:.1f}°E")
        ax.set(title=PLANT_NAMES[pid], xlabel="Longitude (°E)")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=8, labelcolor=INK_2)
        best = g.iloc[0]
        est[PLANT_NAMES[pid]] = {"est_latitude": round(float(lat), 2), "est_longitude": round(float(lon), 2),
                                 "best_cell": best["location"], "best_r": round(float(best["r_ghi"]), 3),
                                 "cells_within_0.05": int((g["r_ghi"] > best["r_ghi"] - 0.05).sum()),
                                 "median_r": round(float(g["r_ghi"].median()), 3)}
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(mesh, ax=axes, label="Correlation of daily sunlight with the plant (15 May–17 Jun 2020)", fraction=0.025)
    fig.suptitle("Plant location on NASA's native 1° grid (estimate, not a confirmed site)", x=0.06, ha="left", fontweight="semibold")
    path = save_figure(fig, out / "i08_plant_location_grid.png")
    p1, p2 = est["Plant 1"], est["Plant 2"]
    return _result("plant_location_grid", "Estimated plant location on a 1° grid", (
        f"On {grid_match['location'].nunique()} 1° cells (10–22°N, 72–84°E) the r-weighted centre of the 5 best cells is "
        f"{p1['est_latitude']:.1f}°N {p1['est_longitude']:.1f}°E for Plant 1 (best r = {p1['best_r']:.2f}, {p1['cells_within_0.05']} cells within 0.05) and "
        f"{p2['est_latitude']:.1f}°N {p2['est_longitude']:.1f}°E for Plant 2 (best r = {p2['best_r']:.2f}, {p2['cells_within_0.05']} cells within 0.05), "
        "both in the southern interior Deccan and consistent with the city match (i07). Plant 1 is the less certain of the two; "
        "34 days cannot pin a site more precisely than a region."), path, estimates=est)


def climate_lines(clim: pd.DataFrame, out: Path, cities: tuple[str, ...] = ("jaisalmer", "delhi", "anantapur")) -> dict:
    """Sun and rain through the year for three contrasting cities (desert, north, Deccan)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    stats = {}
    for i, loc in enumerate(cities):
        c = clim[clim["location"] == loc].sort_values("month")
        ax1.plot(c["month"], c["ghi_kwh_m2_day"], marker="o", markersize=4, color=SERIES[i], label=_label(loc))
        ax2.bar(c["month"] + (i - 1) * 0.27, c["rain_mm_month"], width=0.27, color=SERIES[i], label=_label(loc))
        stats[loc] = {"best_month": MONTHS[int(c.loc[c["ghi_kwh_m2_day"].idxmax(), "month"]) - 1],
                      "worst_month": MONTHS[int(c.loc[c["ghi_kwh_m2_day"].idxmin(), "month"]) - 1],
                      "rain_mm_year": round(float(c["rain_mm_month"].sum()), 0)}
    ax1.set(title="Through the year: sunshine (top) and rainfall (bottom), 2001–2025 averages", ylabel="GHI (kWh/m²/day)", ylim=(0, None))
    ax2.set(xlabel="Month", ylabel="Rainfall (mm/month)")
    ax2.set_xticks(range(1, 13), MONTHS)
    ax1.legend(loc="lower left")
    path = save_figure(fig, out / "i02_monthly_sun_and_rain.png")
    j, d, a = (stats[c] for c in cities)
    return _result("monthly_climate", "Monthly sun and rain", (
        f"Jaisalmer (desert, {j['rain_mm_year']:.0f} mm/year), Delhi ({d['rain_mm_year']:.0f} mm) and Anantapur ({a['rain_mm_year']:.0f} mm, near the estimated "
        f"plant location) all peak in spring ({j['best_month']}, {d['best_month']}, {a['best_month']}), lose sunshine when the monsoon rain arrives, and are darkest "
        f"in early winter ({j['worst_month']}, {d['worst_month']}, {a['worst_month']}). The sunny, nearly rainless spring is when dust builds up unwashed."),
        path, **stats)
