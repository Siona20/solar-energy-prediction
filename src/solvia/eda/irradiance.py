"""EDA figures for Indian irradiance and rainfall (context for the energy balance and cleaning rules)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from solvia.eda.consumption import MONTHS, _result
from solvia.viz import INK_2, SERIES, save_figure


def district_ranking(districts: pd.DataFrame, out: Path, n: int = 15) -> dict:
    d = districts.assign(label=districts["district"] + ", " + districts["state"]).sort_values("insolation_kwh_m2_day")
    show = pd.concat([d.head(n), d.tail(n)])
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [SERIES[0]] * len(show)
    ax.barh(range(len(show)), show["insolation_kwh_m2_day"], color=colors, height=0.7)
    ax.axhline(n - 0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.text(0.02, n - 0.2, f"▲ top {n}   ▼ bottom {n} of {len(d)} districts", fontsize=8, color=INK_2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(show)), show["label"], fontsize=7)
    ax.set(title="Indian districts ranked by solar potential (December 2024 only)",
           xlabel="Mean daily insolation in Dec 2024 (kWh/m²/day)")
    path = save_figure(fig, out / "i01_district_ranking.png")
    top, bottom = d.iloc[-1], d.iloc[0]
    return _result("district_ranking", "District ranking", (
        f"In December 2024 the sunniest district was {top['district']} ({top['state']}, {top['insolation_kwh_m2_day']:.2f} kWh/m²/day) "
        f"and the least sunny {bottom['district']} ({bottom['state']}, {bottom['insolation_kwh_m2_day']:.2f}); one winter month only, so use it to rank, not as an annual value."),
        path, n_districts=len(d), top=top["district"], bottom=bottom["district"])
