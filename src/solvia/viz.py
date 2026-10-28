"""Shared figure style. Colors follow the validated reference palette (dataviz skill).

Color follows the entity: Plant 1 is always slot 1 (blue), Plant 2 always slot 2 (orange).
Status red is reserved for faults and always comes with a text label.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
NEUTRAL = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_DARK = "#1c5cab"
CRITICAL = "#d03b3b"
MUTED = "#a3a29c"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#a12a2a", "#d03b3b", "#ec9a95", NEUTRAL, "#86b6ef", "#2a78d6", "#184f95"])
PLANT_COLORS = {4135001: SERIES[0], 4136001: SERIES[1]}
PLANT_NAMES = {4135001: "Plant 1", 4136001: "Plant 2"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.6,
        "font.family": "sans-serif", "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    log.info("figure saved: %s", path.name)
    return path
