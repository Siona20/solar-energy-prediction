"""Shared helpers for the SOLVIA dashboard pages: page setup, labels, cached data, chart defaults."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ and app/ are on sys.path regardless of execution working directory
_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
_APP = str(_ROOT / "app")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import plotly.graph_objects as go
import streamlit as st

from solvia.dashboard import data as D

# Color follows what a number is, on every page (reference palette, see src/solvia/viz.py).
MEASURED = "#2a78d6"
MODEL = "#eb6834"
BASELINE = "#a3a29c"
SECOND = "#1baf7a"
BAND = "rgba(235, 104, 52, 0.18)"
KIND_BADGE = {"MEASURED": "blue", "MODEL": "orange", "ESTIMATE": "violet", "SYNTHETIC": "red"}
STATUS = {"ok": "🟢 ok", "watch": "🟠 watch", "alert": "🔴 alert"}
SEVERITY = {"info": "🟢 info", "low": "🟡 low", "medium": "🟠 medium", "high": "🔴 high"}
PLANTS = {4135001: "Plant 1", 4136001: "Plant 2"}


def setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · SOLVIA", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def kinds(*names: str) -> None:
    """Badges saying what kind of numbers the section shows, with the meaning underneath."""
    cols = st.columns(len(names) + 1)
    for c, n in zip(cols, names):
        with c:
            st.badge(n, color=KIND_BADGE[n])
    st.caption(" · ".join(f"**{n}**: {D.KINDS[n]}" for n in names))


def chart(fig: go.Figure, height: int = 380, y_title: str | None = None, x_title: str | None = None) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    st.plotly_chart(fig, width="stretch")


def footer() -> None:
    st.divider()
    st.caption("Data: Prayas (Energy Group) eMARC smart meters (non-commercial academic use, credit Prayas); Kaggle "
               "Solar Power Generation Data; NASA POWER (NASA Langley Research Center); UCI Individual household "
               "electric power consumption. SOLVIA, B.Tech CSE final-year project.")


@st.cache_data(show_spinner="Loading forecasts…")
def consumption(track: str):
    return D.consumption_replay(track)


@st.cache_data(show_spinner="Loading meters…")
def meters():
    return D.india_meters(D.consumption_replay("india"))


@st.cache_data(show_spinner="Loading solar data…")
def solar():
    return D.solar_replay()


@st.cache_data
def json_(name: str) -> dict:
    return D.metrics_json(name)


@st.cache_data
def csv_(name: str):
    return D.metrics_csv(name)


@st.cache_data
def interim(name: str):
    return D.interim(name)


@st.cache_data
def inverter_labels() -> dict:
    return D.inverter_labels()


@st.cache_data
def maintenance():
    return D.maintenance()


@st.cache_data
def locations():
    return D.locations()


@st.cache_data
def pv360():
    return D.pv360_daily()
