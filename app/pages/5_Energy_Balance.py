"""Rooftop-solar scenario for an Indian home: monthly estimate, direct use, export and grid import."""
import math

import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("Energy Balance", "⚖️")
C.kinds("ESTIMATE")
st.warning("Everything on this page is an **ESTIMATE** for a scenario home, not a measurement of a real one. "
           "Solar = kWp × long-term NASA POWER sunlight (2001–2025) × performance ratio.", icon="⚠️")

loc = C.locations()
typical = C.D.typical_use_by_type()
cfg = C.D.load_config()["energy_balance"]

c1, c2 = st.columns(2)
city = c1.selectbox("City", loc["location"], index=int(loc.index[loc["location"] == "pune"][0]),
                    format_func=dict(zip(loc["location"], loc["label"])).get)
htype = c2.selectbox("Household type (sets the daily load shape)", list(typical),
                     help="Load shape = when in the day the home uses electricity, measured on eMARC homes of this type.")
c1, c2, c3 = st.columns(3)
daily = c1.number_input("Electricity use (kWh per day)", 0.5, 60.0, float(typical[htype]), 0.1,
                        help=f"Default: median of eMARC '{htype}' homes ({typical[htype]:.2f} kWh/day).")
pr = c3.slider("Performance ratio", 0.60, 0.90, float(cfg["default_performance_ratio"]), 0.01,
               help="Share of the panels' rated output that reaches the home after heat, dust, wiring and inverter losses.")
need = C.D.energy_scenario(city, 1.0, daily, pr, htype)[1]["kwp_to_cover_consumption"]
kwp = c2.number_input("System size (kWp)", 0.5, 20.0, max(0.5, math.ceil(need * 2) / 2), 0.5,
                      help=f"Default: smallest size (in 0.5 kWp steps) that covers the yearly use: {need:.2f} kWp.")

m, a = C.D.energy_scenario(city, kwp, daily, pr, htype)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Solar per year", f"{a['solar_kwh']:,.0f} kWh", f"use {a['consumption_kwh']:,.0f} kWh", delta_color="off", border=True)
c2.metric("Use covered directly by the panels", f"{a['self_sufficiency_pct']:.0f} %", border=True,
          help="Self-sufficiency: solar used at the moment it is produced ÷ the home's use.")
c3.metric("Panel output used in the home", f"{a['self_consumption_pct']:.0f} %", f"{a['export_kwh']:,.0f} kWh exported",
          delta_color="off", border=True, help="Self-consumption. The rest goes to the grid (no battery).")
c4.metric("Size that covers the yearly use", f"{a['kwp_to_cover_consumption']:.2f} kWp", border=True)

st.subheader("Month by month (kWh per day)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=m["month_name"], y=m["solar_high_kwh_day"], line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=m["month_name"], y=m["solar_low_kwh_day"], fill="tonexty", fillcolor=C.BAND, line=dict(width=0),
                         name="solar, poor to good month (10th–90th pct)", hovertemplate="%{y:.1f}"))
fig.add_trace(go.Scatter(x=m["month_name"], y=m["solar_kwh_day"], name="solar, typical month", mode="lines+markers",
                         line=dict(color=C.MODEL, width=2), hovertemplate="%{y:.1f} kWh"))
fig.add_trace(go.Scatter(x=m["month_name"], y=m["consumption_kwh_day"], name="home's use", mode="lines",
                         line=dict(color=C.MEASURED, width=2, dash="dash"), hovertemplate="%{y:.1f} kWh"))
C.chart(fig, 340, "kWh per day")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Where the solar goes**")
    fig = go.Figure([go.Bar(x=m["month_name"], y=m["direct_use_kwh_day"], name="used in the home", marker_color=C.MODEL),
                     go.Bar(x=m["month_name"], y=m["export_kwh_day"], name="exported to the grid", marker_color=C.BASELINE)])
    fig.update_layout(barmode="stack", bargap=0.2)
    C.chart(fig, 300, "kWh per day")
with c2:
    st.markdown("**Where the home's electricity comes from**")
    fig = go.Figure([go.Bar(x=m["month_name"], y=m["direct_use_kwh_day"], name="own panels", marker_color=C.MODEL),
                     go.Bar(x=m["month_name"], y=m["grid_import_kwh_day"], name="grid", marker_color=C.MEASURED)])
    fig.update_layout(barmode="stack", bargap=0.2)
    C.chart(fig, 300, "kWh per day")

with st.expander("How this is calculated, and its limits"):
    for n in a["notes"]:
        st.markdown(f"- {n}")
    st.markdown("- Same daily use every day of the year; real homes use more in summer (cooling) or winter (water heating).\n"
                "- No battery, no tariffs or net-metering rules.")
C.footer()
