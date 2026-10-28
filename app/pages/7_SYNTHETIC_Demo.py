"""SYNTHETIC demo: PV360 full-year home PV data of unverified origin, shown only to test layouts."""
import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("SYNTHETIC demo (PV360)", "🧩")
C.kinds("SYNTHETIC")
st.error("**SYNTHETIC / UNVERIFIED DATA.** PV360 claims to be one year of a 5.4 kWp home system in the Philippines, "
         "but no public source could be found (see `reports/data_audit.md`). It is used here only to test how a "
         "full-year home view looks. It is not used in any model, result or report.", icon="🚫")

d = C.pv360()
c1, c2, c3 = st.columns(3)
c1.metric("PV per year (synthetic)", f"{d['pv_kwh'].sum():,.0f} kWh", border=True)
c2.metric("Use per year (synthetic)", f"{d['consumption_kwh'].sum():,.0f} kWh", border=True)
c3.metric("Days", f"{len(d)}", border=True)

w = d.set_index("date")[["pv_kwh", "consumption_kwh"]].rolling(7, min_periods=4).mean().reset_index()
fig = go.Figure([go.Scatter(x=w["date"], y=w["pv_kwh"], name="PV (7-day mean)", line=dict(color=C.MODEL, width=2)),
                 go.Scatter(x=w["date"], y=w["consumption_kwh"], name="use (7-day mean)", line=dict(color=C.MEASURED, width=2))])
fig.add_annotation(text="SYNTHETIC", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                   font=dict(size=56, color="rgba(208, 59, 59, 0.15)"))
C.chart(fig, 380, "kWh per day (synthetic)")
st.caption("Source file: `data/raw/unverified/pv360_philippines/PV360_data.zip`, read without changes.")
C.footer()
