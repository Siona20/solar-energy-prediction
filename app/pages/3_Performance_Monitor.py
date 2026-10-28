"""Inverter health: actual ÷ out-of-sample expected output, peer comparison, shutdowns vs dust."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("Performance Monitor", "📈")
C.kinds("MEASURED", "MODEL", "ESTIMATE")
st.caption("Performance ratio (PR) = actual ÷ expected energy. Expected comes from a model refitted every day on "
           "**earlier days only**, so no day is judged by a model that saw it. Running PR leaves out the intervals "
           "when the inverter was switched off, which separates shutdowns from slow losses such as dust.")

inv = C.interim("monitoring_inverter_day")
pdays = C.interim("monitoring_plant_day")
names = C.inverter_labels()
cfg = C.D.load_config()["monitoring"]

plant = st.radio("Plant", list(C.PLANTS), format_func=C.PLANTS.get, horizontal=True)
p, iv = pdays[pdays["plant_id"] == plant].sort_values("date"), inv[inv["plant_id"] == plant].copy()
iv["name"] = iv["inverter_id"].map(names)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean plant PR", f"{p['plant_pr'].mean():.3f}", border=True)
c2.metric("Mean running PR", f"{p['plant_running_pr'].mean():.3f}", border=True,
          help="PR while inverters are switched on. Close to 1 with a low plant PR = losses come from shutdowns.")
c3.metric("Inverter-days on alert", int((iv["status"] == "alert").sum()), f"{iv.loc[iv['status'] == 'alert', 'inverter_id'].nunique()} inverters",
          delta_color="off", border=True, help=f"PR < {cfg['ratio_threshold']} for ≥ {cfg['min_consecutive_days']} days in a row.")
c4.metric("Hours switched off in daylight", f"{p['off_hours'].sum():,.0f} h", border=True)

st.subheader("PR of every inverter, every day")
h = iv.pivot_table(index="name", columns="date", values="pr").sort_index(ascending=False)
fig = go.Figure(go.Heatmap(z=h.to_numpy(), x=h.columns, y=h.index, zmin=0.6, zmax=1.4, zmid=1.0,
                           colorscale=[[0, "#a12a2a"], [0.25, "#ec9a95"], [0.5, "#e8e7e3"], [0.75, "#86b6ef"], [1, "#184f95"]],
                           colorbar=dict(title="PR"), hovertemplate="%{y}, %{x|%d %b}: PR %{z:.2f}<extra></extra>"))
fig.update_layout(hovermode="closest")
C.chart(fig, 520)
st.caption("Red = below expected, blue = above. Blank = no data that day (inverter not reporting). Colors stop at 0.6 and 1.4.")

st.subheader("Plant PR vs running PR")
fig = go.Figure()
fig.add_trace(go.Scatter(x=p["date"], y=p["plant_pr"], name="plant PR", mode="lines+markers", line=dict(color=C.MEASURED, width=2)))
fig.add_trace(go.Scatter(x=p["date"], y=p["plant_running_pr"], name="running PR (inverters on)", mode="lines+markers",
                         line=dict(color=C.SECOND, width=2)))
fig.add_hline(y=cfg["ratio_threshold"], line=dict(color=C.BASELINE, dash="dash", width=1))
C.chart(fig, 320, "actual ÷ expected")
fig = go.Figure(go.Bar(x=p["date"], y=p["est_rain_mm"], name="estimated rain", marker_color=C.BASELINE,
                       hovertemplate="%{y:.1f} mm"))
C.chart(fig, 180, "rain (mm)")
st.caption("Rain is an ESTIMATE (NASA POWER at Anantapur, the city whose sunlight best matches the plants); the plants' "
           "real location is not published.")

st.subheader("Inverters on one day")
day = st.select_slider("Day", options=sorted(iv["date"].unique()), value=iv["date"].max(), format_func=lambda d: f"{d:%d %b %Y}")
t = iv[iv["date"] == day].sort_values("pr")
t = t.assign(status=t["status"].map(C.STATUS), off_hours=t["off_intervals"] / 4)
st.dataframe(t[["name", "status", "pr", "running_pr", "peer_z", "low_run_days", "off_hours", "actual_kwh", "expected_kwh"]],
             hide_index=True, column_config={
                 "name": "inverter", "pr": st.column_config.NumberColumn("PR", format="%.2f"),
                 "running_pr": st.column_config.NumberColumn("running PR", format="%.2f"),
                 "peer_z": st.column_config.NumberColumn("vs peers (z)", format="%.1f"),
                 "low_run_days": st.column_config.NumberColumn("days low in a row"),
                 "off_hours": st.column_config.NumberColumn("off in daylight (h)", format="%.2f"),
                 "actual_kwh": st.column_config.NumberColumn("measured (kWh)", format="%,.0f"),
                 "expected_kwh": st.column_config.NumberColumn("expected (kWh)", format="%,.0f")})
st.caption(f"🟠 watch = PR < {cfg['ratio_threshold']} or far below its peers today; 🔴 alert = low for "
           f"{cfg['min_consecutive_days']}+ days in a row. Rows without a value: no data.")
if np.isfinite(t["pr"]).sum() == 0:
    st.info("No inverter data for this day.")
C.footer()
