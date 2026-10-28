"""Measured plant output next to what the saved model expects a healthy inverter to produce."""
import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("Solar Generation", "🔆")
C.kinds("MEASURED", "MODEL")
st.caption("Two utility plants in India (Kaggle), 22 inverters each, 15-minute readings, 15 May – 17 Jun 2020. "
           "Expected output = what a **healthy** inverter of that plant would produce in the measured weather.")

s = C.solar()
names = C.inverter_labels()
c1, c2 = st.columns([1, 2])
plant = c1.radio("Plant", list(C.PLANTS), format_func=C.PLANTS.get, horizontal=True)
invs = sorted(s.loc[s["plant_id"] == plant, "inverter_id"].unique(), key=names.get)
inv = c2.selectbox("Inverter", [None, *invs], format_func=lambda i: "Whole plant (sum of 22 inverters)" if i is None else names[i])

days = sorted(s["date"].unique())
split_of = s.drop_duplicates("date").set_index("date")["split"]
start, end = st.select_slider("Days", options=days, value=(days[-10], days[-1]),
                              format_func=lambda d: f"{d:%d %b} ({split_of[d]})")
chosen = [d for d in days if start <= d <= end]
if any(split_of[d] == "train" for d in chosen):
    st.warning("Some chosen days are training days: the model has seen them, so its fit there is optimistic. "
               "Out-of-sample days are 8–17 June (validation and test).", icon="⚠️")

v = C.D.solar_view(s, plant, inv, chosen)
fig = go.Figure()
fig.add_trace(go.Scatter(x=v["timestamp"], y=v["expected_kw"], name="expected (model)", line=dict(color=C.MODEL, width=1.6),
                         hovertemplate="%{y:,.0f} kW"))
fig.add_trace(go.Scatter(x=v["timestamp"], y=v["actual_kw"], name="actual (measured)", line=dict(color=C.MEASURED, width=1.4),
                         hovertemplate="%{y:,.0f} kW"))
C.chart(fig, 380, "AC power (kW)")

fig = go.Figure(go.Scatter(x=v["timestamp"], y=v["irradiation_kw_m2"], name="irradiation (measured)",
                           line=dict(color=C.SECOND, width=1.2), fill="tozeroy", hovertemplate="%{y:.2f} kW/m²"))
C.chart(fig, 200, "Irradiation (kW/m²)")

d = C.D.solar_daily(s, plant, inv)
d = d[d["date"].isin(chosen)]
tot_a, tot_e = d["actual_kwh"].sum(), d["expected_kwh"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Measured energy", f"{tot_a:,.0f} kWh", border=True)
c2.metric("Expected energy (model)", f"{tot_e:,.0f} kWh", border=True)
c3.metric("Actual ÷ expected", f"{tot_a / tot_e:.2f}", f"{tot_a - tot_e:+,.0f} kWh", delta_color="normal", border=True,
          help="Below 1 = energy lost compared with a healthy inverter. See Performance Monitor for why.")
st.dataframe(d.assign(date=d["date"].dt.date), hide_index=True, column_config={
    "actual_kwh": st.column_config.NumberColumn("measured (kWh)", format="%,.0f"),
    "expected_kwh": st.column_config.NumberColumn("expected (kWh)", format="%,.0f"),
    "ratio": st.column_config.ProgressColumn("actual ÷ expected", min_value=0.0, max_value=1.2, format="%.2f")})
C.footer()
