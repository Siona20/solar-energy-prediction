"""Rule-based maintenance recommendations, each with its action, reason and evidence."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("Maintenance", "🛠️")
C.kinds("MODEL", "ESTIMATE")
st.caption("Rules R0–R7 turn the monitoring results into actions. Every recommendation says why and shows the numbers "
           "behind it. Rain-related advice uses estimated rain (NASA POWER near the plants).")

r = C.maintenance()
names = C.inverter_labels()
r = r.assign(target_name=r["target"].map(lambda t: names.get(t, t)))

c1, c2 = st.columns([1, 2])
plant = c1.radio("Plant", list(C.PLANTS), format_func=C.PLANTS.get, horizontal=True)
rp = r[r["plant_id"] == plant]
days = sorted(rp["date"].unique())
day = c2.select_slider("Day", options=days, value=days[-1], format_func=lambda d: f"{d:%d %b %Y}")

today = rp[rp["date"] == day]
order = {"high": 0, "medium": 1, "low": 2, "info": 3}
today = today.sort_values("severity", key=lambda s: s.map(order))
st.subheader(f"{C.PLANTS[plant]}, {pd.Timestamp(day):%d %B %Y}: {len(today)} recommendation(s)")
for row in today.itertuples():
    with st.container(border=True):
        a, b = st.columns([1, 5])
        a.markdown(f"**{C.SEVERITY.get(row.severity, row.severity)}**")
        a.caption(row.rule)
        b.markdown(f"**{row.target_name}**: {row.action}")
        b.write(row.reason)
        with b.expander("Evidence"):
            st.json(row.evidence)

st.subheader("Recommendations over the monitored period")
counts = rp[rp["rule"] != "R0 no_action"].groupby(["date", "rule"]).size().unstack(fill_value=0)
fig = go.Figure()
for k, rule in enumerate(counts.columns):
    fig.add_trace(go.Bar(x=counts.index, y=counts[rule], name=rule, marker_color=[C.MEASURED, C.MODEL, C.SECOND, "#eda100"][k % 4]))
fig.update_layout(barmode="stack", bargap=0.15)
C.chart(fig, 320, "recommendations per day")

rules = st.multiselect("Rules", sorted(rp["rule"].unique()), default=[x for x in sorted(rp["rule"].unique()) if x != "R0 no_action"])
h = rp[rp["rule"].isin(rules)].sort_values(["date", "severity"], ascending=[False, True])
st.dataframe(h.assign(date=h["date"].dt.date, severity=h["severity"].map(C.SEVERITY))[
    ["date", "target_name", "rule", "severity", "action", "reason"]], hide_index=True,
    column_config={"target_name": "target"})

with st.expander("What each rule means"):
    st.markdown("""
- **R0 no_action**: everything is within the expected range.
- **R1 inverter_offline**: an inverter was switched off for at least an hour of daylight.
- **R2 inverter_underperforming**: one inverter low for 3+ days while the rest of the plant is fine. Shutdowns → read its event log; steady shortfall → inspect its strings.
- **R3 plant_wide_loss**: 60 % or more of the inverters low for 3+ days in a row; recent rain decides between "not dust" and cleaning.
- **R4 gradual_decline**: running PR (inverters on) falling steadily over the last days with no washing rain (soiling-like); trips never trigger it.
- **R5 sensor_check**: PR far above 1, the irradiance sensor may be wrong.
- **R6 seasonal_cleaning**: information only: the estimated location is in a dry month (no rain to wash the panels), so clean on a schedule.
- **R7 not_reporting**: an inverter sent no data.
""")
C.footer()
