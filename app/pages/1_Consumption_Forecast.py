"""Next-day household consumption: replay the saved model's forecasts on days it never trained on."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C

C.setup("Consumption Forecast", "🏠")
C.kinds("MEASURED", "MODEL")

track = st.radio("Households", ["India (Prayas eMARC)", "France (UCI, one home)"], horizontal=True,
                 help="India first. The two are separate tracks and are never combined.")
key = "india" if track.startswith("India") else "france"
r = C.consumption(key)
model, baseline = r.attrs["model"], r.attrs["baseline"]
split = st.segmented_control("Days", ["Test", "Validation", "Both"], default="Test",
                             help="Both splits are out-of-sample. Test was scored once in Phase 5.") or "Test"

if key == "india":
    m = C.meters()
    m = m[m["test_days"] > 0] if split == "Test" else m[m["valid_days"] > 0] if split == "Validation" else m
    c1, c2, c3 = st.columns(3)
    region = c1.selectbox("Region", sorted(m["region"].unique()), index=sorted(m["region"].unique()).index("Pune city"))
    types = sorted(m.loc[m["region"] == region, "household_type"].unique())
    htype = c2.selectbox("Household type", types)
    mm = m[(m["region"] == region) & (m["household_type"] == htype)]
    label = {row.deployment_id: f"{row.deployment_id} · {row.mean_kwh:.1f} kWh/day avg"
             + ("" if row.seen_in_training else " · new meter") for row in mm.itertuples()}
    meter = c3.selectbox("Meter (one home)", list(label), format_func=label.get)
    r = r[r["deployment_id"] == meter]
    st.caption("Each meter is its own series: the forecast for this home uses only this home's own history, its type and "
               "region, the calendar and yesterday's weather. Homes are never added together."
               + ("" if mm.set_index("deployment_id").loc[meter, "seen_in_training"]
                  else " **This meter was not in the training data**: the model has never seen this home."))

sel = r if split == "Both" else r[r["split"] == ("test" if split == "Test" else "valid")]
if sel.empty:
    st.warning("This meter has no days in the chosen split.")
    st.stop()
show_base = st.toggle(f"Show the best simple baseline ({baseline.replace('_', ' ')})", value=False)

g = sel.set_index("date")[["actual_kwh", "forecast_kwh", "low_kwh", "high_kwh", "baseline_kwh"]].asfreq("D").reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=g["date"], y=g["high_kwh"], line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=g["date"], y=g["low_kwh"], fill="tonexty", fillcolor=C.BAND, line=dict(width=0),
                         name="80 % interval (model)", hovertemplate="%{y:.2f}"))
fig.add_trace(go.Scatter(x=g["date"], y=g["actual_kwh"], name="actual (measured)", line=dict(color=C.MEASURED, width=1.6),
                         hovertemplate="%{y:.2f} kWh"))
fig.add_trace(go.Scatter(x=g["date"], y=g["forecast_kwh"], name=f"forecast ({model})", line=dict(color=C.MODEL, width=1.6),
                         hovertemplate="%{y:.2f} kWh"))
if show_base:
    fig.add_trace(go.Scatter(x=g["date"], y=g["baseline_kwh"], name="baseline", line=dict(color=C.BASELINE, width=1.2, dash="dot"),
                             hovertemplate="%{y:.2f} kWh"))
if key == "india" and sel["date"].max() >= pd.Timestamp("2020-03-25"):
    fig.add_vline(x=pd.Timestamp("2020-03-25"), line=dict(color=C.BASELINE, dash="dash", width=1))
    fig.add_annotation(x=pd.Timestamp("2020-03-25"), y=1, yref="paper", text="lockdown", showarrow=False, xanchor="left")
C.chart(fig, 400, "kWh per day")
st.caption("Each forecast is made the evening before, from data up to that day. Gaps are days without a valid reading.")

e = C.D.error_summary(sel)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model error (MAE)", f"{e['mae_model']:.2f} kWh/day", border=True)
c2.metric("Baseline error (MAE)", f"{e['mae_baseline']:.2f} kWh/day",
          f"model {100 * (1 - e['mae_model'] / e['mae_baseline']):+.0f} %", delta_color="off", border=True)
c3.metric("Days inside the 80 % interval", f"{e['coverage']:.0%}", f"{e['days']} days", delta_color="off", border=True)
c4.metric("Average use", f"{e['mean_kwh']:.2f} kWh/day", border=True)

last = sel.sort_values("date").iloc[-1]
st.subheader(f"Forecast for {last['date']:%d %b %Y} (last day in the data)")
st.write(f"Forecast **{last['forecast_kwh']:.2f} kWh** (80 % interval {last['low_kwh']:.2f}–{last['high_kwh']:.2f}); "
         f"the meter measured **{last['actual_kwh']:.2f} kWh**.")

if key == "india" and split != "Validation":
    t = sel[sel["split"] == "test"]
    if t["period"].nunique() == 2:
        st.subheader("Normal days vs lockdown (this home, test)")
        rows = [{"period": p, **C.D.error_summary(g_)} for p, g_ in t.groupby("period")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "mae_model": st.column_config.NumberColumn("model MAE (kWh)", format="%.2f"),
            "mae_baseline": st.column_config.NumberColumn("baseline MAE (kWh)", format="%.2f"),
            "coverage": st.column_config.NumberColumn("in 80 % interval", format="percent"),
            "mean_kwh": st.column_config.NumberColumn("mean use (kWh/day)", format="%.2f")})
C.footer()
