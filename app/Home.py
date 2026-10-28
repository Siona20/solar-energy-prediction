"""SOLVIA dashboard, Overview page. Run with: streamlit run app/Home.py"""
import pandas as pd
import streamlit as st

import common as C

C.setup("SOLVIA: Smart Solar Energy Forecasting & Maintenance", "☀️")

st.markdown(
    "SOLVIA forecasts a home's electricity use for the next day, estimates what a solar plant **should** produce, "
    "spots inverters that produce less than expected, suggests maintenance with reasons, and estimates how much of a "
    "home's use rooftop solar could cover. **Indian data comes first**; the French household is a comparison track.")

st.subheader("What every number on this dashboard is")
for kind, text in C.D.KINDS.items():
    a, b = st.columns([1, 7])
    with a:
        st.badge(kind, color=C.KIND_BADGE[kind])
    b.write(text)

st.subheader("Headline results (test split, scored once in Phase 5)")
ev_solar, ev_india, ev_france = C.csv_("evaluation_solar"), C.csv_("evaluation_consumption_india"), C.csv_("evaluation_consumption_france")
mon = C.json_("monitoring_summary")


def _cons(t: pd.DataFrame, subset: str) -> pd.Series:
    return t[(t["selected"]) & (t["subset"] == subset)].iloc[0]


s = ev_solar[(ev_solar["selected"]) & (ev_solar["split"] == "test") & (ev_solar["eval_set"] == "healthy")
             & (ev_solar["plant"] == "both") & (ev_solar["level"] == "daily")].iloc[0]
i, f = _cons(ev_india, "test"), _cons(ev_france, "test")
c1, c2, c3, c4 = st.columns(4)
c1.metric("India next-day use, error", f"{i['mae']:.2f} kWh/day", f"{i['skill_vs_best_baseline_pct']:+.1f} % vs 7-day mean",
          delta_color="off", border=True, help="Mean absolute error on 104 eMARC meters, Jan–Jun 2020 (includes lockdown).")
c2.metric("Expected solar output, error", f"{s['mape_pct']:.1f} %", f"{s['mae']:.0f} kWh per inverter-day", delta_color="off",
          border=True, help="Healthy inverter-days, both Kaggle plants, 5 test days.")
c3.metric("Plant 2 output ratio", f"{mon['mean_plant_pr']['Plant 2']:.2f}",
          f"{mon['mean_plant_running_pr']['Plant 2']:.2f} while inverters run", delta_color="off", border=True,
          help="Actual ÷ expected. The gap is inverters switching off in bright sun, not dust.")
c4.metric("France next-day use, error", f"{f['mae']:.2f} kWh/day", f"{f['skill_vs_best_baseline_pct']:+.1f} % vs 7-day mean",
          delta_color="off", border=True, help="One household in Sceaux, test year 2010.")

st.subheader("Pages")
st.markdown("""
| Page | What it answers | Data |
|---|---|---|
| **Consumption Forecast** | How much will this home use tomorrow, and how sure are we? | eMARC (India), UCI (France) |
| **Solar Generation** | What should the plant produce in today's weather, and did it? | Kaggle plants |
| **Performance Monitor** | Which inverters are under-performing, and is it dust or shutdowns? | Kaggle plants + estimated rain |
| **Maintenance** | What should the operator do today, and why? | Monitoring results + rules |
| **Energy Balance** | How much of a home's use could rooftop solar cover in an Indian city? | NASA POWER + eMARC shapes (estimate) |
| **Model Performance** | How good are the models, honestly? | Phase 5 test results, model cards |
| **SYNTHETIC demo** | Layout test with a full-year home PV dataset of unverified origin | PV360 (synthetic) |
""")
st.info("Models are loaded from `models/`; the dashboard never retrains them. Each forecast shown is out-of-sample "
        "unless a page says otherwise.", icon="ℹ️")
C.footer()
