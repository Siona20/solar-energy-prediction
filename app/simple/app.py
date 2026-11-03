"""SOLVIA: tomorrow and the next 7 days for your home.

Run with: streamlit run app/simple/app.py
Enter the place, the solar system size (kW), the type of home and recent use; the page forecasts the weather, the
solar units and the units the home will use. It never trains a model; the only file it writes is the NASA POWER
weather cache in data/cache/. Colours and fonts come from .streamlit/config.toml.
"""
import altair as alt
import pandas as pd
import streamlit as st

from solvia import forecast as F
from solvia.dashboard import data as D

st.set_page_config(page_title="SOLVIA · your solar forecast", page_icon=":material/wb_sunny:", layout="centered")


def space(size: str = "small") -> None:
    fn = getattr(st, "space", None)
    if fn is not None:
        fn(size)
    else:
        st.write("")

SOLAR, USE = "Solar units", "Home use"


@st.cache_data
def forecast_facts() -> dict:
    """Accuracy figures quoted on the page, read from the saved model reports."""
    wx = D.metrics_csv("weather_model_scores")
    wx = wx[(wx["split"] == "test") & (wx["model"] == "model")]
    home = D.model_metadata("home_use_simple")
    return {"wx_err": wx[wx["h"].between(8, 14)].groupby("target")["mae"].mean().to_dict(),
            "wx_err_h1": wx[wx["h"] == 1].set_index("target")["mae"].to_dict(),
            "home_err": home["valid_mae_1day"][home["selected"]], "home_types": home["household_types"],
            "typical_use": D.typical_use_by_type()}


@st.cache_resource
def forecast_models():
    return F.load_models()


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching the latest NASA POWER weather…")
def place_history(lat: float, lon: float, location: str | None, today: pd.Timestamp):
    from solvia.data import live_weather
    return live_weather.history(lat, lon, today, location)


def week_chart(t: pd.DataFrame) -> alt.LayerChart:
    """Solar bars with their likely range, home use as a line; colours come from the theme."""
    long = pd.concat([
        pd.DataFrame({"day": t["date"], "series": SOLAR, "units": t["solar_kwh"], "low": t["solar_low_kwh"],
                      "high": t["solar_high_kwh"]}),
        pd.DataFrame({"day": t["date"], "series": USE, "units": t["use_kwh"], "low": t["use_low_kwh"],
                      "high": t["use_high_kwh"]})])
    color = alt.Color("series:N", title=None, scale=alt.Scale(domain=[SOLAR, USE]),
                      legend=alt.Legend(orient="top", direction="horizontal"))
    x = alt.X("yearmonthdate(day):O", title=None, axis=alt.Axis(format="%a %d", labelAngle=0))
    tip = [alt.Tooltip("day:T", title="Day", format="%A %d %b"), alt.Tooltip("series:N", title=" "),
           alt.Tooltip("units:Q", title="Units", format=".1f"), alt.Tooltip("low:Q", title="Likely from", format=".1f"),
           alt.Tooltip("high:Q", title="Likely to", format=".1f")]
    base = alt.Chart(long).encode(x=x)
    bars = base.transform_filter(alt.datum.series == SOLAR).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=34).encode(
        y=alt.Y("units:Q", title="units (kWh) per day", scale=alt.Scale(domain=[0, float(long["high"].max()) * 1.05])),
        color=color, tooltip=tip)
    rng = base.transform_filter(alt.datum.series == SOLAR).mark_rule(opacity=.45, strokeWidth=2).encode(
        y="low:Q", y2="high:Q", color=alt.value("gray"))
    line = base.transform_filter(alt.datum.series == USE).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(
        size=70, filled=True)).encode(y="units:Q", color=color, tooltip=tip)
    return alt.layer(bars, rng, line).properties(height=300)


d = forecast_facts()
today = pd.Timestamp.today().normalize()

st.title("SOLVIA", anchor=False)
st.markdown(":gray[Your solar units and your home's electricity use, for tomorrow and the next 7 days.]")

cities = D.locations()
other = "Other place (latitude / longitude)"
with st.container(border=True):
    a, b = st.columns(2)
    pick = a.selectbox("Where is your home?", [*cities["location"], other],
                       index=int(cities.index[cities["location"] == "pune"][0]),
                       format_func=lambda x: x if x == other else dict(zip(cities["location"], cities["label"]))[x])
    kwp = b.number_input("Solar system size (kW)", 0.5, 100.0, 3.0, 0.5,
                         help="The size written on your solar quotation or inverter, e.g. 3 kW or 12 kW.")
    if pick == other:
        c1, c2 = a.columns(2)
        lat = c1.number_input("Latitude", 6.0, 37.0, 23.03, 0.01, format="%.2f")
        lon = c2.number_input("Longitude", 68.0, 98.0, 72.58, 0.01, format="%.2f")
        place = F.Place(f"{lat:.2f}° N, {lon:.2f}° E", lat, lon)
    else:
        row = cities.set_index("location").loc[pick]
        place = F.Place(row["label"], float(row["lat"]), float(row["lon"]), pick)
    htype = a.selectbox("Type of home", d["home_types"],
                        help="Basic: fans, lights, TV, fridge. Water heater: also a geyser. AC: has air conditioners.")
    sunny = b.number_input("Units on a clear sunny day (optional)", 0.0, 1000.0, 0.0, 1.0,
                           help="If you know it from your meter or app (e.g. a 12 kW system giving 60 units), "
                                "SOLVIA adjusts its estimate to your system. Leave 0 if you don't know.")
    mode = st.segmented_control("Your electricity use", ["A typical day", "The last 7 days"], default="A typical day") or "A typical day"
    if mode == "A typical day":
        typical = st.number_input("Units used on a typical day", 0.5, 200.0, float(d["typical_use"][htype]), 0.5,
                                  key=f"typical_{htype}")       # new default when the type of home changes
        last7 = [typical] * 7
    else:
        dates = pd.date_range(today - pd.Timedelta(days=6), today)
        ed = st.data_editor(pd.DataFrame([[float(d["typical_use"][htype])] * 7], columns=dates.strftime("%a %d")),
                            hide_index=True, width="stretch", key=f"last7_{htype}")   # one row: units per day
        last7 = ed.iloc[0].fillna(0).astype(float).tolist()
    run = st.button("Show my forecast", type="primary", icon=":material/bolt:", width="stretch")

# run only on the button; keep the result until the next press
inputs = (place.lat, place.lon, kwp, sunny, htype, tuple(last7), str(today.date()))
if run:
    try:
        with st.spinner("Fetching the latest weather and running the models…"):
            hist = place_history(place.lat, place.lon, place.location, today)
            res = F.plan(forecast_models(), place, kwp, last7, htype, today, sunny or None, w=hist)
        st.session_state["my_forecast"] = {"inputs": inputs, "r": res, "place": place.name, "kwp": kwp,
                                           "sunny": sunny, "htype": htype}
    except Exception as exc:   # network down, NASA not reachable, or data too old
        st.session_state.pop("my_forecast", None)
        st.error(str(exc), title="Could not make the forecast", icon=":material/cloud_off:")
saved = st.session_state.get("my_forecast")

if saved is None:
    space("small")
    st.markdown(":gray[:material/arrow_upward: Fill in your details and press **Show my forecast**.]")
else:
    if saved["inputs"] != inputs:
        st.warning("Press **Show my forecast** again to update the result below.", title="You changed your details",
                   icon=":material/refresh:")
    r, kwp_s, sunny_s = saved["r"], saved["kwp"], saved["sunny"]
    t = r["days"]
    tm = t.iloc[0]
    net = tm["net_kwh"]

    space("medium")
    st.header(f"Tomorrow, {tm['date']:%A %d %B}", anchor=False)
    st.markdown(f":gray[{kwp_s:g} kW solar · {saved['htype']} home · {saved['place']} · forecasts, not measurements]")
    with st.container(horizontal=True):
        st.metric("From your solar", f"{tm['solar_kwh']:.1f} units", border=True, icon=":material/wb_sunny:",
                  help=f"Likely {tm['solar_low_kwh']:.0f}–{tm['solar_high_kwh']:.0f} units.")
        st.metric("Your home will use", f"{tm['use_kwh']:.1f} units", border=True, icon=":material/home:",
                  help=f"Likely {tm['use_low_kwh']:.1f}–{tm['use_high_kwh']:.1f} units.")
        st.metric("Extra to export" if net >= 0 else "Needed from the grid", f"{abs(net):.1f} units", border=True,
                  icon=":material/bolt:" if net >= 0 else ":material/electrical_services:")
    with st.container(horizontal=True):
        st.metric("Temperature", f"{tm['tmax']:.0f}° / {tm['tmin']:.0f}°", border=True,
                  help=f"Max / min °C. Normal for the date: {tm['tmax_normal']:.0f}° / {tm['tmin_normal']:.0f}°.")
        st.metric("Sunlight", f"{tm['ghi']:.1f} kWh/m²", border=True,
                  help=f"Normal for the date: {tm['ghi_normal']:.1f} kWh/m².")

    space("medium")
    st.header("The next 7 days", anchor=False)
    st.markdown(f":gray[About **{t['solar_kwh'].sum():.0f} units** from solar and **{t['use_kwh'].sum():.0f} units** "
                "used in total. The grey line on each bar is the likely range.]")
    st.altair_chart(week_chart(t), width="stretch", alt="Solar units and home use for each of the next 7 days")
    table = pd.DataFrame({
        "Day": t["date"].dt.strftime("%a %d %b"), "Max °C": t["tmax"], "Min °C": t["tmin"], "Sunlight": t["ghi"],
        "Solar": t["solar_kwh"], "Use": t["use_kwh"], "Extra / grid": t["net_kwh"]})
    with st.expander("Day by day", icon=":material/table_rows:"):
        st.dataframe(table, hide_index=True, width="stretch", alt="Forecast for each of the next 7 days",
                     column_config={
                         "Max °C": st.column_config.NumberColumn(format="%.0f°"),
                         "Min °C": st.column_config.NumberColumn(format="%.0f°"),
                         "Sunlight": st.column_config.NumberColumn("Sunlight kWh/m²", format="%.1f"),
                         "Solar": st.column_config.NumberColumn("Solar units", format="%.1f"),
                         "Use": st.column_config.NumberColumn("Home units", format="%.1f"),
                         "Extra / grid": st.column_config.NumberColumn("Extra (+) / grid (−)", format="%+.1f")})

    e = r["data_end"]
    with st.expander("How these numbers are made", icon=":material/info:"):
        cal = (f"Calibrated to your {sunny_s:.0f} units on a sunny day (factor {r['calibration_scale']:.2f})." if sunny_s
               else f"Not calibrated: on a clear summer day here the model expects about {r['sunny_day_model_units']:.0f} "
                    f"units from {kwp_s:g} kW. Enter your own sunny-day units above to adjust.")
        st.markdown(f"""
- **How far ahead**: NASA POWER has published temperature up to {e['temperature_data_end']:%d %b} and sunlight up
  to {e['sunlight_data_end']:%d %b}, so tomorrow is {int(tm['h_temp'])} and {int(tm['h_sun'])} days after the latest
  data. At that range the forecast is close to the normal for the date.
- **Weather**: a model trained on 25 years of NASA POWER data for 50 Indian cities forecasts each day's difference
  from the normal for that place and date (test 2022–2025: ±{d['wx_err_h1']['tmax']:.1f} °C one day ahead,
  ±{d['wx_err']['tmax']:.1f} °C 8–14 days ahead; sunlight is about as good as the long-term normal at that range).
- **Solar units** = kW × sunlight × losses (0.86) × a heat loss of 0.4 % per °C of panel temperature; how hot panels
  get was measured on two Indian solar plants. {cal} Flat panels assumed (no tilt gain).
- **Home use**: a model trained on Indian smart-meter homes (Prayas eMARC, 2018–2019) from your last 7 days, the
  type of home, the date and the temperature (validation error ±{d['home_err']:.2f} units one day ahead). Festivals
  after 2020 are not in its calendar.
""")

space("large")
st.caption("Data: NASA POWER (weather) · Prayas (Energy Group) eMARC smart meters (home use, non-commercial academic "
           "use) · Kaggle Solar Power Generation Data (panel heating). SOLVIA, B.Tech CSE final-year project.")
