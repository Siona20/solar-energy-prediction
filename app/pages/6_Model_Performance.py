"""Phase 5 results: validation and test metrics, figures, model cards and leakage checks."""
import streamlit as st

import common as C

C.setup("Model Performance", "🧪")
log = C.json_("test_run_log")
st.caption(f"Models were chosen on validation (Phase 4). The test split was scored once on {log['first_run_utc'][:16].replace('T', ' ')} UTC "
           "with the saved model files (their SHA-256 hashes are recorded; a run with changed models is refused).")
fig_dir = C.D.get_path("figures")
reports = C.D.project_path("reports")

tabs = st.tabs(["India consumption", "Solar expected output", "France consumption", "Leakage checks"])


def cons_tab(track: str, figs: list[str]) -> None:
    t = C.csv_(f"evaluation_consumption_{track}")
    piv = t.pivot_table(index=["model", "kind"], columns="subset", values="mae").reset_index().sort_values("valid")
    sel = t[t["selected"]].iloc[0]["model"]
    piv.insert(0, "selected", piv["model"].eq(sel).map({True: "✔", False: ""}))
    st.markdown(f"**Mean absolute error (kWh/day).** Selected on validation: `{sel}`.")
    st.dataframe(piv, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.3f")
                                                     for c in piv.columns if c.startswith(("valid", "test"))})
    iv = C.json_("evaluation_summary")[f"{track}_interval"]
    st.markdown(f"**80 % interval on test:** {iv['test_raw']['coverage']:.1%} raw → **{iv['test_calibrated']['coverage']:.1%}** "
                f"after the conformal adjustment from validation (width {iv['test_calibrated']['mean_width_kwh']:.2f} kWh).")
    for f in figs:
        st.image(str(fig_dir / f))
    with st.expander("Model card"):
        st.markdown((reports / f"model_card_consumption_{track}.md").read_text(encoding="utf-8"))


with tabs[0]:
    cons_tab("india", ["p04_india_daily_error.png", "p05_india_test_predicted_vs_actual.png", "p06_india_error_by_month_weekday.png"])
with tabs[1]:
    t = C.csv_("evaluation_solar")
    d = t[(t["level"] == "daily") & (t["eval_set"] == "healthy") & (t["plant"] == "both")]
    piv = d.pivot_table(index="model", columns="split", values="mae").reset_index().sort_values("valid")
    piv.insert(0, "selected", piv["model"].eq(t[t["selected"]].iloc[0]["model"]).map({True: "✔", False: ""}))
    st.markdown("**Daily error per inverter-day (kWh), healthy inverter-days, both plants.** Only 5 test days.")
    st.dataframe(piv, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("valid", "test")})
    for f in ["p01_solar_test_expected_vs_actual.png", "p02_solar_test_day_curves.png", "p03_solar_error_by_hour.png"]:
        st.image(str(fig_dir / f))
    with st.expander("Model card"):
        st.markdown((reports / "model_card_solar_expected_output.md").read_text(encoding="utf-8"))
with tabs[2]:
    cons_tab("france", ["p07_france_test_timeseries.png", "p08_france_residuals_and_error.png"])
with tabs[3]:
    st.markdown((reports / "leakage_checks.md").read_text(encoding="utf-8"))
st.subheader("What drives each model (permutation importance, validation)")
st.image(str(fig_dir / "p09_permutation_importance.png"))
C.footer()
