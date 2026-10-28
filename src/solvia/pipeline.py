"""Run the SOLVIA pipeline end to end.

    python -m solvia.pipeline                 # all implemented steps
    python -m solvia.pipeline --steps clean   # Phase 1 only
    python -m solvia.pipeline --steps model_solar   # Phase 4B
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from solvia.config import get_path, load_config, project_path
from solvia.data import download_nasa_power, emarc, kaggle_solar, nasa_power, niti, uci
from solvia.preprocessing import emarc as emarc_clean
from solvia.preprocessing import household, india_weather, irradiance, solar
from solvia.preprocessing.cleaning_log import CleaningLog

log = logging.getLogger("solvia.pipeline")


def save(df: pd.DataFrame, name: str) -> None:
    path = get_path("interim") / f"{name}.parquet"
    df.to_parquet(path)
    log.info("saved %s (%d rows)", path.relative_to(get_path("interim").parent.parent), len(df))


def step_clean() -> None:
    """Phase 1: load every dataset assigned to an objective, clean it, write data/interim/."""
    cfg = load_config()
    clog = CleaningLog()

    # Household consumption (UCI)
    minute = uci.load_uci()
    save(minute, "household_minute")
    hc = cfg["consumption"]
    save(household.clean_household(minute, hc["min_daily_coverage"], hc["max_interpolate_days"], clog),
         "household_daily")
    weather = nasa_power.load_nasa_power(project_path(cfg["datasets"]["household_weather"]))
    save(household.tidy_household_weather(weather, hc["hdd_base_c"], clog), "household_weather_daily")

    # Solar plants (Kaggle)
    sc = cfg["solar"]
    inv_daily = []
    for p in kaggle_solar.PLANTS:
        df = solar.clean_plant(kaggle_solar.load_generation(p), kaggle_solar.load_weather(p), p,
                               sc["interval_hours"], sc["off_irradiation_threshold"], clog)
        save(df, f"solar_plant{p}_15min")
        d = solar.daily_per_inverter(df, sc["interval_hours"])
        solar.yield_crosscheck(d, p, clog)
        inv_daily.append(d)
    inv_daily = pd.concat(inv_daily, ignore_index=True)
    save(inv_daily, "solar_daily_inverter")
    daily_plant = solar.daily_per_plant(inv_daily)
    save(daily_plant, "solar_daily_plant")

    # India weather: NASA POWER daily for every city in configs/india_locations.yaml, 2001-2025
    api, locations = download_nasa_power.load_locations()
    india_dir = project_path(cfg["datasets"]["india_weather_dir"])
    frames = {name: nasa_power.load_nasa_power(india_dir / download_nasa_power.file_name(name, lat, lon, api["start"], api["end"]))
              for name, (_, _, lat, lon) in locations.items()}
    india = india_weather.tidy_india_daily(frames, locations, clog)
    clim = india_weather.monthly_climatology(india)
    save(india, "india_weather_daily")
    save(clim, "india_monthly_climatology")
    save(india_weather.location_climate(india, clim), "india_location_climate")
    match = india_weather.match_plants_to_locations(india, daily_plant)
    save(match, "plant_location_match")
    save(india_weather.plant_estimated_weather(india, match), "plant_estimated_weather")
    grid_dir = project_path(cfg["datasets"]["plant_search_grid_dir"])
    grid = india_weather.tidy_grid({p.name.split("_daily")[0]: nasa_power.load_nasa_power(p)
                                    for p in sorted(grid_dir.glob("grid_*.csv"))}, clog)
    save(india_weather.match_plants_to_grid(grid, daily_plant), "plant_grid_match")

    # Indian households: Prayas eMARC smart meters (each meter stays a separate series)
    ec = cfg["emarc"]
    e_daily, e_info = emarc.load_daily(), emarc.load_info()
    clean = emarc_clean.clean_emarc_daily(e_daily, emarc_clean.deployment_meta(e_daily, e_info, clog), ec, clog)
    blocks = emarc.load_blocks()
    block_daily = emarc_clean.daily_from_blocks(blocks, tuple(ec["solar_hours"]))
    emarc_clean.crosscheck_blocks(clean, block_daily, ec["min_blocks_per_day"], clog)
    mains = emarc_clean.mains_table(clean, block_daily, ec, clog)
    save(clean, "emarc_daily_all_meters")
    save(block_daily, "emarc_block_daily")
    save(mains, "emarc_mains_daily")
    save(emarc_clean.typical_profiles(blocks, mains, ec["min_blocks_per_day"]), "emarc_load_profiles")
    del blocks

    # NITI Aayog ICED (validation of NASA POWER and December-2024 district ranking)
    districts = irradiance.tidy_niti_districts(niti.load_niti_sheet(project_path(cfg["datasets"]["niti_districts_dec2024"])), clog)
    kheda = irradiance.tidy_niti_monthly(niti.load_niti_sheet(project_path(cfg["datasets"]["niti_kheda_monthly"])), clog)
    save(districts, "niti_districts_dec2024")
    save(kheda, "niti_kheda_monthly")

    clog.write(project_path("reports/audit/cleaning_log.json"), project_path("reports/cleaning_report.md"))


def step_eda() -> None:
    """Phase 2: figures in reports/figures/, takeaways in reports/eda_summary.md, EDA notebooks."""
    from solvia.eda.report import run_eda
    run_eda()


def step_features() -> None:
    """Phase 3: model-ready feature tables with chronological split labels in data/processed/."""
    from solvia.features import consumption as cons_f
    from solvia.features import india_consumption as india_f
    from solvia.features import solar as solar_f
    from solvia.features.calendar import french_school_holidays

    cfg = load_config()
    clog = CleaningLog()
    interim, processed = get_path("interim"), get_path("processed")

    daily = pd.read_parquet(interim / "household_daily.parquet")
    weather = pd.read_parquet(interim / "household_weather_daily.parquet")
    missing_weather = int((~daily.index.isin(weather.index)).sum())
    clog.add("consumption features", "join Sceaux daily NASA POWER weather on date", len(daily), len(daily), 0,
             f"{missing_weather} household days without weather")
    daily = daily.join(weather, how="left")
    school = french_school_holidays(project_path(cfg["datasets"]["french_school_holidays"]), zone="c")
    cons = cons_f.finalize(cons_f.build_consumption_features(daily, school), cfg["consumption"]["split"], clog)
    cons.to_parquet(processed / "consumption_features.parquet")

    ec = cfg["emarc"]
    mains = india_f.join_weather(pd.read_parquet(interim / "emarc_mains_daily.parquet"),
                                 pd.read_parquet(interim / "india_weather_daily.parquet"), ec["region_weather"],
                                 ec["cdd_base_c"], clog)
    india_cons = india_f.finalize(india_f.build_india_consumption_features(mains), ec, clog)
    india_cons.to_parquet(processed / "india_consumption_features.parquet")

    df15 = pd.concat([pd.read_parquet(interim / f"solar_plant{p}_15min.parquet") for p in kaggle_solar.PLANTS],
                     ignore_index=True)
    sol = solar_f.finalize(solar_f.build_solar_features(df15), cfg["solar"]["split_fractions"], clog)
    sol.to_parquet(processed / "solar_features_15min.parquet")

    clog.fact("consumption_features", cons_f.FEATURES)
    clog.fact("consumption_forecast_proxy_features (experiment only)", cons_f.FORECAST_PROXY_FEATURES)
    clog.fact("india_consumption_features", india_f.FEATURES)
    clog.fact("india_consumption_forecast_proxy_features (experiment only)", india_f.FORECAST_PROXY_FEATURES)
    clog.fact("solar_features", solar_f.FEATURES)
    clog.write(project_path("reports/audit/features_log.json"), project_path("reports/features_report.md"),
               title="Feature report (Phase 3)", command="python -m solvia.pipeline --steps features")
    log.info("features: consumption %d rows, India consumption %d rows, solar %d rows", len(cons), len(india_cons), len(sol))


def step_model_solar() -> None:
    """Phase 4B: expected-solar-output model (validation only; test untouched until Phase 5)."""
    from solvia.models.solar import run_solar_models
    run_solar_models()


def step_monitor() -> None:
    """Phase 4C: performance monitoring with out-of-sample expected output."""
    from solvia.models.monitoring import run_monitoring
    run_monitoring()


def step_maintenance() -> None:
    """Phase 4D: rule-based maintenance recommendations with reasons and evidence."""
    from solvia.models.monitoring_report import write_monitoring_report
    from solvia.rules.maintenance import run_maintenance
    run_maintenance()
    write_monitoring_report()


def step_model_consumption_india() -> None:
    """Phase 4A (India first): next-day consumption of the eMARC households."""
    from solvia.models.consumption import run_consumption
    run_consumption("india")


def step_model_consumption_france() -> None:
    """Phase 4A: next-day consumption of the UCI household (France)."""
    from solvia.models.consumption import run_consumption
    run_consumption("france")


def step_energy_balance() -> None:
    """Phase 4E: measured solar and load shapes + example scenarios (direct use, export, import)."""
    import json
    import math

    import numpy as np

    from solvia.models.solar import healthy_inverter_days
    from solvia.rules import energy_balance as eb

    cfg = load_config()
    interim = get_path("interim")
    df15 = pd.concat([pd.read_parquet(interim / f"solar_plant{p}_15min.parquet") for p in kaggle_solar.PLANTS], ignore_index=True)
    df15["date"] = df15["timestamp"].dt.floor("D")
    healthy = healthy_inverter_days(pd.read_parquet(interim / "solar_daily_inverter.parquet"), cfg["solar"]["healthy_min_ratio"])
    pv = eb.pv_shape_from_plants(df15, healthy)
    loads = eb.load_shapes_from_profiles(pd.read_parquet(interim / "emarc_load_profiles.parquet"))
    shapes = pd.DataFrame({"block": range(96), "hour": np.arange(96) / 4, "pv_shape": pv, **{f"load_{k}": v for k, v in loads.items()}})
    shapes.to_parquet(interim / "energy_balance_shapes.parquet")

    mains = pd.read_parquet(interim / "emarc_mains_daily.parquet")
    measured = mains[mains["kwh"].notna() & ~mains["is_interpolated"]]
    typical = measured.groupby("deployment_id").agg(t=("household_type", "first"), kwh=("kwh", "mean")).groupby("t")["kwh"].median()
    clim = pd.read_parquet(interim / "india_monthly_climatology.parquet")
    pr = cfg["energy_balance"]["default_performance_ratio"]
    rows = []
    for htype, kwh in typical.items():
        need = eb.annual_summary(eb.Scenario("pune", 1.0, float(kwh), pr), clim)["kwp_to_cover_consumption"]
        kwp = math.ceil(need * 2) / 2                                  # round up to 0.5 kWp
        a = eb.annual_summary(eb.Scenario("pune", kwp, float(kwh), pr), clim, (pv, loads[htype]))
        rows.append({"household_type": htype, "daily_use_kwh": round(float(kwh), 2), "kwp": kwp,
                     **{k: round(float(a[k]), 1) for k in ("solar_kwh", "consumption_kwh", "direct_use_kwh", "export_kwh",
                                                          "grid_import_kwh", "self_sufficiency_pct", "self_consumption_pct")}})
    pv_9_16 = float(pv[36:64].sum())
    out = {"pv_share_09_16": round(pv_9_16, 3), "scenarios_pune": rows, "performance_ratio": pr}
    (get_path("metrics") / "energy_balance_examples.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    cols = list(rows[0])
    lines = ["# Energy balance with direct use (Phase 4E upgrade)", "",
             "*Generated by `python -m solvia.pipeline --steps energy_balance`. All solar figures are ESTIMATES.*", "",
             "Method: monthly kWp × NASA POWER GHI × performance ratio "
             f"{pr}; each day is split into 96 blocks with the measured solar output shape (healthy Kaggle inverters, "
             f"{pv_9_16:.0%} of their energy falls in 09:00–16:00) and the measured eMARC load shape of the household type. "
             "Direct use = Σ min(solar, load); the rest of the solar is exported, the rest of the load comes from the grid. No battery.", "",
             "## Example: a typical Pune home of each type, sized to cover its yearly use (rounded up to 0.5 kWp)", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
             *["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows], "",
             "kWh per year. `self_sufficiency_pct` = share of the home's use covered directly by its panels; "
             "`self_consumption_pct` = share of the panels' output used in the home.", "",
             *[f"- {n}" for n in eb.NOTES], ""]
    project_path("reports/energy_balance_report.md").write_text("\n".join(lines), encoding="utf-8")


def step_evaluate() -> None:
    """Phase 5: the single test-set evaluation, figures p01-p09, evaluation report and model cards."""
    from solvia.models.evaluation import run_evaluation
    run_evaluation()


def step_improve() -> None:
    """Phase 8 experiments: more accurate candidates, compared on validation only (not in the default run)."""
    from solvia.models.improve import run_improvement_experiments
    run_improvement_experiments()


STEPS = {"clean": step_clean, "eda": step_eda, "features": step_features, "model_solar": step_model_solar,
         "monitor": step_monitor, "maintenance": step_maintenance,
         "energy_balance": step_energy_balance,
         "model_consumption_india": step_model_consumption_india, "model_consumption_france": step_model_consumption_france,
         "evaluate": step_evaluate, "improve": step_improve}
DEFAULT_STEPS = [s for s in STEPS if s != "improve"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=DEFAULT_STEPS)
    for name in parser.parse_args().steps:
        log.info("=== step: %s ===", name)
        STEPS[name]()


if __name__ == "__main__":
    main()
