"""Clean the eMARC Indian smart-meter data.

Every meter (deployment) stays its own series. Households are never added together or treated as
one household; models later learn across many separate series with the meter id kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from solvia.preprocessing.cleaning_log import CleaningLog
from solvia.preprocessing.household import interpolate_short_gaps

BLOCKS_PER_DAY = 96
EVENING_HOURS = (18, 22)


def deployment_meta(daily: pd.DataFrame, info: pd.DataFrame, clog: CleaningLog) -> pd.DataFrame:
    """One row per deployment. The info sheet is the master metadata; conflicts with the daily file are flagged."""
    in_daily = daily.groupby("deployment_id")[["household_id", "region", "household_type", "deployment_type"]].first()
    meta = info.set_index("deployment_id")[["household_id", "region", "household_type", "deployment_type"]]
    meta = meta.loc[meta.index.isin(in_daily.index)].copy()
    conflict = (in_daily.loc[meta.index] != meta).any(axis=1)
    meta["meta_conflict"] = conflict
    meta["deployment_type"] = meta["deployment_type"].str.lower()        # 'Mainline' / 'appliance' -> 'mainline' / 'appliance'
    clog.add("emarc", "deployment metadata from the info sheet (daily-file labels compared)", len(info), len(meta),
             int(conflict.sum()),
             "deployments with no daily data dropped; conflicts: "
             + ", ".join(f"{d} ({meta.loc[d, 'household_id']}: daily says '{in_daily.loc[d, 'household_type']}', "
                         f"info says '{meta.loc[d, 'household_type']}')" for d in meta.index[conflict]))
    return meta.reset_index()


def _zero_runs(s: pd.Series, min_run: int) -> pd.Series:
    """True on days that are part of a run of >= min_run consecutive exact-zero days."""
    z = s.eq(0)
    run_id = (z != z.shift()).cumsum()
    return z & z.groupby(run_id).transform("sum").ge(min_run)


def _clean_one(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = g.set_index("date").sort_index()
    cal = pd.date_range(g.index.min(), g.index.max(), freq="D", name="date")
    out = pd.DataFrame(index=cal)
    out["kwh_raw"] = g["kwh"].reindex(cal)
    out["is_reported"] = out["kwh_raw"].notna()
    out["is_zero_run"] = _zero_runs(out["kwh_raw"], cfg["zero_run_days"])
    kwh = out["kwh_raw"].mask(out["is_zero_run"])
    q1, q3 = kwh.quantile([0.25, 0.75])
    out["is_outlier"] = kwh > q3 + cfg["outlier_iqr_k"] * (q3 - q1)
    out["kwh"], out["is_interpolated"] = interpolate_short_gaps(kwh, cfg["max_interpolate_days"])
    out["is_missing"] = out["kwh"].isna()
    return out


def clean_emarc_daily(daily: pd.DataFrame, meta: pd.DataFrame, cfg: dict, clog: CleaningLog) -> pd.DataFrame:
    """Long table: one row per deployment and calendar day between its first and last report."""
    n = len(daily)
    daily = daily.drop_duplicates()
    clog.add("emarc", "drop exact duplicate rows", n, len(daily), n - len(daily))
    dup = daily.duplicated(["deployment_id", "date"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} conflicting (deployment, date) rows remain")

    parts = [_clean_one(g, cfg).assign(deployment_id=dep)
             for dep, g in daily[["deployment_id", "date", "kwh"]].groupby("deployment_id")]
    out = pd.concat(parts).reset_index().merge(meta, on="deployment_id", how="left")
    clog.add("emarc", "reindex every meter to a complete calendar (first..last report)", len(daily), len(out),
             int((~out["is_reported"]).sum()), "added days are missing (not reported)")
    clog.add("emarc", f"runs of >= {cfg['zero_run_days']} consecutive 0 kWh days -> missing (meter/supply off or house empty)",
             len(out), len(out), int(out["is_zero_run"].sum()),
             f"shorter zero runs kept as real (e.g. full-day power cuts); {int(out['kwh_raw'].eq(0).sum())} zero days in total")
    clog.add("emarc", f"flag days above Q3 + {cfg['outlier_iqr_k']} x IQR of that meter (kept, not removed)",
             len(out), len(out), int(out["is_outlier"].sum()))
    clog.add("emarc", f"interpolate gaps of <= {cfg['max_interpolate_days']} days within each meter", len(out), len(out),
             int(out["is_interpolated"].sum()), f"{int(out['is_missing'].sum())} meter-days remain missing")
    cols = ["deployment_id", "household_id", "deployment_type", "region", "household_type", "meta_conflict", "date",
            "kwh", "kwh_raw", "is_reported", "is_zero_run", "is_outlier", "is_interpolated", "is_missing"]
    return out[cols].sort_values(["deployment_id", "date"]).reset_index(drop=True)


def daily_from_blocks(blocks: pd.DataFrame, solar_hours: tuple[int, int]) -> pd.DataFrame:
    """Per meter and day from the 15-minute block loads (kW): energy, peak, base load and time-of-day shares.
    Energy = mean(block kW) x 24 h, so a day with a few missing blocks is not under-counted."""
    b = blocks
    h = b["block"] // 4
    kwh = b["load_kw"] * 0.25
    tmp = pd.DataFrame({"deployment_id": b["deployment_id"].astype(str), "date": b["date"], "load_kw": b["load_kw"],
                        "solar_kwh": kwh.where(h.between(solar_hours[0], solar_hours[1] - 1), 0.0),
                        "evening_kwh": kwh.where(h.between(EVENING_HOURS[0], EVENING_HOURS[1] - 1), 0.0),
                        "block_kwh": kwh})
    g = tmp.groupby(["deployment_id", "date"], sort=True)
    d = g.agg(n_blocks=("load_kw", "count"), mean_kw=("load_kw", "mean"), peak_kw=("load_kw", "max"),
              base_kw=("load_kw", "min"), block_kwh=("block_kwh", "sum"), solar_kwh=("solar_kwh", "sum"),
              evening_kwh=("evening_kwh", "sum")).reset_index()
    d["kwh_blocks"] = d.pop("mean_kw") * 24
    total = d.pop("block_kwh").replace(0, np.nan)
    d["solar_hours_share"] = d.pop("solar_kwh") / total
    d["evening_share"] = d.pop("evening_kwh") / total
    return d


def crosscheck_blocks(clean: pd.DataFrame, block_daily: pd.DataFrame, min_blocks: int, clog: CleaningLog) -> pd.DataFrame:
    """Daily kWh from the cumulative meter (daily file) vs mean block load x 24 on complete-enough days."""
    j = clean.merge(block_daily, on=["deployment_id", "date"], how="inner")
    j = j[(j["n_blocks"] >= min_blocks) & (j["kwh_raw"] > 0.5)]
    ratio = j["kwh_blocks"] / j["kwh_raw"]
    within = ratio.between(0.9, 1.1).mean()
    clog.add("emarc", f"cross-check daily kWh vs 15-minute blocks (days with >= {min_blocks} blocks, > 0.5 kWh)",
             len(j), len(j), int((~ratio.between(0.9, 1.1)).sum()),
             f"median ratio {ratio.median():.3f}; {within:.1%} of days within ±10%")
    clog.fact("emarc_blocks_vs_daily", {"days": len(j), "median_ratio": round(float(ratio.median()), 4),
                                        "share_within_10pct": round(float(within), 4)})
    return j


def mains_table(clean: pd.DataFrame, block_daily: pd.DataFrame, cfg: dict, clog: CleaningLog) -> pd.DataFrame:
    """Whole-home (mainline) meters with the same household's appliance meter and the block features joined."""
    mains = clean[clean["deployment_type"] == "mainline"].copy()
    app = clean[clean["deployment_type"] == "appliance"][["household_id", "date", "kwh"]].rename(columns={"kwh": "appliance_kwh"})
    mains = mains.merge(app, on=["household_id", "date"], how="left")
    mains["appliance_gt_mains"] = mains["appliance_kwh"] > mains["kwh"] * 1.01
    clog.add("emarc", "join each household's appliance meter to its mains meter (same household, same day)",
             len(mains), len(mains), int(mains["appliance_kwh"].notna().sum()),
             f"{int(mains['appliance_gt_mains'].sum())} days where appliance > mains (flagged)")

    bd = block_daily.copy()
    full = bd["n_blocks"] >= cfg["min_blocks_per_day"]
    for col in ["peak_kw", "base_kw", "solar_hours_share", "evening_share"]:
        bd[col] = bd[col].where(full)
    mains = mains.merge(bd[["deployment_id", "date", "n_blocks", "peak_kw", "base_kw", "solar_hours_share", "evening_share"]],
                        on=["deployment_id", "date"], how="left")
    mains["n_blocks"] = mains["n_blocks"].fillna(0).astype(int)
    clog.add("emarc", f"join 15-minute load-shape features (days with < {cfg['min_blocks_per_day']} of 96 blocks -> missing)",
             len(mains), len(mains), int(mains["peak_kw"].isna().sum()), "meter-days without usable load shape")

    measured = mains["kwh"].notna() & ~mains["is_interpolated"]
    valid = measured.groupby(mains["deployment_id"]).sum()
    eligible = valid[valid >= cfg["min_valid_days"]].index
    mains["model_eligible"] = mains["deployment_id"].isin(eligible)
    clog.add("emarc", f"mains meters with >= {cfg['min_valid_days']} measured days are model-eligible",
             int(valid.size), int(len(eligible)), int(valid.size - len(eligible)),
             f"{int(measured[mains['model_eligible']].sum()):,} measured meter-days kept for modelling")
    clog.fact("emarc_mains_summary", {"meters": int(valid.size), "eligible": int(len(eligible)),
                                      "daily_kwh": mains["kwh"].describe().round(2).to_dict()})
    return mains


def typical_profiles(blocks: pd.DataFrame, mains: pd.DataFrame, min_blocks: int) -> pd.DataFrame:
    """Average 15-minute load shape (kW) per household type and region from mains meters on complete days.
    Each meter is averaged first, so long-running meters do not dominate."""
    ok = mains.loc[mains["n_blocks"] >= min_blocks, ["deployment_id", "date", "household_type", "region"]]
    b = blocks.assign(deployment_id=blocks["deployment_id"].astype(str)).merge(ok, on=["deployment_id", "date"])
    per_meter = b.groupby(["household_type", "region", "deployment_id", "block"], observed=True)["load_kw"].mean().reset_index()
    prof = per_meter.groupby(["household_type", "region", "block"])["load_kw"].agg(["mean", "count"]).reset_index()
    prof = prof.rename(columns={"mean": "load_kw", "count": "n_meters"})
    prof["hour"] = prof["block"] / 4
    return prof
