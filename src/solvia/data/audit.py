"""Phase 0 data audit: inventory and profile every file under data/raw/ (read-only).

Run with:  python -m solvia.data.audit
Writes reports/audit/inventory.json, which reports/data_audit.md is written from.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from solvia.config import get_path, project_path

log = logging.getLogger(__name__)

SKIP_NAMES = {".DS_Store", ".gitkeep"}


# --------------------------------------------------------------------------- helpers
def md5_stream(chunks: Iterable[bytes]) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return md5_stream(iter(lambda: fh.read(1 << 22), b""))


def md5_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as fh:
        return md5_stream(iter(lambda: fh.read(1 << 22), b""))


def rel(path: Path) -> str:
    return str(path.relative_to(get_path("raw")))


def frame_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Generic shape / dtype / missingness / duplicate summary."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_pct": {c: round(float(v) * 100, 3) for c, v in df.isna().mean().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def interval_summary(ts: pd.Series) -> dict[str, Any]:
    """Most common sampling interval and how regular the series is."""
    ts = pd.Series(ts.dropna().sort_values().unique())
    d = ts.diff().dropna()
    if d.empty:
        return {}
    counts = d.value_counts()
    return {
        "start": str(ts.iloc[0]),
        "end": str(ts.iloc[-1]),
        "unique_timestamps": int(len(ts)),
        "modal_interval": str(counts.index[0]),
        "share_at_modal_interval": round(float(counts.iloc[0] / len(d)), 4),
        "largest_gap": str(d.max()),
    }


# --------------------------------------------------------------------------- inventory
def file_inventory(raw: Path) -> list[dict[str, Any]]:
    """Every file under raw/ with size and checksum; zip members are listed too."""
    rows = []
    for p in sorted(raw.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        entry: dict[str, Any] = {
            "path": rel(p),
            "size_mb": round(p.stat().st_size / 1e6, 2),
            "ext": p.suffix.lower(),
            "md5": md5_file(p),
        }
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                entry["members"] = [
                    {"name": i.filename, "size_mb": round(i.file_size / 1e6, 2),
                     "date": "%04d-%02d-%02d" % i.date_time[:3],
                     "md5": md5_zip_member(zf, i.filename) if not i.is_dir() else None}
                    for i in zf.infolist()
                ]
        rows.append(entry)
        log.info("inventoried %s", entry["path"])
    return rows


def duplicate_groups(inventory: list[dict[str, Any]]) -> list[list[str]]:
    """Groups of byte-identical files, including loose files identical to zip members."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for e in inventory:
        by_hash[e["md5"]].append(e["path"])
        for m in e.get("members", []):
            if m["md5"]:
                by_hash[m["md5"]].append(f"{e['path']}::{m['name']}")
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


# --------------------------------------------------------------------------- UCI household
def profile_uci(path: Path) -> dict[str, Any]:
    """UCI Individual Household Electric Power Consumption (1-minute)."""
    with open(path, "rb") as fh:
        head = fh.read(400).decode("utf-8")
    df = pd.read_csv(path, sep=";", na_values=["?"], dtype={"Date": str, "Time": str},
                     low_memory=False)
    out = {"header_sample": head.splitlines()[:3], **frame_profile(df)}
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    out["datetime_parse_failures"] = int(ts.isna().sum())
    out["duplicate_timestamps"] = int(ts.duplicated().sum())
    out["time"] = interval_summary(ts)
    num = df.drop(columns=["Date", "Time"]).astype("float64")
    out["rows_all_measurements_missing"] = int(num.isna().all(axis=1).sum())
    out["describe"] = num.describe().round(3).to_dict()

    # Unit checks. P[kW] <= V*I/1000 if P is in kW and I in A (power factor <= 1).
    ok = num.dropna()
    pf = ok["Global_active_power"] * 1000 / (ok["Voltage"] * ok["Global_intensity"])
    out["implied_power_factor_median"] = round(float(pf.median()), 4)
    out["implied_power_factor_share_gt_1.05"] = round(float((pf > 1.05).mean()), 5)
    # Sub-metering in Wh per minute: P[kW]*1000/60 should exceed the sum of sub-meters.
    resid = ok["Global_active_power"] * 1000 / 60 - ok[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)
    out["unmetered_wh_per_min_median"] = round(float(resid.median()), 3)
    out["unmetered_share_below_minus1wh"] = round(float((resid < -1).mean()), 5)

    # Daily coverage and a preview of daily kWh.
    present = num["Global_active_power"].notna()
    day = ts.dt.floor("D")
    cov = present.groupby(day).sum() / 1440
    full_days = pd.date_range(day.min(), day.max(), freq="D")
    cov = cov.reindex(full_days, fill_value=0)
    out["daily_coverage"] = {
        "calendar_days": int(len(full_days)),
        "days_ge_90pct": int((cov >= 0.90).sum()),
        "days_lt_90pct": int((cov < 0.90).sum()),
        "days_zero": int((cov == 0).sum()),
        "first_day_coverage": round(float(cov.iloc[0]), 3),
        "last_day_coverage": round(float(cov.iloc[-1]), 3),
    }
    kwh = (num["Global_active_power"].groupby(day).sum() / 60)[cov[cov >= 0.9].index]
    out["daily_kwh_on_good_days"] = kwh.describe().round(2).to_dict()
    out["missing_minutes_by_year"] = (~present).groupby(ts.dt.year).sum().astype(int).to_dict()
    return out


# --------------------------------------------------------------------------- Kaggle solar plants
KAGGLE_FORMATS = ["%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S"]


def parse_with_known_format(s: pd.Series) -> tuple[pd.Series, str | None]:
    """Try each explicit format; return the first that parses every value."""
    for fmt in KAGGLE_FORMATS:
        ts = pd.to_datetime(s, format=fmt, errors="coerce")
        if ts.notna().all():
            return ts, fmt
    return pd.to_datetime(s, errors="coerce"), None


def profile_kaggle_solar(zip_path: Path) -> dict[str, Any]:
    """Kaggle 'Solar Power Generation Data': 2 plants x (generation, weather)."""
    out: dict[str, Any] = {"zip": rel(zip_path), "files": {}}
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            with zf.open(name) as fh:
                df = pd.read_csv(fh, dtype={"DATE_TIME": str, "SOURCE_KEY": str})
            sample = df["DATE_TIME"].iloc[0]
            ts, fmt = parse_with_known_format(df["DATE_TIME"])
            df["DATE_TIME"] = ts
            frames[name] = df
            info = {"datetime_sample": sample, "datetime_format": fmt, **frame_profile(df),
                    "plant_ids": sorted(df["PLANT_ID"].unique().tolist()),
                    "n_source_keys": int(df["SOURCE_KEY"].nunique()),
                    "time": interval_summary(df["DATE_TIME"]),
                    "duplicate_timestamp_source": int(df.duplicated(["DATE_TIME", "SOURCE_KEY"]).sum())}
            numeric = df.select_dtypes("number").drop(columns=["PLANT_ID"])
            info["negative_values"] = {c: int((numeric[c] < 0).sum()) for c in numeric if (numeric[c] < 0).any()}
            info["describe"] = numeric.describe().round(3).to_dict()
            if "AC_POWER" in df:
                slots = pd.date_range(ts.min().floor("D"), ts.max().floor("D") + pd.Timedelta("23h45min"), freq="15min")
                per_inv = df.groupby("SOURCE_KEY").size()
                info["expected_slots_per_inverter"] = int(len(slots))
                info["rows_per_inverter"] = {"min": int(per_inv.min()), "median": int(per_inv.median()), "max": int(per_inv.max())}
                info["missing_slot_pct_per_inverter_median"] = round(float(1 - per_inv.median() / len(slots)) * 100, 2)
                on = df[df["AC_POWER"] > 0]
                ratio = on["DC_POWER"] / on["AC_POWER"]
                info["dc_ac_ratio_when_on"] = {"median": round(float(ratio.median()), 3),
                                               "p05": round(float(ratio.quantile(0.05)), 3),
                                               "p95": round(float(ratio.quantile(0.95)), 3)}
                info["max_ac_power_per_inverter_median"] = round(float(df.groupby("SOURCE_KEY")["AC_POWER"].max().median()), 2)
            else:
                slots = pd.date_range(ts.min().floor("D"), ts.max().floor("D") + pd.Timedelta("23h45min"), freq="15min")
                info["expected_slots"] = int(len(slots))
                info["missing_slot_pct"] = round(float(1 - df["DATE_TIME"].nunique() / len(slots)) * 100, 2)
            out["files"][name] = info

    for p in (1, 2):
        gen, wx = frames[f"Plant_{p}_Generation_Data.csv"], frames[f"Plant_{p}_Weather_Sensor_Data.csv"]
        m = gen.merge(wx[["DATE_TIME", "PLANT_ID", "IRRADIATION"]], on=["DATE_TIME", "PLANT_ID"], how="left")
        out[f"plant_{p}_join"] = {
            "gen_rows": int(len(gen)),
            "gen_rows_without_weather": int(m["IRRADIATION"].isna().sum()),
            "ac_power_gt0_at_zero_irradiation": int(((m["AC_POWER"] > 0) & (m["IRRADIATION"] == 0)).sum()),
            "ac_power_zero_at_irradiation_gt_0.1": int(((m["AC_POWER"] == 0) & (m["IRRADIATION"] > 0.1)).sum()),
            "corr_ac_irradiation_daytime": round(float(m.loc[m["IRRADIATION"] > 0, ["AC_POWER", "IRRADIATION"]].corr().iloc[0, 1]), 4),
        }
    return out


# --------------------------------------------------------------------------- NASA POWER
def profile_nasa_power(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("-END HEADER-")
    header = lines[1:end]
    loc = re.search(r"[Ll]atitude\s+([-\d.]+)\s+[Ll]ongitude\s+([-\d.]+)", " ".join(header))
    params = [h.strip() for h in header if re.match(r"^[A-Z0-9_]{3,}\s{2,}", h)]
    df = pd.read_csv(path, skiprows=end + 1)
    date_cols = [c for c in ["YEAR", "MO", "DY", "HR"] if c in df]
    ts = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.get("HR", 0)))
    values = df.drop(columns=date_cols)
    return {"title": header[0].strip(), "latitude": float(loc.group(1)), "longitude": float(loc.group(2)),
            "parameters": params, **frame_profile(df), "time": interval_summary(ts),
            "missing_minus999": {c: int((values[c] == -999).sum()) for c in values},
            "describe": values.replace(-999, np.nan).describe().round(3).to_dict()}


# --------------------------------------------------------------------------- NCA solar radiation
def profile_nca(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce")
    val = df["Solar Radiation (Watt/m2)"]
    stations = (df.assign(ts=ts).groupby(["Station", "District", "Latitude", "Longitude"])
                .agg(rows=("ts", "size"), start=("ts", "min"), end=("ts", "max")).reset_index())
    hours = ts.dt.hour.value_counts().sort_index()
    return {**frame_profile(df), "datetime_parse_failures": int(ts.isna().sum()),
            "duplicate_station_timestamp": int(df.assign(ts=ts).duplicated(["Station", "ts"]).sum()),
            "stations": stations.astype(str).to_dict("records"),
            "agencies": df["Agency"].unique().tolist(), "states": df["State"].unique().tolist(),
            "value_describe": val.describe().round(2).to_dict(),
            "share_zero": round(float((val == 0).mean()), 4),
            "rows_by_hour": {int(k): int(v) for k, v in hours.items()},
            "max_value_by_hour": {int(k): float(v) for k, v in val.groupby(ts.dt.hour).max().items()}}


# --------------------------------------------------------------------------- PVDAQ systems metadata
def profile_pvdaq_systems(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    lat = next(c for c in df.columns if "latitude" in c)
    lon = next(c for c in df.columns if "longitude" in c)
    india = df[df[lat].between(6, 37) & df[lon].between(68, 98)]
    out = {**frame_profile(df), "unique_system_ids": int(df["system_id"].nunique()),
           "systems_in_india_bbox": int(len(india)),
           "has_time_series_columns": False}
    if "dc_capacity_kW" in df:
        out["dc_capacity_kW_describe"] = df["dc_capacity_kW"].describe().round(2).to_dict()
        out["qa_status_counts"] = df["qa_status"].value_counts().to_dict()
    return out


# --------------------------------------------------------------------------- NITI Aayog insolation
def profile_niti_xlsx(path: Path) -> dict[str, Any]:
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet)
    footer_start = raw.index[raw.iloc[:, 0].astype(str).str.startswith("Copyright")]
    footer = raw.iloc[footer_start[0]:, 0].dropna().tolist() if len(footer_start) else []
    df = raw.iloc[: footer_start[0]] if len(footer_start) else raw
    df = df.dropna(how="all")
    value_col = df.columns[-1]
    out = {"sheets": xl.sheet_names, "footer": footer, **frame_profile(df),
           "value_column": value_col, "value_describe": pd.to_numeric(df[value_col]).describe().round(2).to_dict()}
    for c in df.columns[:-1]:
        out[f"unique_{c}"] = int(df[c].nunique())
        if df[c].nunique() <= 15:
            out[f"values_{c}"] = sorted(map(str, df[c].unique()))
    return out


# --------------------------------------------------------------------------- Grid-India SCADA
GRID_REPORT_RENAME = {"Demand (MW)": "NLDC_DEMAND", "Wind (MW)": "ALL_INDIA_WIND",
                      "Solar (MW)": "ALL_IND_SOLAR", "Total Generation (MW)": "TOTAL_GENERATION"}


def read_grid_india_file(path: Path) -> tuple[pd.DataFrame, str]:
    """Read one Grid-India workbook into Time + MW columns.

    Two layouts exist: SCADA exports ('Sheet1', tag row + label row, formula columns
    ignored) and a later hourly 'Report' sheet with a header row and summary rows at the end.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if "Sheet1" in wb.sheetnames:
        rows = wb["Sheet1"].iter_rows(values_only=True)
        next(rows)
        labels = next(rows)
        keep = [i for i, t in enumerate(labels) if t and "|P" in str(t)]
        data = [(r[0], *[r[i] for i in keep]) for r in rows if r[0] is not None]
        df = pd.DataFrame(data, columns=["Time", *[str(labels[i]).split("|")[0] for i in keep]])
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        layout = "scada_sheet1"
    else:
        rows = list(wb["Report"].iter_rows(values_only=True))
        df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=GRID_REPORT_RENAME)
        df["Time"] = pd.to_datetime(df.pop("Timestamp"), format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df = df[df["Time"].notna()]  # drops the trailing Timestamp/Average/Sum summary rows
        df = df[["Time", *GRID_REPORT_RENAME.values()]]
        layout = "hourly_report"
    wb.close()
    return df, layout


def profile_grid_india(folder: Path) -> dict[str, Any]:
    """Grid-India (NLDC) all-India demand and generation, in MW."""
    files, frames = [], []
    for p in sorted(folder.glob("*.xlsx")):
        df, layout = read_grid_india_file(p)
        num = df.drop(columns="Time").apply(pd.to_numeric, errors="coerce")
        files.append({"file": p.name, "layout": layout, "size_mb": round(p.stat().st_size / 1e6, 1),
                      "rows": len(df), **interval_summary(df["Time"]),
                      "duplicate_timestamps": int(df["Time"].duplicated().sum()),
                      "missing_pct": {c: round(float(v) * 100, 3) for c, v in num.isna().mean().items() if v > 0},
                      "columns": df.columns.tolist()[1:]})
        # Compare files on a common hourly basis (mean MW per hour).
        hourly = num.assign(Time=df["Time"]).set_index("Time")[["NLDC_DEMAND", "ALL_INDIA_WIND", "ALL_IND_SOLAR"]].resample("1h").mean()
        frames.append(hourly)
        log.info("grid-india %s rows=%d", p.name, len(df))
    allh = pd.concat(frames).sort_index()
    allh = allh[allh.notna().any(axis=1)]
    return {"files": files, "hourly_rows": int(len(allh)),
            "hourly_overlapping_hours": int(allh.index.duplicated().sum()),
            "time_hourly": interval_summary(allh.index.to_series()),
            "describe_hourly_MW": allh.describe().round(1).to_dict(),
            "negative_solar_hours": int((allh["ALL_IND_SOLAR"] < 0).sum())}


# --------------------------------------------------------------------------- Solar_PV capacity factors
def profile_solar_pv(folder: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"files": []}
    first_cols = None
    for p in sorted(folder.rglob("*.csv")):
        header = p.open().readline().strip().split(",")
        types = {c: pa.float32() for c in header[1:]}
        tbl = pacsv.read_csv(p, convert_options=pacsv.ConvertOptions(column_types=types))
        time = pd.to_datetime(tbl.column("time").to_pandas(), format="%Y-%m-%d %H:%M:%S")
        vals = np.column_stack([tbl.column(c).to_numpy(zero_copy_only=False) for c in header[1:]])
        col_mean = np.nanmean(vals, axis=0)
        out["files"].append({
            "file": rel(p), "rows": int(len(time)), "region_columns": len(header) - 1,
            "time": interval_summary(time), "missing_values": int(np.isnan(vals).sum()),
            "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
            "mean_capacity_factor": round(float(np.nanmean(vals)), 4),
            "region_mean_cf_range": [round(float(col_mean.min()), 4), round(float(col_mean.max()), 4)],
            "all_zero_regions": int((np.nanmax(vals, axis=0) == 0).sum()),
        })
        same = first_cols is None or header == first_cols
        first_cols = first_cols or header
        out["files"][-1]["same_columns_as_first"] = same
        log.info("solar_pv %s", p.name)
        del tbl, vals
    region_ids = sorted(int(c.split("_")[1]) for c in first_cols[1:])
    out["region_id_range"] = [region_ids[0], region_ids[-1]]
    out["region_ids_missing_in_range"] = len(set(range(region_ids[0], region_ids[-1] + 1)) - set(region_ids))
    out["lat_lon_lookup_present"] = False
    return out


# --------------------------------------------------------------------------- PV360 (claimed household data)
def profile_pv360(zip_path: Path) -> dict[str, Any]:
    """Profile PV360 and run checks that help separate measured from generated data."""
    out: dict[str, Any] = {"zip": rel(zip_path)}
    with zipfile.ZipFile(zip_path) as zf:
        out["member_dates"] = sorted({"%04d-%02d-%02d %02d:%02d" % i.date_time[:5] for i in zf.infolist()})
        m5 = pd.read_csv(zf.open("PV360_data/data/measured_5min.csv"))
        daily = pd.read_csv(zf.open("PV360_data/data/daily_with_weather.csv"))
        nasa = pd.read_csv(zf.open("PV360_data/data/nasa_power_daily.csv"))
        out["specs"] = json.loads(zf.read("PV360_data/data/system_specifications.json"))
        out["results_forecast"] = zf.read("PV360_data/results/forecast_model_comparison.csv").decode()
        out["code_head"] = zf.read("PV360_data/code/pv_analytics.py").decode()[:1500]
    out["measured_5min"] = frame_profile(m5)
    ts = pd.to_datetime(m5["timestamp"], format="%Y-%m-%d %H:%M:%S")
    out["measured_5min"]["time"] = interval_summary(ts)
    out["measured_5min"]["timestamps_on_exact_5min_grid"] = round(float((ts.dt.minute % 5 == 0).mean() * (ts.dt.second == 0).mean()), 4)
    out["measured_5min"]["describe"] = m5.drop(columns="timestamp").describe().round(2).to_dict()

    checks: dict[str, Any] = {}
    for c in m5.columns[1:]:
        v = m5[c].dropna()
        checks[c] = {
            "unique_values": int(v.nunique()),
            "share_integer": round(float((v == v.round()).mean()), 4),
            "share_multiple_of_10": round(float((v % 10 == 0).mean()), 4),
            "last_digit_share": {int(k): round(float(s), 3) for k, s in (v.round().abs() % 10).value_counts(normalize=True).sort_index().items()},
        }
    # Net grid identity: grid = import - feedin (sign convention unknown)
    g = m5.dropna(subset=["grid_power_W", "import_power_W", "feedin_power_W"])
    checks["grid_minus_import_plus_feedin_abs_median"] = round(float((g.grid_power_W - g.import_power_W + g.feedin_power_W).abs().median()), 3)
    # PV dynamics: lag-1 autocorrelation of 5-min changes at midday. Real cloud transients
    # tend to persist; independent random noise around a smooth curve gives about -0.5.
    pv = m5.assign(ts=ts).set_index("ts")["pv_production_W"]
    mid = pv[(pv.index.hour >= 10) & (pv.index.hour < 14)]
    d = mid.diff()
    d = d[mid.index.to_series().diff() == pd.Timedelta("5min")]
    checks["pv_midday_diff_lag1_autocorr"] = round(float(d.autocorr(1)), 4)
    checks["pv_night_share_zero"] = round(float((pv[(pv.index.hour < 5) | (pv.index.hour >= 19)] <= 0).mean()), 4)
    checks["pv_max_W_vs_rated_5400"] = float(pv.max())
    out["synthetic_checks"] = checks

    # Daily consistency and weather link.
    out["daily_rows"] = int(len(daily))
    out["daily_columns"] = daily.columns.tolist()
    out["daily_corr_prod_ghi"] = round(float(daily[["prod_e", "ghi"]].corr().iloc[0, 1]), 4)
    out["daily_prod_kwh_sum"] = round(float(daily["prod_e"].sum()), 1)
    out["daily_cons_kwh_sum"] = round(float(daily["cons_e"].sum()), 1)
    out["nasa_rows"] = int(len(nasa))
    return out


# --------------------------------------------------------------------------- main
def run_audit() -> dict[str, Any]:
    # Paths follow the layout approved after the first audit (reports/data_audit.md §5).
    raw = get_path("raw")
    irr = raw / "india_solar_irradiance"
    result: dict[str, Any] = {}
    result["inventory"] = file_inventory(raw)
    result["duplicate_groups"] = duplicate_groups(result["inventory"])
    result["uci"] = profile_uci(raw / "uci_household_power" / "household_power_consumption.txt")
    result["kaggle_solar"] = profile_kaggle_solar(raw / "kaggle_solar_plants" / "archive.zip")
    result["nasa_power"] = {rel(p): profile_nasa_power(p) for p in sorted((raw / "nasa_power").glob("*.csv"))}
    result["nca_radiation"] = profile_nca(raw / "unusable" / "nca_sardar_sarovar_solar_radiation_2022-2025.csv")
    result["pvdaq_systems"] = {p.name: profile_pvdaq_systems(p) for p in sorted((raw / "nrel_pvdaq_metadata").glob("systems*.csv"))}
    result["niti"] = {p.name: profile_niti_xlsx(p) for p in sorted(irr.glob("*.xlsx"))}
    result["pv360"] = profile_pv360(raw / "unverified" / "pv360_philippines" / "PV360_data.zip")
    result["grid_india"] = profile_grid_india(raw / "electricity_demand_solar_wind")
    result["solar_pv"] = profile_solar_pv(raw / "modelled_pv_capacity_factors")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_audit()
    out = project_path("reports/audit/inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
