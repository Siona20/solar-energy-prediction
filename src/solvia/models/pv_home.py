"""Daily solar units (kWh) of a home system from the day's sunlight and max / min temperature.

    kWh = kWp × sunlight (kWh/m²/day) × other_losses × (1 + γ (T_cell − 25 °C))
    T_cell = T_air_sunny + k × G_sunny,  T_air_sunny = T_min + a (T_max − T_min),  G_sunny = b × sunlight

k, a and b are fitted on the TRAIN days of the two Kaggle plants in India (15-minute measured irradiance, ambient
and module temperature, AC power). γ is the usual crystalline-silicon datasheet value: fitting it on 24 days gave
-0.15 to -0.75 %/°C depending on the irradiance band (hot hours are also bright hours), too unstable to use. With
daily inputs only, the error on the 5 validation days falls from 5.63 % without a temperature effect to 4.05 % with
it (healthy inverter-days, reports/pv_home_report.md). Sunlight on the plants is plane-of-array; for homes the
formula is used with NASA POWER horizontal sunlight, so tilt gain is not modelled: the result is an ESTIMATE, and the
user can calibrate it with the units their own system gives on a clear sunny day.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from solvia.config import get_path, project_path

log = logging.getLogger(__name__)

GAMMA_DATASHEET = -0.004       # per °C, typical crystalline silicon module
OTHER_LOSSES = 0.86            # inverter ~0.97 × wiring ~0.98 × dust ~0.95 × mismatch/other ~0.95 (assumption)
INTERVAL_H = 0.25


@dataclass
class PVHomeModel:
    k_c_per_kw_m2: float       # module heating above air per kW/m² of sunlight
    a_sunny_air: float         # share of the day's temperature range reached, sunlight-weighted
    b_sunny_irradiance: float  # sunlight-weighted irradiance (kW/m²) per kWh/m²/day
    gamma_per_c: float = GAMMA_DATASHEET
    other_losses: float = OTHER_LOSSES

    def temp_factor(self, sunlight, tmax, tmin):
        t_air = np.asarray(tmin) + self.a_sunny_air * (np.asarray(tmax) - np.asarray(tmin))
        t_cell = t_air + self.k_c_per_kw_m2 * self.b_sunny_irradiance * np.asarray(sunlight)
        return 1 + self.gamma_per_c * (t_cell - 25)

    def daily_kwh(self, kwp: float, sunlight, tmax, tmin, scale: float = 1.0):
        """Units per day; `scale` is the user's calibration factor (1 = no calibration)."""
        s = np.clip(np.asarray(sunlight, dtype=float), 0, None)
        return kwp * s * self.other_losses * self.temp_factor(s, tmax, tmin) * scale

    def calibration_scale(self, kwp: float, sunny_day_units: float, sunlight, tmax, tmin) -> float:
        """Factor that makes the model give `sunny_day_units` on a reference clear day (sunlight, tmax, tmin)."""
        return float(sunny_day_units / self.daily_kwh(kwp, sunlight, tmax, tmin))


def _daily_plant_weather(f: pd.DataFrame) -> pd.DataFrame:
    w = f.drop_duplicates(["plant_id", "timestamp"]).assign(G=lambda x: x["irradiation_kw_m2"].clip(lower=0))
    w = w.assign(GG=w["G"] ** 2, GT=w["G"] * w["ambient_temp_c"])
    d = w.groupby(["split", "plant_id", "date"]).agg(G=("G", "sum"), GG=("GG", "sum"), GT=("GT", "sum"),
                                                      tmax=("ambient_temp_c", "max"), tmin=("ambient_temp_c", "min"))
    d["sunlight"] = d["G"] * INTERVAL_H
    d["g_sunny"], d["t_sunny"] = d["GG"] / d["G"], d["GT"] / d["G"]
    return d.reset_index()


def fit(f: pd.DataFrame) -> tuple[PVHomeModel, dict]:
    """Fit k, a, b on train days; report the daily error on validation days (test days are not used)."""
    t = f[(f["split"] == "train") & f["train_eligible"] & (f["irradiation_kw_m2"] > 0.05)]
    k = float((t["module_minus_ambient_c"] * t["irradiation_kw_m2"]).sum() / (t["irradiation_kw_m2"] ** 2).sum())
    d = _daily_plant_weather(f[f["split"] != "test"])
    tr = d[d["split"] == "train"]
    b = float((tr["g_sunny"] * tr["sunlight"]).sum() / (tr["sunlight"] ** 2).sum())
    rng = tr["tmax"] - tr["tmin"]
    a = float(((tr["t_sunny"] - tr["tmin"]) * rng).sum() / (rng ** 2).sum())
    model = PVHomeModel(k, a, b)

    # daily check with daily inputs only, per inverter scale fitted on train days; only inverter-days without a
    # shutdown or bad interval in daylight (a switched-off inverter is a fault, not a weather effect)
    e = f[f["split"] != "test"].assign(e=lambda x: x["ac_power_kw"] * INTERVAL_H,
                                       bad=lambda x: x["is_daylight"] & ~x["train_eligible"])
    inv = e.groupby(["split", "plant_id", "inverter_id", "date"]).agg(kwh=("e", "sum"), bad=("bad", "sum")).reset_index()
    inv = inv[inv["bad"] == 0].drop(columns="bad")
    inv = inv.merge(d[["split", "plant_id", "date", "sunlight", "tmax", "tmin"]], on=["split", "plant_id", "date"])
    checks = {}
    for name, gamma in [("no_temperature_effect", 0.0), ("datasheet_gamma", GAMMA_DATASHEET)]:
        m = PVHomeModel(k, a, b, gamma, 1.0)
        inv["x"] = m.daily_kwh(1.0, inv["sunlight"], inv["tmax"], inv["tmin"])
        c = inv[inv["split"] == "train"].groupby(["plant_id", "inverter_id"]).apply(
            lambda g: (g["kwh"] * g["x"]).sum() / (g["x"] ** 2).sum(), include_groups=False).rename("c")
        v = inv[inv["split"] == "valid"].join(c, on=["plant_id", "inverter_id"])
        err = (v["kwh"] - v["c"] * v["x"]).abs()
        checks[name] = {"valid_mape_pct": round(float((err / v["kwh"]).mean() * 100), 2),
                        "valid_mae_kwh_per_inverter_day": round(float(err.mean()), 1), "valid_inverter_days": len(v)}
    info = {"k_r2_train": round(float(1 - np.var(t["module_minus_ambient_c"] - k * t["irradiation_kw_m2"])
                                      / np.var(t["module_minus_ambient_c"])), 3),
            "train_days": int(tr["date"].nunique()), "daily_check": checks}
    log.info("pv_home: k=%.2f a=%.3f b=%.4f; validation %s", k, a, b, checks)
    return model, info


def run_pv_home() -> dict:
    f = pd.read_parquet(get_path("processed") / "solar_features_15min.parquet")
    model, info = fit(f)
    out = {"params": asdict(model), **info, "label": "ESTIMATE for home systems (fitted on two Indian utility plants)"}
    (get_path("models") / "pv_home.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    c = info["daily_check"]
    lines = ["# Home solar units model", "",
             "*Generated by `python -m solvia.pipeline --steps model_pv_home`.*", "",
             "kWh = kWp × sunlight × other losses × (1 + γ (T_cell − 25 °C)); T_cell from the day's sunlight and max / min "
             "air temperature.", "",
             "| Parameter | Value | From |", "|---|---|---|",
             f"| k, module heating | {model.k_c_per_kw_m2:.1f} °C per kW/m² (R² {info['k_r2_train']}) | Kaggle plants, {info['train_days']} train days |",
             f"| a, air temperature while sunny | T_min + {model.a_sunny_air:.2f} × (T_max − T_min) | same |",
             f"| b, sunlight-weighted irradiance | {model.b_sunny_irradiance:.3f} kW/m² per kWh/m²/day | same |",
             f"| γ, temperature coefficient | {model.gamma_per_c * 100:.1f} % per °C | crystalline-silicon datasheet value |",
             f"| other losses | {model.other_losses:.2f} | assumption: inverter, wiring, dust, mismatch |", "",
             "## Daily check on the 5 validation days (per-inverter scale fitted on train days)", "",
             "| Model | Daily error | MAE per inverter-day |", "|---|---|---|",
             *[f"| {n.replace('_', ' ')} | {v['valid_mape_pct']} % | {v['valid_mae_kwh_per_inverter_day']} kWh |" for n, v in c.items()], "",
             "The plants' 5 test days are not used. Limits: utility plants, May–June only (26–39 °C), plane-of-array "
             "sunlight; homes use NASA POWER horizontal sunlight, so the tilt gain is not modelled. A user calibration "
             "(units on a clear sunny day) corrects the overall level.", ""]
    project_path("reports/pv_home_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def load() -> PVHomeModel:
    return PVHomeModel(**json.loads((get_path("models") / "pv_home.json").read_text(encoding="utf-8"))["params"])
