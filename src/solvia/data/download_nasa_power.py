"""Download NASA POWER daily point data for the Indian locations in configs/india_locations.yaml.

    python -m solvia.data.download_nasa_power            # skips files that already exist
    python -m solvia.data.download_nasa_power --force    # re-download everything

Also downloads the `extra_points` (e.g. the UCI household at Sceaux) with their own periods and
the 1° `search_grid` used for the plant-location estimate.

Each file is downloaded to a temporary name, checked (NASA header, row count, no HTML error
page) and only then renamed into place. A manifest records URL, time, size, rows and MD5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from solvia.config import PROJECT_ROOT, project_path

log = logging.getLogger(__name__)
LOCATIONS_PATH = PROJECT_ROOT / "configs" / "india_locations.yaml"


def load_locations() -> tuple[dict, dict[str, list]]:
    cfg = yaml.safe_load(LOCATIONS_PATH.read_text(encoding="utf-8"))
    return cfg["nasa_power"], cfg["locations"]


def load_extra_points() -> dict[str, dict]:
    return yaml.safe_load(LOCATIONS_PATH.read_text(encoding="utf-8")).get("extra_points", {})


def file_name(name: str, lat: float, lon: float, start: str, end: str, tag: str | None = None) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{name}_{lat:.2f}N_{lon:.2f}E_daily_{start[:4]}-{end[:4]}{suffix}.csv"


def build_url(api: dict, lat: float, lon: float) -> str:
    query = {"parameters": ",".join(api["parameters"]), "community": api["community"],
             "longitude": f"{lon:.2f}", "latitude": f"{lat:.2f}",
             "start": api["start"], "end": api["end"], "format": "CSV"}
    return f"{api['base_url']}?{urllib.parse.urlencode(query, safe=',')}"


def validate(text: str, expected_rows: int) -> int:
    """Return the number of data rows, or raise if this is not a complete NASA POWER CSV."""
    if not text.startswith("-BEGIN HEADER-") or "-END HEADER-" not in text:
        raise ValueError("response is not a NASA POWER CSV (missing header block)")
    body = text.split("-END HEADER-", 1)[1].strip().splitlines()
    rows = len(body) - 1
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {rows}")
    return rows


def _n_days(api: dict) -> int:
    return (datetime.strptime(api["end"], "%Y%m%d") - datetime.strptime(api["start"], "%Y%m%d")).days + 1


def _fetch(url: str, days: int, retries: int, label: str) -> tuple[bytes, int]:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                raw = resp.read()
            return raw, validate(raw.decode("utf-8"), days)
        except Exception as exc:  # network errors and validation failures are retried
            log.warning("%s attempt %d failed: %s", label, attempt, exc)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def _download_one(target: Path, api: dict, lat: float, lon: float, retries: int, label: str) -> dict:
    url = build_url(api, lat, lon)
    raw, rows = _fetch(url, _n_days(api), retries, label)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(raw)
    tmp.rename(target)
    log.info("downloaded %s (%d rows, %.0f KB)", target.name, rows, len(raw) / 1024)
    return {"file": target.name, "latitude": lat, "longitude": lon, "url": url, "rows": rows,
            "bytes": len(raw), "md5": hashlib.md5(raw).hexdigest(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(sorted(manifest.values(), key=lambda m: m["file"]), indent=1))


def _read_manifest(path: Path) -> dict:
    return {m["file"]: m for m in json.loads(path.read_text())} if path.exists() else {}


def download_all(force: bool = False, pause_s: float = 1.0, retries: int = 3) -> list[dict]:
    api, locations = load_locations()
    out_dir = project_path(api["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.json"
    manifest = _read_manifest(manifest_path)

    for name, (state, region, lat, lon) in locations.items():
        target = out_dir / file_name(name, lat, lon, api["start"], api["end"])
        if target.exists() and not force:
            log.info("skip %s (exists)", target.name)
            continue
        entry = _download_one(target, api, lat, lon, retries, name)
        manifest[target.name] = {**entry, "location": name, "state": state, "region": region}
        _write_manifest(manifest_path, manifest)
        time.sleep(pause_s)
    return list(manifest.values())


def download_extra_points(force: bool = False, retries: int = 3) -> list[dict]:
    """Points outside the India list, each with its own period (configs/india_locations.yaml: extra_points)."""
    api, _ = load_locations()
    done = []
    for name, pt in load_extra_points().items():
        point_api = {**api, "start": pt["start"], "end": pt["end"]}
        out_dir = project_path(pt["out_dir"])
        target = out_dir / file_name(name, pt["latitude"], pt["longitude"], pt["start"], pt["end"], pt.get("tag"))
        if target.exists() and not force:
            log.info("skip %s (exists)", target.name)
            continue
        manifest_path = out_dir / "_manifest_extra_points.json"
        manifest = _read_manifest(manifest_path)
        manifest[target.name] = {**_download_one(target, point_api, pt["latitude"], pt["longitude"], retries, name),
                                 "location": name}
        _write_manifest(manifest_path, manifest)
        done.append(manifest[target.name])
    return done


def download_search_grid(force: bool = False, pause_s: float = 0.5, retries: int = 3) -> int:
    """1° grid over the region in configs/india_locations.yaml: search_grid (one file per point)."""
    api, _ = load_locations()
    g = yaml.safe_load(LOCATIONS_PATH.read_text(encoding="utf-8")).get("search_grid")
    if not g:
        return 0
    grid_api = {**api, "start": g["start"], "end": g["end"]}
    out_dir = project_path(g["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.json"
    manifest = _read_manifest(manifest_path)
    n = 0
    for lat in range(g["lat"][0], g["lat"][1] + 1):
        for lon in range(g["lon"][0], g["lon"][1] + 1):
            target = out_dir / file_name("grid", lat, lon, g["start"], g["end"])
            if target.exists() and not force:
                continue
            manifest[target.name] = _download_one(target, grid_api, float(lat), float(lon), retries, target.name)
            n += 1
            if n % 20 == 0:
                _write_manifest(manifest_path, manifest)
            time.sleep(pause_s)
    _write_manifest(manifest_path, manifest)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    force = parser.parse_args().force
    download_all(force=force)
    download_extra_points(force=force)
    download_search_grid(force=force)


if __name__ == "__main__":
    main()
