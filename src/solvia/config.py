"""Load configs/config.yaml and resolve project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the parsed YAML config."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str | Path) -> Path:
    """Resolve a config path (relative to the project root) to an absolute Path."""
    return PROJECT_ROOT / relative


def get_path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config."""
    return project_path(load_config()["paths"][key])
