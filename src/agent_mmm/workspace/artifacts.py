from __future__ import annotations

import json
from pathlib import Path

from .paths import ensure_workspace


def save_json(data: dict, path: Path) -> None:
    ensure_workspace()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_idata(idata, path: Path) -> None:
    """Save ArviZ InferenceData to NetCDF."""
    ensure_workspace()
    idata.to_netcdf(str(path))


def load_idata(path: Path):
    """Load ArviZ InferenceData from NetCDF."""
    import arviz as az
    return az.from_netcdf(str(path))


def save_report(content: str, path: Path) -> None:
    ensure_workspace()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
