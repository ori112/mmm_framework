"""Budget bound helpers: derive per-channel min/max from spec and observed spend."""
from __future__ import annotations

import pandas as pd


def channel_bounds(
    spec,
    X: pd.DataFrame,
    strategy: str = "moderate",
) -> dict[str, tuple[float, float]]:
    """Return (min, max) spend bounds per channel column.

    strategy: "conservative" ±20%, "moderate" ±50%, "aggressive" ±80%
    """
    widths = {"conservative": 0.20, "moderate": 0.50, "aggressive": 0.80}
    w = widths.get(strategy, 0.50)

    channel_cols = [ch.spend_col for ch in spec.channels]
    bounds: dict[str, tuple[float, float]] = {}
    for col in channel_cols:
        if col not in X.columns:
            continue
        total = float(X[col].sum())
        bounds[col] = (total * (1 - w), total * (1 + w))
    return bounds


def make_budget_bounds_xarray(bounds: dict[str, tuple[float, float]]):
    """Convert bounds dict to xr.DataArray with dims=["channel","bound"]."""
    import xarray as xr
    channels = list(bounds.keys())
    data = [[lo, hi] for lo, hi in bounds.values()]
    return xr.DataArray(
        data=data,
        dims=["channel", "bound"],
        coords={"channel": channels, "bound": ["lower", "upper"]},
    )
