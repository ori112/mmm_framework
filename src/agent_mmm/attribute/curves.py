"""Response and saturation curves per channel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_saturation_curves(mmm, num_points: int = 100, max_value: float = 2.0) -> pd.DataFrame:
    """Sample saturation curves with posterior uncertainty.

    Returns a tidy DataFrame: channel, x, mean, hdi_low, hdi_high.
    x is normalized spend (0 → max_value × observed mean spend per channel).
    """
    import arviz as az

    curve = mmm.sample_saturation_curve(max_value=max_value, num_points=num_points)
    # curve dims: (chain, draw, x, channel)

    mean_curve = curve.mean(dim=("chain", "draw"))
    hdi_curve = az.hdi(curve, hdi_prob=0.89)

    rows = []
    for ch in curve.coords["channel"].values:
        x_vals = curve.coords["x"].values if "x" in curve.coords else np.linspace(0, max_value, num_points)
        for i, xv in enumerate(x_vals):
            rows.append({
                "channel": str(ch),
                "x_normalized": float(xv),
                "mean": float(mean_curve.sel(channel=ch).isel(x=i).values),
                "hdi_low": float(hdi_curve.sel(channel=ch).isel(x=i)[curve.name].values[0]),
                "hdi_high": float(hdi_curve.sel(channel=ch).isel(x=i)[curve.name].values[1]),
            })
    return pd.DataFrame(rows)


def saturation_point(mmm, channel: str, threshold: float = 0.90) -> float:
    """Return normalized spend at which the saturation curve reaches `threshold`
    of its maximum (i.e. approximate knee/ceiling).

    Returns 0.0 if the curve cannot be sampled.
    """
    try:
        curve = mmm.sample_saturation_curve(max_value=3.0, num_points=200)
        mean_curve = curve.mean(dim=("chain", "draw")).sel(channel=channel)
        y = mean_curve.values
        x = np.linspace(0, 3.0, len(y))
        ceiling = threshold * y.max()
        idx = np.searchsorted(y, ceiling)
        if idx >= len(x):
            return float(x[-1])
        return float(x[idx])
    except Exception:
        return 0.0
