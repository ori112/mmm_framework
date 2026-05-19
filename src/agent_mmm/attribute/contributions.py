"""Per-sample channel contributions in original (ILS) scale."""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


def get_contributions(mmm, hdi_prob: float = 0.89) -> pd.DataFrame:
    """Return mean and HDI contributions per channel, summed over the data period.

    Returns a DataFrame with columns: channel, mean, hdi_low, hdi_high (ILS).
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    cc = mmm.idata.posterior["channel_contribution"]
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    contrib = cc * target_scale  # (chain, draw, date, channel)

    # Per-sample total across dates → (chain, draw, channel)
    contrib_total = contrib.sum("date")

    mean_vals = contrib_total.mean(dim=("chain", "draw"))
    hdi_vals = az.hdi(contrib_total, hdi_prob=hdi_prob)

    channels = mean_vals.coords["channel"].values.tolist()
    rows = []
    for ch in channels:
        rows.append({
            "channel": ch,
            "mean": float(mean_vals.sel(channel=ch).values),
            "hdi_low": float(hdi_vals.sel(channel=ch)["channel_contribution"].values[0]),
            "hdi_high": float(hdi_vals.sel(channel=ch)["channel_contribution"].values[1]),
        })
    return pd.DataFrame(rows)


def get_contributions_timeseries(mmm) -> pd.DataFrame:
    """Return mean contributions per channel per date (wide format, ILS)."""
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    cc = mmm.idata.posterior["channel_contribution"]
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    contrib = (cc * target_scale).mean(dim=("chain", "draw"))  # (date, channel)

    df = contrib.to_dataframe("contribution").reset_index()
    return df.pivot(index="date", columns="channel", values="contribution").reset_index()
