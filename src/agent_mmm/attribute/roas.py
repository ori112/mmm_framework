"""ROAS calculation with posterior uncertainty (ILS-native)."""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from agent_mmm.attribute.contributions import get_contributions


def compute_roas(mmm, X: pd.DataFrame, hdi_prob: float = 0.89) -> pd.DataFrame:
    """Return per-channel ROAS with credible intervals.

    ROAS = total_contribution / total_spend (over the data period).
    All values are ILS/ILS (dimensionless ratio).

    Returns a DataFrame with columns: channel, spend, contribution_mean,
    roas_mean, roas_hdi_low, roas_hdi_high.
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    cc = mmm.idata.posterior["channel_contribution"]
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    contrib = cc * target_scale  # (chain, draw, date, channel)

    # Per-sample totals across dates → (chain, draw, channel)
    contrib_total = contrib.sum("date")

    channels_in_model = contrib_total.coords["channel"].values.tolist()

    rows = []
    for ch in channels_in_model:
        if ch not in X.columns:
            continue
        spend = float(X[ch].sum())
        if spend == 0:
            continue

        # ROAS samples: (chain, draw)
        roas_samples = contrib_total.sel(channel=ch) / spend
        roas_mean = float(roas_samples.mean(dim=("chain", "draw")).values)
        hdi = az.hdi(roas_samples, hdi_prob=hdi_prob)
        roas_low = float(hdi["channel_contribution"].values[0])
        roas_high = float(hdi["channel_contribution"].values[1])

        contrib_mean = float(contrib_total.sel(channel=ch).mean(dim=("chain", "draw")).values)

        rows.append({
            "channel": ch,
            "spend_ils": spend,
            "contribution_mean_ils": contrib_mean,
            "roas_mean": roas_mean,
            f"roas_hdi{int(hdi_prob * 100)}_low": roas_low,
            f"roas_hdi{int(hdi_prob * 100)}_high": roas_high,
        })

    return pd.DataFrame(rows)


def compute_cpa(mmm, X: pd.DataFrame, hdi_prob: float = 0.89) -> pd.DataFrame:
    """CPA = spend / incremental_units, for acquisition-type targets.

    Returns DataFrame with same structure as compute_roas but cpa_mean/hdi columns.
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    cc = mmm.idata.posterior["channel_contribution"]
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    contrib = cc * target_scale

    contrib_total = contrib.sum("date")
    channels_in_model = contrib_total.coords["channel"].values.tolist()

    rows = []
    for ch in channels_in_model:
        if ch not in X.columns:
            continue
        spend = float(X[ch].sum())
        contrib_samp = contrib_total.sel(channel=ch)
        units_mean = float(contrib_samp.mean(dim=("chain", "draw")).values)
        if units_mean == 0:
            continue

        cpa_samples = spend / contrib_samp.where(contrib_samp > 0)
        cpa_mean = float(cpa_samples.mean(dim=("chain", "draw")).values)
        hdi = az.hdi(cpa_samples.fillna(0), hdi_prob=hdi_prob)
        rows.append({
            "channel": ch,
            "spend_ils": spend,
            "units_mean": units_mean,
            "cpa_mean_ils": cpa_mean,
            f"cpa_hdi{int(hdi_prob * 100)}_low": float(hdi["channel_contribution"].values[0]),
            f"cpa_hdi{int(hdi_prob * 100)}_high": float(hdi["channel_contribution"].values[1]),
        })
    return pd.DataFrame(rows)
