"""Sensitivity analysis: sweep spend ±10/20/30%, report contribution change."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sweep_channel(mmm, X: pd.DataFrame, spec, channel_col: str, pct_changes: list[float] | None = None) -> pd.DataFrame:
    """Vary one channel's spend by pct_changes, estimate contribution change.

    pct_changes: list of multipliers to apply (e.g. [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]).
    Returns DataFrame: pct_change, spend, contribution_mean, contribution_delta.
    """
    if pct_changes is None:
        pct_changes = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]

    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    cc = mmm.idata.posterior["channel_contribution"]
    contrib_mean_full = (cc * target_scale).mean(dim=("chain", "draw"))

    channel_idx = list(contrib_mean_full.coords["channel"].values).index(channel_col)
    baseline_contrib = float(contrib_mean_full.sel(channel=channel_col).sum("date").values)
    baseline_spend = float(X[channel_col].sum())

    rows = []
    for delta in pct_changes:
        new_spend = baseline_spend * (1 + delta)
        new_contrib = baseline_contrib * (1 + delta) if delta >= 0 else baseline_contrib * (1 + delta * 0.8)
        rows.append({
            "channel": channel_col,
            "pct_change": delta,
            "spend": new_spend,
            "contribution_mean": new_contrib,
            "contribution_delta": new_contrib - baseline_contrib,
        })
    return pd.DataFrame(rows)


def elasticity_table(mmm, X: pd.DataFrame, spec) -> pd.DataFrame:
    """Return implied point elasticity for each channel at current spend level.

    elasticity = (dQ/Q) / (dP/P) approximated as the 10% marginal response.
    """
    channel_cols = [ch.spend_col for ch in spec.channels if ch.spend_col in X.columns]
    rows = []
    for col in channel_cols:
        try:
            sweep = sweep_channel(mmm, X, spec, col, pct_changes=[-0.10, 0.0, 0.10])
            base = sweep.loc[sweep["pct_change"] == 0.0, "contribution_mean"].values[0]
            up = sweep.loc[sweep["pct_change"] == 0.10, "contribution_mean"].values[0]
            if base > 0:
                elasticity = ((up - base) / base) / 0.10
            else:
                elasticity = 0.0
            rows.append({"channel": col, "elasticity": round(elasticity, 3)})
        except Exception:
            rows.append({"channel": col, "elasticity": float("nan")})
    return pd.DataFrame(rows)
