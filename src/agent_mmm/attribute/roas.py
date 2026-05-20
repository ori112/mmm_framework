"""Effectiveness metrics: ROAS for revenue targets, cost-per-unit for acquisitions/volume.

All per-sample, never mean-then-divide.
"""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _contrib_samples(mmm):
    """Return (contrib_total, channels) where contrib_total has dims (chain, draw, channel)."""
    cc = mmm.idata.posterior["channel_contribution"]
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    contrib = cc * target_scale  # (chain, draw, date, channel)
    return contrib.sum("date"), contrib.coords["channel"].values.tolist()


def _metric_label(spec) -> str:
    """Return the short metric label for this target: ROAS, CPL, CPA, CPP, CPU …"""
    if spec.target.type == "revenue":
        return "ROAS"
    unit = (spec.target.unit_name or "").strip().lower()
    if unit:
        return f"CP{unit[0].upper()}"   # lead → CPL, policy → CPP, install → CPI
    # Fallback by type
    return {"acquisitions": "CPA", "volume": "CPU"}.get(spec.target.type, "CPA")


# ── Public API ────────────────────────────────────────────────────────────────

def compute_effectiveness(mmm, X: pd.DataFrame, spec, hdi_prob: float = 0.89) -> pd.DataFrame:
    """Unified effectiveness metric dispatcher.

    Returns a normalised DataFrame regardless of target type:

    Columns:
      channel            – spend column name (matches X)
      spend_ils          – total spend over the period (ILS)
      contribution_mean  – mean incremental contribution (ILS for revenue, units otherwise)
      metric_label       – "ROAS", "CPL", "CPA", "CPP", "CPU" …
      metric_value_mean  – mean effectiveness value
      metric_hdi89_low   – 89% HDI lower bound
      metric_hdi89_high  – 89% HDI upper bound
      implied_roas_mean  – only present when value_per_unit > 1 on a non-revenue target

    ROAS = contribution / spend  (revenue targets, dimensionless ratio)
    CPA/CPL/… = spend / contribution  (unit targets, ILS per unit)
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    contrib_total, channels_in_model = _contrib_samples(mmm)
    label = _metric_label(spec)
    is_revenue = spec.target.type == "revenue"
    value_per_unit = spec.target.value_per_unit or 1.0

    rows = []
    for ch in channels_in_model:
        if ch not in X.columns:
            continue
        spend = float(X[ch].sum())
        if spend == 0:
            continue

        ch_contrib = contrib_total.sel(channel=ch)  # (chain, draw)
        contrib_mean = float(ch_contrib.mean(dim=("chain", "draw")).values)

        if is_revenue:
            # ROAS samples
            metric_samples = ch_contrib / spend
        else:
            # cost-per-unit samples — guard against zero contribution draws
            metric_samples = spend / ch_contrib.where(ch_contrib > 0)

        metric_mean = float(metric_samples.mean(dim=("chain", "draw")).values)

        # HDI (fill NaN for cost-per-unit edge case)
        samples_for_hdi = metric_samples if is_revenue else metric_samples.fillna(metric_mean)
        hdi = az.hdi(samples_for_hdi, hdi_prob=hdi_prob)
        hdi_vals = hdi["channel_contribution"].values
        hdi_low, hdi_high = float(hdi_vals[0]), float(hdi_vals[1])

        row = {
            "channel": ch,
            "spend_ils": spend,
            "contribution_mean": contrib_mean,
            "metric_label": label,
            "metric_value_mean": metric_mean,
            f"metric_hdi{int(hdi_prob * 100)}_low": hdi_low,
            f"metric_hdi{int(hdi_prob * 100)}_high": hdi_high,
        }

        # Implied ROAS for non-revenue targets with a known value per unit
        if not is_revenue and value_per_unit > 1.0 and metric_mean > 0:
            row["implied_roas_mean"] = round(value_per_unit / metric_mean, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Legacy wrappers (kept for backward compatibility) ─────────────────────────

def compute_roas(mmm, X: pd.DataFrame, hdi_prob: float = 0.89) -> pd.DataFrame:
    """Compute ROAS. For revenue targets only.

    Kept for backward compatibility — prefer compute_effectiveness().
    Returns DataFrame with roas_mean and roas_hdi<p>_low/high columns.
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    contrib_total, channels_in_model = _contrib_samples(mmm)
    rows = []
    for ch in channels_in_model:
        if ch not in X.columns:
            continue
        spend = float(X[ch].sum())
        if spend == 0:
            continue
        roas_samples = contrib_total.sel(channel=ch) / spend
        roas_mean = float(roas_samples.mean(dim=("chain", "draw")).values)
        hdi = az.hdi(roas_samples, hdi_prob=hdi_prob)
        contrib_mean = float(contrib_total.sel(channel=ch).mean(dim=("chain", "draw")).values)
        rows.append({
            "channel": ch,
            "spend_ils": spend,
            "contribution_mean_ils": contrib_mean,
            "roas_mean": roas_mean,
            f"roas_hdi{int(hdi_prob * 100)}_low": float(hdi["channel_contribution"].values[0]),
            f"roas_hdi{int(hdi_prob * 100)}_high": float(hdi["channel_contribution"].values[1]),
        })
    return pd.DataFrame(rows)


def compute_cpa(mmm, X: pd.DataFrame, hdi_prob: float = 0.89) -> pd.DataFrame:
    """Compute CPA. For non-revenue targets.

    Kept for backward compatibility — prefer compute_effectiveness().
    """
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError("Model has no idata — run fit() first.")

    contrib_total, channels_in_model = _contrib_samples(mmm)
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
