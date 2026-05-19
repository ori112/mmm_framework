"""In-sample fit metrics (R2, MAPE, wMAPE) from posterior predictive."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else float("nan")


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def compute_in_sample_metrics(mmm: MMM, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute in-sample R2, MAPE, wMAPE using posterior predictive mean.

    Note: per-sample metrics are more correct (E[f(x)] != f(E[x])) but
    require pymc_marketing.mmm.evaluation.compute_summary_metrics — added in Phase 2.
    """
    y_true = y.values.astype(float)

    pp = mmm.idata.posterior_predictive
    target_var = mmm.target_column
    if target_var not in pp:
        # Fall back to first variable
        target_var = list(pp.data_vars)[0]

    arr = pp[target_var]

    # Average over sample dims (handles both (chain,draw,...) and (sample,...))
    reduce_dims = [d for d in arr.dims if d in ("chain", "draw", "sample")]
    if reduce_dims:
        y_pred_norm = arr.mean(dim=reduce_dims).values.flatten()
    else:
        y_pred_norm = arr.values.flatten()

    # Un-normalize
    try:
        target_scale = float(mmm.get_scales_as_xarray()["target_scale"].values)
    except Exception:
        target_scale = float(y_true.max()) if y_true.max() > 0 else 1.0

    y_pred = y_pred_norm * target_scale

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    return {
        "in_sample_r2": round(_r2(y_true, y_pred), 4),
        "in_sample_mape": round(_mape(y_true, y_pred), 4),
        "in_sample_wmape": round(_wmape(y_true, y_pred), 4),
    }
