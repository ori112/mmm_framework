"""Budget allocation optimization using MultiDimensionalBudgetOptimizerWrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_mmm.optimize.bounds import channel_bounds, make_budget_bounds_xarray


def optimize(
    mmm,
    X: pd.DataFrame,
    spec,
    strategy: str = "moderate",
    total_budget: float | None = None,
) -> dict:
    """Run budget optimization and return allocation + uplift summary.

    Returns dict with keys: optimal_allocation, current_allocation,
    expected_uplift_mean, expected_uplift_p5, expected_uplift_p95, p_positive_uplift.
    All monetary values in spec.currency (ILS by default).
    """
    from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper
    import xarray as xr

    channel_cols = [ch.spend_col for ch in spec.channels if ch.spend_col in X.columns]
    current_alloc = {ch: float(X[ch].sum()) for ch in channel_cols}

    if total_budget is None:
        total_budget = sum(current_alloc.values())

    bounds = channel_bounds(spec, X, strategy=strategy)
    budget_bounds = make_budget_bounds_xarray(bounds)

    date_col = spec.data.date_col
    start_date = pd.to_datetime(X[date_col].min()).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(X[date_col].max()).strftime("%Y-%m-%d")

    wrapper = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm,
        start_date=start_date,
        end_date=end_date,
    )

    opt_alloc, opt_result = wrapper.optimize_budget(
        budget=total_budget,
        budget_bounds=budget_bounds,
    )

    # Sample response distribution for current vs optimal
    n_periods = len(X)
    current_alloc_da = xr.DataArray(
        [current_alloc[ch] / n_periods for ch in channel_cols],
        dims=["channel"],
        coords={"channel": channel_cols},
    )
    optimal_alloc_da = xr.DataArray(
        [float(opt_alloc.sel(channel=ch).values) / n_periods for ch in channel_cols],
        dims=["channel"],
        coords={"channel": channel_cols},
    )

    current_resp = wrapper.sample_response_distribution(current_alloc_da)
    optimal_resp = wrapper.sample_response_distribution(optimal_alloc_da)

    resp_var = "total_media_contribution_original_scale"
    current_total = current_resp[resp_var].values.flatten()
    optimal_total = optimal_resp[resp_var].values.flatten()
    uplift = optimal_total - current_total

    return {
        "optimal_allocation": {ch: float(opt_alloc.sel(channel=ch).values) for ch in channel_cols},
        "current_allocation": current_alloc,
        "expected_uplift_mean": float(uplift.mean()),
        "expected_uplift_p5": float(np.percentile(uplift, 5)),
        "expected_uplift_p95": float(np.percentile(uplift, 95)),
        "p_positive_uplift": float((uplift > 0).mean()),
        "currency": getattr(spec, "currency", "ILS"),
    }
