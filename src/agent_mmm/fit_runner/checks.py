"""Prior and posterior predictive checks."""
from __future__ import annotations

import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM


def run_prior_predictive(mmm: MMM, X: pd.DataFrame, y: pd.Series, samples: int = 200):
    """Run prior predictive check. Returns prior idata."""
    y_named = y.rename(mmm.target_column) if y.name != mmm.target_column else y
    return mmm.sample_prior_predictive(X=X, y=y_named, samples=samples)


def run_posterior_predictive(mmm: MMM, X: pd.DataFrame):
    """Run in-sample posterior predictive check. Extends mmm.idata in-place."""
    return mmm.sample_posterior_predictive(X, extend_idata=True)
