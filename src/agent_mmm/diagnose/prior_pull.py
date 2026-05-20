"""Prior dominance check: posterior vs prior moment shift per parameter.

Posterior should be narrower than prior (data updated beliefs).
posterior_std / prior_std < 0.8 means the data informed beliefs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prior_pull_ratio(idata, param: str) -> dict:
    """Return posterior_std / prior_std for `param` across channels.

    Returns {channel: ratio} or {scalar: ratio} for scalar params.
    Values < 0.8 indicate the data pulled the posterior.
    Values > 1.0 indicate the prior is dominating (possible too-tight prior).
    Returns empty dict if param not in both prior and posterior.
    """
    if idata is None:
        return {}

    posterior = idata.posterior if hasattr(idata, "posterior") else {}
    prior_group = idata.prior if hasattr(idata, "prior") else {}

    if param not in posterior or param not in prior_group:
        return {}

    post = posterior[param]
    pri = prior_group[param]

    post_std = post.std(dim=("chain", "draw"))
    pri_std = pri.std(dim=("chain", "draw"))

    if "channel" in post.dims:
        channels = post.coords["channel"].values.tolist()
        return {
            ch: round(float(post_std.sel(channel=ch).values) / float(pri_std.sel(channel=ch).values + 1e-8), 4)
            for ch in channels
        }
    else:
        return {"scalar": round(float(post_std.values) / float(pri_std.values + 1e-8), 4)}


def audit_prior_pull(idata, params: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame summarising prior pull for all key parameters.

    Columns: param, channel/scalar, ratio, flag ("INFORMED"|"DOMINATED").
    """
    params = params or ["adstock_alpha", "saturation_lam", "intercept"]
    rows = []
    for p in params:
        ratios = prior_pull_ratio(idata, p)
        for label, ratio in ratios.items():
            rows.append({
                "param": p,
                "label": label,
                "ratio": ratio,
                "flag": "INFORMED" if ratio < 0.8 else "DOMINATED",
            })
    return pd.DataFrame(rows)
