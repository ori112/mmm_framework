"""Tighten priors from a previous model's posterior (brownfield / iterative improvement)."""
from __future__ import annotations

import arviz as az
import numpy as np
from pymc_extras.prior import Prior

from agent_mmm.prior_engine.moments import beta_moment_match, gamma_moment_match


TIGHTEN_FACTOR = 0.7
MIN_SIGMA = 0.01


def tighten_priors_from_idata(idata, model_config: dict, tighten_factor: float = TIGHTEN_FACTOR) -> dict:
    """Return a new model_config with priors tightened from the posterior.

    For each channel parameter:
    - new_mu = posterior mean
    - new_sigma = posterior std * tighten_factor (floored at MIN_SIGMA)
    - adstock_alpha → Beta re-matched via beta_moment_match
    - saturation_lam → Gamma re-matched via gamma_moment_match

    Unchanged parameters are carried through as-is.
    """
    import copy
    mc = copy.deepcopy(model_config)

    posterior = idata.posterior

    # adstock_alpha: (chain, draw, channel) → per-channel mean/std
    if "adstock_alpha" in posterior:
        alpha_samples = posterior["adstock_alpha"]
        channels = alpha_samples.coords["channel"].values.tolist()
        mu_vals = alpha_samples.mean(dim=("chain", "draw")).values
        sd_vals = alpha_samples.std(dim=("chain", "draw")).values

        # Per-channel Beta params
        alpha_a_list, alpha_b_list = [], []
        for mu, sd in zip(mu_vals, sd_vals):
            sd = max(sd * tighten_factor, MIN_SIGMA)
            mu = float(np.clip(mu, 0.01, 0.99))
            a, b = beta_moment_match(mu, sd)
            alpha_a_list.append(a)
            alpha_b_list.append(b)

        mc["adstock_alpha"] = Prior(
            "Beta",
            alpha=np.array(alpha_a_list),
            beta=np.array(alpha_b_list),
            dims="channel",
        )

    # saturation_lam: (chain, draw, channel)
    if "saturation_lam" in posterior:
        lam_samples = posterior["saturation_lam"]
        mu_vals = lam_samples.mean(dim=("chain", "draw")).values
        sd_vals = lam_samples.std(dim=("chain", "draw")).values

        lam_a_list, lam_b_list = [], []
        for mu, sd in zip(mu_vals, sd_vals):
            sd = max(sd * tighten_factor, MIN_SIGMA)
            mu = max(float(mu), 0.01)
            a, b = gamma_moment_match(mu, sd)
            lam_a_list.append(a)
            lam_b_list.append(b)

        mc["saturation_lam"] = Prior(
            "Gamma",
            alpha=np.array(lam_a_list),
            beta=np.array(lam_b_list),
            dims="channel",
        )

    return mc
