"""Recommend pymc-marketing model_config priors from spec channels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior

from ..spec.schema import Channel, Spec
from .moments import beta_moment_match, gamma_moment_match

# (keywords, (alpha_mu, alpha_sigma, lam_mu, lam_sigma))
_CHANNEL_TABLE: list[tuple[tuple[str, ...], tuple[float, float, float, float]]] = [
    (("sem", "search", "google", "bing", "ppc", "paid_search"),  (0.10, 0.07, 4.0, 1.0)),
    (("youtube", "video"),                                         (0.40, 0.15, 2.5, 0.65)),
    (("meta", "facebook", "instagram", "fb", "ig"),               (0.30, 0.12, 3.0, 0.8)),
    (("tiktok", "snapchat", "social"),                            (0.25, 0.10, 3.0, 0.8)),
    (("display", "banner", "programmatic", "dsp"),                (0.25, 0.12, 3.0, 0.8)),
    (("ooh", "outdoor", "billboard", "digital_ooh", "dooh"),      (0.60, 0.20, 2.0, 0.55)),
    (("tv", "television", "ctv", "streaming_tv"),                 (0.65, 0.20, 1.5, 0.45)),
    (("radio", "audio", "podcast", "spotify"),                    (0.45, 0.15, 2.0, 0.55)),
    (("print", "newspaper", "magazine"),                           (0.50, 0.17, 2.0, 0.55)),
    (("email", "crm"),                                             (0.05, 0.04, 4.5, 1.0)),
    (("affiliate", "referral"),                                    (0.15, 0.08, 3.5, 0.9)),
]

_DEFAULT_DIGITAL: tuple[float, float, float, float] = (0.20, 0.10, 3.0, 0.8)
_DEFAULT_OFFLINE: tuple[float, float, float, float] = (0.60, 0.20, 1.5, 0.45)


def _classify(ch: Channel) -> tuple[float, float, float, float]:
    """Return (alpha_mu, alpha_sigma, lam_mu, lam_sigma) for a channel."""
    # Respect manually-set spec priors
    if ch.alpha_mean != 0.5 or ch.alpha_sigma != 0.15:
        return ch.alpha_mean, ch.alpha_sigma, ch.lam_mean, ch.lam_sigma

    name = (ch.name + " " + ch.spend_col).lower()
    for keywords, params in _CHANNEL_TABLE:
        if any(kw in name for kw in keywords):
            return params

    return _DEFAULT_OFFLINE if ch.channel_type == "offline" else _DEFAULT_DIGITAL


def recommend_priors(spec: Spec, df: pd.DataFrame | None = None) -> dict:
    """
    Build model_config dict with moment-matched Beta/Gamma priors per channel.

    If df is provided, spend-share sigma is used for saturation_beta (recommended).
    Returns dict suitable for MMM(model_config=...).
    """
    channels = spec.channels
    n = len(channels)

    alpha_a = np.zeros(n)
    alpha_b = np.zeros(n)
    lam_a = np.zeros(n)
    lam_b = np.zeros(n)

    for i, ch in enumerate(channels):
        a_mu, a_sig, l_mu, l_sig = _classify(ch)

        # Widen priors for offline channels (more uncertainty)
        if ch.channel_type == "offline":
            a_sig = min(a_sig * 1.2, 0.30)
            l_sig = min(l_sig * 1.2, 1.5)

        alpha_a[i], alpha_b[i] = beta_moment_match(a_mu, a_sig)
        lam_a[i], lam_b[i] = gamma_moment_match(l_mu, l_sig)

    # saturation_beta: spend-share sigma (wider for high-spend channels)
    if df is not None:
        ch_sums = np.array([df[ch.spend_col].sum() for ch in channels], dtype=float)
        total = ch_sums.sum()
        spend_shares = ch_sums / total if total > 0 else np.ones(n) / n
        beta_sigma = np.maximum(spend_shares, 0.05)
    else:
        beta_sigma = np.ones(n) * 0.5

    model_config = {
        "adstock_alpha": Prior("Beta", alpha=alpha_a, beta=alpha_b, dims="channel"),
        "saturation_lam": Prior("Gamma", alpha=lam_a, beta=lam_b, dims="channel"),
        "saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel"),
        "intercept": Prior("Normal", mu=0.5, sigma=0.5),
        "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.3, dims="fourier_mode"),
        # StudentT likelihood for robustness to IL seasonal outliers
        "likelihood": Prior("StudentT", nu=5, sigma=Prior("HalfNormal", sigma=0.5)),
    }

    return model_config
