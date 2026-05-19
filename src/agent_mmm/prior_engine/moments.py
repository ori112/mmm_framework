"""Moment-matching helpers to convert (mean, std) to distribution parameters."""
from __future__ import annotations

import numpy as np


def beta_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """
    Convert (mean, std) to Beta(alpha, beta) parameters.
    Ensures mu in (0,1) and sigma is achievable.
    Returns (alpha, beta).
    """
    mu = float(np.clip(mu, 0.01, 0.99))
    sigma_max = np.sqrt(mu * (1.0 - mu))
    sigma = float(np.clip(sigma, 1e-4, sigma_max * 0.95))
    C = max(mu * (1.0 - mu) / sigma**2 - 1.0, 0.5)
    return mu * C, (1.0 - mu) * C


def gamma_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """
    Convert (mean, std) to Gamma(alpha, beta) where beta is the rate (1/scale).
    Returns (alpha, beta_rate).
    """
    mu = max(float(mu), 1e-6)
    sigma = max(float(sigma), 1e-6)
    alpha = (mu / sigma) ** 2
    beta_rate = mu / sigma**2
    return alpha, beta_rate


def normal_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """Identity — Normal is already parameterized by (mu, sigma)."""
    return float(mu), float(sigma)
