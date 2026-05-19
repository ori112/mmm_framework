"""Unit tests for moment-matching functions."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from agent_mmm.prior_engine.moments import beta_moment_match, gamma_moment_match, normal_moment_match


def _beta_stats(alpha, beta):
    d = stats.beta(alpha, beta)
    return d.mean(), d.std()


def _gamma_stats(alpha, beta_rate):
    d = stats.gamma(alpha, scale=1.0 / beta_rate)
    return d.mean(), d.std()


@pytest.mark.parametrize("mu,sigma", [
    (0.10, 0.07),
    (0.30, 0.12),
    (0.60, 0.20),
    (0.50, 0.15),
])
def test_beta_moment_match_mean(mu, sigma):
    a, b = beta_moment_match(mu, sigma)
    recovered_mu, _ = _beta_stats(a, b)
    assert abs(recovered_mu - mu) < 1e-6


@pytest.mark.parametrize("mu,sigma", [
    (0.10, 0.07),
    (0.50, 0.15),
])
def test_beta_moment_match_std_approx(mu, sigma):
    a, b = beta_moment_match(mu, sigma)
    _, recovered_sigma = _beta_stats(a, b)
    assert abs(recovered_sigma - sigma) / sigma < 0.05  # within 5%


def test_beta_moment_match_valid_params():
    a, b = beta_moment_match(0.5, 0.15)
    assert a > 0 and b > 0


@pytest.mark.parametrize("mu,sigma", [
    (4.0, 1.0),
    (2.5, 0.65),
    (1.5, 0.45),
])
def test_gamma_moment_match_mean(mu, sigma):
    a, b = gamma_moment_match(mu, sigma)
    recovered_mu, _ = _gamma_stats(a, b)
    assert abs(recovered_mu - mu) < 1e-4


def test_gamma_moment_match_positive():
    a, b = gamma_moment_match(3.0, 0.8)
    assert a > 0 and b > 0


def test_normal_moment_match_identity():
    assert normal_moment_match(0.5, 0.5) == (0.5, 0.5)
    assert normal_moment_match(0.0, 1.0) == (0.0, 1.0)
