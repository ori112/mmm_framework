"""Unit tests for optimization, scoring, variants, and leaderboard."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from agent_mmm.iter_loop.scoring import composite_score, is_plausible
from agent_mmm.iter_loop.variants import Variant, generate_variants, apply_variant_to_spec


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_composite_score_perfect():
    score = composite_score(cv_r2=1.0, in_sample_r2=1.0, converged=True, plausible=True)
    assert abs(score - 1.0) < 1e-6


def test_composite_score_not_converged():
    score = composite_score(cv_r2=0.8, in_sample_r2=0.82, converged=False, plausible=True)
    converged_score = composite_score(cv_r2=0.8, in_sample_r2=0.82, converged=True, plausible=True)
    assert score < converged_score


def test_composite_score_overfit_penalty():
    # Large overfit gap should reduce score toward 0
    score = composite_score(cv_r2=0.5, in_sample_r2=0.95, converged=True, plausible=True)
    assert score < 0.5  # overfit_gap=0.45 → penalty = max(0, 1 - 0.9) = 0.1


def test_composite_score_non_negative():
    score = composite_score(cv_r2=-0.1, in_sample_r2=0.8, converged=False, plausible=False)
    assert score >= 0.0


def test_is_plausible_single_channel_dominates():
    contrib = pd.DataFrame([
        {"channel": "google_spend", "mean": 950_000, "hdi_low": 0, "hdi_high": 0},
        {"channel": "tv_spend", "mean": 50_000, "hdi_low": 0, "hdi_high": 0},
    ])
    roas = pd.DataFrame([
        {"channel": "google_spend", "roas_mean": 2.0},
        {"channel": "tv_spend", "roas_mean": 1.5},
    ])
    assert not is_plausible(contrib, roas)


def test_is_plausible_negative_roas():
    contrib = pd.DataFrame([{"channel": "google_spend", "mean": 500_000, "hdi_low": 0, "hdi_high": 0}])
    roas = pd.DataFrame([{"channel": "google_spend", "roas_mean": -0.5}])
    assert not is_plausible(contrib, roas)


def test_is_plausible_balanced():
    contrib = pd.DataFrame([
        {"channel": "google_spend", "mean": 400_000, "hdi_low": 0, "hdi_high": 0},
        {"channel": "tv_spend", "mean": 600_000, "hdi_low": 0, "hdi_high": 0},
    ])
    roas = pd.DataFrame([
        {"channel": "google_spend", "roas_mean": 2.0},
        {"channel": "tv_spend", "roas_mean": 1.5},
    ])
    assert is_plausible(contrib, roas)


# ── Variants ─────────────────────────────────────────────────────────────────

def test_generate_variants_returns_list():
    variants = generate_variants(
        l_max_options=[4, 8],
        fourier_options=[2],
        width_options=[1.0],
    )
    assert len(variants) == 2
    assert all(isinstance(v, Variant) for v in variants)


def test_generate_variants_n_limit():
    variants = generate_variants(n=3)
    assert len(variants) == 3


def test_variant_name():
    v = Variant(l_max=8, fourier_order=4, prior_width_factor=1.0)
    assert "l_max=8" in v.name
    assert "fourier=4" in v.name


def test_apply_variant_to_spec():
    from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit
    spec = Spec(
        data=DataCfg(source="csv", path="data.csv", date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend", l_max=8)],
        fourier_order=2,
    )
    v = Variant(l_max=13, fourier_order=6, prior_width_factor=1.5)
    v_spec = apply_variant_to_spec(spec, v)
    assert v_spec.fourier_order == 6
    assert v_spec.channels[0].l_max == 13
    assert spec.fourier_order == 2  # original unchanged


# ── Leaderboard ───────────────────────────────────────────────────────────────

def test_leaderboard_append_and_sort(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_mmm.iter_loop.leaderboard.leaderboard_path", lambda: tmp_path / "leaderboard.json")

    from agent_mmm.iter_loop.leaderboard import append_run, load_leaderboard
    append_run("run1", 1, "v1", 0.5, 0.6, 0.65, 0.05, True, 0)
    append_run("run2", 1, "v2", 0.8, 0.85, 0.87, 0.02, True, 0)
    append_run("run3", 1, "v3", 0.3, 0.4, 0.5, 0.1, False, 5)

    data = load_leaderboard()
    assert data["best_score"] == 0.8
    assert data["best_run_id"] == "run2"
    assert data["runs"][0]["run_id"] == "run2"  # sorted desc
