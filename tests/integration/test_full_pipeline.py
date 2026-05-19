"""Integration test: full Phase 1 pipeline on synthetic data.

Run with: pytest -m integration
Skipped in default (fast) test runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_df():
    from tests.fixtures.synthetic.generate import generate
    df, truth = generate(FIXTURES)
    return df, truth


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    from agent_mmm.spec.schema import Channel, DataCfg, Spec, SamplerCfg, TargetUnit

    csv_path = FIXTURES / "data.csv"
    return Spec(
        company="Test Co",
        industry="retail",
        data=DataCfg(source="csv", path=str(csv_path), date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[
            Channel(name="google", spend_col="google_spend", channel_type="digital", adstock="geometric", l_max=5),
            Channel(name="facebook", spend_col="facebook_spend", channel_type="digital", adstock="geometric", l_max=8),
            Channel(name="tv", spend_col="tv_spend", channel_type="offline", adstock="geometric", l_max=13),
        ],
        # Fast sampling for tests — override in real use
        sampler=SamplerCfg(draws=100, tune=100, chains=2, target_accept=0.9, random_seed=42),
    )


@pytest.mark.integration
def test_data_audit_passes(synthetic_df, spec):
    from agent_mmm.data.quality import audit_data
    df, _ = synthetic_df
    result = audit_data(df, spec)
    assert result.tier in ("PASS", "WARN"), f"Audit FAIL: {result.findings}"


@pytest.mark.integration
def test_holiday_controls_added(synthetic_df, spec):
    from agent_mmm.data.controls import recommend_controls
    df, _ = synthetic_df
    out = recommend_controls(df, spec)
    assert "is_rosh_hashana" in out["df"].columns
    assert len(out["controls"]) > 0


@pytest.mark.integration
def test_priors_built(synthetic_df, spec):
    from agent_mmm.prior_engine.recommender import recommend_priors
    from pymc_extras.prior import Prior
    df, _ = synthetic_df
    mc = recommend_priors(spec, df)
    assert "adstock_alpha" in mc
    assert "saturation_lam" in mc
    assert "saturation_beta" in mc
    assert isinstance(mc["adstock_alpha"], Prior)


@pytest.mark.integration
def test_build_mmm(synthetic_df, spec):
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.model_factory.builder import build_mmm
    df, _ = synthetic_df
    mc = recommend_priors(spec, df)
    mmm, df_out, control_cols = build_mmm(spec, mc, df)
    assert mmm is not None
    assert "is_rosh_hashana" in df_out.columns
    assert len(control_cols) > 0


@pytest.mark.integration
def test_full_fit_and_diagnose(synthetic_df, spec, tmp_path):
    """End-to-end: fit model and check convergence + artifacts."""
    import os
    os.chdir(tmp_path)  # redirect mmm-workspace/ to tmp

    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.fit_runner.runner import run_fit
    from agent_mmm.diagnose.report import write_diagnostics
    from agent_mmm.workspace.paths import idata_path, diagnostics_path

    df, _ = synthetic_df

    mc = recommend_priors(spec, df)
    mmm = run_fit(spec, mc, df, skip_prior_check=True)

    assert hasattr(mmm, "idata")
    assert idata_path().exists()

    # Diagnostics
    y = df[spec.target.column]
    X = df.drop(columns=[spec.target.column], errors="ignore")
    report = write_diagnostics(mmm, X, y, spec)

    assert diagnostics_path().exists()
    assert "convergence" in report

    conv = report["convergence"]
    rhat = conv.get("rhat_max")
    if rhat is not None:
        assert rhat < 1.5, f"rhat={rhat:.4f} suspiciously high (even for fast test)"

    metrics = report.get("fit_metrics", {})
    r2 = metrics.get("in_sample_r2")
    if r2 is not None:
        assert r2 > 0.0, f"R2={r2} — model not learning anything"
