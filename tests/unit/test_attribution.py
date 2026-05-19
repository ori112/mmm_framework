"""Unit tests for attribution and report rendering utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from agent_mmm.report.renderer import fmt_currency, fmt_hdi, fmt_roas


def test_fmt_currency_ils():
    assert fmt_currency(1_000_000, "ILS") == "₪1,000,000"


def test_fmt_currency_decimals():
    assert fmt_currency(3.14, "ILS", decimals=2) == "₪3.14"


def test_fmt_hdi_no_currency():
    result = fmt_hdi(2.5, 1.8, 3.2)
    assert "2.50" in result
    assert "1.80" in result
    assert "3.20" in result
    assert "89% CI" in result


def test_fmt_hdi_with_currency():
    result = fmt_hdi(100_000, 80_000, 120_000, currency="ILS", decimals=0)
    assert "₪" in result
    assert "89% CI" in result


def test_fmt_roas():
    result = fmt_roas(2.5, 1.8, 3.2)
    assert "2.50x" in result
    assert "89% CI" in result


def test_cmo_report_structure():
    from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit

    spec = Spec(
        company="TestCo",
        industry="retail",
        data=DataCfg(source="csv", path="data.csv", date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[
            Channel(name="google", spend_col="google_spend"),
            Channel(name="tv", spend_col="tv_spend"),
        ],
    )

    contributions_df = pd.DataFrame([
        {"channel": "google_spend", "mean": 500_000, "hdi_low": 400_000, "hdi_high": 600_000},
        {"channel": "tv_spend", "mean": 300_000, "hdi_low": 200_000, "hdi_high": 400_000},
    ])
    roas_df = pd.DataFrame([
        {"channel": "google_spend", "spend_ils": 200_000, "contribution_mean_ils": 500_000,
         "roas_mean": 2.5, "roas_hdi89_low": 2.0, "roas_hdi89_high": 3.0},
        {"channel": "tv_spend", "spend_ils": 300_000, "contribution_mean_ils": 300_000,
         "roas_mean": 1.0, "roas_hdi89_low": 0.7, "roas_hdi89_high": 1.3},
    ])
    diagnostics = {"convergence": {"tier": "PASS", "rhat_max": 1.01}}

    from unittest.mock import patch
    from pathlib import Path
    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cmo.md")):
        from agent_mmm.report.cmo import generate_cmo_report
        md = generate_cmo_report(spec, diagnostics, contributions_df, roas_df)

    assert "CMO" in md
    assert "TestCo" in md
    assert "ILS" in md
    assert "google" in md.lower() or "google_spend" in md.lower()


def test_cfo_report_structure():
    from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit

    spec = Spec(
        company="TestCo",
        data=DataCfg(source="csv", path="data.csv", date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend")],
    )

    contributions_df = pd.DataFrame([
        {"channel": "google_spend", "mean": 500_000, "hdi_low": 400_000, "hdi_high": 600_000},
    ])
    roas_df = pd.DataFrame([
        {"channel": "google_spend", "spend_ils": 200_000, "contribution_mean_ils": 500_000,
         "roas_mean": 2.5, "roas_hdi89_low": 2.0, "roas_hdi89_high": 3.0},
    ])
    diagnostics = {"convergence": {"tier": "PASS"}, "fit_metrics": {"in_sample_r2": 0.85}}

    from unittest.mock import patch
    from pathlib import Path
    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cfo.md")):
        from agent_mmm.report.cfo import generate_cfo_report
        md = generate_cfo_report(spec, diagnostics, contributions_df, roas_df)

    assert "CFO" in md
    assert "ROAS" in md
    assert "ILS" in md
    assert "₪" in md
