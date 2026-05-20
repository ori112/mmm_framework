"""Unit tests for attribution, effectiveness metrics, and report rendering."""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch
from pathlib import Path

from agent_mmm.report.renderer import fmt_currency, fmt_hdi, fmt_roas
from agent_mmm.attribute.roas import _metric_label


# ── Renderer helpers ──────────────────────────────────────────────────────────

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


# ── Metric label derivation ───────────────────────────────────────────────────

def _make_spec(target_type="revenue", unit_name=None, value_per_unit=1.0):
    from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit
    return Spec(
        company="TestCo",
        industry="retail",
        data=DataCfg(source="csv", path="data.csv", date_col="date"),
        target=TargetUnit(column="revenue", type=target_type, unit_name=unit_name, value_per_unit=value_per_unit),
        channels=[
            Channel(name="google", spend_col="google_spend"),
            Channel(name="tv", spend_col="tv_spend"),
        ],
    )


def test_metric_label_revenue():
    spec = _make_spec("revenue")
    assert _metric_label(spec) == "ROAS"


def test_metric_label_acquisitions_no_unit():
    spec = _make_spec("acquisitions")
    assert _metric_label(spec) == "CPA"


def test_metric_label_leads():
    spec = _make_spec("acquisitions", unit_name="lead")
    assert _metric_label(spec) == "CPL"


def test_metric_label_policy():
    spec = _make_spec("acquisitions", unit_name="policy")
    assert _metric_label(spec) == "CPP"


def test_metric_label_volume():
    spec = _make_spec("volume")
    assert _metric_label(spec) == "CPU"


# ── CMO report (revenue) ──────────────────────────────────────────────────────

def _make_effectiveness_df_revenue():
    return pd.DataFrame([
        {"channel": "google_spend", "spend_ils": 200_000, "contribution_mean": 500_000,
         "metric_label": "ROAS", "metric_value_mean": 2.5,
         "metric_hdi89_low": 2.0, "metric_hdi89_high": 3.0},
        {"channel": "tv_spend", "spend_ils": 300_000, "contribution_mean": 300_000,
         "metric_label": "ROAS", "metric_value_mean": 1.0,
         "metric_hdi89_low": 0.7, "metric_hdi89_high": 1.3},
    ])


def _make_effectiveness_df_leads():
    return pd.DataFrame([
        {"channel": "google_spend", "spend_ils": 200_000, "contribution_mean": 2_000,
         "metric_label": "CPL", "metric_value_mean": 100.0,
         "metric_hdi89_low": 80.0, "metric_hdi89_high": 120.0,
         "implied_roas_mean": 2.5},
        {"channel": "tv_spend", "spend_ils": 300_000, "contribution_mean": 1_500,
         "metric_label": "CPL", "metric_value_mean": 200.0,
         "metric_hdi89_low": 160.0, "metric_hdi89_high": 240.0,
         "implied_roas_mean": 1.25},
    ])


def _make_contributions_df():
    return pd.DataFrame([
        {"channel": "google_spend", "mean": 500_000, "hdi_low": 400_000, "hdi_high": 600_000},
        {"channel": "tv_spend", "mean": 300_000, "hdi_low": 200_000, "hdi_high": 400_000},
    ])


def test_cmo_report_revenue():
    spec = _make_spec("revenue")
    diagnostics = {"convergence": {"tier": "PASS", "rhat_max": 1.01}}

    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cmo.md")):
        from agent_mmm.report.cmo import generate_cmo_report
        md = generate_cmo_report(spec, diagnostics, _make_contributions_df(), _make_effectiveness_df_revenue())

    assert "CMO" in md
    assert "ROAS" in md
    assert "₪" in md
    assert "google" in md.lower()


def test_cmo_report_leads_shows_cpl():
    spec = _make_spec("acquisitions", unit_name="lead", value_per_unit=250.0)
    diagnostics = {"convergence": {"tier": "PASS"}}

    contrib = pd.DataFrame([
        {"channel": "google_spend", "mean": 2_000, "hdi_low": 1_500, "hdi_high": 2_500},
        {"channel": "tv_spend", "mean": 1_500, "hdi_low": 1_000, "hdi_high": 2_000},
    ])

    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cmo.md")):
        from agent_mmm.report.cmo import generate_cmo_report
        md = generate_cmo_report(spec, diagnostics, contrib, _make_effectiveness_df_leads())

    assert "CPL" in md
    assert "lead" in md.lower()
    # The effectiveness section title should mention CPL, not "Return on Ad Spend"
    assert "Return on Ad Spend" not in md
    # Implied ROAS footnote is allowed since value_per_unit=250
    assert "Implied ROAS" in md or "implied" in md.lower()


def test_cfo_report_revenue():
    spec = _make_spec("revenue")
    diagnostics = {"convergence": {"tier": "PASS"}, "fit_metrics": {"in_sample_r2": 0.85}}

    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cfo.md")):
        from agent_mmm.report.cfo import generate_cfo_report
        md = generate_cfo_report(spec, diagnostics, _make_contributions_df(), _make_effectiveness_df_revenue())

    assert "CFO" in md
    assert "ROAS" in md
    assert "₪" in md


def test_cfo_report_leads_shows_cpl_and_implied_roas():
    spec = _make_spec("acquisitions", unit_name="lead", value_per_unit=250.0)
    diagnostics = {"convergence": {"tier": "PASS"}, "fit_metrics": {"in_sample_r2": 0.80}}

    contrib = pd.DataFrame([
        {"channel": "google_spend", "mean": 2_000, "hdi_low": 1_500, "hdi_high": 2_500},
        {"channel": "tv_spend", "mean": 1_500, "hdi_low": 1_000, "hdi_high": 2_000},
    ])

    with patch("agent_mmm.report.renderer.save_report", return_value=Path("/tmp/cfo.md")):
        from agent_mmm.report.cfo import generate_cfo_report
        md = generate_cfo_report(spec, diagnostics, contrib, _make_effectiveness_df_leads())

    assert "CPL" in md
    assert "lead" in md.lower()
    assert "Implied" in md or "implied" in md


# ── Schema validation ─────────────────────────────────────────────────────────

def test_spec_unit_name_accepted():
    from agent_mmm.spec.schema import TargetUnit
    t = TargetUnit(column="leads", type="acquisitions", unit_name="lead", value_per_unit=250.0)
    assert t.unit_name == "lead"
    assert t.value_per_unit == 250.0


def test_spec_revenue_no_unit_name():
    from agent_mmm.spec.schema import TargetUnit
    t = TargetUnit(column="revenue", type="revenue")
    assert t.unit_name is None
    assert t.value_per_unit == 1.0
