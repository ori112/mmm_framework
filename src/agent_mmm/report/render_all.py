"""Render all stakeholder reports from fitted MMM and diagnostics."""
from __future__ import annotations

import pandas as pd

from agent_mmm.attribute.contributions import get_contributions
from agent_mmm.attribute.roas import compute_roas
from agent_mmm.report.cmo import generate_cmo_report
from agent_mmm.report.cfo import generate_cfo_report


def render_all(mmm, X: pd.DataFrame, y: pd.Series, spec, diagnostics: dict) -> dict[str, str]:
    """Compute attribution and render CMO + CFO reports.

    Returns dict mapping role → markdown content.
    """
    contributions_df = get_contributions(mmm)
    roas_df = compute_roas(mmm, X)

    reports = {}
    reports["cmo"] = generate_cmo_report(spec, diagnostics, contributions_df, roas_df)
    reports["cfo"] = generate_cfo_report(spec, diagnostics, contributions_df, roas_df)
    return reports
