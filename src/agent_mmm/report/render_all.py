"""Render all stakeholder reports from fitted MMM and diagnostics."""
from __future__ import annotations

import pandas as pd

from agent_mmm.attribute.contributions import get_contributions
from agent_mmm.attribute.roas import compute_roas
from agent_mmm.diagnose.prior_pull import audit_prior_pull
from agent_mmm.report.cmo import generate_cmo_report
from agent_mmm.report.cfo import generate_cfo_report
from agent_mmm.report.mops import generate_mops_report
from agent_mmm.report.ds import generate_ds_report


def render_all(
    mmm,
    X: pd.DataFrame,
    y: pd.Series,
    spec,
    diagnostics: dict,
    optimize_result: dict | None = None,
    overfit_result: dict | None = None,
) -> dict[str, str]:
    """Compute attribution and render all four stakeholder reports.

    Returns dict mapping role → markdown content.
    Roles: "cmo", "cfo", "mops", "ds"
    """
    contributions_df = get_contributions(mmm)
    roas_df = compute_roas(mmm, X)

    prior_pull_df = None
    try:
        prior_pull_df = audit_prior_pull(mmm.idata)
    except Exception:
        pass

    reports = {}
    reports["cmo"] = generate_cmo_report(spec, diagnostics, contributions_df, roas_df)
    reports["cfo"] = generate_cfo_report(spec, diagnostics, contributions_df, roas_df)
    reports["mops"] = generate_mops_report(spec, diagnostics, contributions_df, roas_df, optimize_result)
    reports["ds"] = generate_ds_report(spec, diagnostics, prior_pull_df, overfit_result)
    return reports
