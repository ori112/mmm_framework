"""Convergence diagnostics: rhat, ESS, divergences, BFMI."""
from __future__ import annotations

import numpy as np
import arviz as az

RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 400


def check_convergence(idata) -> dict:
    """
    Run convergence checks on InferenceData.

    Returns dict with:
      tier, rhat_max, ess_bulk_min, ess_tail_min, divergences, bfmi_min, issues
    """
    summary = az.summary(idata)

    rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary.columns else None
    ess_bulk_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else None
    ess_tail_min = float(summary["ess_tail"].min()) if "ess_tail" in summary.columns else None

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().values)

    bfmi_min = None
    try:
        bfmi = az.bfmi(idata)
        bfmi_min = float(np.min(bfmi))
    except Exception:
        pass

    issues: list[str] = []
    tier = "PASS"

    if rhat_max is not None and rhat_max > RHAT_THRESHOLD:
        issues.append(f"rhat_max={rhat_max:.4f} > {RHAT_THRESHOLD} — chains not converged")
        tier = "FAIL"

    if ess_bulk_min is not None and ess_bulk_min < ESS_THRESHOLD:
        issues.append(f"ESS bulk min={ess_bulk_min:.0f} < {ESS_THRESHOLD}")
        if tier == "PASS":
            tier = "WARN"

    if divergences > 0:
        issues.append(f"{divergences} divergences detected")
        if tier == "PASS":
            tier = "WARN"

    if bfmi_min is not None and bfmi_min < 0.3:
        issues.append(f"BFMI={bfmi_min:.3f} < 0.3 — poor energy efficiency")
        if tier == "PASS":
            tier = "WARN"

    return {
        "tier": tier,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "divergences": divergences,
        "bfmi_min": bfmi_min,
        "issues": issues,
    }
