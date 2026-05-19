"""Write diagnostics.json and diagnostics_report.md."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM

from ..workspace.paths import diagnostics_path, ensure_workspace, report_path
from .convergence import check_convergence
from .fit_metrics import compute_in_sample_metrics


def _fmt(val, fmt=".4f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return format(val, fmt)


def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.1%}"


def write_diagnostics(mmm: MMM, X: pd.DataFrame, y: pd.Series, spec) -> dict:
    """
    Run convergence + fit checks and write:
      - mmm-workspace/diagnostics.json
      - mmm-workspace/reports/diagnostics.md

    Returns the full diagnostics dict.
    """
    ensure_workspace()

    conv = check_convergence(mmm.idata)

    try:
        metrics = compute_in_sample_metrics(mmm, X, y)
    except Exception as exc:
        metrics = {"error": str(exc)}

    report = {
        "run_date": date.today().isoformat(),
        "spec_company": spec.company,
        "n_observations": int(len(y)),
        "channels": [ch.name for ch in spec.channels],
        "convergence": conv,
        "fit_metrics": metrics,
    }

    # Write JSON
    path = diagnostics_path()
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  diagnostics.json -> {path}")

    # Write markdown
    _write_md(report)

    return report


def _write_md(report: dict) -> None:
    conv = report["convergence"]
    m = report.get("fit_metrics", {})
    tier_tag = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}.get(conv["tier"], conv["tier"])

    lines = [
        f"# Diagnostics — {report['run_date']}",
        f"Company: {report['spec_company']}",
        f"Observations: {report['n_observations']}",
        "",
        f"## Convergence: {tier_tag}",
        f"- rhat max: {_fmt(conv.get('rhat_max'))}",
        f"- ESS bulk min: {_fmt(conv.get('ess_bulk_min'), '.0f')}",
        f"- ESS tail min: {_fmt(conv.get('ess_tail_min'), '.0f')}",
        f"- Divergences: {conv.get('divergences', 'N/A')}",
        f"- BFMI min: {_fmt(conv.get('bfmi_min'))}",
    ]
    if conv.get("issues"):
        lines += ["", "**Issues:**"] + [f"- {i}" for i in conv["issues"]]

    lines += [
        "",
        "## Fit Metrics (in-sample, posterior mean)",
        f"- R2: {_fmt(m.get('in_sample_r2'))}",
        f"- MAPE: {_fmt_pct(m.get('in_sample_mape'))}",
        f"- wMAPE: {_fmt_pct(m.get('in_sample_wmape'))}",
    ]

    path = report_path("diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  diagnostics.md -> {path}")
