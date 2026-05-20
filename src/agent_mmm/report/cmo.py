"""CMO report: plain-English channel attribution narrative."""
from __future__ import annotations

from datetime import date

import pandas as pd

from agent_mmm.report.renderer import fmt_currency, fmt_hdi, save_report


def _effectiveness_header(eff_df: pd.DataFrame, is_revenue: bool) -> tuple[str, str]:
    """Return (section_title, column_header) based on the metric in eff_df."""
    if eff_df is None or len(eff_df) == 0:
        return "Effectiveness", "Value"
    label = eff_df["metric_label"].iloc[0]
    if is_revenue:
        return "Return on Ad Spend (ROAS)", "ROAS"
    return f"Cost per Unit ({label})", label


def generate_cmo_report(
    spec,
    diagnostics: dict,
    contributions_df: pd.DataFrame,
    effectiveness_df: pd.DataFrame,
) -> str:
    """Generate the CMO markdown report and write it to mmm-workspace/reports/cmo.md.

    Returns the rendered markdown string.
    `effectiveness_df` is the normalised output of compute_effectiveness().
    """
    currency = getattr(spec, "currency", "ILS")
    company = getattr(spec, "company", "")
    is_revenue = spec.target.type == "revenue"
    unit_name = spec.target.unit_name or ("unit" if not is_revenue else "")

    total_contrib = contributions_df["mean"].sum()
    contrib = contributions_df.copy()
    contrib["share_pct"] = (contrib["mean"] / total_contrib * 100).round(1) if total_contrib > 0 else 0.0
    contrib = contrib.sort_values("mean", ascending=False)

    conv = diagnostics.get("convergence", {})
    tier = conv.get("tier", "UNKNOWN")
    confidence = {"PASS": "High", "WARN": "Medium", "FAIL": "Low"}.get(tier, "Unknown")

    top = contrib.iloc[0] if len(contrib) > 0 else None
    bottom = contrib.iloc[-1] if len(contrib) > 1 else None

    metric_desc = f"incremental revenue ({currency})" if is_revenue else f"incremental {unit_name}s"

    lines = [
        "# CMO Marketing Effectiveness Report",
        f"**{company}** | Generated {date.today().isoformat()} | Confidence: **{confidence}**",
        "",
        "## Executive Summary",
    ]

    if top is not None:
        contrib_fmt = fmt_currency(total_contrib, currency, 0) if is_revenue else f"{total_contrib:,.0f} {unit_name}s"
        lines.append(
            f"Marketing contributed a total of {contrib_fmt} in {metric_desc} during the "
            f"analysis period. **{top['channel'].replace('_spend','')}** was the strongest "
            f"performer ({top['share_pct']:.0f}% of media contribution)."
        )
    if bottom is not None:
        lines.append(
            f"**{bottom['channel'].replace('_spend','')}** delivered the lowest contribution "
            f"({bottom['share_pct']:.0f}%) and warrants review."
        )
    lines.append("")

    # Contribution table
    contrib_col = "Incremental revenue" if is_revenue else f"Incremental {unit_name}s"
    ci_note = f"({currency})" if is_revenue else "(units)"
    lines += [
        "## Channel Contributions",
        "",
        f"| Channel | {contrib_col} | Share | 89% CI {ci_note} |",
        f"|---------|{'—' * len(contrib_col)}|-------|{'—' * (len(ci_note)+9)}|",
    ]
    for _, row in contrib.iterrows():
        ch = row["channel"].replace("_spend", "")
        mean_fmt = fmt_currency(row["mean"], currency, 0) if is_revenue else f"{row['mean']:,.0f}"
        lo = fmt_currency(row["hdi_low"], currency, 0) if is_revenue else f"{row['hdi_low']:,.0f}"
        hi = fmt_currency(row["hdi_high"], currency, 0) if is_revenue else f"{row['hdi_high']:,.0f}"
        lines.append(f"| {ch} | {mean_fmt} | {row['share_pct']:.1f}% | {lo}–{hi} |")
    lines.append("")

    # Effectiveness table (ROAS or CPL/CPA/…)
    if effectiveness_df is not None and len(effectiveness_df) > 0:
        label = effectiveness_df["metric_label"].iloc[0]
        hdi_low_col = next((c for c in effectiveness_df.columns if "hdi" in c and "low" in c), None)
        hdi_high_col = next((c for c in effectiveness_df.columns if "hdi" in c and "high" in c), None)

        section_title, col_header = _effectiveness_header(effectiveness_df, is_revenue)
        unit_suffix = "x" if is_revenue else f" {currency}/{unit_name}"
        lines += [
            f"## {section_title}",
            "",
            f"| Channel | {col_header} | 89% CI |",
            f"|---------|{'—' * len(col_header)}|--------|",
        ]
        eff_sorted = effectiveness_df.sort_values("metric_value_mean", ascending=is_revenue is False)
        for _, row in eff_sorted.iterrows():
            ch = row["channel"].replace("_spend", "")
            lo = row[hdi_low_col] if hdi_low_col else row["metric_value_mean"]
            hi = row[hdi_high_col] if hdi_high_col else row["metric_value_mean"]
            val = row["metric_value_mean"]
            if is_revenue:
                lines.append(f"| {ch} | {val:.2f}{unit_suffix} | {lo:.2f}{unit_suffix}–{hi:.2f}{unit_suffix} |")
            else:
                lines.append(f"| {ch} | {fmt_currency(val, currency, 0)}/{unit_name} | "
                              f"{fmt_currency(lo, currency, 0)}–{fmt_currency(hi, currency, 0)} |")
        if "implied_roas_mean" in effectiveness_df.columns:
            lines.append("")
            lines.append("*Implied ROAS assumes "
                         f"{fmt_currency(spec.target.value_per_unit, currency)} per {unit_name}.*")
        lines.append("")

    lines += [
        "## Model Confidence",
        f"Convergence tier: **{tier}**. Results reliability is **{confidence.lower()}**.",
        "",
        "_Report generated by agent_mmm. All monetary values in " + currency + "._",
    ]

    content = "\n".join(lines)
    save_report("cmo", content)
    return content
