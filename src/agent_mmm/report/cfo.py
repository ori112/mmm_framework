"""CFO report: financial ROI with credible intervals (ROAS or CPA/CPL)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from agent_mmm.report.renderer import fmt_currency, fmt_roas, save_report


def generate_cfo_report(
    spec,
    diagnostics: dict,
    contributions_df: pd.DataFrame,
    effectiveness_df: pd.DataFrame,
) -> str:
    """Generate the CFO markdown report and write it to mmm-workspace/reports/cfo.md.

    Returns the rendered markdown string.
    `effectiveness_df` is the normalised output of compute_effectiveness().
    """
    currency = getattr(spec, "currency", "ILS")
    company = getattr(spec, "company", "")
    is_revenue = spec.target.type == "revenue"
    unit_name = spec.target.unit_name or ("unit" if not is_revenue else "")
    value_per_unit = spec.target.value_per_unit or 1.0

    total_spend = effectiveness_df["spend_ils"].sum() if effectiveness_df is not None and len(effectiveness_df) > 0 else 0.0
    total_contrib = contributions_df["mean"].sum()

    conv = diagnostics.get("convergence", {})
    metrics = diagnostics.get("fit_metrics", {})
    r2 = metrics.get("in_sample_r2")

    hdi_low_col = next((c for c in effectiveness_df.columns if "hdi" in c and "low" in c), None) if effectiveness_df is not None else None
    hdi_high_col = next((c for c in effectiveness_df.columns if "hdi" in c and "high" in c), None) if effectiveness_df is not None else None

    # Portfolio summary
    if is_revenue:
        blended = total_contrib / total_spend if total_spend > 0 else 0.0
        summary_rows = [
            f"| Total media spend | {fmt_currency(total_spend, currency, 0)} |",
            f"| Total incremental revenue | {fmt_currency(total_contrib, currency, 0)} |",
            f"| Blended ROAS | {blended:.2f}x |",
        ]
    else:
        blended_cpa = total_spend / total_contrib if total_contrib > 0 else 0.0
        summary_rows = [
            f"| Total media spend | {fmt_currency(total_spend, currency, 0)} |",
            f"| Total incremental {unit_name}s | {total_contrib:,.0f} |",
            f"| Blended {effectiveness_df['metric_label'].iloc[0] if effectiveness_df is not None and len(effectiveness_df) > 0 else 'CPA'} | {fmt_currency(blended_cpa, currency, 2)}/{unit_name} |",
        ]
        if value_per_unit > 1.0 and total_contrib > 0:
            implied_roas = value_per_unit / blended_cpa if blended_cpa > 0 else 0.0
            summary_rows.append(f"| Implied blended ROAS (at {fmt_currency(value_per_unit, currency)}/{unit_name}) | {implied_roas:.2f}x |")

    if r2 is not None:
        summary_rows.append(f"| Model in-sample R² | {r2:.3f} |")

    lines = [
        "# CFO Financial ROI Report",
        f"**{company}** | Generated {date.today().isoformat()}",
        "",
        "## Portfolio Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ] + summary_rows + [""]

    # Per-channel table
    if effectiveness_df is not None and len(effectiveness_df) > 0:
        label = effectiveness_df["metric_label"].iloc[0]
        eff_sorted = effectiveness_df.sort_values("metric_value_mean", ascending=not is_revenue)

        if is_revenue:
            lines += [
                "## Per-Channel ROAS",
                "",
                "| Channel | Spend | Contribution | ROAS | 89% CI |",
                "|---------|-------|--------------|------|--------|",
            ]
            for _, row in eff_sorted.iterrows():
                ch = row["channel"].replace("_spend", "")
                lo = row[hdi_low_col] if hdi_low_col else row["metric_value_mean"]
                hi = row[hdi_high_col] if hdi_high_col else row["metric_value_mean"]
                lines.append(
                    f"| {ch} | {fmt_currency(row['spend_ils'], currency, 0)} "
                    f"| {fmt_currency(row['contribution_mean'], currency, 0)} "
                    f"| {row['metric_value_mean']:.2f}x | {lo:.2f}x–{hi:.2f}x |"
                )
        else:
            lines += [
                f"## Per-Channel {label}",
                "",
                f"| Channel | Spend | {unit_name.capitalize()}s | {label} | 89% CI |" +
                (" Implied ROAS |" if value_per_unit > 1.0 else ""),
                "|---------|-------|" + "-" * (len(unit_name) + 3) + "|-----|--------|" +
                ("-------------|" if value_per_unit > 1.0 else ""),
            ]
            for _, row in eff_sorted.iterrows():
                ch = row["channel"].replace("_spend", "")
                lo = row[hdi_low_col] if hdi_low_col else row["metric_value_mean"]
                hi = row[hdi_high_col] if hdi_high_col else row["metric_value_mean"]
                implied = row.get("implied_roas_mean", "")
                implied_col = f" {implied:.2f}x |" if implied != "" and value_per_unit > 1.0 else (" – |" if value_per_unit > 1.0 else "")
                lines.append(
                    f"| {ch} | {fmt_currency(row['spend_ils'], currency, 0)} "
                    f"| {row['contribution_mean']:,.0f} "
                    f"| {fmt_currency(row['metric_value_mean'], currency, 2)}/{unit_name} "
                    f"| {fmt_currency(lo, currency, 2)}–{fmt_currency(hi, currency, 2)} |"
                    + implied_col
                )
        lines.append("")

    lines += [
        "## Sensitivity Note",
        f"{'ROAS' if is_revenue else label} estimates reflect *incremental* (marginal) attribution — "
        "not total platform-reported figures. "
        "Uncertainty intervals (89% CI) capture model posterior uncertainty.",
        "",
        "_Report generated by agent_mmm. All monetary values in " + currency + "._",
    ]

    content = "\n".join(lines)
    save_report("cfo", content)
    return content
