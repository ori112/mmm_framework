"""Data Science report: full diagnostics, convergence, prior/posterior, adstock, saturation."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from agent_mmm.report.renderer import save_report


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return format(float(val), fmt)


def _posterior_summary(idata, var: str) -> pd.DataFrame | None:
    """Return mean / std / hdi_5 / hdi_94 per channel for a named variable."""
    try:
        import arviz as az
        if var not in idata.posterior:
            return None
        da = idata.posterior[var]           # (chain, draw, channel)
        summary = az.summary(idata, var_names=[var], hdi_prob=0.89)
        return summary[["mean", "sd", "hdi_5.5%", "hdi_94.5%"]]
    except Exception:
        return None


def _adstock_half_life(alpha: float) -> float:
    """Return half-life in periods: log(0.5) / log(alpha)."""
    if alpha <= 0 or alpha >= 1:
        return float("nan")
    return math.log(0.5) / math.log(alpha)


def _prior_posterior_table(idata, spec) -> list[str]:
    """Side-by-side prior mean/std vs posterior mean/std per channel param."""
    import arviz as az

    lines: list[str] = []
    rows: list[dict] = []

    for var in ("adstock_alpha", "saturation_lam", "saturation_beta"):
        if var not in idata.posterior:
            continue
        post_summary = az.summary(idata, var_names=[var], hdi_prob=0.89)
        for idx in post_summary.index:
            rows.append({
                "param": idx,
                "post_mean": post_summary.loc[idx, "mean"],
                "post_sd": post_summary.loc[idx, "sd"],
            })

    if not rows:
        return []

    lines += [
        "",
        "## Posterior Parameter Summary",
        "",
        "| Parameter | Post mean | Post sd | 89% HDI low | 89% HDI high |",
        "|-----------|-----------|---------|-------------|--------------|",
    ]
    for var in ("adstock_alpha", "saturation_lam", "saturation_beta"):
        if var not in idata.posterior:
            continue
        try:
            summary = az.summary(idata, var_names=[var], hdi_prob=0.89)
            for idx in summary.index:
                r = summary.loc[idx]
                lines.append(
                    f"| {idx} | {_fmt(r['mean'])} | {_fmt(r['sd'])} "
                    f"| {_fmt(r['hdi_5.5%'])} | {_fmt(r['hdi_94.5%'])} |"
                )
        except Exception:
            pass
    return lines


def _adstock_table(idata, spec) -> list[str]:
    """Per-channel adstock alpha posterior + derived half-life."""
    try:
        import arviz as az
        if "adstock_alpha" not in idata.posterior:
            return []
        summary = az.summary(idata, var_names=["adstock_alpha"], hdi_prob=0.89)
    except Exception:
        return []

    lines = [
        "",
        "## Adstock Decay (per channel)",
        "",
        "Half-life = log(0.5) / log(alpha_mean) — weeks until 50% of effect decays.",
        "",
        "| Channel | alpha mean | alpha sd | Half-life (wks) | l_max |",
        "|---------|-----------|----------|-----------------|-------|",
    ]

    channel_map = {ch.name: ch for ch in spec.channels}
    for idx in summary.index:
        mean = summary.loc[idx, "mean"]
        sd = summary.loc[idx, "sd"]
        hl = _adstock_half_life(mean)
        # idx is like "adstock_alpha[google_spend]" — extract channel name
        ch_raw = idx.split("[")[-1].rstrip("]") if "[" in idx else idx
        ch_name = ch_raw.replace("_spend", "")
        l_max = channel_map[ch_name].l_max if ch_name in channel_map else "?"
        lines.append(
            f"| {ch_name} | {_fmt(mean)} | {_fmt(sd)} "
            f"| {_fmt(hl, '.1f')} | {l_max} |"
        )
    return lines


def _saturation_table(idata, spec) -> list[str]:
    """Per-channel saturation lambda posterior."""
    try:
        import arviz as az
        if "saturation_lam" not in idata.posterior:
            return []
        summary = az.summary(idata, var_names=["saturation_lam"], hdi_prob=0.89)
    except Exception:
        return []

    lines = [
        "",
        "## Saturation Parameters (per channel)",
        "",
        "lambda controls the spend level at half-maximum response.",
        "Lower lambda → channel saturates at lower spend levels.",
        "",
        "| Channel | lam mean | lam sd | 89% HDI |",
        "|---------|----------|--------|---------|",
    ]
    for idx in summary.index:
        mean = summary.loc[idx, "mean"]
        sd = summary.loc[idx, "sd"]
        lo = summary.loc[idx, "hdi_5.5%"]
        hi = summary.loc[idx, "hdi_94.5%"]
        ch_raw = idx.split("[")[-1].rstrip("]") if "[" in idx else idx
        ch_name = ch_raw.replace("_spend", "")
        lines.append(f"| {ch_name} | {_fmt(mean)} | {_fmt(sd)} | [{_fmt(lo)}, {_fmt(hi)}] |")
    return lines


def _prior_predictive_section(mmm) -> list[str]:
    """Prior predictive check: compare prior predictive range vs observed."""
    try:
        idata = mmm.idata
        if "prior_predictive" not in idata:
            return [
                "",
                "## Prior Predictive Check",
                "",
                "_Not available — run `fit_runner.checks.run_prior_predictive()` before fitting._",
            ]

        pp = idata.prior_predictive
        obs_var = [v for v in pp.data_vars if "obs" in v.lower() or "y" in v.lower()]
        if not obs_var:
            return []
        pp_vals = pp[obs_var[0]].values.flatten()

        # Observed
        obs_vals = None
        if "observed_data" in idata:
            obs_data = idata.observed_data
            obs_key = list(obs_data.data_vars)[0] if obs_data.data_vars else None
            if obs_key:
                obs_vals = obs_data[obs_key].values.flatten()

        lines = [
            "",
            "## Prior Predictive Check",
            "",
            "| Statistic | Prior predictive | Observed |",
            "|-----------|-----------------|----------|",
            f"| Mean      | {_fmt(np.nanmean(pp_vals), ',.0f')} "
            f"| {_fmt(np.nanmean(obs_vals) if obs_vals is not None else None, ',.0f')} |",
            f"| Std       | {_fmt(np.nanstd(pp_vals), ',.0f')} "
            f"| {_fmt(np.nanstd(obs_vals) if obs_vals is not None else None, ',.0f')} |",
            f"| 5th pct   | {_fmt(np.nanpercentile(pp_vals, 5), ',.0f')} "
            f"| {_fmt(np.nanpercentile(obs_vals, 5) if obs_vals is not None else None, ',.0f')} |",
            f"| 95th pct  | {_fmt(np.nanpercentile(pp_vals, 95), ',.0f')} "
            f"| {_fmt(np.nanpercentile(obs_vals, 95) if obs_vals is not None else None, ',.0f')} |",
        ]
        return lines
    except Exception:
        return []


def _posterior_predictive_section(mmm) -> list[str]:
    """Posterior predictive check: in-sample fitted vs actual."""
    try:
        idata = mmm.idata
        if "posterior_predictive" not in idata:
            return [
                "",
                "## Posterior Predictive Check",
                "",
                "_Not available — run `fit_runner.checks.run_posterior_predictive()` to add._",
            ]

        pp = idata.posterior_predictive
        obs_var = [v for v in pp.data_vars if "obs" in v.lower() or "y" in v.lower()]
        if not obs_var:
            return []
        fitted = pp[obs_var[0]].mean(dim=("chain", "draw")).values.flatten()

        obs_vals = None
        if "observed_data" in idata:
            obs_data = idata.observed_data
            obs_key = list(obs_data.data_vars)[0] if obs_data.data_vars else None
            if obs_key:
                obs_vals = obs_data[obs_key].values.flatten()

        if obs_vals is None:
            return []

        residuals = obs_vals - fitted
        mape = float(np.mean(np.abs(residuals / np.where(obs_vals == 0, 1, obs_vals))))

        lines = [
            "",
            "## Posterior Predictive Check (in-sample)",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Fitted mean | {_fmt(np.mean(fitted), ',.0f')} |",
            f"| Observed mean | {_fmt(np.mean(obs_vals), ',.0f')} |",
            f"| Residual mean | {_fmt(np.mean(residuals), ',.0f')} |",
            f"| Residual std | {_fmt(np.std(residuals), ',.0f')} |",
            f"| MAPE (post mean) | {mape:.1%} |",
        ]
        return lines
    except Exception:
        return []


# ── main report ───────────────────────────────────────────────────────────────

def generate_ds_report(
    spec,
    diagnostics: dict,
    prior_pull_df: pd.DataFrame | None = None,
    overfit_result: dict | None = None,
    mmm=None,
) -> str:
    """Generate Data Science markdown report.

    Parameters
    ----------
    spec        : fitted Spec
    diagnostics : dict from write_diagnostics()
    prior_pull_df : DataFrame from audit_prior_pull() (optional)
    overfit_result : dict from overfit module (optional)
    mmm         : fitted MMM object — enables posterior/prior sections (optional)

    Returns the rendered markdown string.
    """
    import pymc_marketing

    company = getattr(spec, "company", "")
    currency = getattr(spec, "currency", "ILS")
    conv = diagnostics.get("convergence", {})
    metrics = diagnostics.get("fit_metrics", {})

    lines = [
        "# Data Science Diagnostics Report",
        f"**{company}** | Generated {date.today().isoformat()}",
        f"pymc-marketing version: {pymc_marketing.__version__}",
        "",
        "## Model Specification",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Channels | {', '.join(ch.name for ch in spec.channels)} |",
        f"| Target column | `{spec.target.column}` ({spec.target.type}) |",
        f"| Currency | {currency} |",
        f"| Fourier modes | {spec.fourier_order} |",
        f"| Sampler draws | {spec.sampler.draws} |",
        f"| Sampler tune | {spec.sampler.tune} |",
        f"| Sampler chains | {spec.sampler.chains} |",
        f"| target_accept | {spec.sampler.target_accept} |",
        f"| Random seed | {spec.sampler.random_seed} |",
    ]
    for ch in spec.channels:
        lines.append(f"| {ch.name} adstock | {ch.adstock} (l_max={ch.l_max}) |")
        lines.append(f"| {ch.name} saturation | {ch.saturation} |")

    # ── Convergence ────────────────────────────────────────────────────────────
    lines += [
        "",
        "## Convergence",
        "",
        f"Overall tier: **{conv.get('tier', 'UNKNOWN')}**",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
    ]
    rhat = conv.get("rhat_max")
    ess_bulk = conv.get("ess_bulk_min")
    ess_tail = conv.get("ess_tail_min")
    divs = conv.get("divergences", 0)
    bfmi = conv.get("bfmi_min")

    if rhat is not None:
        flag = "✅" if rhat < 1.05 else "❌"
        lines.append(f"| rhat_max | {rhat:.4f} | < 1.05 | {flag} |")
    if ess_bulk is not None:
        flag = "✅" if ess_bulk >= 400 else "❌"
        lines.append(f"| ess_bulk_min | {ess_bulk:.0f} | ≥ 400 | {flag} |")
    if ess_tail is not None:
        flag = "✅" if ess_tail >= 400 else "❌"
        lines.append(f"| ess_tail_min | {ess_tail:.0f} | ≥ 400 | {flag} |")
    lines.append(f"| divergences | {divs} | 0 | {'✅' if divs == 0 else '❌'} |")
    if bfmi is not None:
        flag = "✅" if bfmi > 0.2 else "❌"
        lines.append(f"| bfmi_min | {bfmi:.4f} | > 0.2 | {flag} |")

    issues = conv.get("issues", [])
    if issues:
        lines += ["", "**Convergence issues:**"] + [f"- {i}" for i in issues]

    # ── Fit Metrics ────────────────────────────────────────────────────────────
    lines += [
        "",
        "## Fit Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        if v is not None and k != "error":
            lines.append(f"| {k} | {_fmt(v)} |")

    # ── Overfit ────────────────────────────────────────────────────────────────
    if overfit_result is not None:
        gap = overfit_result.get("gap")
        tier = overfit_result.get("tier", "N/A")
        flag = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(tier, "")
        lines += [
            "",
            "## Overfit Analysis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| In-sample R² | {_fmt(overfit_result.get('in_sample_r2'))} |",
            f"| CV R² | {_fmt(overfit_result.get('cv_r2'))} |",
            f"| Overfit gap | {_fmt(gap)} |",
            f"| Tier | {tier} {flag} |",
        ]

    # ── Prior predictive check ─────────────────────────────────────────────────
    if mmm is not None:
        lines += _prior_predictive_section(mmm)

    # ── Posterior predictive check ─────────────────────────────────────────────
    if mmm is not None:
        lines += _posterior_predictive_section(mmm)

    # ── Adstock decay ──────────────────────────────────────────────────────────
    if mmm is not None:
        lines += _adstock_table(mmm.idata, spec)

    # ── Saturation params ──────────────────────────────────────────────────────
    if mmm is not None:
        lines += _saturation_table(mmm.idata, spec)

    # ── Posterior parameter summary ────────────────────────────────────────────
    if mmm is not None:
        lines += _prior_posterior_table(mmm.idata, spec)

    # ── Prior pull ─────────────────────────────────────────────────────────────
    if prior_pull_df is not None and len(prior_pull_df) > 0:
        lines += [
            "",
            "## Prior Pull (posterior_std / prior_std)",
            "",
            "< 0.8 → data informed ✅  |  0.8–1.0 → weak pull ⚠️  |  > 1.0 → prior dominated ❌",
            "",
            "| Parameter | Ratio | Flag |",
            "|-----------|-------|------|",
        ]
        for _, row in prior_pull_df.iterrows():
            flag_icon = {"INFORMED": "✅", "WEAK": "⚠️", "DOMINATED": "❌"}.get(
                str(row.get("flag", "")).upper(), ""
            )
            lines.append(
                f"| {row['param']} | {row['ratio']:.4f} | {row.get('flag','')} {flag_icon} |"
            )

    lines += [
        "",
        "---",
        f"_Report generated by agent_mmm. Region: {spec.region}._",
    ]

    content = "\n".join(lines)
    save_report("ds", content)
    return content
