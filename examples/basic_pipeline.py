"""Basic greenfield pipeline on the bundled synthetic IL dataset.

Runs: load data -> audit -> recommend controls -> recommend priors
      -> build -> fit (fast) -> diagnose -> attribute -> report

Usage:
    uv run python examples/basic_pipeline.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

if __name__ == "__main__":
    # Force UTF-8 stdout so ₪ and other Unicode renders correctly on Windows
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Redirect workspace to a temp dir so this example doesn't pollute the repo
    _tmpdir = tempfile.mkdtemp(prefix="mmm_example_")
    os.chdir(_tmpdir)

    from agent_mmm.spec.schema import Channel, DataCfg, Spec, SamplerCfg, TargetUnit
    from agent_mmm.spec.loader import save_spec
    from agent_mmm.workspace.paths import ensure_workspace, spec_path

    # ── 1. Build spec ────────────────────────────────────────────────────────────
    FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic"

    spec = Spec(
        company="Example Co",
        industry="retail",
        region="IL",
        currency="ILS",
        data=DataCfg(source="csv", path=str(FIXTURES / "data.csv"), date_col="date"),
        target=TargetUnit(column="revenue", type="revenue"),
        channels=[
            Channel(name="google",   spend_col="google_spend",   channel_type="digital",  adstock="geometric", l_max=5),
            Channel(name="facebook", spend_col="facebook_spend", channel_type="digital",  adstock="geometric", l_max=8),
            Channel(name="tv",       spend_col="tv_spend",       channel_type="offline",  adstock="geometric", l_max=13),
        ],
        sampler=SamplerCfg(draws=200, tune=200, chains=2, target_accept=0.9, random_seed=42),
    )

    ensure_workspace()
    save_spec(spec, spec_path())
    print(f"Spec written to {spec_path()}")

    # ── 2. Load and audit data ───────────────────────────────────────────────────
    from agent_mmm.data.io import load_panel
    from agent_mmm.data.quality import audit_data

    df = load_panel(spec)
    audit = audit_data(df, spec)
    print(f"\nAudit tier: {audit.tier}")
    for f in audit.findings:
        print(f"  [{f.tier}] {f.check}: {f.message}")

    if audit.tier == "FAIL":
        raise SystemExit("Data audit FAIL — fix issues before fitting.")

    # ── 3. Recommend controls (IL holidays + industry presets) ───────────────────
    from agent_mmm.data.controls import recommend_controls

    out = recommend_controls(df, spec)
    df_with_controls = out["df"]
    print(f"\nControls recommended: {out['controls']}")

    # ── 4. Recommend priors ──────────────────────────────────────────────────────
    from agent_mmm.prior_engine.recommender import recommend_priors

    mc = recommend_priors(spec, df_with_controls)
    print(f"\nPriors set for {len(spec.channels)} channels.")

    # ── 5. Fit ──────────────────────────────────────────────────────────────────
    from agent_mmm.fit_runner.runner import run_fit

    print("\nFitting model (fast mode: 200 draws, 2 chains) …")
    mmm = run_fit(spec, mc, df_with_controls, skip_prior_check=True)

    # ── 6. Posterior predictive check (populates DS report) ──────────────────────
    from agent_mmm.fit_runner.checks import run_posterior_predictive
    print("\nRunning posterior predictive check …")
    run_posterior_predictive(mmm, X=df_with_controls.drop(columns=[spec.target.column], errors="ignore"))

    # ── 7. Diagnose ──────────────────────────────────────────────────────────────
    from agent_mmm.diagnose.report import write_diagnostics

    target_col = spec.target.column
    y = df_with_controls[target_col].rename(target_col)
    X = df_with_controls.drop(columns=[target_col], errors="ignore")

    report = write_diagnostics(mmm, X, y, spec)
    conv = report["convergence"]
    print(f"\nConvergence tier: {conv['tier']}  rhat_max: {conv.get('rhat_max', 'N/A')}")

    # ── 8. Attribution ───────────────────────────────────────────────────────────
    from agent_mmm.attribute.contributions import get_contributions
    from agent_mmm.attribute.roas import compute_effectiveness

    contrib_df = get_contributions(mmm)
    eff_df = compute_effectiveness(mmm, X, spec)

    print("\nChannel contributions (ILS):")
    for _, row in contrib_df.iterrows():
        ch = row["channel"].replace("_spend", "")
        print(f"  {ch}: ₪{row['mean']:,.0f}  (89% CI: ₪{row['hdi_low']:,.0f}–₪{row['hdi_high']:,.0f})")

    print("\nROAS:")
    for _, row in eff_df.iterrows():
        ch = row["channel"].replace("_spend", "")
        print(f"  {ch}: {row['metric_value_mean']:.2f}x")

    # ── 9. Reports ──────────────────────────────────────────────────────────────
    from agent_mmm.report.render_all import render_all

    reports = render_all(mmm, X, y, spec, report)
    print(f"\nReports written: {list(reports.keys())}")
    print(f"Workspace: {_tmpdir}/mmm-workspace/")
