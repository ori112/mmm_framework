"""Thin CLI shims: one function per /mmm-* skill, callable from agents and __main__."""
from __future__ import annotations


def cmd_intake_quick(args: list[str]) -> None:
    from agent_mmm.spec.intake import build_spec_from_answers
    build_spec_from_answers(quick=True)


def cmd_intake(args: list[str]) -> None:
    from agent_mmm.spec.intake import build_spec_from_answers
    build_spec_from_answers(quick=False)


def cmd_analyze_data(args: list[str]) -> None:
    import json
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.data.quality import audit_data

    spec = load_spec(spec_path())
    df = load_panel(spec)
    result = audit_data(df, spec)
    print(f"Audit tier: {result.tier}")
    for f in result.findings:
        print(f"  [{f.tier}] {f.check}: {f.message}")


def cmd_recommend_controls(args: list[str]) -> None:
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.data.controls import recommend_controls

    spec = load_spec(spec_path())
    df = load_panel(spec)
    out = recommend_controls(df, spec)
    print("Recommended controls:")
    for ctrl in out["controls"]:
        print(f"  {ctrl}")


def cmd_recommend_priors(args: list[str]) -> None:
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors

    spec = load_spec(spec_path())
    df = load_panel(spec)
    mc = recommend_priors(spec, df)
    print("Model config priors:")
    for k, v in mc.items():
        print(f"  {k}: {v}")


def cmd_build(args: list[str]) -> None:
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.model_factory.builder import build_mmm

    spec = load_spec(spec_path())
    df = load_panel(spec)
    mc = recommend_priors(spec, df)
    mmm, df_out, control_cols = build_mmm(spec, mc, df)
    print(f"Model built. Controls: {control_cols}")


def cmd_fit(args: list[str]) -> None:
    from agent_mmm.fit_runner.runner import run_pipeline
    report = run_pipeline()
    print("Fit complete.")
    conv = report.get("convergence", {})
    rhat = conv.get("rhat_max")
    if rhat is not None:
        print(f"  rhat_max: {rhat:.4f}")
    metrics = report.get("fit_metrics", {})
    r2 = metrics.get("in_sample_r2")
    if r2 is not None:
        print(f"  in_sample_r2: {r2:.4f}")


def cmd_diagnose(args: list[str]) -> None:
    import json
    from agent_mmm.workspace.paths import diagnostics_path

    path = diagnostics_path()
    if not path.exists():
        print("No diagnostics.json found. Run 'fit' first.")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    conv = report.get("convergence", {})
    print(f"Convergence tier: {conv.get('tier', 'unknown')}")
    rhat = conv.get("rhat_max")
    if rhat is not None:
        print(f"  rhat_max: {rhat:.4f}")
    metrics = report.get("fit_metrics", {})
    r2 = metrics.get("in_sample_r2")
    if r2 is not None:
        print(f"  in_sample_r2: {r2:.4f}")


def cmd_attribute(args: list[str]) -> None:
    import json
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path, idata_path, diagnostics_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.model_factory.builder import build_mmm
    from agent_mmm.workspace.artifacts import load_idata
    from agent_mmm.attribute.contributions import get_contributions
    from agent_mmm.attribute.roas import compute_effectiveness

    spec = load_spec(spec_path())
    df = load_panel(spec)
    mc = recommend_priors(spec, df)
    mmm, df_out, control_cols = build_mmm(spec, mc, df)
    mmm.idata = load_idata()

    target_col = spec.target.column
    y = df[target_col].rename(target_col)
    X = df_out.drop(columns=[target_col], errors="ignore")

    is_revenue = spec.target.type == "revenue"
    currency = getattr(spec, "currency", "ILS")
    unit_name = spec.target.unit_name or "unit"

    contrib_df = get_contributions(mmm)
    eff_df = compute_effectiveness(mmm, X, spec)

    print("Channel contributions:")
    for _, row in contrib_df.iterrows():
        ch = row["channel"].replace("_spend", "")
        if is_revenue:
            print(f"  {ch}: {currency} {row['mean']:,.0f} (89% CI: {row['hdi_low']:,.0f}–{row['hdi_high']:,.0f})")
        else:
            print(f"  {ch}: {row['mean']:,.0f} {unit_name}s (89% CI: {row['hdi_low']:,.0f}–{row['hdi_high']:,.0f})")

    if len(eff_df) > 0:
        label = eff_df["metric_label"].iloc[0]
        print(f"\n{label}:")
        for _, row in eff_df.iterrows():
            ch = row["channel"].replace("_spend", "")
            val = row["metric_value_mean"]
            implied = row.get("implied_roas_mean", None)
            if is_revenue:
                print(f"  {ch}: {val:.2f}x")
            else:
                suffix = f"  (implied ROAS {implied:.2f}x)" if (implied is not None and not (isinstance(implied, float) and implied != implied)) else ""
                print(f"  {ch}: {currency} {val:,.0f}/{unit_name}{suffix}")


def cmd_optimize(args: list[str]) -> None:
    import json
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.model_factory.builder import build_mmm
    from agent_mmm.workspace.artifacts import load_idata
    from agent_mmm.optimize.budget import optimize

    spec = load_spec(spec_path())
    df = load_panel(spec)
    mc = recommend_priors(spec, df)
    mmm, df_out, _ = build_mmm(spec, mc, df)
    mmm.idata = load_idata()

    target_col = spec.target.column
    X = df_out.drop(columns=[target_col], errors="ignore")
    result = optimize(mmm, X, spec)

    print(f"Total budget: {sum(result['current_allocation'].values()):,.0f} {result['currency']}")
    print("Optimal allocation:")
    for ch, spend in result["optimal_allocation"].items():
        print(f"  {ch}: {spend:,.0f}")
    print(f"Expected uplift: {result['expected_uplift_mean']:,.0f} "
          f"(P5: {result['expected_uplift_p5']:,.0f}, P95: {result['expected_uplift_p95']:,.0f})")
    print(f"P(positive uplift): {result['p_positive_uplift']*100:.1f}%")


def cmd_improve(args: list[str]) -> None:
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.iter_loop.tournament import run_tournament

    spec = load_spec(spec_path())
    df = load_panel(spec)
    result = run_tournament(spec, df, max_rounds=3, n_variants=6, patience=2)
    print(f"Tournament complete. Best score: {result['best_score']:.4f}")
    print(f"Best run ID: {result['best_run_id']}")


def cmd_report(args: list[str]) -> None:
    import json
    from agent_mmm.spec.loader import load_spec
    from agent_mmm.workspace.paths import spec_path, diagnostics_path
    from agent_mmm.data.io import load_panel
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.model_factory.builder import build_mmm
    from agent_mmm.workspace.artifacts import load_idata
    from agent_mmm.report.render_all import render_all

    spec = load_spec(spec_path())
    df = load_panel(spec)
    mc = recommend_priors(spec, df)
    mmm, df_out, control_cols = build_mmm(spec, mc, df)
    mmm.idata = load_idata()

    diagnostics = {}
    if diagnostics_path().exists():
        diagnostics = json.loads(diagnostics_path().read_text(encoding="utf-8"))

    target_col = spec.target.column
    y = df[target_col].rename(target_col)
    X = df_out.drop(columns=[target_col], errors="ignore")

    reports = render_all(mmm, X, y, spec, diagnostics)
    for role, path in reports.items():
        print(f"Wrote {role} report")
    print("Reports written to mmm-workspace/reports/")


COMMANDS: dict[str, object] = {
    "intake-quick": cmd_intake_quick,
    "intake": cmd_intake,
    "analyze-data": cmd_analyze_data,
    "recommend-controls": cmd_recommend_controls,
    "recommend-priors": cmd_recommend_priors,
    "build": cmd_build,
    "fit": cmd_fit,
    "diagnose": cmd_diagnose,
    "attribute": cmd_attribute,
    "optimize": cmd_optimize,
    "improve": cmd_improve,
    "report": cmd_report,
}
