"""Full fit pipeline: prior PC -> NUTS -> posterior PC -> save artifacts."""
from __future__ import annotations

import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM

from ..spec.schema import Spec
from ..spec.loader import freeze_spec
from ..workspace.paths import ensure_workspace, idata_path, spec_used_path
from ..workspace.artifacts import save_idata
from .checks import run_prior_predictive, run_posterior_predictive
from .sampler import get_sampler_kwargs


def run_fit(
    spec: Spec,
    model_config: dict,
    df: pd.DataFrame,
    skip_prior_check: bool = False,
) -> MMM:
    """
    Full fit pipeline:
      1. Build MMM (with control columns)
      2. Prior predictive check (unless skip_prior_check=True)
      3. Fit via NUTS
      4. Posterior predictive check
      5. Save idata.nc + spec_used.yaml

    Returns the fitted MMM instance (access idata via mmm.idata).
    """
    from ..model_factory.builder import build_mmm

    ensure_workspace()

    mmm, df, _ = build_mmm(spec, model_config, df)

    # Separate X and y (y must be named to match target_column)
    y = df[spec.target.column].rename(spec.target.column)
    X = df.drop(columns=[spec.target.column], errors="ignore")

    # Step 1: Prior predictive check
    if not skip_prior_check:
        print("  Prior predictive check...", end=" ", flush=True)
        run_prior_predictive(mmm, X, y, samples=200)
        print("done.")

    # Step 2: Fit
    sampler_kwargs = get_sampler_kwargs(spec.sampler)
    print(
        f"  Fitting: {sampler_kwargs['draws']} draws, "
        f"{sampler_kwargs['tune']} tune, {sampler_kwargs['chains']} chains...",
        end=" ",
        flush=True,
    )
    mmm.fit(X, y, **sampler_kwargs)
    print("done.")

    # Step 3: Posterior predictive check
    print("  Posterior predictive check...", end=" ", flush=True)
    run_posterior_predictive(mmm, X)
    print("done.")

    # Save artifacts
    out = idata_path()
    save_idata(mmm.idata, out)
    freeze_spec(spec, spec_used_path())
    print(f"  Artifacts saved -> {out.parent}/")

    return mmm


def run_pipeline(spec_path=None, skip_prior_check: bool = False) -> dict:
    """
    End-to-end pipeline from spec.yaml:
      load spec -> load data -> audit -> priors -> fit -> diagnose

    Returns the diagnostics dict.
    """
    from ..workspace.paths import spec_path as default_spec_path
    from ..spec.loader import load_spec
    from ..data.io import load_panel
    from ..data.quality import audit_data, print_audit_summary
    from ..prior_engine.recommender import recommend_priors
    from ..diagnose.report import write_diagnostics

    sp = load_spec(spec_path or default_spec_path())
    df = load_panel(sp)

    print("\n[1/5] Auditing data...")
    audit = audit_data(df, sp)
    print_audit_summary(audit)
    if audit.tier == "FAIL":
        raise RuntimeError("Data audit FAIL — fix issues before fitting.")

    print("\n[2/5] Recommending priors...")
    model_config = recommend_priors(sp, df)
    print(f"  Priors set for {len(sp.channels)} channels.")

    print("\n[3/5] Fitting model...")
    mmm = run_fit(sp, model_config, df, skip_prior_check=skip_prior_check)

    print("\n[4/5] Writing diagnostics...")
    y = df[sp.target.column]
    X = df.drop(columns=[sp.target.column], errors="ignore")
    report = write_diagnostics(mmm, X, y, sp)

    print(f"\n[5/5] Pipeline complete. Overall: {report['convergence']['tier']}")
    return report
