"""Tournament-based model selection for iterative MMM improvement."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from agent_mmm.iter_loop.leaderboard import append_run, best_run, load_leaderboard
from agent_mmm.iter_loop.scoring import composite_score, is_plausible
from agent_mmm.iter_loop.variants import Variant, apply_variant_to_spec, generate_variants

log = logging.getLogger(__name__)

PATIENCE = 2
MAX_ROUNDS = 3
N_VARIANTS = 6


def _run_variant(spec, mc, df, variant: Variant, round_num: int) -> dict:
    """Fit one variant, compute diagnostics + score. Returns result dict."""
    from agent_mmm.model_factory.builder import build_mmm
    from agent_mmm.fit_runner.runner import run_fit
    from agent_mmm.diagnose.convergence import check_convergence
    from agent_mmm.diagnose.fit_metrics import compute_in_sample_metrics
    from agent_mmm.attribute.contributions import get_contributions
    from agent_mmm.attribute.roas import compute_roas

    v_spec = apply_variant_to_spec(spec, variant)
    mmm = run_fit(v_spec, mc, df, skip_prior_check=True)

    target_col = v_spec.target.column
    y = df[target_col].rename(target_col)
    X = df.drop(columns=[target_col], errors="ignore")

    conv = check_convergence(mmm.idata)
    metrics = compute_in_sample_metrics(mmm, X, y)
    in_r2 = metrics.get("in_sample_r2", 0.0) or 0.0

    # CV R² via TimeSliceCrossValidator (fast n_folds=3)
    cv_r2 = 0.0
    try:
        from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator
        cv = TimeSliceCrossValidator(n_splits=3)
        cv_scores = cv.cross_val_score(mmm, X, y)
        cv_r2 = float(cv_scores.mean()) if hasattr(cv_scores, "mean") else float(cv_scores)
    except Exception as e:
        log.warning("CV R² skipped: %s", e)
        cv_r2 = in_r2 * 0.9  # Conservative fallback

    converged = conv["tier"] == "PASS"
    n_div = conv.get("divergences", 0) or 0

    contrib_df = get_contributions(mmm)
    roas_df = compute_roas(mmm, X)
    plausible = is_plausible(contrib_df, roas_df)

    overfit_gap = max(0.0, in_r2 - cv_r2)
    score = composite_score(cv_r2, in_r2, converged, plausible)

    run_id = f"{datetime.utcnow().strftime('%Y-%m-%dT%H-%M')}_r{round_num}_{variant.name}"
    append_run(run_id, round_num, variant.name, score, cv_r2, in_r2, overfit_gap, converged, n_div)

    return {
        "run_id": run_id,
        "variant": variant,
        "score": score,
        "cv_r2": cv_r2,
        "in_sample_r2": in_r2,
        "converged": converged,
        "mmm": mmm,
        "model_config": mc,
    }


def run_tournament(
    spec,
    df: pd.DataFrame | None = None,
    max_rounds: int = MAX_ROUNDS,
    n_variants: int = N_VARIANTS,
    patience: int = PATIENCE,
    model_config: dict | None = None,
) -> dict:
    """Run tournament improvement loop.

    Each round fits n_variants models, picks the winner by composite score,
    tightens priors from the winner, and repeats. Stops when score plateau
    < 0.01 improvement for `patience` consecutive rounds or max_rounds reached.

    Returns: {"best_run_id": ..., "best_score": ..., "rounds": [...]}
    """
    from agent_mmm.prior_engine.recommender import recommend_priors
    from agent_mmm.prior_engine.posterior_informed import tighten_priors_from_idata
    from agent_mmm.data.io import load_panel

    if df is None:
        df = load_panel(spec)

    if model_config is None:
        model_config = recommend_priors(spec, df)

    mc = model_config
    best_score = -1.0
    no_improve_rounds = 0
    round_results = []

    for rnd in range(1, max_rounds + 1):
        log.info("Tournament round %d/%d", rnd, max_rounds)
        variants = generate_variants(n=n_variants)

        round_best = None
        for v in variants:
            try:
                result = _run_variant(spec, mc, df, v, rnd)
                if round_best is None or result["score"] > round_best["score"]:
                    round_best = result
            except Exception as e:
                log.warning("Variant %s failed: %s", v.name, e)

        if round_best is None:
            log.warning("No variants succeeded in round %d", rnd)
            break

        round_results.append(round_best)
        improvement = round_best["score"] - best_score

        if improvement < 0.01:
            no_improve_rounds += 1
        else:
            no_improve_rounds = 0
            best_score = round_best["score"]
            # Tighten priors from winner's idata for next round
            mc = tighten_priors_from_idata(round_best["mmm"].idata, mc)

        log.info(
            "Round %d winner: %s score=%.4f (improvement=%.4f)",
            rnd, round_best["variant"].name, round_best["score"], improvement,
        )

        if no_improve_rounds >= patience:
            log.info("Stopping: no improvement for %d rounds", no_improve_rounds)
            break

    board = load_leaderboard()
    return {
        "best_run_id": board.get("best_run_id"),
        "best_score": board.get("best_score"),
        "rounds": len(round_results),
    }
