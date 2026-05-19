"""Composite scoring function for tournament model selection."""
from __future__ import annotations


def composite_score(
    cv_r2: float,
    in_sample_r2: float,
    converged: bool,
    plausible: bool,
) -> float:
    """Score = cv_r2 × overfit_penalty × convergence_factor × plausibility_factor.

    Returns float in [0, 1] (higher = better).
    cv_r2: cross-validated R² (primary signal)
    in_sample_r2: in-sample R² (used to compute overfit gap)
    converged: True if rhat_max < 1.05 and no divergences
    plausible: True if no single channel > 60% contribution AND no negative ROAS
    """
    overfit_gap = max(0.0, in_sample_r2 - cv_r2)
    overfit_penalty = max(0.0, 1.0 - overfit_gap * 2.0)
    convergence_factor = 1.0 if converged else 0.5
    plausibility_factor = 1.0 if plausible else 0.8
    score = cv_r2 * overfit_penalty * convergence_factor * plausibility_factor
    return round(max(0.0, score), 6)


def is_plausible(contributions_df, roas_df) -> bool:
    """Check attribution plausibility: no channel dominates, no negative ROAS."""
    if contributions_df is None or len(contributions_df) == 0:
        return True

    total = contributions_df["mean"].sum()
    if total > 0:
        max_share = contributions_df["mean"].max() / total
        if max_share > 0.60:
            return False

    if roas_df is not None and len(roas_df) > 0:
        if (roas_df["roas_mean"] < 0).any():
            return False

    return True
