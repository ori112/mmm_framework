"""Overfit detection: in-sample vs CV R² gap."""
from __future__ import annotations


def overfit_gap(in_sample_r2: float, cv_r2: float) -> float:
    """Return train-test R² gap. > 0.20 indicates likely overfitting."""
    return max(0.0, in_sample_r2 - cv_r2)


def overfit_tier(gap: float) -> str:
    """Classify overfit gap into PASS / WARN / FAIL."""
    if gap < 0.05:
        return "PASS"
    if gap < 0.20:
        return "WARN"
    return "FAIL"


def compute_overfit(mmm, X, y, n_splits: int = 3) -> dict:
    """Run TimeSliceCrossValidator and compute overfit gap.

    Returns dict: {in_sample_r2, cv_r2, gap, tier}.
    Falls back gracefully if TSCV is unavailable.
    """
    from agent_mmm.diagnose.fit_metrics import compute_in_sample_metrics

    metrics = compute_in_sample_metrics(mmm, X, y)
    in_r2 = metrics.get("in_sample_r2", 0.0) or 0.0

    cv_r2 = None
    try:
        from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator
        cv = TimeSliceCrossValidator(n_splits=n_splits)
        scores = cv.cross_val_score(mmm, X, y)
        cv_r2 = float(scores.mean()) if hasattr(scores, "mean") else float(scores)
    except Exception:
        cv_r2 = None

    gap = overfit_gap(in_r2, cv_r2) if cv_r2 is not None else 0.0
    tier = overfit_tier(gap) if cv_r2 is not None else "UNKNOWN"

    return {
        "in_sample_r2": in_r2,
        "cv_r2": cv_r2,
        "gap": gap,
        "tier": tier,
    }
