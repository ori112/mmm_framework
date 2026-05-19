"""Generate synthetic 104-week IL MMM dataset with known ground-truth parameters.

Run directly:
    python tests/fixtures/synthetic/generate.py

Produces:
    tests/fixtures/synthetic/data.csv
    tests/fixtures/synthetic/truth.json
"""
from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 104
START_DATE = "2022-01-03"  # First Monday of 2022

TRUE_PARAMS: dict = {
    "google": {"alpha": 0.10, "lam": 4.0, "beta_scale": 300_000},
    "facebook": {"alpha": 0.20, "lam": 3.0, "beta_scale": 250_000},
    "tv": {"alpha": 0.60, "lam": 1.5, "beta_scale": 400_000},
    "baseline_ils": 1_000_000,
    "noise_sigma_ils": 80_000,
}


def _geometric_adstock(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(l_max, t + 1)):
            out[t] += x[t - lag] * (alpha**lag)
    return out


def _logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    x_norm = x / (x.max() + 1e-8)
    return 1.0 - np.exp(-lam * x_norm)


def _il_seasonality_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Israeli seasonal multiplier (centred at 1.0)."""
    years = dates.year.unique().tolist()
    il = holidays.Israel(years=years)
    holiday_set = set(il.keys())

    mult = np.ones(len(dates))
    for i, dt in enumerate(dates.date):
        m = dt.month
        if m in (9, 10):    # Tishrei gift season
            mult[i] = 1.15
        elif m in (3, 4):   # Pesach
            mult[i] = 1.08
        elif m in (7, 8):   # Summer vacation dip
            mult[i] = 0.92
        # Slight dip on actual holiday weeks
        if any(abs((dt - h).days) <= 3 for h in holiday_set):
            mult[i] *= 0.95

    return mult


def generate(output_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic data. Returns (df, truth_dict)."""
    rng = np.random.default_rng(SEED)
    output_dir = output_dir or Path(__file__).parent

    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")

    # Spend (ILS/week) — log-normal
    g_spend = rng.lognormal(np.log(150_000), 0.5, N_WEEKS)
    f_spend = rng.lognormal(np.log(100_000), 0.6, N_WEEKS)
    tv_spend = rng.lognormal(np.log(200_000), 0.4, N_WEEKS)
    tv_spend[32:45] *= 0.3  # summer campaign pause (weeks 33-45)

    # Adstock
    g_ad = _geometric_adstock(g_spend, TRUE_PARAMS["google"]["alpha"], l_max=5)
    f_ad = _geometric_adstock(f_spend, TRUE_PARAMS["facebook"]["alpha"], l_max=8)
    tv_ad = _geometric_adstock(tv_spend, TRUE_PARAMS["tv"]["alpha"], l_max=13)

    # Saturation
    g_sat = _logistic_saturation(g_ad, TRUE_PARAMS["google"]["lam"])
    f_sat = _logistic_saturation(f_ad, TRUE_PARAMS["facebook"]["lam"])
    tv_sat = _logistic_saturation(tv_ad, TRUE_PARAMS["tv"]["lam"])

    # Contributions (ILS)
    g_contrib = TRUE_PARAMS["google"]["beta_scale"] * g_sat
    f_contrib = TRUE_PARAMS["facebook"]["beta_scale"] * f_sat
    tv_contrib = TRUE_PARAMS["tv"]["beta_scale"] * tv_sat

    # Revenue = baseline * IL_seasonality + media contributions + noise
    seasonality = _il_seasonality_multiplier(pd.DatetimeIndex(dates))
    noise = rng.normal(0, TRUE_PARAMS["noise_sigma_ils"], N_WEEKS)

    revenue = (
        TRUE_PARAMS["baseline_ils"] * seasonality
        + g_contrib
        + f_contrib
        + tv_contrib
        + noise
    ).clip(min=0)

    df = pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue.round(2),
            "google_spend": g_spend.round(2),
            "facebook_spend": f_spend.round(2),
            "tv_spend": tv_spend.round(2),
        }
    )

    truth = {
        "true_params": TRUE_PARAMS,
        "true_roas": {
            "google": round(float(g_contrib.sum() / g_spend.sum()), 4),
            "facebook": round(float(f_contrib.sum() / f_spend.sum()), 4),
            "tv": round(float(tv_contrib.sum() / tv_spend.sum()), 4),
        },
        "n_weeks": N_WEEKS,
        "start_date": START_DATE,
        "seed": SEED,
    }

    df.to_csv(output_dir / "data.csv", index=False)
    (output_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return df, truth


if __name__ == "__main__":
    df, truth = generate()
    print(f"Generated {len(df)} rows.")
    print(f"True ROAS: {truth['true_roas']}")
    print(f"Revenue range: {df['revenue'].min():,.0f} - {df['revenue'].max():,.0f} ILS")
