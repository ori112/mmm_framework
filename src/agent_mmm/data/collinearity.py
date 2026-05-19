"""Collinearity and structural break checks for MMM channel spend."""
from __future__ import annotations

import numpy as np
import pandas as pd


def vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each column via OLS R^2.
    Returns DataFrame with columns ['feature', 'vif'].
    """
    from sklearn.linear_model import LinearRegression

    X = df[columns].dropna().values.astype(float)
    rows = []
    for i, col in enumerate(columns):
        others = [j for j in range(X.shape[1]) if j != i]
        if not others:
            rows.append({"feature": col, "vif": 1.0})
            continue
        y = X[:, i]
        X_others = X[:, others]
        r2 = LinearRegression().fit(X_others, y).score(X_others, y)
        vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        rows.append({"feature": col, "vif": round(vif_val, 2)})
    return pd.DataFrame(rows)


def pairwise_corr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix for channel spend columns."""
    return df[columns].corr()


def flag_high_vif(vif_df: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    """Return feature names with VIF above threshold (severe collinearity)."""
    return vif_df[vif_df["vif"] > threshold]["feature"].tolist()


def flag_high_corr(corr_df: pd.DataFrame, threshold: float = 0.70) -> list[tuple[str, str, float]]:
    """Return (feat_a, feat_b, corr) pairs with |corr| > threshold."""
    pairs = []
    cols = corr_df.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = corr_df.loc[a, b]
            if abs(c) > threshold:
                pairs.append((a, b, round(float(c), 3)))
    return pairs


def structural_break(series: pd.Series, min_size: int = 26) -> list:
    """
    CUSUM-based structural break detection.
    Returns list of index labels where a break is suspected.
    """
    s = series.dropna().reset_index(drop=True)
    n = len(s)
    if n < min_size * 2:
        return []

    vals = s.values.astype(float)
    mean = np.mean(vals)
    cusum = np.cumsum(vals - mean)
    threshold = 2.0 * np.std(cusum)

    peak_idx = int(np.argmax(np.abs(cusum)))
    if abs(cusum[peak_idx]) > threshold:
        return [series.index[peak_idx] if len(series.index) > peak_idx else peak_idx]
    return []
