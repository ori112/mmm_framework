"""Recommend and add control variables for IL MMM."""
from __future__ import annotations

import pandas as pd

from .providers.holidays_il import add_il_holiday_flags, get_active_holiday_columns


# Industry-specific seasonality presets (IL calendar, per plan spec)
_INDUSTRY_SEASON_MAP: dict[str, list[int]] = {
    "retail": [3, 4, 9, 10, 11, 12],     # Pesach, Tishrei, Hanukkah/Black-Friday-IL
    "automotive": [1, 8],                  # New-year plates, end-of-summer clearance
    "insurance": [1, 4, 5],               # January renewal, post-Pesach churn
    "telco": [1, 4, 7, 9, 10],            # Quarterly fiscal + back-to-school Sep
    "saas": [1, 4, 7, 9, 10],            # Same as telco
    "other": [],
}


def _add_industry_season_flags(df: pd.DataFrame, spec, date_col: str = "date") -> pd.DataFrame:
    """Add binary flag columns for industry-specific peak months."""
    industry = getattr(spec, "industry", "other")
    peak_months = _INDUSTRY_SEASON_MAP.get(industry, [])
    if not peak_months:
        return df

    col_name = f"is_{industry}_peak"
    dates = pd.to_datetime(df[date_col])
    df[col_name] = dates.dt.month.isin(peak_months).astype(int)
    return df


def recommend_controls(df: pd.DataFrame, spec, audit_result=None) -> dict:
    """Recommend control variables:
    1. IL holiday flags (always; no key needed)
    2. Industry-specific seasonal peak flag (based on spec.industry)
    3. External macro/search controls when providers are available

    Returns:
        {
            "controls": list[str],   # column names to add to spec.controls
            "df": pd.DataFrame,      # df with new control columns
        }
    """
    df = add_il_holiday_flags(df, spec.data.date_col)
    active = get_active_holiday_columns(df)

    # Industry seasonal preset
    n_before = len(df.columns)
    df = _add_industry_season_flags(df, spec, date_col=spec.data.date_col)
    new_cols = [c for c in df.columns[n_before:] if c not in active]
    active = active + new_cols

    return {"controls": active, "df": df}
