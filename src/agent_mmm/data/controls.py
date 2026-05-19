"""Recommend and add control variables for IL MMM."""
from __future__ import annotations

import pandas as pd

from .providers.holidays_il import add_il_holiday_flags, get_active_holiday_columns


def recommend_controls(df: pd.DataFrame, spec, audit_result=None) -> dict:
    """
    Recommend control variables. Adds IL holiday flags (always; no key needed).
    External macro/search providers are added in Phase 4.

    Returns:
        {
            "controls": list[str],   # column names to add to spec.controls
            "df": pd.DataFrame,      # df with new control columns
        }
    """
    df = add_il_holiday_flags(df, spec.data.date_col)
    active = get_active_holiday_columns(df)
    return {"controls": active, "df": df}
