"""Build and validate control columns for MMM from spec and data."""
from __future__ import annotations

import pandas as pd

from ..data.providers.holidays_il import add_il_holiday_flags, get_active_holiday_columns


def build_controls(df: pd.DataFrame, spec) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure control columns are present in df.
    - Adds IL holiday flags if not already there.
    - Merges spec.controls with auto-generated holiday controls.
    - Returns (updated_df, sorted list of control column names).

    Only includes columns that exist in df AND have at least one non-zero row.
    """
    # Add holiday flags if not yet present
    if "is_rosh_hashana" not in df.columns:
        df = add_il_holiday_flags(df, spec.data.date_col)

    # Base: columns already declared in spec
    declared = list(spec.controls)

    # Add active holiday columns not already declared
    active_holidays = get_active_holiday_columns(df)
    merged = declared + [c for c in active_holidays if c not in declared]

    # Keep only cols that exist in df
    control_cols = [c for c in merged if c in df.columns]

    return df, control_cols
