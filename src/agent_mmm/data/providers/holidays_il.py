"""Israeli holiday flags for MMM control variables.

Uses the `holidays` package (holidays.Israel) — no API key required.
"""
from __future__ import annotations

from datetime import timedelta

import holidays
import pandas as pd


# Canonical holiday column names
HOLIDAY_COLUMNS = [
    "is_rosh_hashana",
    "is_yom_kippur",
    "is_sukkot",
    "is_pesach",
    "is_shavuot",
    "is_independence_day",
    "is_summer_break",
]

# Map substrings in holiday names (en_US) to column names
_HOLIDAY_MAP = {
    "Rosh Hashanah": "is_rosh_hashana",
    "Yom Kippur": "is_yom_kippur",
    "Sukkot": "is_sukkot",
    "Pesach": "is_pesach",
    "Shavuot": "is_shavuot",
    "Independence Day": "is_independence_day",
}


def add_il_holiday_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add binary flag columns for major Israeli holiday periods.
    Each holiday window is expanded to cover +-3 days (one full week).
    Also adds is_summer_break (July + August).

    Returns a copy of df with added columns.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    years = sorted(dates.dt.year.unique().tolist())

    il = holidays.Israel(years=years, language="en_US")

    # Initialise columns
    for col in HOLIDAY_COLUMNS:
        df[col] = 0

    for holiday_date, holiday_name in il.items():
        window_start = holiday_date - timedelta(days=3)
        window_end = holiday_date + timedelta(days=3)
        mask = (dates.dt.date >= window_start) & (dates.dt.date <= window_end)
        if not mask.any():
            continue
        for keyword, col in _HOLIDAY_MAP.items():
            if keyword in holiday_name:
                df.loc[mask, col] = 1

    # Summer break — July + August
    df.loc[dates.dt.month.isin([7, 8]), "is_summer_break"] = 1

    return df


def get_active_holiday_columns(df: pd.DataFrame) -> list[str]:
    """Return holiday columns that have at least one positive row in df."""
    return [c for c in HOLIDAY_COLUMNS if c in df.columns and df[c].sum() > 0]
