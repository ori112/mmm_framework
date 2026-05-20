"""Recommend and add control variables for IL MMM."""
from __future__ import annotations

import logging

import pandas as pd

from .providers.holidays_il import add_il_holiday_flags, get_active_holiday_columns

log = logging.getLogger(__name__)


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


def _add_external_controls(df: pd.DataFrame, spec) -> tuple[pd.DataFrame, list[str]]:
    """Attempt to fetch and merge BoI and CBS macro controls when providers are available.

    Each provider gates on network availability. Returns (df, new_col_names).
    """
    date_col = spec.data.date_col
    dates = pd.to_datetime(df[date_col])
    start = str(dates.min().date())
    end = str(dates.max().date())
    new_cols: list[str] = []

    from .providers.boi import available as boi_ok, fetch as boi_fetch
    if boi_ok():
        try:
            boi_df = boi_fetch(start, end, series=["policy_rate", "ils_usd"])
            if not boi_df.empty:
                boi_df["date"] = pd.to_datetime(boi_df["date"])
                df = df.merge(boi_df, left_on=date_col, right_on="date", how="left").drop(columns=["date"], errors="ignore")
                new_cols += [c for c in boi_df.columns if c != "date" and c in df.columns]
                log.info("BoI controls added: %s", new_cols[-len(boi_df.columns)+1:])
        except Exception as exc:
            log.warning("BoI provider failed: %s", exc)

    from .providers.cbs import available as cbs_ok, fetch as cbs_fetch
    if cbs_ok():
        try:
            cbs_df = cbs_fetch(start, end, series=["cpi", "consumer_confidence"])
            if not cbs_df.empty:
                cbs_df["date"] = pd.to_datetime(cbs_df["date"])
                df = df.merge(cbs_df, left_on=date_col, right_on="date", how="left").drop(columns=["date"], errors="ignore")
                new_cols += [c for c in cbs_df.columns if c != "date" and c in df.columns]
                log.info("CBS controls added: %s", new_cols[-len(cbs_df.columns)+1:])
        except Exception as exc:
            log.warning("CBS provider failed: %s", exc)

    return df, new_cols


def recommend_controls(df: pd.DataFrame, spec, audit_result=None) -> dict:
    """Recommend control variables:
    1. IL holiday flags (always; no key needed)
    2. Industry-specific seasonal peak flag (based on spec.industry)
    3. BoI and CBS macro controls (when providers are reachable; fail gracefully)

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

    # External macro providers (gated; never raise)
    df, ext_cols = _add_external_controls(df, spec)
    active = active + [c for c in ext_cols if c not in active]

    return {"controls": active, "df": df}
