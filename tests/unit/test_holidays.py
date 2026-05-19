"""Unit tests for Israeli holiday flags."""
from __future__ import annotations

import pandas as pd
import pytest

from agent_mmm.data.providers.holidays_il import (
    HOLIDAY_COLUMNS,
    add_il_holiday_flags,
    get_active_holiday_columns,
)


def _make_df(start="2022-01-03", n=104):
    dates = pd.date_range(start, periods=n, freq="W-MON")
    return pd.DataFrame({"date": dates, "revenue": 1.0})


def test_holiday_columns_added():
    df = _make_df()
    out = add_il_holiday_flags(df)
    for col in HOLIDAY_COLUMNS:
        assert col in out.columns, f"Missing {col}"


def test_returns_copy():
    df = _make_df()
    out = add_il_holiday_flags(df)
    assert "is_rosh_hashana" not in df.columns  # original not mutated


def test_rosh_hashana_2022():
    # Rosh Hashana 2022: Sep 25-27 → week of Sep 26
    df = pd.DataFrame({"date": pd.date_range("2022-09-19", periods=4, freq="W-MON")})
    out = add_il_holiday_flags(df)
    # At least one row should be flagged
    assert out["is_rosh_hashana"].sum() >= 1


def test_pesach_2022():
    # Pesach 2022: Apr 15-22 → weeks around Apr 18
    df = pd.DataFrame({"date": pd.date_range("2022-04-11", periods=3, freq="W-MON")})
    out = add_il_holiday_flags(df)
    assert out["is_pesach"].sum() >= 1


def test_summer_break_july_august():
    df = pd.DataFrame({"date": pd.date_range("2022-07-04", periods=9, freq="W-MON")})
    out = add_il_holiday_flags(df)
    # All July and August rows should have is_summer_break = 1
    summer_rows = out[pd.to_datetime(out["date"]).dt.month.isin([7, 8])]
    assert (summer_rows["is_summer_break"] == 1).all()


def test_get_active_holiday_columns():
    df = _make_df("2022-01-03", 104)
    out = add_il_holiday_flags(df)
    active = get_active_holiday_columns(out)
    # 2 years of data should hit most holidays
    assert len(active) > 3


def test_binary_values():
    df = _make_df()
    out = add_il_holiday_flags(df)
    for col in HOLIDAY_COLUMNS:
        assert out[col].isin([0, 1]).all(), f"{col} has non-binary values"
