"""Unit tests for data quality audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_mmm.data.quality import audit_data, AuditResult
from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit


def _make_spec(channels=None) -> Spec:
    return Spec(
        data=DataCfg(source="csv", path="data.csv", date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=channels or [
            Channel(name="google", spend_col="google_spend"),
            Channel(name="tv", spend_col="tv_spend"),
        ],
    )


def _make_df(n: int = 104) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    return pd.DataFrame(
        {
            "date": dates,
            "revenue": rng.lognormal(14, 0.3, n),
            "google_spend": rng.lognormal(11, 0.5, n),
            "tv_spend": rng.lognormal(12, 0.4, n),
        }
    )


def test_pass_on_good_data():
    df = _make_df(104)
    result = audit_data(df, _make_spec())
    assert result.tier == "PASS"
    tiers = {f.check: f.tier for f in result.findings}
    assert tiers["row_count"] == "PASS"
    assert tiers["date_gaps"] == "PASS"
    assert tiers["target_variation"] == "PASS"


def test_fail_under_52_weeks():
    df = _make_df(40)
    result = audit_data(df, _make_spec())
    tiers = {f.check: f.tier for f in result.findings}
    assert tiers["row_count"] == "FAIL"
    assert result.tier == "FAIL"


def test_warn_between_52_and_104():
    df = _make_df(70)
    result = audit_data(df, _make_spec())
    tiers = {f.check: f.tier for f in result.findings}
    assert tiers["row_count"] == "WARN"


def test_fail_missing_date_column():
    df = _make_df(104).rename(columns={"date": "week"})
    result = audit_data(df, _make_spec())
    assert result.tier == "FAIL"
    assert any(f.check == "date_column" and f.tier == "FAIL" for f in result.findings)


def test_fail_missing_channel_column():
    df = _make_df(104).drop(columns=["tv_spend"])
    result = audit_data(df, _make_spec())
    assert result.tier == "FAIL"


def test_fail_negative_spend():
    df = _make_df(104)
    df.loc[0, "google_spend"] = -100
    result = audit_data(df, _make_spec())
    assert any("neg" in f.check for f in result.findings if f.tier == "FAIL")


def test_fail_channel_all_zeros():
    df = _make_df(104)
    df["tv_spend"] = 0.0
    result = audit_data(df, _make_spec())
    assert any(f.tier == "FAIL" and "tv" in f.check for f in result.findings)


def test_warn_sparse_channel():
    df = _make_df(104)
    # 50% zeros — should WARN
    df.loc[df.index[:52], "tv_spend"] = 0.0
    result = audit_data(df, _make_spec())
    assert any(f.tier == "WARN" and "tv" in f.check for f in result.findings)
