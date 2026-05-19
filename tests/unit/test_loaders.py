"""Unit tests for data loaders."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from agent_mmm.spec.schema import Channel, DataCfg, Spec, TargetUnit
from agent_mmm.data.io import load_panel
from agent_mmm.data.loaders.dataframe import register_dataframe


def _make_df() -> pd.DataFrame:
    import numpy as np
    dates = pd.date_range("2022-01-01", periods=10, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.rand(10) * 1000,
        "google_spend": np.random.rand(10) * 500,
    })


def _spec_csv(path: str) -> Spec:
    return Spec(
        data=DataCfg(source="csv", path=path, date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend")],
    )


def _spec_df(key: str) -> Spec:
    return Spec(
        data=DataCfg(source="dataframe", path=key, date_col="date"),
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend")],
    )


def test_csv_loader():
    df_orig = _make_df()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        df_orig.to_csv(path, index=False)
        spec = _spec_csv(str(path))
        df = load_panel(spec)
    assert len(df) == 10
    assert "revenue" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_csv_loader_sorted():
    df_orig = _make_df().sample(frac=1)  # shuffle
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.csv"
        df_orig.to_csv(path, index=False)
        spec = _spec_csv(str(path))
        df = load_panel(spec)
    assert df["date"].is_monotonic_increasing


def test_dataframe_loader():
    df_orig = _make_df()
    key = "test_df_loader"
    register_dataframe(key, df_orig)
    spec = _spec_df(key)
    df = load_panel(spec)
    assert len(df) == 10


def test_dataframe_loader_missing_key():
    spec = _spec_df("nonexistent_key_xyz")
    with pytest.raises(ValueError, match="No DataFrame registered"):
        load_panel(spec)


def test_unknown_source():
    spec = Spec(
        data=DataCfg(source="bigquery"),  # type: ignore[arg-type]
        target=TargetUnit(column="revenue"),
        channels=[Channel(name="google", spend_col="google_spend")],
    )
    with pytest.raises(ValueError, match="No loader registered"):
        load_panel(spec)
