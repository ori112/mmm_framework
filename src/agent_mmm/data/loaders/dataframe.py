"""In-memory DataFrame loader.

Usage:
    from agent_mmm.data.loaders.dataframe import register_dataframe
    register_dataframe("my_data", df)

    # In spec.yaml: source: dataframe, path: my_data
"""
from __future__ import annotations

import pandas as pd

_REGISTRY: dict[str, pd.DataFrame] = {}


def register_dataframe(key: str, df: pd.DataFrame) -> None:
    """Register a DataFrame under a key matching spec.data.path."""
    _REGISTRY[key] = df


def load_dataframe(spec) -> pd.DataFrame:
    key = spec.data.path or "default"
    if key not in _REGISTRY:
        raise ValueError(
            f"No DataFrame registered with key '{key}'. "
            "Call agent_mmm.data.loaders.dataframe.register_dataframe(key, df) first, "
            "then set spec.data.path = key."
        )
    df = _REGISTRY[key].copy()
    if spec.data.date_col in df.columns:
        df[spec.data.date_col] = pd.to_datetime(df[spec.data.date_col])
        df = df.sort_values(spec.data.date_col).reset_index(drop=True)
    return df
