from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet(spec) -> pd.DataFrame:
    path = Path(spec.data.path)
    df = pd.read_parquet(path)
    if spec.data.date_col in df.columns:
        df[spec.data.date_col] = pd.to_datetime(df[spec.data.date_col])
        df = df.sort_values(spec.data.date_col).reset_index(drop=True)
    return df
