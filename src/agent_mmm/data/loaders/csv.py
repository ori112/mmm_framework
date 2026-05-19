from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(spec) -> pd.DataFrame:
    path = Path(spec.data.path)
    df = pd.read_csv(path, parse_dates=[spec.data.date_col])
    return df.sort_values(spec.data.date_col).reset_index(drop=True)
