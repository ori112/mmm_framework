from __future__ import annotations

from typing import Callable

import pandas as pd

from .loaders.csv import load_csv
from .loaders.dataframe import load_dataframe
from .loaders.parquet import load_parquet
from .loaders.bigquery import load_bigquery

_REGISTRY: dict[str, Callable] = {
    "csv": load_csv,
    "parquet": load_parquet,
    "dataframe": load_dataframe,
    "bigquery": load_bigquery,
}


def register_loader(source: str, loader: Callable) -> None:
    """Register a custom data loader (e.g. bigquery) keyed by source name."""
    _REGISTRY[source] = loader


def load_panel(spec) -> pd.DataFrame:
    """Dispatch to the appropriate loader based on spec.data.source."""
    loader = _REGISTRY.get(spec.data.source)
    if loader is None:
        raise ValueError(
            f"No loader registered for source '{spec.data.source}'. "
            f"Available: {list(_REGISTRY)}"
        )
    return loader(spec)
