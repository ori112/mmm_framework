"""BigQuery data loader (requires google-cloud-bigquery extra)."""
from __future__ import annotations

import pandas as pd


def load_bigquery(spec) -> pd.DataFrame:
    """Load data from BigQuery using spec.data.path as 'project.dataset.table'.

    Requires:
    - google-cloud-bigquery installed (pip install mmm-framework[bigquery])
    - GCP credentials (GOOGLE_APPLICATION_CREDENTIALS env var or gcloud ADC)
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise ImportError(
            "google-cloud-bigquery is not installed. "
            "Install with: pip install mmm-framework[bigquery]"
        ) from e

    table_ref = spec.data.path
    if not table_ref:
        raise ValueError("spec.data.path must be set to 'project.dataset.table' for BigQuery loader.")

    client = bigquery.Client()
    query = f"SELECT * FROM `{table_ref}` ORDER BY {spec.data.date_col}"
    df = client.query(query).to_dataframe()
    df[spec.data.date_col] = pd.to_datetime(df[spec.data.date_col])
    return df.sort_values(spec.data.date_col).reset_index(drop=True)
