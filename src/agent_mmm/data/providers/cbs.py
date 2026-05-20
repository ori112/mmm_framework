"""Israeli Central Bureau of Statistics (CBS / Lishkat HaStatistika) data provider.

Fetches: CPI, unemployment rate, consumer confidence index, retail sales index.
Uses the CBS open API (no API key required).
Gated on network access; returns empty DataFrame if unavailable.

CBS API reference: https://www.cbs.gov.il/en/Pages/default.aspx (open data section)
Series codes use the CBS SDMX / API format.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# CBS SDMX-like endpoints (representative — verify exact endpoint at CBS open data portal)
_CBS_BASE = "https://api.cbs.gov.il/Index/data/series"
_SERIES_MAP = {
    "cpi": "120010",             # Consumer Price Index (monthly)
    "unemployment": "10280011",  # Unemployment rate (quarterly)
    "retail_sales": "110610020", # Retail trade index (monthly)
    "consumer_confidence": "10940105",  # Consumer confidence index (quarterly)
}


def available() -> bool:
    """Return True if the CBS API is reachable."""
    try:
        import requests
        r = requests.get("https://api.cbs.gov.il", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def fetch(
    start: str | date,
    end: str | date,
    series: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch CBS macro series and return a tidy weekly DataFrame.

    series: list of keys from _SERIES_MAP. Defaults to all.
    Returns DataFrame with columns: date, <series_key>... (weekly, Monday-aligned).
    Returns empty DataFrame on failure (provider is gated on connectivity).
    """
    import requests

    series = series or list(_SERIES_MAP.keys())
    start_str = str(start)[:10]
    end_str = str(end)[:10]

    frames = []
    for key in series:
        code = _SERIES_MAP.get(key)
        if code is None:
            log.warning("Unknown CBS series key: %s", key)
            continue
        url = f"{_CBS_BASE}/{code}?startPeriod={start_str}&endPeriod={end_str}&format=json"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            rows = [
                {"date": pd.to_datetime(obs["period"]), key: float(obs.get("value", "nan") or "nan")}
                for obs in data.get("data", [])
                if "period" in obs
            ]
            if rows:
                frames.append(pd.DataFrame(rows).set_index("date"))
        except Exception as exc:
            log.warning("CBS series %s fetch failed: %s", code, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date"})
    df = df.sort_values("date")
    # Resample to weekly Monday (forward-fill monthly/quarterly data)
    df = df.set_index("date").resample("W-MON").ffill().reset_index()
    return df
