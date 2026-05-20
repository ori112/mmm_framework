"""Bank of Israel (BoI) macro data provider.

Fetches: policy interest rate, ILS/USD, ILS/EUR exchange rates.
Uses the BoI public data portal (no API key required).
Endpoint: https://edge.boi.gov.il/FusionEdge/series/...
Gated on network access; returns empty DataFrame if unavailable.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# BoI BDL series codes (confirmed publicly available)
_BOI_BASE = "https://edge.boi.gov.il/FusionEdge/series"
_SERIES_MAP = {
    "policy_rate": "IR01",          # Bank rate (%)
    "ils_usd": "RER_USD",           # ILS per 1 USD (representative rate)
    "ils_eur": "RER_EUR",           # ILS per 1 EUR
}


def available() -> bool:
    """Return True if the BoI API is reachable."""
    try:
        import requests
        r = requests.get(_BOI_BASE, timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def fetch(
    start: str | date,
    end: str | date,
    series: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch BoI macro series and return a tidy weekly DataFrame.

    series: list of keys from _SERIES_MAP. Defaults to all.
    Returns DataFrame with columns: date, <series_key>... (weekly, Monday-aligned).
    Returns empty DataFrame on failure.
    """
    import requests

    series = series or list(_SERIES_MAP.keys())
    start_str = str(start)[:10]
    end_str = str(end)[:10]

    frames = []
    for key in series:
        code = _SERIES_MAP.get(key)
        if code is None:
            log.warning("Unknown BoI series key: %s", key)
            continue
        url = f"{_BOI_BASE}/{code}?startPeriod={start_str}&endPeriod={end_str}&format=json"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            obs = data.get("data", {}).get("dataSets", [{}])[0].get("series", {})
            if not obs:
                log.warning("No data for BoI series %s", code)
                continue
            series_key = list(obs.keys())[0]
            observations = obs[series_key].get("observations", {})
            periods = data["data"]["structure"]["dimensions"]["observation"][0]["values"]
            rows = [
                {"date": pd.to_datetime(p["id"]), key: float(observations.get(str(i), [None])[0] or "nan")}
                for i, p in enumerate(periods)
                if str(i) in observations
            ]
            if rows:
                frames.append(pd.DataFrame(rows).set_index("date"))
        except Exception as exc:
            log.warning("BoI series %s fetch failed: %s", code, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "date"})
    df = df.sort_values("date")
    # Resample to weekly Monday
    df = df.set_index("date").resample("W-MON").last().reset_index()
    return df
