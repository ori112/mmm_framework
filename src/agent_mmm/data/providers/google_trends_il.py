"""Google Trends Israel provider using pytrends (no API key, rate-limited).

Fetches weekly search interest for given keywords in Israel (geo="IL", hl="he").
Gated on pytrends availability and network access.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)


def available() -> bool:
    """Return True if pytrends is installed."""
    try:
        import pytrends  # noqa: F401
        return True
    except ImportError:
        return False


def fetch(
    keywords: list[str],
    start: str | date,
    end: str | date,
    geo: str = "IL",
    hl: str = "he",
    sleep_between: float = 1.0,
) -> pd.DataFrame:
    """Fetch weekly Google Trends data for Israel.

    keywords: list of search terms (max 5 per request due to pytrends limitation).
    Returns DataFrame with columns: date, <keyword>... (indexed weekly on Monday).
    Returns empty DataFrame on failure.
    """
    if not available():
        log.warning("pytrends not installed — Google Trends provider unavailable.")
        return pd.DataFrame()

    try:
        from pytrends.request import TrendReq

        start_str = str(start)[:10]
        end_str = str(end)[:10]
        timeframe = f"{start_str} {end_str}"

        pytrends = TrendReq(hl=hl, tz=120)  # tz=120 → Israel (UTC+2)

        all_frames = []
        # pytrends accepts max 5 keywords per payload
        for i in range(0, len(keywords), 5):
            batch = keywords[i: i + 5]
            try:
                pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()
                if df.empty:
                    continue
                df = df.drop(columns=["isPartial"], errors="ignore")
                df.index = pd.to_datetime(df.index).normalize()
                all_frames.append(df)
                if i + 5 < len(keywords):
                    time.sleep(sleep_between)
            except Exception as exc:
                log.warning("Google Trends batch %s failed: %s", batch, exc)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, axis=1)
        result.index.name = "date"
        return result.reset_index()
    except Exception as exc:
        log.warning("Google Trends fetch failed: %s", exc)
        return pd.DataFrame()
