"""Select adstock and saturation transforms from channel spec."""
from __future__ import annotations

from pymc_marketing.mmm import (
    DelayedAdstock,
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
)

from ..spec.schema import Channel


def adstock_for(channels: list[Channel]):
    """
    Pick a single adstock type for all channels (pymc-marketing limitation).
    Uses DelayedAdstock if any channel prefers it; GeometricAdstock otherwise.
    l_max is the maximum across all channels.
    """
    l_max = max(ch.l_max for ch in channels)
    use_delayed = any(ch.adstock == "delayed" for ch in channels)
    return DelayedAdstock(l_max=l_max) if use_delayed else GeometricAdstock(l_max=l_max)


def saturation_for(channels: list[Channel]):
    """
    Pick a single saturation type for all channels.
    Uses HillSaturation if any channel prefers it; LogisticSaturation otherwise.
    """
    use_hill = any(ch.saturation == "hill" for ch in channels)
    return HillSaturation() if use_hill else LogisticSaturation()
