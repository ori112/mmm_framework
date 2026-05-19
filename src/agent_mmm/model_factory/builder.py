"""Build pymc-marketing MMM instance from spec and model_config."""
from __future__ import annotations

import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM

from ..spec.schema import Spec
from .controls_block import build_controls
from .transforms import adstock_for, saturation_for


def build_mmm(
    spec: Spec,
    model_config: dict,
    df: pd.DataFrame | None = None,
) -> tuple[MMM, pd.DataFrame, list[str]]:
    """
    Construct an MMM instance.

    Args:
        spec: Validated Spec.
        model_config: Prior dict from prior_engine.recommender.
        df: Full data DataFrame (used to add holiday controls). May include target column.

    Returns:
        (mmm, df_with_controls, control_col_names)
        df_with_controls still contains the target column — caller separates X and y.
    """
    channel_cols = [ch.spend_col for ch in spec.channels]

    if df is not None:
        df, control_cols = build_controls(df, spec)
    else:
        control_cols = list(spec.controls)

    adstock = adstock_for(spec.channels)
    saturation = saturation_for(spec.channels)

    mmm = MMM(
        date_column=spec.data.date_col,
        channel_columns=channel_cols,
        target_column=spec.target.column,
        adstock=adstock,
        saturation=saturation,
        control_columns=control_cols if control_cols else None,
        yearly_seasonality=spec.fourier_order,
        model_config=model_config,
        adstock_first=True,
    )

    return mmm, df if df is not None else pd.DataFrame(), control_cols
