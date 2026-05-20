from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Channel(BaseModel):
    name: str
    spend_col: str
    channel_type: Literal["digital", "offline"] = "digital"
    adstock: Literal["geometric", "delayed"] = "geometric"
    saturation: Literal["logistic", "hill"] = "logistic"
    l_max: int = 8
    # Prior hyperparameters — populated by prior_engine, overrideable
    alpha_mean: float = 0.5
    alpha_sigma: float = 0.15
    lam_mean: float = 0.5
    lam_sigma: float = 0.15
    beta_sigma: float | None = None  # None → spend-share derived in prior_engine


class TargetUnit(BaseModel):
    column: str
    type: Literal["revenue", "acquisitions", "volume"] = "revenue"
    # Human-readable unit name for non-revenue targets: "lead", "policy", "install", "visit", etc.
    # Drives the cost-per-X metric label: "lead" → CPL, "policy" → CPP, None → CPA
    unit_name: str | None = None
    # Monetary value per unit — enables implied ROAS on non-revenue targets
    # e.g. value_per_unit=250 on leads → implied ROAS = 250 / CPL
    value_per_unit: float = 1.0


class SamplerCfg(BaseModel):
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    random_seed: int | None = None


class Brownfield(BaseModel):
    idata_path: str  # path to a previous idata.nc for warm-start


class DataCfg(BaseModel):
    source: Literal["csv", "parquet", "dataframe", "bigquery"] = "csv"
    path: str | None = None  # not required for source="dataframe"
    date_col: str = "date"


class Spec(BaseModel):
    company: str = "Unknown"
    industry: Literal["retail", "automotive", "insurance", "telco", "saas", "other"] = "other"
    region: str = "IL"
    currency: str = "ILS"
    date_created: str = Field(default_factory=lambda: date.today().isoformat())
    data: DataCfg = Field(default_factory=DataCfg)
    target: TargetUnit
    channels: list[Channel]
    controls: list[str] = Field(default_factory=list)
    fourier_order: int = 2
    sampler: SamplerCfg = Field(default_factory=SamplerCfg)
    brownfield: Brownfield | None = None
