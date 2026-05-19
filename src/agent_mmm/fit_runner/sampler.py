"""Convert SamplerCfg to PyMC sampling kwargs."""
from __future__ import annotations

from ..spec.schema import SamplerCfg


def get_sampler_kwargs(cfg: SamplerCfg) -> dict:
    kwargs: dict = {
        "draws": cfg.draws,
        "tune": cfg.tune,
        "chains": cfg.chains,
        "target_accept": cfg.target_accept,
    }
    if cfg.random_seed is not None:
        kwargs["random_seed"] = cfg.random_seed
    return kwargs
