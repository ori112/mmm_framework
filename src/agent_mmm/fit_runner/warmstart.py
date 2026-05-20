"""Brownfield warm-start: tighten priors from an existing idata.nc."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_warmstart_config(spec, base_model_config: dict) -> dict:
    """If spec.brownfield is set, load the existing idata and tighten priors.

    Returns a (possibly tightened) model_config dict ready for build_mmm.
    Falls back to base_model_config if the idata path is missing or unreadable.
    """
    if spec.brownfield is None:
        return base_model_config

    idata_path = Path(spec.brownfield.idata_path)
    if not idata_path.exists():
        log.warning(
            "Brownfield idata_path %s does not exist — falling back to greenfield priors.",
            idata_path,
        )
        return base_model_config

    try:
        import arviz as az
        from agent_mmm.prior_engine.posterior_informed import tighten_priors_from_idata

        log.info("Loading brownfield idata from %s", idata_path)
        idata = az.from_netcdf(str(idata_path))
        tight_config = tighten_priors_from_idata(idata, base_model_config)
        log.info("Priors tightened from brownfield posterior.")
        return tight_config
    except Exception as exc:
        log.warning("Brownfield warm-start failed (%s) — using greenfield priors.", exc)
        return base_model_config
