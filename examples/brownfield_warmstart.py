"""Brownfield warm-start: re-fit with tightened priors from a previous run.

Demonstrates:
  1. Greenfield fit → saves idata.nc to mmm-workspace/
  2. Configure spec.brownfield to point at that idata.nc
  3. Re-fit: prior_engine.posterior_informed tightens priors automatically
  4. Compare posterior widths before vs after warm-start

Output artifacts go to mmm-workspace/ in the current directory.

Usage:
    uv run python examples/brownfield_warmstart.py
"""
from __future__ import annotations

from pathlib import Path

from agent_mmm.spec.schema import Brownfield, Channel, DataCfg, Spec, SamplerCfg, TargetUnit
from agent_mmm.spec.loader import save_spec
from agent_mmm.workspace.paths import ensure_workspace, idata_path, spec_path

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic"

_CHANNELS = [
    Channel(name="google",   spend_col="google_spend",   adstock="geometric", l_max=5),
    Channel(name="facebook", spend_col="facebook_spend", adstock="geometric", l_max=8),
    Channel(name="tv",       spend_col="tv_spend",       adstock="geometric", l_max=13),
]
_FAST = SamplerCfg(draws=150, tune=150, chains=2, target_accept=0.9, random_seed=42)

from agent_mmm.data.io import load_panel
from agent_mmm.prior_engine.recommender import recommend_priors
from agent_mmm.fit_runner.runner import run_fit

ensure_workspace()

# ── Pass 1: Greenfield ────────────────────────────────────────────────────────
print("=== Pass 1: Greenfield fit ===")

spec_green = Spec(
    company="Brownfield Example",
    data=DataCfg(source="csv", path=str(FIXTURES / "data.csv"), date_col="date"),
    target=TargetUnit(column="revenue"),
    channels=_CHANNELS,
    sampler=_FAST,
)
save_spec(spec_green, spec_path())

df = load_panel(spec_green)
mc_green = recommend_priors(spec_green, df)
mmm_green = run_fit(spec_green, mc_green, df, skip_prior_check=True)

idata_nc = idata_path()
print(f"  idata saved: {idata_nc}\n")

import arviz as az
summary_green = az.summary(mmm_green.idata, var_names=["adstock_alpha", "saturation_lam"])
print("Greenfield posterior std:")
print(summary_green[["sd"]].to_string())

# ── Pass 2: Brownfield warm-start ─────────────────────────────────────────────
print("\n=== Pass 2: Brownfield warm-start ===")

spec_brown = spec_green.model_copy(update={
    "brownfield": Brownfield(idata_path=str(idata_nc)),
})
save_spec(spec_brown, spec_path())

from agent_mmm.fit_runner.warmstart import load_warmstart_config

mc_base = recommend_priors(spec_brown, df)
mc_tight = load_warmstart_config(spec_brown, mc_base)
mmm_brown = run_fit(spec_brown, mc_tight, df, skip_prior_check=True)

summary_brown = az.summary(mmm_brown.idata, var_names=["adstock_alpha", "saturation_lam"])
print("\nBrownfield posterior std:")
print(summary_brown[["sd"]].to_string())

# ── Compare ───────────────────────────────────────────────────────────────────
print("\n=== Width reduction (greenfield → brownfield) ===")
for idx in summary_green.index:
    g = summary_green.loc[idx, "sd"]
    b = summary_brown.loc[idx, "sd"] if idx in summary_brown.index else float("nan")
    pct = (1 - b / g) * 100 if g > 0 else 0
    print(f"  {idx:40s}  {g:.4f} → {b:.4f}  ({pct:+.1f}%)")

print("\nExpect brownfield posterior to be narrower (tighter priors from warm-start).")
print(f"Artifacts in: {idata_path().parent}/")
