# MMM Framework

Modular Marketing Mix Modeling for Israel (region: IL, currency: ILS).
Built on [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) 0.19+.

## Quickstart

```bash
# Install
uv sync

# Run the bundled example (synthetic IL data, full pipeline)
uv run python examples/basic_pipeline.py

# Or step through the CLI
python -m agent_mmm intake-quick          # writes mmm-workspace/spec.yaml
python -m agent_mmm analyze-data          # audit + collinearity
python -m agent_mmm recommend-controls    # IL holidays + macro controls
python -m agent_mmm recommend-priors      # spend-share-informed Beta/Gamma priors
python -m agent_mmm build                 # assemble pymc-marketing MMM
python -m agent_mmm fit                   # NUTS sampling → idata.nc
python -m agent_mmm diagnose              # rhat, ESS, MAPE, CV-R², overfit gap
python -m agent_mmm attribute             # per-sample ROAS / CPL in ILS
python -m agent_mmm optimize             # budget reallocation with sensitivity
python -m agent_mmm improve              # iterative tournament (tightens priors)
python -m agent_mmm report               # CMO / CFO / MOps / DS reports
```

## Brownfield Warm-start

Re-fit an existing model with priors tightened from a previous run:

```python
# See examples/brownfield_warmstart.py for a full walkthrough
from agent_mmm.spec.schema import Brownfield
spec_v2 = spec_v1.model_copy(update={"brownfield": Brownfield(idata_path="mmm-workspace/idata.nc")})
```

## Python API

```python
from agent_mmm.spec.loader import load_spec
from agent_mmm.data.io import load_panel
from agent_mmm.prior_engine.recommender import recommend_priors
from agent_mmm.fit_runner.runner import run_fit
from agent_mmm.attribute.roas import compute_effectiveness

spec = load_spec("mmm-workspace/spec.yaml")
df   = load_panel(spec)
mc   = recommend_priors(spec, df)
mmm  = run_fit(spec, mc, df)
eff  = compute_effectiveness(mmm, df.drop(columns=[spec.target.column]), spec)
# eff columns: channel, spend_ils, contribution_mean, metric_label, metric_value_mean, ...
```

## Non-revenue KPIs

Set `target.type` to `"acquisitions"` or `"volume"` to get CPL/CPA/CPU instead of ROAS:

```yaml
target:
  column: leads
  type: acquisitions
  unit_name: lead          # drives label: CPL
  value_per_unit: 500.0    # ₪500/lead → implied ROAS column added automatically
```

## Israeli External Controls

`recommend_controls()` layers controls in order of availability:

| Source | Data | Key required |
|---|---|---|
| `holidays.Israel()` | Rosh Hashanah, Yom Kippur, Pesach, Sukkot, … | No |
| Industry preset | Retail/automotive/insurance/telco peak months | No |
| Bank of Israel | Policy rate, ILS/USD | No (public portal; endpoint TBD — see ERRORS.md) |
| CBS Israel | CPI, consumer confidence | No (public API; endpoint TBD) |
| Google Trends IL | Brand/category search `geo="IL"` | No (rate-limited); `SERPAPI_KEY` for paid path |

All external providers fail gracefully — the pipeline runs without any of them.

## Pipeline Overview

```
spec.yaml (single source of truth)
    │
    ▼
load_panel → audit_data → recommend_controls
    │
    ▼
recommend_priors  ←── (brownfield: tighten from previous idata.nc)
    │
    ▼
build_mmm → run_fit → idata.nc
    │
    ├── diagnose  → diagnostics.json / diagnostics_report.md
    ├── attribute → ROAS / CPL / CPA (per-sample, ILS)
    ├── optimize  → budget allocation + sensitivity table
    ├── improve   → tournament → leaderboard.json → tighter priors (loop)
    └── report    → cmo.md / cfo.md / mops.md / ds.md
```

## Project Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout, data flow, design decisions
- [ERRORS.md](ERRORS.md) — known pymc-marketing pitfalls and provider quirks
- [TODO.md](TODO.md) — manual tasks (API keys, accounts)

## Build Status

| Phase | Status | Description |
|---|---|---|
| 0 | Done | Foundations: spec, workspace, CLI, loaders (CSV/Parquet/DataFrame/BigQuery) |
| 1 | Done | Greenfield MVP: data quality, priors, model factory, fit runner, diagnostics |
| 2 | Done | Attribution + CMO/CFO reports (ILS, per-sample ROAS/CPL) |
| 3 | Done | Budget optimization + improvement tournament + leaderboard |
| 4 | Done | Brownfield warm-start + BigQuery loader + BoI/CBS/Trends-IL providers |
| 5 | Done | MOps/DS reports, overfit/prior-pull diagnostics, industry seasonality presets |
