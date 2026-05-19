# MMM Framework

Modular Marketing Mix Modeling for Israel (region: IL, currency: ILS).
Built on [pymc-marketing](https://github.com/pymc-labs/pymc-marketing).

## Quickstart

```bash
# Install
uv sync

# Run intake (creates mmm-workspace/spec.yaml)
python -m agent_mmm intake-quick

# Full pipeline (once Phase 1+ is built)
python -m agent_mmm analyze-data
python -m agent_mmm recommend-priors
python -m agent_mmm build
python -m agent_mmm fit
python -m agent_mmm diagnose
python -m agent_mmm attribute
python -m agent_mmm report
```

## Pipeline Overview

```
intake → analyze-data → recommend-controls → recommend-priors
       → build → fit → diagnose → attribute → optimize → improve → report
```

- **spec.yaml** is the single source of truth (region: IL, currency: ILS by default).
- All artifacts land in `mmm-workspace/` (gitignored).
- Israeli external controls: `holidays.Israel()`, Bank of Israel, CBS, Google Trends `geo="IL"`.

## Python API

```python
from agent_mmm import load_spec, load_panel

spec = load_spec("mmm-workspace/spec.yaml")
df   = load_panel(spec)
```

## Project Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout, data flow, design decisions
- [ERRORS.md](ERRORS.md) — known pymc-marketing pitfalls and provider quirks
- [TODO.md](TODO.md) — manual tasks (API keys, accounts)

## Build Status

| Phase | Status | Description |
|---|---|---|
| 0 | ✓ Done | Foundations: spec, workspace, CLI, loaders |
| 1 | Pending | Greenfield MVP: data quality, priors, model, fit, diagnose |
| 2 | Pending | Attribution + CMO/CFO reports |
| 3 | Pending | Budget optimization + improvement tournament |
| 4 | Pending | Brownfield + BigQuery + IL macro/search providers |
| 5 | Pending | MOps/DS reports, diagnostics polish, industry presets |
