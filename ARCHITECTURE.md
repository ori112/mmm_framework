# Architecture

Modular Marketing Mix Modeling framework for **Israel** (region: IL, currency: ILS).
Single-series only — no multi-geo / panel modeling.

## Data Flow

```
spec.yaml
    │
    ▼
data/io.load_panel()              ← CSV / Parquet / DataFrame / BigQuery
    │
    ▼
data/quality.audit_data()         ← 52+ wk check, completeness, zero-streaks, VIF
data/controls.recommend_controls() ← IL holidays, BoI, CBS, Google Trends IL
    │
    ▼
prior_engine/recommender.recommend_priors()   ← moment-matched Beta / Gamma
    │ (brownfield: posterior_informed instead)
    ▼
model_factory/builder.build_mmm()  ← GeometricAdstock / DelayedAdstock + LogisticSaturation
    │
    ▼
fit_runner/runner.run_fit()        ← prior PC → NUTS → posterior PC
    │
    ├──▶ diagnose/report.write_diagnostics()    → mmm-workspace/diagnostics.json
    ├──▶ attribute/contributions + roas + curves
    ├──▶ optimize/budget.optimize()
    └──▶ iter_loop/tournament.run_tournament()  ← loops back to prior_engine
                                                   (posterior-informed priors)
    │
    ▼
report/renderer.render_all()  → mmm-workspace/reports/{cmo,cfo,mops,ds}.md
```

## Module Layout

```
src/agent_mmm/
├── spec/              spec.yaml parsing and interactive intake
│   ├── schema.py      Pydantic models: Spec, Channel, TargetUnit, SamplerCfg, Brownfield
│   ├── loader.py      load_spec / save_spec / freeze_spec
│   └── intake.py      Interactive CLI questionnaire → spec.yaml
│
├── data/              Data I/O and validation
│   ├── io.py          load_panel() dispatcher (pluggable loader registry)
│   ├── loaders/       csv.py, parquet.py, dataframe.py, bigquery.py (Phase 4)
│   ├── quality.py     audit_data() — Phase 1
│   ├── collinearity.py vif, structural_break — Phase 1
│   ├── controls.py    recommend_controls() — Phase 1/5
│   └── providers/     IL data providers (all gated on env keys)
│       ├── holidays_il.py   holidays.Israel() — no key required
│       ├── boi.py           Bank of Israel: rate, ILS/USD, M1 — Phase 4
│       ├── cbs.py           CBS: CPI, unemployment, consumer confidence — Phase 4
│       └── google_trends_il.py  pytrends geo="IL" — Phase 4
│
├── prior_engine/      Prior recommendation and posterior-informed tightening
│   ├── moments.py     beta_moment_match, gamma_moment_match — Phase 1
│   ├── recommender.py recommend_priors(spec) — Phase 1
│   └── posterior_informed.py tighten_priors_from_idata() — Phase 3
│
├── model_factory/     pymc-marketing MMM construction
│   ├── builder.py     build_mmm(spec, priors) — Phase 1
│   ├── transforms.py  adstock_for(), saturation_for() — Phase 1
│   └── controls_block.py Fourier + IL calendar dummies — Phase 1
│
├── fit_runner/        Sampling and predictive checks
│   ├── runner.py      run_fit(spec, priors, existing_idata=None) — Phase 1
│   ├── checks.py      prior_predictive, posterior_predictive — Phase 1
│   ├── warmstart.py   brownfield: seed from previous idata.nc — Phase 4
│   └── sampler.py     NUTS config builder — Phase 1
│
├── diagnose/          Convergence and fit diagnostics
│   ├── convergence.py rhat, ESS, divergences, BFMI — Phase 1
│   ├── fit_metrics.py R², MAPE, WAIC, cv_r2 — Phase 1
│   ├── overfit.py     overfit_gap = in_sample_r2 - cv_r2 — Phase 5
│   ├── prior_pull.py  posterior vs prior moment shift — Phase 5
│   └── report.py      write diagnostics.json — Phase 1
│
├── attribute/         Channel attribution and ROAS
│   ├── contributions.py per-sample original-scale contributions — Phase 2
│   ├── roas.py          ROAS / CPA per channel (per-sample) — Phase 2
│   └── curves.py        response / saturation curves — Phase 2
│
├── optimize/          Budget optimization
│   ├── budget.py      BudgetOptimizer wrapper — Phase 3
│   ├── bounds.py      min/max per channel — Phase 3
│   └── sensitivity.py ±10/20/30% sweep, elasticity table — Phase 3
│
├── iter_loop/         Iterative improvement (tournament)
│   ├── tournament.py  run_tournament(spec, ...) — Phase 3
│   ├── variants.py    cartesian over (l_max, fourier_modes, prior_width) — Phase 3
│   ├── scoring.py     composite score function — Phase 3
│   └── leaderboard.py leaderboard.json I/O — Phase 3
│
├── report/            Stakeholder reports
│   ├── cmo.py         CMO narrative — Phase 2
│   ├── cfo.py         CFO ROI in ILS — Phase 2
│   ├── mops.py        Marketing Ops saturation + budget — Phase 5
│   ├── ds.py          Data Science diagnostics — Phase 5
│   └── renderer.py    Markdown templates, HDI formatting — Phase 2
│
├── workspace/         Artifact contract
│   ├── paths.py       Canonical paths (spec.yaml, idata.nc, diagnostics.json, ...)
│   └── artifacts.py   save/load InferenceData, JSON, markdown
│
└── cli/               CLI shims
    ├── commands.py    One function per /mmm-* skill
    └── __main__.py    Dispatcher (python -m agent_mmm <cmd>)
```

## Key Design Decisions

- **spec.yaml is the only stateful input.** All modules accept a validated `Spec` pydantic model. `freeze_spec()` saves `spec_used.yaml` alongside `idata.nc` for reproducibility.
- **Single-series only.** `model_factory.builder` builds a scalar-target MMM. The `mmm-multi-geo-panel` skill is intentionally unused.
- **Loader registry.** `data/io.py` dispatches on `spec.data.source`. New backends are added by dropping a file in `data/loaders/` and calling `register_loader()`.
- **Per-sample metrics.** E[f(x)] ≠ f(E[x]). `attribute/` and `diagnose/fit_metrics` always compute on posterior samples, never on posterior means.
- **Greenfield vs brownfield** is a single branch: `spec.brownfield` set → `prior_engine.posterior_informed`; unset → `prior_engine.recommender`. All downstream code is identical.
- **Israeli localization.** Default `region: "IL"`, `currency: "ILS"`. External controls use BoI, CBS, and Google Trends `geo="IL"` providers. Holidays use `holidays.Israel()`.

## Artifacts (mmm-workspace/)

| File | Written by | Read by |
|---|---|---|
| `spec.yaml` | intake CLI / user | all modules |
| `spec_used.yaml` | fit_runner (freeze_spec) | audit/reproducibility |
| `idata.nc` | fit_runner | diagnose, attribute, optimize, improve, report |
| `diagnostics.json` | diagnose | report/ds |
| `leaderboard.json` | iter_loop | report/ds |
| `reports/*.md` | report | stakeholders |

## Skill → Module Mapping

| Slash command | Module | Phase |
|---|---|---|
| `/mmm-intake[-quick]` | `spec.intake` | 0 |
| `/mmm-analyze-data` | `data.quality` + `data.collinearity` | 1 |
| `/mmm-recommend-controls` | `data.controls` | 1 |
| `/mmm-recommend-priors` | `prior_engine.recommender` | 1 |
| `/mmm-build` | `model_factory.builder` | 1 |
| `/mmm-fit` | `fit_runner.runner` | 1 |
| `/mmm-diagnose` | `diagnose.report` | 1 |
| `/mmm-attribute` | `attribute.*` | 2 |
| `/mmm-report` | `report.renderer` | 2 |
| `/mmm-optimize` | `optimize.budget` | 3 |
| `/mmm-improve` | `iter_loop.tournament` | 3 |

Note: `mmm-multi-geo-panel` skill is intentionally out of scope (single-series framework).

## Build Phases

| Phase | Scope |
|---|---|
| 0 (done) | Foundations: spec, workspace, CLI, loaders |
| 1 | Greenfield MVP: data quality, prior_engine, model_factory, fit_runner, diagnose |
| 2 | Attribution + CMO/CFO reports |
| 3 | Budget optimization + iterative improvement tournament |
| 4 | Brownfield warm-start + BigQuery + IL macro/search providers |
| 5 | MOps/DS reports, overfit/prior-pull diagnostics, industry seasonality presets |
