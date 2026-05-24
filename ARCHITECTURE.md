# Architecture

Modular Marketing Mix Modeling framework for **Israel** (region: IL, currency: ILS).
Single-series only — no multi-geo / panel modeling.

## Data Flow

```
spec.yaml  (single source of truth — pydantic Spec model)
    │
    ▼
data/io.load_panel()                ← CSV / Parquet / DataFrame / BigQuery
    │
    ▼
data/quality.audit_data()           ← 52+ wk, completeness, zero-streaks, outliers
data/collinearity.vif()             ← VIF, pairwise correlation, structural break
data/controls.recommend_controls()  ← IL holidays + industry preset + BoI + CBS + Trends-IL
    │
    ▼
prior_engine/recommender.recommend_priors()       ← spend-share-informed Beta / Gamma
    │  (brownfield path: posterior_informed instead ↑)
    ▼
model_factory/builder.build_mmm()   ← Geometric/Delayed adstock + Logistic/Hill saturation
                                        + Fourier seasonality + IL calendar dummies
    │
    ▼
fit_runner/runner.run_fit()         ← NUTS sampling → idata.nc
    │
    ├── fit_runner/checks.py        ← prior predictive / posterior predictive
    │
    ├── diagnose/report.write_diagnostics()    → diagnostics.json + diagnostics.md
    │       ├── convergence: rhat, ESS, divergences, BFMI
    │       ├── fit_metrics: R², MAPE, wMAPE, CV-R²
    │       ├── overfit: in_sample_r2 − cv_r2 gap
    │       └── prior_pull: posterior_std / prior_std per param
    │
    ├── attribute/contributions     ← per-sample original-scale channel contributions
    ├── attribute/roas              ← ROAS / CPL / CPA / CPU dispatcher (per-sample, ILS)
    ├── attribute/curves            ← saturation curves, saturation knee
    │
    ├── optimize/budget             ← MultiDimensionalBudgetOptimizerWrapper
    │   └── optimize/sensitivity    ← ±10/20/30% sweep, elasticity table
    │
    └── iter_loop/tournament        ← variant grid → composite score → leaderboard
            └── loops back to prior_engine (posterior-informed prior tightening)
    │
    ▼
report/render_all()  → mmm-workspace/reports/{cmo,cfo,mops,ds}.md
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
│   ├── loaders/       csv.py, parquet.py, dataframe.py, bigquery.py
│   ├── quality.py     audit_data() — tier PASS / WARN / FAIL
│   ├── collinearity.py vif(), pairwise_corr(), structural_break()
│   ├── controls.py    recommend_controls() — holidays + industry preset + macro
│   └── providers/     IL data providers (all gated on env keys / availability)
│       ├── holidays_il.py    holidays.Israel(language="en_US") — no key required
│       ├── boi.py            Bank of Israel: policy rate, ILS/USD — placeholder URL
│       ├── cbs.py            CBS: CPI, unemployment, consumer confidence — placeholder URL
│       └── google_trends_il.py  pytrends geo="IL" / SERPAPI_KEY for paid path
│
├── prior_engine/      Prior recommendation and posterior-informed tightening
│   ├── moments.py     beta_moment_match, gamma_moment_match, normal_moment_match
│   ├── recommender.py recommend_priors(spec, df) — spend-share-informed
│   └── posterior_informed.py tighten_priors_from_idata(idata, model_config, factor=0.7)
│
├── model_factory/     pymc-marketing MMM construction
│   ├── builder.py     build_mmm(spec, model_config) → MMM instance
│   ├── transforms.py  adstock_for(channel), saturation_for(channel)
│   └── controls_block.py Fourier modes + IL calendar dummies + macro regressors
│
├── fit_runner/        Sampling and predictive checks
│   ├── runner.py      run_fit(spec, model_config, df) → fitted MMM + idata.nc
│   ├── checks.py      run_prior_predictive(), run_posterior_predictive()
│   ├── warmstart.py   brownfield: load_warmstart_config() from previous idata.nc
│   └── sampler.py     NUTS kwargs builder from SamplerCfg
│
├── diagnose/          Convergence and fit diagnostics
│   ├── convergence.py check_convergence() — rhat, ESS, divergences, BFMI
│   ├── fit_metrics.py compute_in_sample_metrics() — R², MAPE, wMAPE, CV-R²
│   ├── overfit.py     compute_overfit_gap(), overfit_tier()
│   ├── prior_pull.py  audit_prior_pull() — posterior_std / prior_std ratios
│   └── report.py      write_diagnostics() → diagnostics.json + diagnostics.md
│
├── attribute/         Channel attribution and effectiveness
│   ├── contributions.py get_contributions() — per-sample original-scale, 89% HDI
│   ├── roas.py          compute_effectiveness() — ROAS / CPL / CPA / CPU dispatcher
│   └── curves.py        get_saturation_curves(), saturation_point()
│
├── optimize/          Budget optimization
│   ├── budget.py      MultiDimensionalBudgetOptimizerWrapper wrapper
│   ├── bounds.py      channel_bounds(spec) — min/max from spec
│   └── sensitivity.py budget_sensitivity() — ±10/20/30% sweep + elasticity
│
├── iter_loop/         Iterative improvement (tournament)
│   ├── tournament.py  run_tournament(spec, n_variants, max_rounds, patience)
│   ├── variants.py    generate_variants() — cartesian over l_max × fourier × prior_width
│   ├── scoring.py     composite_score() — cv_r² × overfit_penalty × convergence × plausibility
│   └── leaderboard.py read/append/sort leaderboard.json
│
├── report/            Stakeholder reports (all ILS-denominated)
│   ├── cmo.py         CMO narrative — contributions, ROAS/CPL, top channel, recommendation
│   ├── cfo.py         CFO ROI — ILS attribution table, implied ROAS for non-revenue KPIs
│   ├── mops.py        Marketing Ops — saturation status, budget reallocation table
│   ├── ds.py          Data Science — convergence, predictive checks, adstock decay,
│   │                               saturation params, posterior summary, prior pull
│   ├── render_all.py  render_all() — computes attribution then calls all four reports
│   └── renderer.py    save_report(), fmt_currency_ils(), fmt_hdi()
│
├── workspace/         Artifact contract
│   ├── paths.py       Canonical paths (spec.yaml, idata.nc, diagnostics.json, ...)
│   └── artifacts.py   save/load InferenceData, JSON, frozen spec_used.yaml
│
└── cli/               CLI shims
    ├── commands.py    One thin function per /mmm-* slash command
    └── __main__.py    Dispatcher (python -m agent_mmm <cmd>)
```

## Key Design Decisions

- **`spec.yaml` is the only stateful input.** All modules accept a validated `Spec` pydantic model. `freeze_spec()` saves `spec_used.yaml` alongside `idata.nc` for reproducibility. Default `region: "IL"`, `currency: "ILS"`.
- **Single-series only.** `model_factory.builder` builds a scalar-target MMM with no `geo` dimension. The `mmm-multi-geo-panel` skill is intentionally out of scope.
- **Loader registry.** `data/io.py` dispatches on `spec.data.source`. New backends: drop a file in `data/loaders/` and call `register_loader()` — no core changes.
- **Per-sample metrics.** E[f(x)] ≠ f(E[x]). `attribute/` and `diagnose/fit_metrics` always compute over posterior samples, never over posterior means.
- **Greenfield vs brownfield** is a single branch in `fit_runner.runner`: `spec.brownfield` set → `posterior_informed`; unset → `recommender`. All downstream code is identical.
- **Israeli localization.** Default `region: "IL"`, `currency: "ILS"`. External controls use BoI, CBS, and Google Trends `geo="IL"` providers. Holidays use `holidays.Israel(language="en_US")` — the `language` kwarg is required; the default is Hebrew.
- **Non-revenue KPIs.** `spec.target.type` drives the effectiveness metric: `revenue` → ROAS, `acquisitions`/`volume` → CPL/CPA/CPU. `unit_name` on `TargetUnit` customises the label (e.g. `"lead"` → CPL). `value_per_unit > 1` adds an implied-ROAS column to CFO/CMO reports.
- **DS report.** `report/ds.py` receives the fitted `MMM` object and reads directly from `idata.posterior` / `idata.posterior_predictive`. Sections: convergence, fit metrics, overfit, prior/posterior predictive checks, adstock decay table with half-lives, saturation parameter table, full posterior summary, prior pull ratios.

## Artifacts (`mmm-workspace/`)

| File | Written by | Content |
|---|---|---|
| `spec.yaml` | intake CLI / user | Editable model spec |
| `spec_used.yaml` | `fit_runner` (freeze_spec) | Frozen copy at fit time |
| `idata.nc` | `fit_runner` | Full ArviZ InferenceData (posterior + predictive) |
| `diagnostics.json` | `diagnose/report` | Convergence + fit metrics JSON |
| `leaderboard.json` | `iter_loop` | Tournament run history |
| `reports/cmo.md` | `report/cmo` | CMO narrative |
| `reports/cfo.md` | `report/cfo` | CFO ROI table |
| `reports/mops.md` | `report/mops` | Marketing Ops saturation + budget |
| `reports/ds.md` | `report/ds` | Full DS diagnostics |

## Skill → Module Mapping

| Slash command | Module entrypoint | Phase |
|---|---|---|
| `/mmm-intake[-quick]` | `spec.intake.build_spec_from_answers` | 0 |
| `/mmm-analyze-data` | `data.quality.audit_data` + `data.collinearity` | 1 |
| `/mmm-recommend-controls` | `data.controls.recommend_controls` | 1 |
| `/mmm-recommend-priors` | `prior_engine.recommender.recommend_priors` | 1 |
| `/mmm-build` | `model_factory.builder.build_mmm` | 1 |
| `/mmm-fit` | `fit_runner.runner.run_fit` | 1 |
| `/mmm-diagnose` | `diagnose.report.write_diagnostics` | 1 |
| `/mmm-attribute` | `attribute.contributions` + `attribute.roas` | 2 |
| `/mmm-report` | `report.render_all.render_all` | 2 |
| `/mmm-optimize` | `optimize.budget` | 3 |
| `/mmm-improve` | `iter_loop.tournament.run_tournament` | 3 |

`mmm-multi-geo-panel` skill is **intentionally out of scope** — single-series framework only.

## Build Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: spec, workspace, CLI, loaders (CSV/Parquet/DataFrame/BigQuery) | ✅ done |
| 1 | Greenfield MVP: data quality, prior_engine, model_factory, fit_runner, diagnose | ✅ done |
| 2 | Attribution + CMO/CFO reports (ILS, per-sample ROAS/CPL) | ✅ done |
| 3 | Budget optimization + iterative improvement tournament | ✅ done |
| 4 | Brownfield warm-start + BigQuery + IL macro/search providers | ✅ done |
| 5 | MOps/DS reports, overfit/prior-pull diagnostics, industry seasonality presets | ✅ done |
