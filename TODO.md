# TODO — Manual Tasks

Actions that require steps outside the codebase (API keys, accounts, infrastructure).

---

## 🔴 Blocking — External Data Providers

### Bank of Israel (BoI) API
- **Problem:** `data/providers/boi.py` uses a placeholder URL that currently returns 404 (see `ERRORS.md`).
- **Data needed:** Policy rate, ILS/USD daily fixing, ILS/EUR, M1/M2, inflation expectations.
- **Action:**
  1. Find the correct open-data endpoint at https://www.boi.org.il/en/economic-roles/financial-markets/
  2. Confirm terms of use (public portal, no key typically needed).
  3. Update the `BASE_URL` and series codes in `src/agent_mmm/data/providers/boi.py`.
  4. Set `BOI_API_URL` in `.env` if endpoint is configurable.

### CBS (Central Bureau of Statistics, Israel)
- **Problem:** `data/providers/cbs.py` uses a placeholder URL.
- **Data needed:** CPI, unemployment, consumer confidence, retail trade index.
- **Action:**
  1. Check https://www.cbs.gov.il/en for open API access.
  2. Confirm endpoint format and authentication (public or key-gated).
  3. Update `src/agent_mmm/data/providers/cbs.py`.
  4. Set `CBS_API_KEY` in `.env` if a key is required.

---

## 🟡 Optional — Performance & Scale

### C++ compiler for PyTensor (Windows)
- **Problem:** Without `g++`, MCMC sampling runs in pure Python mode — 5–10× slower.
- **Action:** Install MinGW-w64 via MSYS2 and add `C:\msys64\ucrt64\bin` to PATH (see `ERRORS.md`).

### Google Trends — higher throughput
- **Problem:** `pytrends` (no key, rate-limited) is fine for occasional pulls but fails under load.
- **Action:** If rate limits are hit, obtain a SerpAPI key and set `SERPAPI_KEY` in `.env`. `google_trends_il.py` uses it automatically.

### BigQuery loader
- **Status:** Implemented in `data/loaders/bigquery.py`. Requires GCP credentials.
- **Action:**
  1. Create a GCP project with BigQuery access.
  2. Either `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json` in `.env`.
  3. Set `spec.data.source = "bigquery"` and configure `project`, `dataset`, `table` in `DataCfg`.

---

## 🟢 Infrastructure

### GitHub repository
- [x] Repo created at https://github.com/ori112/mmm_framework
- [x] `.gitignore` excludes `mmm-workspace/`, `.env`, `*.nc`
- [ ] Add branch protection on `main` (require PR + review before merge)
- [ ] Set up GitHub Actions CI to run `uv run pytest tests/unit/` on every push

---

## 💡 Deferred Features

| Feature | Notes |
|---|---|
| `HillSaturation` in tournament variant grid | Currently only `LogisticSaturation`; Hill is implemented in transforms.py but not in `iter_loop/variants.py` |
| Hebrew-language section titles | CMO/CFO reports could have Hebrew headings; needs `spec.report_language` field |
| Prior predictive check in `basic_pipeline.py` | `run_prior_predictive()` exists but is not called in the example; requires calling before `run_fit()` |
| Structural-break detection wired to controls | `data/collinearity.py` has `structural_break()` but it's not surfaced in `recommend_controls()` |
| FX backup API | Only needed if BoI rates unavailable and non-ILS report currency required |
| Hierarchical priors for multi-geo | Out of scope for v1; would require the `mmm-multi-geo-panel` skill |
