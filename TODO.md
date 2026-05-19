# TODO — Manual Tasks

Actions that require the user to take steps outside the codebase (API keys, accounts, etc.).

---

## Required for External Control Variables

### Bank of Israel (BoI) API
- Endpoint: https://www.boi.org.il/en/economic-roles/financial-markets/exchange-rates/
- Data needed: policy rate, ILS/USD, ILS/EUR daily fixings, M1/M2, inflation expectations
- Action: Confirm open-data endpoint URL and terms of use. Set `BOI_API_URL` in `.env`.

### CBS (Central Bureau of Statistics, Israel)
- Website: https://www.cbs.gov.il
- Data needed: CPI (מדד המחירים לצרכן), unemployment, consumer confidence, retail trade index
- Action: Obtain API access if required; confirm endpoint format. Set `CBS_API_KEY` in `.env` if needed.

### Google Trends — Israel
- Implementation: `pytrends` (no key required, rate-limited)
- Alternative: SerpAPI or DataForSEO for higher throughput / brand + competitor queries
- Action: If rate limits are hit, obtain SerpAPI key and set `SERPAPI_KEY` in `.env`.

---

## Required for BigQuery Loader (Phase 4)

- GCP project with BigQuery access
- Service account JSON or run `gcloud auth application-default login`
- Action: Create GCP project, set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json` in `.env`

---

## Optional

### FX Backup API
- Only needed if BoI rates are insufficient for non-ILS report currency conversion
- Candidates: Open Exchange Rates, Fixer.io
- Action: If needed, obtain key and set `FX_API_KEY` in `.env`

---

## GitHub Setup

- [ ] Confirm private GitHub repo is set up
- [ ] Verify `.gitignore` excludes `mmm-workspace/`, `.env`, `*.nc`
- [ ] Add branch protection on `main`

---

## Deferred Features (future plans)

- HillSaturation variant in tournament variants grid
- Hebrew-language section titles in CMO/CFO reports
- Hierarchical priors for future multi-geo extension
- Automated structural-break detection integration with controls recommendation
