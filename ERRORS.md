# Known Errors and Pitfalls

Document errors and pymc-marketing quirks as they are encountered.

---

## pymc-marketing API Pitfalls

### Legacy MMM import
**Symptom:** `ImportError: cannot import name 'MMM' from 'pymc_marketing.mmm'`
**Cause:** Pre-0.19 API used a different module path.
**Fix:** Use `from pymc_marketing.mmm.multidimensional import MMM` (0.19+). The flat `pymc_marketing.mmm.MMM` path may refer to an older API variant.

### Prior specification — use `pymc_extras.prior.Prior`
**Symptom:** `TypeError` when passing raw dicts or `pm.Beta(...)` objects as model_config values.
**Cause:** pymc-marketing 0.19+ expects `Prior` objects from `pymc_extras`, not raw PyMC distributions.
**Fix:** `from pymc_extras.prior import Prior; Prior("Beta", alpha=2, beta=5)`.

### `holidays.Israel()` returns Hebrew names by default
**Symptom:** Holiday flag columns are all zeros; `_HOLIDAY_MAP` keys don't match.
**Cause:** `holidays.Israel()` without `language="en_US"` returns keys in Hebrew (e.g. `"ראש השנה"` instead of `"Rosh Hashanah"`).
**Fix:** Always pass `language="en_US"`: `holidays.Israel(years=years, language="en_US")`.
Holiday name mapping: `"Rosh Hashanah"` (not "Rosh Hashana"), `"Pesach"` (not "Passover").

### y-series column naming
**Symptom:** `KeyError` or silent wrong column during fit.
**Cause:** `MMM` requires the target column to match exactly what is passed as `y`.
**Fix:** Always use `spec.target.column` consistently; never rename after `build_mmm()`.

### `model.optimize_budget()` does not exist
**Symptom:** `AttributeError: 'MMM' object has no attribute 'optimize_budget'`
**Cause:** Budget optimization is a separate object, not a method on MMM.
**Fix:** Use `from pymc_marketing.mmm import MultiDimensionalBudgetOptimizerWrapper` and wire up separately in `optimize/budget.py`. Pass `budget_bounds` as a dict `{channel_col: (min, max)}`.

### Scaling — contributions are in scaled space
**Symptom:** ROAS values are orders of magnitude off.
**Cause:** pymc-marketing scales inputs internally. `add_original_scale_contribution_variable()` must be called before reading contributions.
**Fix:** Always call `mmm.add_original_scale_contribution_variable()` after sampling; see `attribute/contributions.py`.

### Brownfield idata coordinate mismatch
**Symptom:** `ValueError: conflicting sizes for dimension` when loading previous idata.nc.
**Cause:** Channel list or date range changed between runs.
**Fix:** Verify `spec.channels` names and date range match the previous run before warm-start. `warmstart.py` should validate this.

### Divergences / bad sampling
**Symptom:** High rhat (> 1.05), many divergences.
**Common causes:**
- Prior too wide (alpha_sigma or lam_sigma > 0.3 for most channels)
- l_max too large relative to data length
- Collinear channels (check VIF)
**Fix:** Run `/mmm-diagnose` → check `diagnostics.json` → tighten priors or reduce l_max.

---

## Israeli Data Provider Issues

### BoI data portal endpoint
**Status:** Endpoint URL not yet confirmed. `data/providers/boi.py` uses a placeholder URL.
**Action needed:** Verify correct endpoint and terms of service at Bank of Israel open-data portal.

### CBS API endpoint
**Status:** Endpoint URL not yet confirmed. `data/providers/cbs.py` uses a placeholder URL.
**Action needed:** Verify correct endpoint at Israel CBS (Central Bureau of Statistics) open API.

### Google Trends rate limiting
**Symptom:** `pytrends` raises `TooManyRequestsError` after multiple pulls.
**Cause:** Unofficial API; Google throttles aggressive requests.
**Fix:** Add delays between calls, use a VPN rotation, or switch to a paid provider (SerpAPI / DataForSEO) for production throughput. The `google_trends_il.py` provider gates on a `SERPAPI_KEY` env var for the paid path.

---

## Environment Issues

*(Append as encountered)*
