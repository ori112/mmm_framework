# Known Errors and Pitfalls

Append errors and quirks as encountered. Do not remove resolved entries — note the fix.

---

## pymc-marketing API

### Wrong MMM import path
**Symptom:** `ImportError: cannot import name 'MMM' from 'pymc_marketing.mmm'`
**Cause:** Pre-0.19 API used a different module path.
**Fix:** `from pymc_marketing.mmm.multidimensional import MMM` (0.19+).

### Prior specification — must use `pymc_extras.prior.Prior`
**Symptom:** `TypeError` when passing raw dicts or `pm.Beta(...)` objects as model_config values.
**Cause:** pymc-marketing 0.19+ expects `Prior` objects from `pymc_extras`, not raw PyMC distributions.
**Fix:** `from pymc_extras.prior import Prior; Prior("Beta", alpha=2, beta=5)`.

### `UserWarning: Implicit conversion of array-like parameter`
**Symptom:** Repeated warnings like `Implicit conversion of array-like parameter alpha to DataArray with dims ('channel',)`.
**Cause:** Passing plain Python arrays as Prior parameter values; pymc_extras auto-wraps them but warns.
**Status:** Harmless. Suppress by passing `xr.DataArray` with explicit dims, or ignore — does not affect results.

### y-series column naming
**Symptom:** `KeyError` or silent wrong column during fit.
**Cause:** `MMM` requires the target column name to match `y.name` exactly.
**Fix:** Always use `spec.target.column` consistently. Never rename `y` after `build_mmm()`.

### `model.optimize_budget()` does not exist
**Symptom:** `AttributeError: 'MMM' object has no attribute 'optimize_budget'`
**Cause:** Budget optimization is a separate wrapper class, not a method on MMM.
**Fix:** Use `from pymc_marketing.mmm import MultiDimensionalBudgetOptimizerWrapper`. Pass `budget_bounds` as `{channel_col: (min, max)}`. See `optimize/budget.py`.

### Scaling — contributions are in scaled space by default
**Symptom:** ROAS values are orders of magnitude off.
**Cause:** pymc-marketing scales inputs internally. Raw `channel_contribution` values are in scaled space.
**Fix:** Call `mmm.add_original_scale_contribution_variable()` after sampling before reading contributions. See `attribute/contributions.py`.

### Brownfield idata coordinate mismatch
**Symptom:** `ValueError: conflicting sizes for dimension` when loading previous idata.nc.
**Cause:** Channel list or date range changed between the original and warm-start run.
**Fix:** Verify `spec.channels` names and date range exactly match the previous run before warm-starting. See `fit_runner/warmstart.py`.

### Divergences / poor sampling
**Symptom:** rhat > 1.05, many divergences, low ESS.
**Common causes:**
- Prior too wide (`alpha_sigma` or `lam_sigma` > 0.3 for most channels)
- `l_max` too large relative to dataset length
- Collinear channels (run VIF check)
**Fix:** `/mmm-diagnose` → inspect `diagnostics.json` → tighten priors or reduce `l_max` → re-fit.

---

## Israeli Data Providers

### BoI data portal — endpoint not confirmed
**Status:** `data/providers/boi.py` uses a placeholder URL that returns 404.
**Error seen:** `BoI series IR01 fetch failed: 404 Client Error: Not Found for url: https://edge.boi.gov.il/FusionEdge/series/IR01?...`
**Action:** Verify correct endpoint and terms of service at the Bank of Israel open-data portal. Update `boi.py` and `TODO.md`.

### CBS API — endpoint not confirmed
**Status:** `data/providers/cbs.py` uses a placeholder URL.
**Action:** Verify correct endpoint at Israel Central Bureau of Statistics open API. Update `cbs.py` and `TODO.md`.

### Google Trends rate limiting
**Symptom:** `pytrends` raises `TooManyRequestsError` after multiple pulls.
**Cause:** Unofficial API; Google throttles aggressive requests.
**Fix:** Add delays between calls, or use SerpAPI / DataForSEO (paid). Set `SERPAPI_KEY` in `.env` — `google_trends_il.py` uses it automatically when present.

---

## Windows / Environment

### Multiprocessing crash on Windows — missing `if __name__ == '__main__'`
**Symptom:** `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.`
**Cause:** Windows uses `spawn` (not `fork`) for multiprocessing. PyMC spawns worker processes, which re-import the script. Top-level code that calls `run_fit()` runs again inside each worker, triggering infinite recursion.
**Fix:** Wrap all executable code in script files with `if __name__ == '__main__':`. Both example scripts already include this guard.

### g++ not available — PyTensor pure Python mode
**Symptom:** `WARNING (pytensor.configdefaults): g++ not detected! PyTensor will be unable to compile C-implementations and will default to Python. Performance may be severely degraded.`
**Cause:** No C++ compiler found in PATH on Windows.
**Impact:** MCMC sampling is significantly slower (5–10×). Results are identical.
**Fix (recommended):** Install MinGW-w64 via MSYS2:
```
winget install -e --id MSYS2.MSYS2
# Then in MSYS2 UCRT64 terminal:
pacman -S mingw-w64-ucrt-x86_64-gcc
# Add C:\msys64\ucrt64\bin to PATH
```
**Quick suppress (no perf gain):** Create `pytensor.ini` in project root:
```ini
[pytensor]
cxx =
```

### Experimental module warnings (harmless)
**Symptom:** Warnings like `xtensor module is experimental and full of bugs` and `pymc.dims module is experimental`.
**Cause:** PyTensor and PyMC ship experimental submodules that print warnings on import.
**Status:** Harmless — these are emitted by the libraries, not the framework. No action needed.
