# Known Errors and Pitfalls

Document errors and pymc-marketing quirks as they are encountered.

---

## pymc-marketing API Pitfalls

### Legacy MMM import
**Symptom:** `ImportError: cannot import name 'MMM' from 'pymc_marketing.mmm'`
**Cause:** Pre-0.8 API used a different module path.
**Fix:** Use `from pymc_marketing.mmm import MMM` (0.8+), not the old multidimensional path.

### y-series column naming
**Symptom:** `KeyError` or silent wrong column during fit.
**Cause:** `MMM` requires the target column to match exactly what is passed as `y`.
**Fix:** Always use `spec.target.column` consistently; never rename after `build_mmm()`.

### `model.optimize_budget()` does not exist
**Symptom:** `AttributeError: 'MMM' object has no attribute 'optimize_budget'`
**Cause:** Budget optimization is a separate object (`BudgetOptimizer`), not a method on MMM.
**Fix:** Use `from pymc_marketing.mmm import BudgetOptimizer` and wire up separately in `optimize/budget.py`.

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

*(Append as encountered)*

---

## Environment Issues

*(Append as encountered)*
