CURSOR RESULT - Science Audit Tranche 2 (2026-07-30)

Base: `06ed950`. Implemented **P-10 fix** + **I-12 logging**. **I-11 deferred** (separate change from P-10). **No anchor re-cut** (P-10 invalidates all photometry outputs - measurement pass required first).

---

## P-10 - sky-surface sign error (FIXED)

**Root cause:** `_fit_subtract_preprocess_sky_surface` fitted `z = bg_median - work` (negative deviation) then subtracted `surf`, doubling the gradient: `out = work - (-g) = bg_median + 2g`.

**Fix:** Fit target changed to `z = work - bg_median`; subtract step unchanged (`out = work - surf`). Docstring aligned.

**File:** `src_py/pipeline.py` (`_fit_subtract_preprocess_sky_surface`, ~line 16765)

**Regression tests:** `dev/tests/test_preprocess_sky_surface.py`
- `test_order1_gradient_removal_ratio_and_pedestal` - INV-FLAT-style p99 residual drops >85%; not doubled
- `test_order2_flattens_gradient_only_frame` - strengthened (was passing on 2x amplification)

**NOT done (per audit):** anchor re-run, flux/err/?_bkg_ap/?- delta table, anchor re-cut.

---

## I-11 - Howell on sky-subtracted frames (NOT IMPLEMENTED)

Documented for Milan decision. Options unchanged from audit:
1. Use `sky_surface_bg_median_adu` for sky Poisson term (preferred)
2. Raise `BKG_SCALE_R_CLAMP_HI` with measurement justification
3. Refuse legacy err on sky-subtracted frames

Keep separate from P-10 delta measurement.

---

## I-12 - PM silent no-op (logging FIXED)

**Change:** `_apply_pm_to_gaia_rows` logs WARNING once when rows exist but no finite `pmra`/`pmdec`; stops treating missing PM as 0.0 mas/yr for threshold logic.

**File:** `src_py/vyvar_platesolver.py`

**Test:** `dev/tests/test_pm_correction_logging.py`

**Flagged not fixed:** `_obs_year_from_header` UTC-now fallback when DATE-OBS unparseable - note for DR4 PM enablement.

**Not implemented:** centroid-vs-catalog residual gate (audit suggested; separate task).

---

## Verified correct (unchanged)

Per audit table: Labb- ?_bkg, `_sigma_by_r`, PM cos ? math, hybrid design (clamp still wrong - I-11), sky-surface fit-set construction (only sign was wrong).

---

## Doctrine note (audit recommendation)

Byte-identity gates prove reproducibility, not physical correctness. Recommend at least one gate per physical step compare against independent expectation - e.g. synthetic gradient test now in `test_preprocess_sky_surface.py`.

---

## Files changed

| File | Change |
|------|--------|
| `src_py/pipeline.py` | P-10 sign fix + docstring |
| `src_py/vyvar_platesolver.py` | I-12 PM unavailable warning |
| `dev/tests/test_preprocess_sky_surface.py` | P-10 regression tests |
| `dev/tests/test_pm_correction_logging.py` | I-12 test |
| `dev/results/CURSOR_RESULT_audit_t2.md` | this report |

---

## Next steps (Milan)

1. Re-run anchor input end-to-end after P-10; measure ?flux, ?err, ?_bkg_ap, ?- per setup - **before** re-cut.
2. Decide I-11 option; implement in follow-up commit.
3. Consider INV-FLAT-01 WARN ? FAIL when P-10-class defects are possible.
