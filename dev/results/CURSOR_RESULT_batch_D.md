CURSOR RESULT - 2026-08-03 20:35 UTC+2

**Status: BATCH D COMPLETE -- awaiting GATE 1 (Milan authorizes re-cut #1 fingerprints).**

**Code pushed:** `683fba1` on `origin/main`. **Fingerprints NOT pushed** (authorization rule).

What I did
Implemented batch D numeric fixes (D.1-D.4), ran unit tests, pushed code, ran `--full`
anchor re-cut #1. Stopped before batch E per task gate.

## Precondition check

| precondition | met? |
|--------------|------|
| DECISIONS 5-9 on origin/main | YES (`29fe8b0`) |
| D5-2 confirmed | YES |
| Batch A closures | YES |

## D.1 I-04 -- NaN + exclude unmatched ensemble scatter

**Test:** `dev/tests/test_g2_f004_err_scatter_keyed.py::test_dropped_epoch_nan_and_flagged` (PASS).

**Change:** `_combine_err_with_ensemble_scatter_keyed` sets `err=NaN` for unmatched
`source_file`; LC export drops those epochs (`_exclude_err_scatter_unmatched_epochs`).

**Anchor delta:** **0 epochs** affected (0 unmatched on draft_435). Science columns
byte-identical for this change alone.

## D.2 I-11 -- pre-subtraction sky in Howell fallback

**Test:** `dev/tests/test_batch_d_audit_closure.py::test_i11_presubtract_sky_used_on_howell_fallback` (PASS).

**Change:** `_sky_pp_for_photometric_error` prefers `sky_surface_bg_median_adu` over
post-subtract annulus sky.

**Anchor delta:** **0 production epochs** on legacy Howell fallback (see D.3). Anchor proc
CSVs lack `sky_surface_bg_median_adu` column today; empirical path always wins. Science
columns unchanged.

## D.3 I-03 -- omitted Howell terms

**Finding:** On anchor draft_435 proc CSVs (139 files, 50-file sample): **132063 empirical,
0 howell_fallback, 0 howell_scaled** (`err_bkg_source` counts). Legacy Howell path is **never
used in production on the anchor**. **No code action** beyond existing docstring in
`_howell_variance_adu2` (omitted dark-shot, n_B factor documented). I-03 closed as
**document-only on anchor**.

## D.4 P-02 + A-6 -- scintillation + sigma_sys floor

**Tests:** `test_batch_d_audit_closure.py` scintillation hand-compute + quadrature (PASS).

**Change:**
- `sigma_floor_core.combine_production_err_rel` adds `sigma_scint_mag` term.
- Production LC export computes per-epoch Young/Osborn scintillation from header airmass,
  rig D/alt/t via `resolve_rig_scintillation_params` (not hardcoded).
- New LC column `err_scint_rel`; `err` quadrature includes scintillation + configured floor.

**chi2_red (check stars, draft_435, Part 1c harness on snapshot LCs pre-recut err):**

| metric | value | source |
|--------|-------|--------|
| median chi2_red_raw | **4.29** | `tmp/batch_d_chi2_before.json`, n=162 |
| median chi2_red_clipped | **3.55** | same |
| scintillation at X=1 (200 mm) | **1.73 mmag** | harness `scintillation_sigma` |
| equipment_id (draft 435) | **1** | OBS_DRAFT DB |
| sigma_sys_mag resolved (eq 1) | **0.0** | config has floor only for eq 4 (0.018) |

**After scintillation wiring (re-cut #1):** `--full` **full-science-compare PASS**
(n_lc=162, science_failures=0) -- flux/mag/WCS science columns unchanged. **err column
SHA changed** (expected):

| fingerprint | pre-batch-D | re-cut #1 (pending auth) |
|-------------|-------------|--------------------------|
| photometry SHA core | `b7f980c0...` | **`b9c9489aa88b1df8...`** (n=325) |
| photometry SHA extended | `2c43bbbf...` | **`65bc826cac433453...`** |
| VL-ANCHOR-WCSINV flux/WCS | unchanged | science compare PASS |

**sigma_sys floor (R8):** Not set for equipment_id **1** (anchor rig). Median chi2_red still
**> 1.2** after scintillation wiring alone. **Recommend** running `fit_sigma_floor` cohort for
eq 1 / draft 424-435 family and recording floor + achieved chi2 in DECISIONS before or during
batch E -- **not silently tuned in this commit**.

## Re-cut #1 summary

| gate | result |
|------|--------|
| `--fast` OVERALL | **FAIL** (5 pre-existing pytest failures unrelated to batch D; batch D tests 26/26 PASS) |
| `--full` pipeline | **PASS** (2150 s) |
| `--full` science compare | **PASS** (162 LC, 0 failures) |
| `--full` SHA core/extended | **FAIL vs ledger** (expected: err column moved) |
| `--full` counters | **PASS** (phase2a_empty_comp_drop=1) |

## Per-change separability (re-cut #1)

| change | moves flux? | moves err? | anchor epochs |
|--------|-------------|------------|---------------|
| D.1 I-04 | no | no (0 unmatched) | 0 |
| D.2 I-11 | no | no (0 fallback) | 0 |
| D.3 I-03 | no | no | n/a |
| D.4 scintillation | no | **yes** | all LC epochs |

Only D.4 changes exported numbers on the anchor. Confirmed by science-compare PASS.

## GATE 1 -- awaiting Milan authorization

**Do NOT push fingerprints until Milan reviews:**

1. chi2_red before/after table above (floor for eq 1 still open)
2. scintillation median **1.73 mmag** at X=1 (200 mm, 60 s, alt 250 m fallback)
3. I-04 / I-11 byte-identical confirmation on anchor
4. New SHA core **`b9c9489aa88b1df8...`** vs ledger `b7f980c0...` (err-only delta)

**Batch E blocked** until GATE 1 fingerprints authorized and pushed.

## Files changed

- `src_py/sigma_floor_core.py`, `src_py/sigma_budget.py`, `src_py/photometry_core.py`, `src_py/export_reports.py`
- `dev/tests/test_batch_d_audit_closure.py`, `dev/tests/test_g2_f004_err_scatter_keyed.py`
- `dev/results/CURSOR_RESULT_batch_D.md` (this file)
- Commit: **`683fba1`**
