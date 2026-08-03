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

---

## GATE 1 authorized append (2026-08-03 22:15 UTC+2)

**Status: BATCH D GATE 1 CLOSED -- fingerprints pushed; batch E blocked on floor anomaly.**

What I did
- Pushed batch D anchor fingerprints (SHA core/extended + ledger + baseline tests).
- Fitted wide-rig `sigma_sys_mag` for equipment_id 1 per R8 (Part 1c check stars, n=162).
- **Did not apply floor** (outside 2-5 mmag sanity). **Did not start batch E.**

### Fingerprints pushed

| tier | value | n |
|------|-------|---|
| core (new) | `b9c9489aa88b1df815bf6157911b35af5bb1c42a3b0eaf58995042fcdd007a39` | 325 |
| extended (new) | `65bc826cac433453f689dbc5ab2883e783b7a7c7563092c02cfa443058f48cc2` | 487 |
| core (superseded) | `b7f980c09e238b855c2ee1b9518061777934d8f0a61eaec7431cda4f537aed52` | 325 |
| extended (superseded) | `2c43bbbf06921fbef46fb6a4ed1f8afccdabacaa5827b8ec50372de0e3816205` | 487 |

Science compare **PASS** (162 LC, flux/mag/WCS unchanged). Delta **err column only**
(scintillation wired in D.4).

### Wide-rig sigma_sys floor fit (equipment_id 1, R8)

**Median quoted err (post-scintillation):** **9.4 mmag** (Part 1c `err_median` across 162 check fields).
Scintillation at X=1: **1.73 mmag** (negligible vs quoted err).

| stage | median chi2_red_clipped | source |
|-------|-------------------------|--------|
| before batch D (no scint) | **3.55** | prior Part 1c / `tmp/batch_d_chi2_before.json` |
| scintillation only | **3.55** | `tmp/batch_d_part1c_post_scint.json` (2026-08-03) |
| scintillation + fitted floor | **~1.0** (simulated) | needs **~15 mmag** floor -- **NOT applied** |

**Fitted floor (NOT applied):**

| method | sigma_sys (mmag) | sanity 2-5 mmag? |
|--------|------------------|------------------|
| chi2 scaling (`sqrt(chi2) x err`) | **14.7** | NO |
| measured residual RMS (median scatter 20.1 mmag) | **15.7** | NO |
| constant-calibrator cohort (n=12, separate harness) | **8.33** | NO |

**R8 verdict:** Floor outside Everett & Howell (2001) 2-5 mmag band on the check-star population.
Per task rule: **do not apply**; **stop before batch E**. Likely causes: photon or ensemble term
mis-scaled, and/or frame-correlated scatter not captured by a constant quadrature floor.

Harness: `dev/tools/batch_d_wide_floor_fit.py` (cohort fit); Part 1c:
`dev/scripts/audit_stage3_part1c_robust_chi2.py`.

### Pre-existing `--fast` failures (separate from batch D)

Batch D tests **26/26 PASS**. `--fast` OVERALL still **FAIL** on 5 pre-existing pytest issues
(unrelated to batch D; fix outside this thread):

1. `dev/tests/test_invariants_p1_seed.py` -- requires Archive snapshot on disk (environment)
2. Additional legacy failures logged at re-cut #1 (see prior session baseline output)

These do **not** block fingerprint push.

### Docs updated

- `docs/VYVAR_DECISIONS.md` -- decision 7 addendum (floor fit table)
- `docs/VYVAR_AUDIT_FINAL.md` -- error budget section
- `docs/VYVAR_VALIDATION.md` -- anchor fingerprint table
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` -- items 21, 22, 25 FIXED; item 29 batch D done
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`

## Files changed (GATE 1 append)

- `dev/scripts/session_baseline_check.py` -- EXPECTED SHA
- `dev/tests/test_invariants_p1_seed.py` -- EXPECTED SHA
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` -- VL-ANCHOR-WCSINV
- `dev/tools/batch_d_wide_floor_fit.py` -- floor fit harness (new)
- `dev/results/CURSOR_RESULT_batch_D.md` (this append)
- Docs listed above

## Errors (if any)

None blocking. Floor fit **anomaly** reported; batch E not started.

