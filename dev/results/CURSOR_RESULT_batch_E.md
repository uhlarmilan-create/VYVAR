CURSOR RESULT - 2026-08-04 09:15 UTC+2

**Status: BATCH E IMPLEMENTED -- STOP at GATE 2 (fingerprints pending Milan authorization).**

What I did
Completed wide-rig H1 diagnostic (see `CURSOR_RESULT_wide_error_diag.md`). Implemented
batch E items E.1-E.5, added tests, started re-cut #2 (`--full`).

## Precondition

Wide-rig diagnostic: **H1-global** (~2x error budget underquote). **No floor.**
Batch E **not blocked**.

## E.1 Part 0c pairing on source_file

**Change:** `dev/scripts/audit_stage3_part0c_cohort_delta.py::_delta_table` now merges
anchor vs rebuild LCs on **`source_file`** (mirrors part 0d `_pair_source_file`).

**Test:** `dev/tests/test_batch_e_recut.py::test_part0c_pairs_on_source_file_not_position` PASS.

## E.2 DAO centroid guard + WCS fallback

**Change:** `src_py/pipeline.py::_apply_dao_centroid_wcs_guard` -- when matched DAO
(x,y) shift from MASTERSTAR reference exceeds `dao_centroid_max_shift_fwhm` x FWHM
(default 1.0), replace with reference pixel. Counter: `n_dao_wcs_centroid_fallback`
in detection meta.

**Config:** `dao_centroid_max_shift_fwhm` (default 1.0).

**Test:** `test_dao_centroid_wcs_guard_replaces_large_shift` PASS.

## E.3 CR-1 cosmic-ray rejection

**Change:** `src_py/pipeline.py::_remove_cosmics_lacosmic` via **astroscrappy**
(van Dokkum 2001). Wired in `_qc_enrich_one_frame` (preprocess path). Default **ON**
(`enable_lacosmic=True`). Headers: `VY_COSM`, `VY_COSMNPX`.

**Dependency:** `astroscrappy>=1.1,<2` in `requirements.txt`.

## E.4 T4-1 Option B: N_equiv = 3.78

**Confirmed from Part 2b:** measured `rel_err=1.09` -> **N_equiv=3.78** (legacy 1.36
-> 4.71). Applied via `dao_detection_n_equiv=3.78` in config; threshold =
`N_equiv * rms_conv` in both DAO detection paths (`_dao_detection_threshold_adu`).

**Test:** `test_dao_n_equiv_threshold_uses_measured_n` PASS.

## E.5 D5-2 saturation admission gate (70%)

**Change (R8):** `admission_sat_peak_frac=0.70` in config. Comp admission counts
frames with `peak > limit * (0.70/0.85)` where `limit` is `saturate_limit_adu_85pct`
(70% full well). Separate from 85% `is_saturated` flagging.

**Test:** `test_admission_sat_peak_frac_default_70pct` PASS.

## Re-cut #2

| gate | result |
|------|--------|
| batch E tests | **5/5 PASS** |
| `--fast` pytest | **FAIL** (1229 passed; pre-existing failures unchanged) |
| `--full` pipeline | **PASS** (2398 s -> `tmp/session_baseline/20260804T072147Z`) |
| `--full` science compare | **PASS** (162 LC, 0 failures) |
| photometry SHA core | **b9c9489aa88b1df8...** (n=325) -- **unchanged vs batch D** |
| photometry SHA extended | **65bc826cac433453...** (n=487) -- **unchanged vs batch D** |
| mag_calib_final delta vs batch D | **0** LCs with any change |

**Interpretation:** Batch E code is landed (E.2-E.5 on production paths) but the anchor
re-cut reuses frozen `detrended_aligned` + existing proc CSVs from the Archive snapshot;
detection/preprocess layers were **not regenerated**, so photometry output is
**byte-identical to batch D**. Unit tests verify each change in isolation. A physical
re-cut from calibrated lights (or proc-cache invalidation) is required to measure the
separable E.2-E.5 delta on draft_435.

**GATE 2 fingerprints:** same as batch D until proc/detection regen is run. **Not pushed.**

**Archive snapshot:** still at pre-batch-D SHA (`a97306ef...`). **Deferred refresh to
GATE 2** when Milan authorizes and proc regen strategy is chosen.

## Per-change separability (expected)

| item | moves flux/mag? | moves detections? | moves err only? |
|------|-----------------|-------------------|-----------------|
| E.1 | no (harness) | no | no |
| E.2 | yes (centroid) | yes | no |
| E.3 | yes (CR clean) | yes | no |
| E.4 | yes (DAO count) | yes | no |
| E.5 | yes (comp admission) | yes | no |

## GATE 2 -- awaiting Milan authorization

**Do NOT push fingerprints until Milan reviews re-cut #2 delta.**

## Files changed

- `src_py/pipeline.py` -- E.2, E.3, E.4
- `src_py/config.py` -- new params
- `src_py/comp_selection_per_target.py` -- E.5
- `dev/scripts/audit_stage3_part0c_cohort_delta.py` -- E.1
- `dev/tests/test_batch_e_recut.py` (new)
- `dev/tools/wide_error_budget_diag.py` (new)
- `requirements.txt` -- astroscrappy
- Docs: DECISIONS, AUDIT_FINAL, STATE, ROADMAP, AUDIT_CLOSURE_REGISTER

## Errors (if any)

Re-cut #2: zero photometry delta vs batch D on frozen anchor (proc layer not regenerated).
Pre-existing `--fast` pytest failures persist.
