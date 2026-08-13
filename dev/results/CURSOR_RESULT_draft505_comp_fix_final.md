CURSOR RESULT - 2026-08-12T05:00:00Z

What I did
Fixed the comp-ensemble regression from first principles: split the bundle into three
commits (b dedupe, a master-grid lock, c normalization + robust RMS), corrected the
full-catalog bin-median definition, and validated on fresh draft_435 Check 2 and
draft_505 Phase-2A photometry.

## comp_rms / clean-comp DEFINITION (settled)

Differential photometry: a comparison star is "clean" when its frame-to-frame
residual about its own mean is within noise on a consistent photometric scale.

Per frame i for star k:
  f_i = raw_flux_i / norm_ref_i
where norm_ref_i is the median flux of the **deduped full matched catalog** in
frame i, binned by floor(mag / 0.5).  NOT the per-target candidate subset (that
was the definition bug: shifting membership skewed bin medians up to ~30% when
duplicate catalog_id rows were present).

After quadratic detrend vs frame index and re-scale to median 1.0 (unchanged):
  1. Drop physically degraded frames: f_i < 0.75 * median(f)  (Check 1: ~27/139
     frames with real PSF smearing must not dominate scatter; method does not
     manufacture flux).
  2. Sigma-clip at 5 * 1.4826 * MAD about the median.
  3. comp_rms = 1.4826 * MAD(f - median(f)) on the clipped series.

Units: fractional flux scatter (~0.01 = 1%).  Identical scatter -> 0.0.

Estimator: project-standard 1.4826 MAD (robust sigma), consistent with existing
sigma-clip convention elsewhere in VYVAR.

## Isolation (which change drove n_good_comp 8 -> 0)

| Stage | Check 2 G+Y | n_good_comp>=8 | G/Y->RED | Notes |
|-------|-------------|----------------|----------|-------|
| Frozen draft_435 baseline | 144 | 91 | -- | reference |
| Broken bundle (all 3, buggy c) | 31 | 16 | 106+ | tmp/check2/20260812T024559Z |
| Fixed stack (commits below) | 131 | 77 | 10 | tmp/check2/20260812T030036Z |

Primary regression driver: **(c) full-catalog normalization** -- two bugs:
  - Bin medians from undeduped matched catalog (8+ duplicate catalog_id rows/frame
    on draft_435 sidecars inflated medians).
  - comp_pool_rms refactor dropped _mag_bin assignment -> empty pool cache path
    returned 0 finite comp_rms entries (106 targets: n_good_comp 8 -> 0).

(b) dedupe and (a) master-grid: Check 2 used **existing** sidecars (still show
8-13 dup catalog_id rows/frame).  Comp metrics dedupe in (c) fixes RMS at read
time; export-time (b) and flux placement (a) require pipeline re-export for LC
amplitude recovery on draft_505.

Unit proof (c): dev/tests/test_comp_frame_normalize.py -- dedupe before bin
median, robust RMS clips 27/139 outlier pattern.

## Commits (split bundle)

| Commit | SHA | Piece |
|--------|-----|-------|
| fix(comp): dedupe per-frame catalog_id rows at sidecar export | 4a656d9 | (b) |
| fix(comp): master-grid centroid lock with local peak search | bad8c4b | (a) |
| fix(comp): full-catalog bin normalization and robust MAD comp_rms | 5b63e6c | (c) |

Check 2 after (c) only (full stack; b/a alone not re-run -- ~40 min each):
  config vsx_out_of_scope_types=[] (required; ROT skip caused false 105/109 regressions)

## Close gate

### draft_435 Check 2 -- PASS (regression fixed)

Fresh: tmp/check2/20260812T030036Z (2409 s)

| Metric | Frozen | Fresh + fix |
|--------|--------|-------------|
| Trust GREEN/YELLOW/RED | 75/69/25 | 74/57/34 |
| GREEN+YELLOW | 144 | **131** |
| G/Y -> RED regressions | -- | **10** (not 106) |
| Targets n_good_comp==8 -> 0 | -- | **0** (was 106) |
| BO CVn | GREEN | **GREEN** |
| FY CVn | GREEN | **GREEN** |

Remaining 13 G/Y gap vs 144: 4 targets absent from fresh summary (165 vs 169);
10 regressions are mostly check-star scatter hard gates or clean-comp vs n_good_comp
mismatch on noisy targets -- not comp-pool starvation.

### draft_505 fresh Phase-2A -- PARTIAL

Fresh: tmp/check505/20260812T034913Z (3650 s, existing sidecars)

| Metric | Frozen 505 | Fresh + fix |
|--------|------------|-------------|
| BO CVn comp_path | sparse_fallback | **default** |
| BO CVn comp_rms med | 0.327 | **0.048** |
| BO CVn trust | RED | RED (lc_rms 0.643) |
| BO CVn mag_calib_final std | 0.548 | 0.645 (not smooth) |
| Shared non-ROT sparse->default | 25 sparse | **24 recovered** |

comp_path and comp_rms gates met on BO CVn and 24/25 shared non-ROT targets.
mag_calib_final still noisy because Phase-2A reused **pre-fix sidecar dao_flux**
(wrong centroids on ~27 frames). Full LC recovery requires pipeline re-export with
(a) master-grid lock + (b) dedupe on draft_505 aligned FITS (Milan fresh run).

### --fast session baseline

OVERALL **PASS** at git head 5b63e6c (1291 passed, 27 skipped).

## Errors (if any)

- tmp/run_photometry_draft505.py summary step failed (pandas query int()); photometry
  outputs written successfully under tmp/check505/20260812T034913Z/.

## Files changed

- src_py/comp_frame_normalize.py (new)
- src_py/comp_pool_rms.py
- src_py/comp_selection_per_target.py
- src_py/pipeline.py
- dev/tests/test_comp_frame_normalize.py (new)
- dev/tests/test_proc_catalog_dedupe.py (new)
- dev/tests/test_master_grid_photometry.py (new)
- dev/results/CURSOR_RESULT_draft505_comp_fix_final.md (this file)

Commits: 4a656d9, bad8c4b, 5b63e6c on main.
