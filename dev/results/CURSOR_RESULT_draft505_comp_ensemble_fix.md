CURSOR RESULT - 2026-08-11

What I did
Diagnosed draft_505 vs draft_435 LC regression for non-ROT targets (BO CVn) and
implemented photometry/comp-selection fixes in the pipeline.

## Output / findings

### Root cause (evidence on disk)
- Raw `mag_inst` was fine; `mag_calib_final` / ensemble was broken by bad comp stars.
- Same Gaia comp IDs in both drafts, but draft_505 `comp_rms` ~0.27-0.54 vs ~0.008 in 435.
- Default comp path returned 0 gate-passers ? `sparse_fallback` ? RED trust / noisy LCs.
- **27 / 139 frames**: aligned FITS at MASTERSTAR grid positions show ~5- lower peak
  flux (PSF smeared / DAO matched faint neighbour). Example comp star
  `1499200223486564608` frame 012: peak 15699 (435) vs 3241 (505).
- **Duplicate `catalog_id` rows** in per-frame CSV (9 frames for that comp) inflated
  Phase-1 RMS scatter.
- **Phase-1 brightness-bin normalization** used only the per-target candidate subset,
  not the full matched catalog ? unstable `comp_rms` when candidate pools differ
  (505 dense-field adaptive + ROT skip changed pools vs 435).

ROT skip is NOT the cause; BO CVn (EW) was measured in both runs.

### Fixes implemented
1. **`pipeline.py`**: `_lock_matched_centroids_to_master_grid` - on VY_ALGN frames,
   matched catalog stars snap to MASTERSTAR (x,y) + local peak search (~2.5 FWHM).
2. **`pipeline.py`**: `_proc_deduplicate_matched_catalog_rows` - one row per
   `catalog_id` (brightest peak wins) before writing per-frame CSV.
3. **`comp_selection_per_target.py`** + **`comp_pool_rms.py`**: brightness-bin
   medians from full matched catalog; dedupe one flux point per star per frame in
   comp metrics accumulation.

### Tests
- `dev/tests/test_master_grid_photometry.py` (2 tests) - PASS

### Validation (Milan)
Re-run BO CVn (draft copy OK) with `vsx_out_of_scope_types: ["ROT"]` kept.
Expect for shared non-ROT targets vs draft_435:
- `comp_path: default` (not sparse_fallback)
- `comp_rms` ~0.01-0.03 (not ~0.3)
- `trust` GREEN/YELLOW, smooth `mag_calib_final`

## Errors (if any)
None.

## Files changed
- src_py/pipeline.py
- src_py/comp_selection_per_target.py
- src_py/comp_pool_rms.py
- dev/tests/test_master_grid_photometry.py
