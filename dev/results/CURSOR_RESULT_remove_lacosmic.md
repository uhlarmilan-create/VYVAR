CURSOR RESULT - 2026-08-12 remove L.A.Cosmic

What I did
Removed L.A.Cosmic / destructive per-frame CR cleaning entirely (Milan decision).
Kept master dark/flat stacking and non-destructive stats clips.

## Inventory (destructive vs keep)

### REMOVED (destructive on light science pixels)
1. `_remove_cosmics_lacosmic` + `_lacosmic_gain_readnoise_from_header` (`pipeline.py`)
2. Call site in `_qc_enrich_one_frame` / QC in-place preprocess
3. `enable_lacosmic`, `lacosmic_sigclip`, `lacosmic_objlim` from AppConfig / load / dump
4. Registry entries for those 3 keys; Settings UI had no dedicated widgets (registry-driven)
5. `VY_COSM` / `VY_COSMNPX` writes (calibration stamp + QC stamp); QC now deletes legacy keys
6. `astroscrappy` from `requirements.txt`
7. `dev/tests/test_lacosmic_star_core.py`

### Other candidates checked
- `temporal_sigma_clip` / `temporal_sigma` on `preprocess_calibrated_to_processed`: **dead API**
  (always False from app/night_run; body never applies). Not removing (not destructive today;
  out of scope API cleanup).
- Sky-surface subtract: modifies lights but is gradient removal, not CR/sigma pixel cleaning.
  KEPT.
- No other per-frame sigma/clip pixel overwrite of lights found.

### KEPT (non-destructive / masters)
- Master dark mean stack + master flat median stack (`importer.py`; no light-pixel CR clean)
- Robust `comp_rms` MAD / LC outlier stats
- Background / DAO `sigma_clipped_stats` estimation

## Registry
- Before: 282 entries (config_runtime 254)
- After: **279** entries (config_runtime **251**; db_static 9; fits_dynamic 6; internal 13)

## Tests / gate
- `test_no_destructive_lacosmic.py` + registry/UI/batch-E tests: PASS
- `--fast`: **OVERALL PASS** (1293 passed, 27 skipped; lacosmic star-core tests removed)

## Close gate note
Fresh BO CVn re-run still required by Milan: on-disk 505/506 calibrated files remain
CR-damaged. New preprocess must not write VY_COSM and must keep frame-012 peaks ~17246.

## Files changed
- src_py/pipeline.py
- src_py/config.py
- requirements.txt
- dev/validation/params_registry.json
- docs/VYVAR_PARAMS.md (regen)
- docs/VYVAR_LIMITATIONS.md
- dev/tests/test_batch_e_recut.py
- dev/tests/test_ui_params_dashboard.py
- dev/tests/test_no_destructive_lacosmic.py (new)
- deleted: dev/tests/test_lacosmic_star_core.py
