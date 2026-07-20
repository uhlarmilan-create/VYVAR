CURSOR RESULT - 2026-07-08 (EXCEPT-FIX-2)

What I did
Stage A probe + Stage B fixes for Tranche-2 TOP-10 pipeline sites (EXC-0275, 0312, 0317,
0331, 0339, 0342, 0350, 0389, 0415, 0433). Uniform ERROR + `except_fix_counters`; census
rows ? FIXED; validated on drafts 424/425/427.

## Stage A - firing probe

| Site | Context | Fires |
|------|---------|-------|
| EXC-0312/0342/0275/0317/0331/0415 | draft_424 NoFilter + draft_427 g platesolve/catalog | NEVER |
| EXC-0339/0350 | draft_425 B variable_targets build | NEVER |
| EXC-0389 | draft_424 stress_test (standard pass) + flatness utility | NEVER |
| EXC-0433 | unit-test path (425/427 pre-calibrated - no calibrate run) | N/A ? unit tests |

Artifact: `tmp/except_fix2_probe.json` - **17 probe rows, 0 fires**.

Key natural-path metrics:
- draft_424 optics floor: **13.635 deg** (Newton-class); draft_427 g: exercised separately
- draft_425 variable_targets: **unchanged SHA** after `write_photometry_plan_files` exercise

## Stage B - fixes

| Site | Fix |
|------|-----|
| EXC-0275 | ERROR + `catalog_bpm_enhance_fail`; return input df |
| EXC-0312 | ERROR + `plate_solve_bundle_fail`; `bundle_error` in out dict |
| EXC-0317 | ERROR + `masterstars_rescale_coords_fail` |
| EXC-0331 | ERROR + `vytarg_header_write_fail` |
| EXC-0339 | ERROR + `vsx_frame_bbox_wcs_fail` |
| EXC-0342 | ERROR + `gaia_cone_optics_floor_fail`; **FOV fallback cone** (never 0.0) |
| EXC-0350 | ERROR summary + `vsx_variable_coord_drop` with VSX IDs |
| EXC-0389 | ERROR + `stress_sidecar_skip` in `validate_comparison_ensemble_flatness` |
| EXC-0415 | ERROR + `masterstar_ref_swap_fail` |
| EXC-0433 | `_sync_obs_calibration_state_with_retry` - retry once, ERROR + `calibrate_db_sync_fail` + stats |

## Validation

- `tests/test_except_fix2_top10.py`: **9 passed**
- Full pytest: **604 passed**; ruff green
- NEVER-FIRE: MASTERSTAR/catalog artifact SHA256 unchanged on probe exercise (see probe JSON)
- EXC-0433: `test_exc0433_db_sync_retry_then_count` + succeed-on-retry test

## Errors (if any)

None.

## Files changed

- `except_fix_counters.py`
- `pipeline.py`
- `tests/test_except_fix2_top10.py`
- `docs/VYVAR_EXCEPT_CENSUS.md`
- `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_except_fix2.md`
- `sandbox/_except_fix2_probe.py` (local)
- `tmp/except_fix2_probe.json`
