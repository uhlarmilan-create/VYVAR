CURSOR RESULT — 2026-06-25 (F-HOWELL-3 Stage C)

What I did
Implemented `sky_adu_per_px_annulus` column + err call-site preference with legacy fallback.
Verified on real draft_424 via production `run_full_photometry_pipeline`. Updated docs (Stage B
number corrections + Stage C FIXED).

## Output / findings

### C1 implementation
- `enhance_catalog_dataframe_aperture_bpm`: writes `sky_adu_per_px_annulus` (both branches)
- `read_flux_from_csv` / `_sky_pp_for_photometric_error`: prefers annulus column
- `proc_frame_store.py`, `gaia_catalog_id.py`, Phase-2A column lists updated
- `tests/test_photometric_error_sky_column.py` (4 tests)

### C2 verification (draft_424, production path)
- **C2a:** 178/178 LCs `science_ok` true (mag/flux/delta_mag unchanged; err deltas only)
- **C2b:** sky-dominated faint targets: err ratio detection/annulus **1.12–1.14** (~12–14%)
- **C2c:** `photometry_mode=epsf` without ePSF → aperture OFF; no annulus column (rare edge)
- Report: `tmp/phaseHowell3/stage_c_verify.json`

### C3 docs
- Corrected Stage B addendum numbers (1.30 ratio, 1.5% bright-star inflation)
- STATE / DECISIONS / JOURNAL / ROADMAP / AUDIT_LEDGER updated

## Errors (if any)
None.

## Files changed
- `photometry_core.py`, `proc_frame_store.py`, `gaia_catalog_id.py`
- `tests/test_photometric_error_sky_column.py`
- `docs/VYVAR_MATH_PHYS_AUDIT.md`, `VYVAR_STATE.md`, `VYVAR_DECISIONS.md`, `VYVAR_JOURNAL.md`, `VYVAR_ROADMAP.md`, `VYVAR_AUDIT_LEDGER.md`
- Commit message: `fix(err): annulus-sky column for Howell sky term; deglobalize noise_floor_adu`
