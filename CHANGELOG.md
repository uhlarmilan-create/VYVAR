# CHANGELOG — VYVAR

All notable changes to VYVAR are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Unit test suite `tests/test_photometry_core.py` (11 tests, Howell/Broeg/Stetson validation)
- Scientific citations in AAVSO and VAR.ASTRO export headers
- `resolve_draft_dir()` canonical utility in `utils.py`
- `read_vyvar_csv()` + `VYVAR_CSV_DTYPE` in `gaia_catalog_id.py`
- `_PhotometryReportBuilder` class — `generate_photometry_report()` refactored (3384 → 63 lines)
- RGB camera support planned (TODO-45, IMX533 de-Bayer → G channel)

### Changed
- Airmass fit now runs AFTER outlier detection (TODO-29)
- Airmass detrend applied on CT-corrected magnitudes (TODO-30)
- Complete UI translation to English (~766 strings)
- Gaia ID normalization consolidated to `gaia_catalog_id.normalize_gaia_source_id()`
- Draft path resolution unified via `resolve_draft_dir()`
- 38 silent `except: pass` blocks now log WARNING/DEBUG

### Fixed
- `variability_candidates.csv` missing TESS columns (`vsx_known_variable`, `vsx_match`, `gaia_dr3_variable_catalog`)
- Float64 catalog_id precision loss in proc CSV (19-digit Gaia IDs)
- Howell (1989) sky term: `sky_pp/gain × area` (was `sky_pp × area`)
- ZP MAD sigma-clip per frame (DAOPHOT standard)
- TESS duplicate runs eliminated (result.json check before auto-trigger)

### Removed (CSV schema cleanup)
- 26 obsolete columns from proc_*.csv, photometry_summary.csv,
  active_targets.csv, masterstars_full_match.csv, comparison_stars_per_target.csv

---

## [0.9.0] — 2026-05-17

### Added
- Cross-validation: photutils (2.0% scatter), SExtractor (6% offset), IRAF (2.2% scatter)
- Per-star SNR-optimal aperture selection (TODO-21)
- Gain/RN Settings UI + DB storage (TODO-22)
- Comp star P90 noise floor for variability envelope (TODO-26)
- FWHM priority: `VY_FWHM_GAUSS` before `VY_FWHM×0.667`
- 2-pass iterative DAO detection with Gaia-targeted pass 2 (TODO-13)
- TESS blend check + period reliability classification
- Summary Measure Report PDF redesign (TODO-15)

### Fixed
- Float64 catalog_id precision loss (Gaia 19-digit IDs)
- catalog_only exclusion from comp pool, summary, hockey stick (TODO-24)
- TESS duplicate auto-run prevention
- Variability crossmatch stale cache fix (3-layer fix)

---

## [0.8.0] — 2026-05-14

### Added
- TESS auto-trigger for all variability candidates
- Global comp pool (TODO-3)
- BP-RP sliders in UI settings (TODO-6)
- VSX crossmatch bug fix
- RUN VYVAR e2e pipeline button

### Fixed
- Double MASTERSTAR issue
- Comp stars = variable targets proximity veto
- Border filter (aligned_files after RAM flush)
- Stale x/y coordinates after MAKE MASTERSTAR
