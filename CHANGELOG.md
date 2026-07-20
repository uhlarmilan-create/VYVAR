# CHANGELOG - VYVAR

All notable changes to VYVAR are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- **CONFIG-HUMAN-EDIT (2026-07-18):** `config.json` is now a grouped, commented, `//`-tolerant
  JSONC-lite document that can be edited without the UI; standalone validator
  `dev/scripts/validate_config.py`; registry `help` / `__meta__.phase_help` as the single source
  of config comments, tooltips and guides.
- **README front door (2026-07-18):** rewritten `README.md` plus Czech twin `README_CZ.md`
  (user docs EN+CZ); root `LICENSE` (proprietary, all rights reserved).
- **VALIDATE-429 (2026-07-16):** Forensics scripts `validate_429_wcsinv.py`, `pass2_contamination_428_429.py`,
  `anchor_pair_430_431.py`; CURSOR_RESULT_429_validate_anchor.md.
- **F-429 fixes (2026-07-16):** VSX stamp wired post-finalize; `[AC] run summary` infolog; Gaia TAP
  retry INFO lines (`fc177be`).
- **F-428 WCS-INV (2026-07-16):** Round-trip invertibility gate, SIP inverse regen, coordinate
  finalization (`coord_source`), post-match pixel identity gate (`wcs_invertibility.py`).
- **F-428 COORD v5 (2026-07-16):** MASTERSTAR peak test + direction stats
  (`scripts/diag_428_coord_forensics_v5.py`); RECLASSIFY-PROJECTION - Gaia WCS->pixel agrees with
  ms x/y (~1.3 px); v4 angular MISASSIGNED reframed as coordinate bookkeeping offset; T4 control
  SPURIOUS-UNIFORM (ratio 1.07).
- **F-428 COORD v4 (2026-07-16):** Pixel-space identity forensics (`scripts/diag_428_coord_forensics_v4.py`);
  164 MISASSIGNED-ID STOP verdict on draft_428.
- **F-428 MS-STAMP (2026-07-15):** `stamp_vsx_known_variable_on_masterstars()` - catalog_id join
  for masterstars VSX flag; diag v3 forensics (`scripts/diag_428_unmatched_sep.py --forensics`).
- **F-428 fix batch (2026-07-15):** VSX `variable_targets.csv` path resolution; repair-catalog-ID
  placeholder skip + summary line; HRD Gaia TAP retry (`hrd_enrich_tap_timeout_s`); UTC infolog;
  `photometry_summary.csv` AC status columns; `excluded_targets.csv` sidecar; diagnostic
  `scripts/diag_428_unmatched_sep.py`.
- **Sparse trust (2026-07-14):** check-star ensemble at n>=2 with Howell 1988 triangulation;
  CI-based trust bands; sidecar columns (`check_sparse`, `trust_R`, `trust_band`, ...); external K
  sourcing on sparse branch (`sparse_trust_core.py`, `docs/VYVAR_SPARSE_TRUST_SPEC.md`).
- **CAL-DIAG gate (2026-07-14):** calibration-time radiometry gate default ON; `VY_DKRSMP`
  provenance in `cal_diag.json` (`docs/VYVAR_CAL_DIAG_SPEC.md`).
- **k2 cohort diagnostics (2026-07-14):** full-cohort k'' signature report; bootstrap CIs with
  overdispersion-honest weighting; internal-consistency warning (`k2_cohort_core.py`).
- **Wide slope noise study (2026-07-14):** report-only diagnostic on draft_424 wide_CLEAR
  (`wide_slope_noise_core.py`; spec `docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md`; parked verdict).
- Unit test suite `tests/test_photometry_core.py` (11 tests, Howell/Broeg/Stetson validation)
- Scientific citations in AAVSO and VAR.ASTRO export headers
- `resolve_draft_dir()` canonical utility in `utils.py`
- `read_vyvar_csv()` + `VYVAR_CSV_DTYPE` in `gaia_catalog_id.py`
- `_PhotometryReportBuilder` class - `generate_photometry_report()` refactored (3384 -> 63 lines)
- RGB camera support planned (TODO-45, IMX533 de-Bayer -> G channel)

### Changed
- **WAVE-B parameter reduction (2026-07-18):** registered configuration parameters consolidated
  to 269 (config.json persists 249); dead, hardcoded and DB-duplicated keys deleted or merged;
  Config <-> registry parity kept green as a tested property.
- **REPO-REORG (2026-07-17):** all production modules moved under `src_py/` (imports stay flat);
  developer material (tests/tools/validation/scripts/sandbox/orchestrator/results) moved under
  `dev/`; root `app.py` is now a thin shim onto `src_py/app.py`. Anchor #3 byte-identical gate
  PASS on `8f4d7b4`. (Older entries above/below that cite `scripts/` predate this move.)
- **Err model re-anchor (2026-07-13):** c4-corrected ensemble SEM; per-rig `sigma_sys` floor in
  production LC `err` (`sigma_sys_mag` column); Newton eq4 floor 18.0 mmag; wide eq1 un-floored;
  accepted anchor `draft_000424_snapshot_sigma_floor_20260713` (core `bf3743a1`, git `8fb21b3`).
- Airmass fit now runs AFTER outlier detection (TODO-29)
- Airmass detrend applied on CT-corrected magnitudes (TODO-30)
- Complete UI translation to English (~766 strings)
- Gaia ID normalization consolidated to `gaia_catalog_id.normalize_gaia_source_id()`
- Draft path resolution unified via `resolve_draft_dir()`
- 38 silent `except: pass` blocks now log WARNING/DEBUG

### Fixed
- `variability_candidates.csv` missing TESS columns (`vsx_known_variable`, `vsx_match`, `gaia_dr3_variable_catalog`)
- Float64 catalog_id precision loss in proc CSV (19-digit Gaia IDs)
- Howell (1989) sky term: `sky_pp/gain x area` (was `sky_pp x area`)
- ZP MAD sigma-clip per frame (DAOPHOT standard)
- TESS duplicate runs eliminated (result.json check before auto-trigger)

### Removed (CSV schema cleanup)
- 26 obsolete columns from proc_*.csv, photometry_summary.csv,
  active_targets.csv, masterstars_full_match.csv, comparison_stars_per_target.csv

---

## [0.9.0] - 2026-05-17

### Added
- Cross-validation: photutils (2.0% scatter), SExtractor (6% offset), IRAF (2.2% scatter)
- Per-star SNR-optimal aperture selection (TODO-21)
- Gain/RN Settings UI + DB storage (TODO-22)
- Comp star P90 noise floor for variability envelope (TODO-26)
- FWHM priority: `VY_FWHM_GAUSS` before `VY_FWHMx0.667`
- 2-pass iterative DAO detection with Gaia-targeted pass 2 (TODO-13)
- TESS blend check + period reliability classification
- Summary Measure Report PDF redesign (TODO-15)

### Fixed
- Float64 catalog_id precision loss (Gaia 19-digit IDs)
- catalog_only exclusion from comp pool, summary, hockey stick (TODO-24)
- TESS duplicate auto-run prevention
- Variability crossmatch stale cache fix (3-layer fix)

---

## [0.8.0] - 2026-05-14

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
