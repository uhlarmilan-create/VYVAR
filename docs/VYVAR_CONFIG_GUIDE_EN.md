# VYVAR Configuration Guide (config.json) - EN

_Companion document: `VYVAR_CONFIG_GUIDE_CZ.md` (Czech). Generated from the
parameter registry (`dev/validation/params_registry.json`, 264 entries) and
the parameter source audit (`dev/results/PARAM_SOURCE_AUDIT.md`), state as of
2026-07-18 (post WAVE-B parameter reduction). When parameters change, update
this guide together with the registry._

## What is config.json?

`config.json` sits in the VYVAR root directory and is the main settings file
of the pipeline. It stores the knobs that control HOW your images are
processed: how stars are detected, how comparison stars are chosen, how the
light curve is built, and what ends up in reports. You normally edit it
through the Settings page in the application, not by hand.

Not every value VYVAR uses lives in this file. There are three kinds of
parameters, and this guide labels every one of them:

- **Static (database)** - facts about your observatory: observing site,
  telescope, camera, catalogs. They live in database tables (LOCATION,
  TELESCOPE, EQUIPMENTS) and are managed in the app, which is the
  authoritative source. WAVE-B removed nine config.json copies of such facts
  so there is only one place to edit them: the observer site coordinates and
  name (observer_lat / observer_lon / observer_alt_m / observer_location_name,
  hydrated from the default LOCATION row, IS_DEFAULT), the
  detector gain and read_noise (resolved DB/FITS-first), and the plate-scale
  labels (plate_scale_arcsec_per_px, phase01_plate_scale_arcsec_per_px,
  export_arcsec_per_px, derived from the WCS/optics). CFG-REVIEW-2026-10
  moved two more camera properties into EQUIPMENTS: ADC_BITS (was
  gain_container_scale: native gain / 2^(16-ADC_BITS) is the gain on the
  16-bit frames when photon transfer is unavailable) and SIGMA_SYS_MAG (was
  the sigma_sys_mag map: the error floor when a night has too few comparison
  stars to measure it).
- **Dynamic (FITS / runtime)** - values measured or computed for each run:
  detector gain, read noise, frame size, plate scale, binning, filter,
  exposure time. VYVAR reads them from the FITS file headers or derives
  them (e.g. from the astrometric solution). These no longer carry
  config.json fallbacks after WAVE-B.
- **Setting (config.json)** - the genuine user-tunable behavior of the
  pipeline. This is the majority of the 264 registered parameters (config.json
  persists 249 of them; the rest are database facts, FITS/runtime values, or
  internal plumbing).
  WAVE-B also hardcoded 20 blind/plate-solve solver internals that were never
  tuned in practice, and merged 14 tier/aperture scalars into 3 structured
  keys (comp_color_tiers, phase01_tiers, aperture_snr_sizing).
- **Internal** - plumbing (file paths, machine-specific values). Leave
  these alone unless you know exactly why.

## How values are resolved

For each parameter the pipeline applies a clear precedence. In general:
code defaults -> config.json -> database facts -> FITS-measured values,
with the most specific source winning for dynamic parameters (a gain
measured in the FITS header beats a config fallback). Every report contains
a Configuration section with the FULL snapshot of the configuration as the
run actually used it, including the resolved dynamic values - so you can
always see how a result was produced, even years later.

Safety: config.json can only be written by an explicit Save action in the
UI. Pipeline runs never modify it.

## Editing without the UI

You can open config.json in any text editor and change values by hand. The
file is written grouped into sections (one per pipeline stage) with a short
comment above every group and every key, so you can see what each setting
does without leaving the editor. A few rules:

- `//` line comments are allowed and ignored on load (block comments `/* */`
  and trailing commas are NOT allowed - keep it otherwise strict JSON).
- Unknown keys are ignored with a warning that suggests the closest real key
  (a typo safety net); the pipeline still runs with defaults for them.
- After editing, check your file with the standalone validator:

    python dev/scripts/validate_config.py

  It reports syntax errors (with line numbers), unknown keys (with
  suggestions), out-of-range values and type mismatches, and exits non-zero
  if anything is wrong.
- Saving from the UI regenerates the file (grouping and comments included)
  from the parameter registry, so any custom comments you add by hand are not
  preserved across a UI save.

## How to read the tables

Type column: see the four categories above. Source column: where the value
actually comes from (with a note when FITS can override it). "Used in"
column: the area of the code that consumes it, with one representative
code reference. Range: hard limits enforced by the app where defined.


## Observer & site

Who observed and from where. These are observatory facts: they identify you in AAVSO exports and give the site coordinates used for airmass and time corrections. Managed via the location picker; the database LOCATION table is the source of truth.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `aavso_filter_map` | {} | Static (database) | config.json | config assembly & validation (`config.py:1196`) | Optional mapping of your local filter names to official AAVSO filter codes used in exports (e.g. 'NoFilter' -> 'CV'). |
| `observer_alt_m` | 275.0 | Static (database) | database (LOCATION) | main application UI (`app.py:2045`) | Altitude of the observing site above sea level in meters; part of the site definition used for airmass and time corrections. WAVE-B removed the config.json copy - hydrated from the LOCATION row. |
| `observer_code` | (empty) | Static (database) | config.json | photometry engine (Phase 2A) (`photometry_core.py:9690`) | Official AAVSO observer code stamped into every AAVSO export as #OBSCODE (and printed in reports). A warning is emitted only when this value is missing/empty. |
| `observer_lat` | 50.1121658 | Static (database) | database (LOCATION) | main application UI (`app.py:2043`) | Latitude of the observing site in degrees. Hydrated from the draft's LOCATION record; WAVE-B removed the config.json copy. |
| `observer_location_id` | - | Static (database) | database (LOCATION.IS_DEFAULT) | import page / night runner | Not in config.json: the default site is the LOCATION row with IS_DEFAULT (Database Explorer); a run may choose another site on the import page or with --location-id. |
| `observer_location_name` | (empty) | Static (database) | database (LOCATION) | main application UI (`app.py:2046`) | Human-readable name of the selected observing site (e.g. Jirny, Dablice). WAVE-B removed the config.json copy - hydrated from the LOCATION row. |
| `observer_lon` | 14.6982547 | Static (database) | database (LOCATION) | main application UI (`app.py:2044`) | Longitude of the observing site in degrees; hydrated from the LOCATION row (WAVE-B removed the config.json copy). |
| `observer_name` | Unknown Observer | Static (database) | config.json | photometry engine (Phase 2A) (`photometry_core.py:9691`) | Full observer name printed in reports. |

## File & catalog paths

Where VYVAR finds its data on disk: the star catalogs (Gaia, VSX, exoplanets), the plate-solve indexes, the archive and the calibration library. Machine-specific plumbing - set once, then forget.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `archive_root` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:76`) | Root folder of the observation archive (raw and processed drafts). Resolved per machine at startup. |
| `calibration_library_root` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:233`) | Root folder of the calibration library (master darks/flats organized by camera, binning and temperature). Resolved per machine. |
| `database_path` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:1918`) | Path to the main VYVAR SQLite database (drafts, equipment, locations, results). Resolved per machine. |
| `gaia_db_path` | (empty) | Internal | config.json | comparison-star selection (`comp_selection_per_target.py:151`) | Path to the local full-sky Gaia DR3 SQLite database (40M+ stars) used for star identification and comparison selection. |
| `objects_db_path` | catalogs/vyvar_objects.db | Internal | config.json | catalog access (`catalog_db.py`) | Path to the local objects catalog: table vsx_data (AAVSO VSX variable stars) and table exoplanet_data (NASA Exoplanet Archive hosts), built by catalogs/objects_make.py. Used to identify known variables and exoplanet hosts in the field. |
| `project_root` | VYVAR | Internal | code default only | main application UI (`app.py:2047`) | VYVAR installation root; derived from the code location, never edit. |

## Calibration

Removing the camera signature from raw frames using master darks and flats, plus sanity gates that catch a wrong or stale master before it silently damages the science.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:743`) | Maximum allowed CCD temperature difference (deg C) between a master dark and the light frames it calibrates. |
| `masterdark_validity_days` | 90 | Setting (config.json) | config.json | main application UI (`app.py:1853`) | How many days a master dark stays valid; older masters trigger a warning to shoot new darks (currently ~21.7. deadline). |
| `masterflat_validity_days` | 200 | Setting (config.json) | config.json | main application UI (`app.py:1854`) | How many days a master flat stays valid before VYVAR asks for a fresh one. |

## Frame quality control (QC)

Automatic checks of each frame: star sharpness (FWHM/HFR), elongation, background, minimum star counts, and optional gates that drop bad frames from photometry. Also includes the sky-surface background preprocess.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `auto_fwhm_enabled` | True | Setting (config.json) | config.json | main application UI (`app.py:408`) | Automatically derives the frame-quality FWHM limit from the night's own seeing statistics instead of a fixed number. |
| `auto_fwhm_k_factor` | 1.5 | Setting (config.json) | config.json | main application UI (`app.py:418`) | Multiplier applied to the night's median FWHM when deriving the automatic quality limit. |
| `preprocess_sky_surface_order` | 2; range 0 .. 2 | Setting (config.json) | config.json | config assembly & validation (`config.py:1330`) | Polynomial order of the flux-conserving sky-surface background model subtracted before photometry (2 = gentle 2D gradient removal; part of the draft_435 anchor). |
| `qc_after_calibrate_enabled` | True | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14449`) | Runs the full QC pass right after calibration and stores per-frame quality metrics. |
| `detection_threshold_sigma` | 4.5; range 2 .. 20 | Setting (config.json) | config.json | Settings UI (`ui_settings.py`) | The one star-detection threshold, N x sky sigma, used by every DAO detection (QC, alignment, plate solve, MASTERSTAR, per-frame catalogs). Replaces sips_dao_threshold_sigma, masterstar_dao_threshold_sigma, dao_detection_n_equiv, alignment_detection_sigma and qc_dao_detection_sigma (CFG-CLEAN 2026-10). |

## Alignment

Registering all frames of a series onto a common pixel grid so that each star stays at the same coordinates throughout the night.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `alignment_max_control_points` | 80 | Setting (config.json) | config.json | main application UI (`app.py:2163`) | Maximum number of control points the aligner may use per frame; more is robuster but slower. |
| `alignment_max_stars` | 160; range 10 .. 5000 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:122`) | Cap on stars considered by the alignment matcher. |

## Detection, plate solving & masterstar

Finding stars on the frames, solving the sky coordinates (plate solving, including the blind solver), building the masterstar reference catalog, matching against Gaia, and classifying field density and variability candidates.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `epsf_min_stars` | 30 | Setting (config.json) | config.json | config assembly & validation (`config.py:1335`) | Minimum suitable stars required to build an empirical PSF model (ePSF) for PSF photometry. |
| `exoplanet_match_max_sep_arcsec` | 3.0; range 0.5 .. 30 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:5026`) | Maximum sky separation (arcsec) for identifying a detected star with a known exoplanet host. |
| `frame_height_px` | 1397 | Internal | FITS header | photometry engine (Phase 2A) (`photometry_core.py:14713`) | Frame height in pixels; measured from FITS NAXIS2 at run time. WAVE-B internalized it (no longer stored in config.json). |
| `frame_width_px` | 2082 | Internal | FITS header | photometry engine (Phase 2A) (`photometry_core.py:14712`) | Frame width in pixels; measured from FITS NAXIS1 at run time. WAVE-B internalized it (no longer stored in config.json). |
| `masterstar_best_of_n` | 10; range 1 .. 25 | Setting (config.json) | config.json | config assembly & validation (`config.py:1476`) | How many best frames are stacked/considered when building the masterstar reference. |
| `masterstar_dao_pass2_sigma` | 1.9 | Setting (config.json) | code default only | calibration & frame processing (`pipeline.py:7411`) | Detection sigma of the second, deeper DAO pass on the masterstar stack. |
| `masterstar_use_best_frame_fwhm` | True | Setting (config.json) | code default only | calibration & frame processing (`pipeline.py:11889`) | Uses the best frame's FWHM for masterstar detection kernels instead of an average. |
| `phase01_chip_interior_margin_px` | 50 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14732`) | Margin (px) from the chip edge inside which stars are excluded from comparison selection (edge effects). |
| `phase01_plate_scale_arcsec_per_px` | 1.3; range 0 .. 30 | Dynamic (FITS / runtime) | computed from WCS | photometry engine (Phase 2A) (`photometry_core.py:10063`) | Phase-1 plate scale; resolved from the WCS solution at run time. WAVE-B removed its config.json fallback - the resolver is authoritative. |
| `plate_scale_arcsec_per_px` | 1.3; range 0.1 .. 30 | Dynamic (FITS / runtime) | computed from WCS | photometry engine (Phase 2A) (`photometry_core.py:9934`) | Global plate scale (arcsec/px); resolved from the WCS at run time - the number that converts pixels to sky angles. WAVE-B removed its config.json fallback. |
| `plate_solve_fov_deg` | 1.0 | Dynamic (FITS / runtime) | computed (FITS + DB optics) | main application UI (`app.py:2155`) | Field-of-view estimate (degrees) fed to the plate solver; computed from frame size and optics. |
| `sips_dao_fwhm_px` | 2.5; range 1 .. 8 | Setting (config.json) | config.json | main application UI (`app.py:529`) | Assumed star FWHM (px) for the SIPS-style DAO detection preset. |
## Photometry

Measuring star brightness: aperture sizing, sky annulus, error model, aperture correction, optional PSF photometry and neighbor subtraction, and optional detrending methods. The heart of the pipeline.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `annulus_inner_fwhm` | 2.7; range 1 .. 10 | Setting | config.json | PSF photometry (`psf_photometry.py:1940`) | Inner radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Inner 14 px on draft 516). |
| `annulus_outer_fwhm` | 5.2; range 1.5 .. 12 | Setting | config.json | PSF photometry (`psf_photometry.py:1941`) | Outer radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Outer 27 px on draft 516). |
| `aperture_fwhm_factor` | 1.35; range 0.5 .. 6 | Setting | config.json | per-frame catalogues (`photometry_shared.enhance_catalog_dataframe_aperture`) | Aperture radius of the per-frame catalogues: r = f x frame QC FWHM (APERTURE-01). The light curves use k* per target from the fixed grid 1.0-2.0 x FWHM (APERTURE-K-2026-10). |
| `epsf_auto_run` | False | Setting (config.json) | config.json | night-run orchestration (`night_run.py`) | When true, RUN VYVAR / night_run automatically runs the ePSF stage (**beta**) after aperture photometry. Default OFF. Aperture is validated end-to-end (AstroImageJ 4.86 mmag RMS, 134 epochs, BO CVn). ePSF is internally consistent with aperture on the reference night (era06); independent external and multi-rig validation pending. UI ePSF buttons still force the stage. |
| `epsf_targets` | ALL | Setting (config.json) | config.json | ePSF stage (`epsf_stage.py`, `epsf_science_set.py`) | Targets RUN ePSF measures: `ALL` = every VSX target with aperture photometry (after `vsx_out_of_scope_types`; exoplanet hosts and variability-search candidates are not catalogue variables), or comma-separated VSX types / star names / catalog ids with `*` wildcard (`EA, EB, EW`, `E*`). Used by the night run when `epsf_auto_run` is on; the ePSF tab can override it per run (EPSF-TARGETS-2026-10). |
| `epsf_export_auto` | true | Setting (config.json) | config.json | ePSF stage (`epsf_stage.py`, `epsf_export.py`) | Write the ePSF light-curve files for the user (`photometry/lightcurves_reports/epsf/<star>_<date>_<setup>_epsf.txt`: bjd_tdb, delta_mag, err with a descriptive header) automatically when RUN ePSF finishes. false: the ePSF tab's "Generate ePSF light-curve files" button writes them. Not an AAVSO / VarAstro submission (EPSF-EXPORT-2026-10). |
| `err_empty_apertures_min` | 16 | Setting (config.json) | config.json | config assembly & validation (`config.py:1469`) | Minimum empty apertures required for a valid empirical background estimate. |
| `err_empty_apertures_n` | 64 | Setting (config.json) | config.json | config assembly & validation (`config.py:1462`) | How many empty apertures are placed per frame for the empirical background noise measurement. |
| `gain` | 1.0 | Dynamic (FITS / runtime) | FITS header | calibration & frame processing (`pipeline.py:309`) | Detector gain (e-/ADU) converting counts to electrons in the error model; resolved from the FITS header (cross-checked against the DB). WAVE-B removed its config.json fallback - the resolver is authoritative. |
| `gs11_dilution_aperture_arcsec` | 0.0; range 0 .. 120 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:165`) | Aperture (arcsec) used when computing catalog-based dilution; 0 derives it from the photometric aperture. |
| `gs11_dilution_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:144`) | Enables catalog-based dilution estimation (how much neighbour light leaks into apertures). |
| `gs11_dilution_mag_limit_delta` | 5.0; range 0.5 .. 15 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:183`) | How many magnitudes fainter than the star the dilution census still counts neighbours. |
| `gs11_target_min_dilution` | 0.5; range 0.01 .. 1 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:6096`) | Minimum acceptable target dilution before the target is flagged as badly blended. |
| `lc_outlier_enabled` | True | Setting (config.json) | config.json + Settings UI | LC-OUTLIER-01 (`lc_outlier.py`) | Enable isolated-spike + image-evidence + frame_qc flagging. Never deletes photometry. |
| `lc_outlier_n_sigma` | 5.0 | Setting (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Isolated-spike residual threshold (|r| > n_sigma). Statistical convention. |
| `lc_outlier_adjacent_sigma` | 3.0 | Setting (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Adjacent same-sign residual that protects flares/eclipses. |
| `lc_outlier_frame_qc_n_sigma` | 5.0 | Setting (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Night MAD-sigma for whole-frame QC (FWHM/elong/sky). |
| `lc_outlier_evidence_n_sigma` | 5.0 | Setting (config.json) | config.json + Settings UI | LC-OUTLIER-01 | Image-evidence MAD-sigma vs same-frame similar-flux peers. |
| `lc_high_err_nsigma` | 5.0 | Setting (config.json) | config.json + Settings UI | LC-FLAG-ERR-01 | Flag epoch when `err_i > median(err) + n_sigma x 1.4826 x MAD(err)` on that star's LC. `flag_reason` names the dominant err component. Statistical convention (default 5). |
| `psf_chi2_locus_nsigma` | 5.0 | Setting (config.json) | config.json | config assembly & validation | N-sigma cut on residual from the chi2(flux) locus for psf_fit_ok. |
| `read_noise` | 10.0 | Dynamic (FITS / runtime) | database (EQUIPMENTS) | calibration & frame processing (`pipeline.py:310`) | Detector read noise (electrons) in the error model; resolved DB-first (equipment-intrinsic), then FITS. WAVE-B removed its config.json fallback. |
| `save_lightcurve_png` | False | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7386`) | Also saves per-target light-curve PNG previews during the run. |

## Comparison-star selection

Choosing the ensemble of constant stars against which the target is measured (differential photometry). Criteria: color match (Gaia BP-RP tiers), brightness difference, distance, stability (RMS), isolation and frame coverage.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `comp_contamination_penalty_k` | 3.0; range 0 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:2359`) | Weight penalty strength for contaminated comparison stars in ensemble weighting. |
| `comp_max_delta_bprp` | 0.79; range 0 .. 5 | Setting | config.json | comparison-star selection (`photometry_comp.py`) | Upper colour limit of the comparison ladder (last rung after colour classes 1-3). Wider only through the T4 fallback when fewer than n_comp_min comps exist inside it; also the row-4 boundary of comp_color_tiers. |
| `comp_max_slope_mmag_hr` | 5.0; range 0 .. 500 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:116`) | Maximum linear trend (mmag/hour) a comparison star may show before being rejected as drifting. |
| `comp_select_rms_floor` | 1e-06 | Setting (config.json) | config.json | config assembly & validation (`config.py:2211`) | Numerical floor of comparison RMS in weighting (prevents division blow-ups). |
| `comp_slope_significance_k` | 3.0; range 0 .. 10 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1092`) | Statistical significance required before a comparison star's trend counts as real drift. |
| `comp_catalog_offset_max_mag` | 0.20; range 0 .. 2 (0 = off) | Setting (config.json) | config.json | Phase 2A comparison selection (`comp_smart.smart_select`) | A comparison star whose catalogue minus measured magnitude differs from the other comparison stars by more than this (mag) is excluded and replaced from the reserve (wrong catalogue value, misidentification, unseen neighbour). |
| `comp_color_tiers` | [{bprp: 0.15}, {bprp: 0.3}, {bprp: 0.55}, {bprp: 0.79}] | Setting (config.json) | config.json | comparison-star selection (`photometry_comp.py`) | Colour classes \|BP-RP(comp) - BP-RP(target)\| in row order. Selection uses the first class (rows 1-3, then comp_max_delta_bprp) holding at least n_comp_min comps and orders inside it by LOO RMS, colour, distance. Row 4 is everything above comp_max_delta_bprp (red-target T4 fallback); its 'bprp' follows comp_max_delta_bprp. The ensemble weights are Broeg weights, not set per class. |
| `comp_pool_derived_admission` | True | Setting (config.json) | config.json | global comp pool (`photometry_core.build_global_comp_pool`) | COMP-POOL-01: admit pool via draft-derived noise/stability/dilution criteria. |
| `comparison_stars_pool_n` | 0 | Setting (config.json) | config.json | plan comparison pool (`pipeline.select_comparison_stars_spatial_grid`) | Plan/spatial pool size; 0 = uncapped (COMP-POOL-01). |
| `phase01_comparison_isolation_radius_px` | 25.0; range 1 .. 200 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14762`) | Isolation radius (px): a comparison must have no bright neighbour inside it. |
| `phase01_comparison_max_comp_rms` | 0.1; range 0.01 .. 0.5 | Setting | config.json | photometry engine (Phase 2A) (`photometry_core.py:14750`) | Absolute upper bound on comparison-star LOO mag MAD (INV-COMP-RMS-01). The gate is min(this, k x photon). |
| `phase01_comparison_min_dist_arcsec` | 60.0; range 0 .. 600 | Setting | config.json | photometry engine (Phase 2A) (`photometry_core.py:14751`) | Minimum distance (arcsec) between comparison and target to avoid mutual contamination. |
| `phase01_comparison_min_frames_frac` | 0.2; range 0.05 .. 0.95 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14752`) | Minimum fraction of frames a comparison must be measured on. |
| `phase01_comparison_n_comp_max` | 8 | Setting (config.json) | config.json | check-star handling (`check_star_kmag.py:537`) | Maximum ensemble size; literature shows scintillation gains saturate around 6-8 comparisons. |
| `snr_min` | 10.0; range 3 .. 1000 | Setting (config.json) | config.json | Settings UI (`ui_settings.py`) | Faint limit for every star: aperture SNR below this -> star not measured; a frame where the target SNR falls below it is not exported. 10 = 0.11 mag (SNR-MIN-2026-10). |
| `phase01_comparison_n_comp_min` | 3 | Setting | config.json | photometry engine (Phase 2A) (`photometry_core.py:14748`) | Minimum number of comparison stars per target (the colour ladder widens until it is reached). |
| `phase01_ct_extrapolation_tol` | 0.0 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8927`) | Allowed color-range extrapolation of the color-term relation (0 = no extrapolation). |
| `phase01_ct_min_comp` | 7; range 2 .. 30 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:249`) | Minimum comparisons required to fit a color-term relation. |

## Trust & quality flags

How VYVAR grades its own results: GREEN/YELLOW/RED trust of the comparison ensemble and check star, minimum epochs and frames, and light-curve quality thresholds. Since 2026-06 the min-comps value is the GREEN threshold, not a hard floor - fewer comps degrade gracefully to YELLOW with errors scaled accordingly.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `check_select_rms_floor` | 0.0001; range 0 .. 0.01 | Setting (config.json) | config.json | config assembly & validation (`config.py:2210`) | RMS floor in check-star selection scoring. |
| `check_star_min_epochs` | 5 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1100`) | Minimum epochs the check star must cover for its quality verdict to count. |
| `comp_trust_min_comps` | 5 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1097`) | GREEN trust threshold of ensemble size (default 5 per literature; production runs 3 with sigma scaled by N - see DECISIONS). Fewer comps degrade to YELLOW, not RED. |
| `lc_quality_min_frames` | 20 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1094`) | Minimum frames for a full-quality light-curve verdict. |
| `lc_quality_min_normal_frac` | 0.5; range 0.1 .. 1 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1096`) | Minimum fraction of normal (unflagged) points a light curve needs. |
| `lc_quality_short_min_frames` | 3 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1093`) | Frame floor of the short-baseline quality track for brief series. |
| `sparse_trust_T_green` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2207`) | Check-star T-statistic bound for GREEN trust on sparse fields. |
| `sparse_trust_T_red` | 4.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2208`) | Check-star T-statistic bound beyond which sparse-field trust turns RED. |
| `sparse_trust_X2_RED` | 0.0004 | Setting (config.json) | config.json | config assembly & validation (`config.py:2209`) | Excess-variance bound turning sparse-field trust RED. |
| `trust_flag_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:1183`) | Master switch of the per-epoch trust flag written into results and exports. |

## Atmospheric extinction & color

Correcting for the atmosphere: second-order extinction (k2) that depends on star color and airmass, and optional color-term handling. Mostly relevant for unfiltered or wide-band observations.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `apply_color_term` | auto | Setting (config.json) | config.json | band classification (`band_classify.py:348`) | Whether a color-term transformation is applied to magnitudes ('off' keeps instrumental system). |
| `k2_defaults_bprp` | {} | Setting (config.json) | config.json | config assembly & validation (`config.py:1508`) | Per-band literature defaults of k2 keyed by BP-RP, used when fitting is off or fails. |
| `k2_mode` | literature | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1091`) | Source of k2: 'literature' (defaults) or fitted per night when the fit program is enabled. |

## Reports & HRD

What appears in the PDF summary report: the color HR diagram of the field, online enrichment of interesting objects (SIMBAD, Gaia), and its visual tuning.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `hrd_dsc_confirm_prob` | 0.9; range 0.5 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:803`) | Probability threshold for confirming a candidate's HRD-based classification. |
| `hrd_enrich_max_candidates` | 20; range 1 .. 100 | Setting (config.json) | config.json | config assembly & validation (`config.py:760`) | Cap on objects sent to online enrichment services per report. |
| `hrd_enrich_tap_timeout_s` | 60.0; range 5 .. 120 | Setting (config.json) | config.json | config assembly & validation (`config.py:767`) | Timeout (s) of TAP queries during online enrichment. |
| `hrd_max_per_category` | 3; range 1 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:788`) | Maximum highlighted objects per category in the HRD legend. |
| `hrd_min_per_net` | 4; range 0 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:795`) | Minimum objects kept per detection net when trimming HRD highlights. |
| `hrd_nss_category_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:799`) | Adds the Gaia non-single-star category to HRD highlights. |
| `hrd_online_enrich_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:753`) | Online enrichment of interesting HRD objects from Gaia archive services. |
| `hrd_parallax_min_mas` | 0.15; range 0 .. 10 | Setting (config.json) | config.json | config assembly & validation (`config.py:774`) | Minimum parallax (mas) for placing a star in the absolute-magnitude HRD. |
| `hrd_parallax_snr_min` | 5.0; range 1 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:781`) | Minimum parallax SNR for a trustworthy HRD position. |
| `hrd_simbad_enrich_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:756`) | SIMBAD lookups for object types of HRD highlights. |

## Export

Output for the outside world: AAVSO/VarAstro submission and TESS cross-analysis.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `tess_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:2100`) | Enables the TESS cross-analysis block (comparing your light curve against TESS data). |

## System & performance

Machine-level behavior: parallel workers and RAM reserves. Computed from your hardware; override only when you must.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `per_frame_mp_reserve_ram_gb` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:1028`) | RAM (GB) kept free per worker when sizing per-frame parallelism. |
| `qc_preprocess_workers` | 1 | Internal | environment / machine | calibration & frame processing (`pipeline.py:15384`) | Number of parallel preprocessing workers; computed from CPU/RAM at startup, overridable by the VYVAR_PARALLEL_WORKERS environment variable. |
