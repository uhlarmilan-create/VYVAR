# VYVAR Configuration Guide (config.json) - EN

_Companion document: `VYVAR_CONFIG_GUIDE_CZ.md` (Czech). Generated from the
parameter registry (`dev/validation/params_registry.json`, 55 entries) and
the parameter source audit (`dev/results/PARAM_SOURCE_AUDIT.md`), state as of
2026-10-22 (CODE-REVIEW-2026-10 step 7: 58 keys became fixed constants). When parameters change, update
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
  pipeline. This is the majority of the 55 registered parameters (config.json
  persists 49 of them; the rest are database facts, FITS/runtime values, or
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
  (a typo safety net); the pipeline still runs with defaults for them. Keys
  that were removed on purpose are ignored with a short INFO line instead.
- A number outside its allowed range is clamped to the nearest limit, a value
  that is not a number (or not true / false for an on / off key) keeps the
  default; both with a warning naming the key. The range of every numeric key
  is defined once in the code (`PARAM_RANGES` in `src_py/config.py`) and the
  Settings page and the validator use the same range.
- After editing, check your file with the standalone validator:

    python dev/scripts/validate_config.py

  It reports syntax errors (with line numbers), unknown keys (with
  suggestions), out-of-range values and type mismatches, and exits non-zero
  if anything is wrong.
- Saving from the UI regenerates the file (grouping and comments included)
  from the parameter registry, so any custom comments you add by hand are not
  preserved across a UI save.

## Fixed constants (CODE-REVIEW-2026-10 step 7)

58 former config.json keys that nobody tuned are fixed constants in the code now, with the same
values (the results do not change). A config.json that still carries them loads normally: each
such key is reported once as an INFO line and ignored. The main ones:

| Former keys | Now |
|---|---|
| lc_outlier_n_sigma, lc_outlier_frame_qc_n_sigma, lc_outlier_evidence_n_sigma, lc_high_err_nsigma | one 5-sigma flag threshold (`lc_outlier.OUTLIER_N_SIGMA`); lc_outlier_adjacent_sigma = 3 (`ADJACENT_SIGMA`); flagging is always on |
| lc_quality_short_min_frames, lc_quality_min_normal_frac | 3 frames, 0.5 (defaults of `classify_lc_quality`) |
| hrd_parallax_min_mas, hrd_parallax_snr_min, hrd_dsc_confirm_prob, hrd_enrich_max_candidates, hrd_max_per_category, hrd_min_per_net, hrd_enrich_tap_timeout_s | `hrd_analysis.HRD_*` (0.15 mas, 5, 0.90, 20, 3, 4, 60 s); hrd_simbad_enrich_enabled merged into hrd_online_enrich_enabled |
| masterstar_dao_*, masterstar_forced_seed_*, masterstar_lock_*, masterstar_gaia_census_edge_margin_px, masterstar_use_best_frame_fwhm | `pipeline_constants.MS_*`; forced seeds never enter the comparison pool |
| sparse_trust_T_green / T_red / X2_RED, check_star_min_epochs, check_select_rms_floor, comp_select_rms_floor, comp_rms_loo_photon_k, comp_slope_significance_k, phase01_ct_min_comp, phase01_ct_extrapolation_tol | trust and comparison constants (1.5 / 4.0 / 0.0004, 5, 1e-4, 1e-6, 5, 3 sigma, 7, 0) |
| psf_chi2_locus_nsigma, epsf_min_stars, dao_centroid_max_shift_fwhm, alignment_max_control_points, exoplanet_match_max_sep_arcsec, tess_contam_max, err_empty_apertures_n / _min, gs11_dilution_mag_limit_delta, gs11_target_min_dilution | 5, 30, 1 FWHM, 80, 3 arcsec, 0.10, 64 / 16, 5 mag, 0.5 |
| plate_solve_fov_deg, frame_width_px, frame_height_px, comparison_stars_pool_n, forced_photometry_enabled / _margin_px, preprocess_sky_surface_force_reapply, phase01_comparison_min_dist_arcsec, comp_contamination_penalty_k, gs11_dilution_aperture_arcsec | removed: the value was never used, or always the same (FITS NAXIS, uncapped pool, forced photometry on, one sky-surface pass, geometric separation, target aperture) |

The full list with the reason per key is `KNOWN_REMOVED_KEYS` in `src_py/config.py`.

### Phase B (results may change; each with its own A/B)

- `phase01_comparison_isolation_radius_px` (25 px) is gone together with the crude neighbour-flux
  index it sized and the check-star crowding cut that used it; neighbour light is handled only by
  the comparison pool's aperture blend model (Moffat, |blend| > 1 % removed).
- `phase01_chip_interior_margin_px` (50 px) is gone: a star must have its sky annulus on the chip -
  inside the annulus-aware safe box (targets, comparison stars and now also suspected variables),
  else at least annulus_outer_fwhm x FWHM from the edge.
- `sips_dao_fwhm_px` is gone: the DAO FWHM is each frame's measured VY_FWHM; a frame without one
  uses the fixed 3 px kernel.
- `gain` / `read_noise` are gone from the config: they come from the FITS header or EQUIPMENTS;
  when neither knows them the run stops with an error naming the camera (no silent 1.0 / 10 e-).
- There is no built-in observing site: without the draft's site or a default LOCATION row the
  run stops with an error (before, the Jirny coordinates were a silent fallback).

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
| `observer_code` | (empty) | Static (database) | config.json | aperture photometry (`photometry_core.py:9690`) | Official AAVSO observer code stamped into every AAVSO export as #OBSCODE (and printed in reports). A warning is emitted only when this value is missing/empty. |
| `observer_lat` | 50.1121658 | Static (database) | database (LOCATION) | main application UI (`app.py:2043`) | Latitude of the observing site in degrees. Hydrated from the draft's LOCATION record; WAVE-B removed the config.json copy. |
| `observer_location_id` | - | Static (database) | database (LOCATION.IS_DEFAULT) | import page / night runner | Not in config.json: the default site is the LOCATION row with IS_DEFAULT (Database Explorer); a run may choose another site on the import page or with --location-id. |
| `observer_location_name` | (empty) | Static (database) | database (LOCATION) | main application UI (`app.py:2046`) | Human-readable name of the selected observing site (e.g. Jirny, Dablice). WAVE-B removed the config.json copy - hydrated from the LOCATION row. |
| `observer_lon` | 14.6982547 | Static (database) | database (LOCATION) | main application UI (`app.py:2044`) | Longitude of the observing site in degrees; hydrated from the LOCATION row (WAVE-B removed the config.json copy). |
| `observer_name` | Unknown Observer | Static (database) | config.json | aperture photometry (`photometry_core.py:9691`) | Full observer name printed in reports. |

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
| `alignment_max_stars` | 160; range 10 .. 200 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:122`) | Cap on stars considered by the alignment matcher. |

## Detection, plate solving & masterstar

Finding stars on the frames, solving the sky coordinates (plate solving, including the blind solver), building the masterstar reference catalog, matching against Gaia, and classifying field density and variability candidates.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `masterstar_best_of_n` | 10; range 1 .. 25 | Setting (config.json) | config.json | config assembly & validation (`config.py:1476`) | How many best frames are stacked/considered when building the masterstar reference. |
## Photometry

Measuring star brightness: aperture sizing, sky annulus, error model, aperture correction, optional PSF photometry and neighbor subtraction, and optional detrending methods. The heart of the pipeline.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `annulus_inner_fwhm` | 2.7; range 1 .. 10 | Setting | config.json | PSF photometry (`psf_photometry.py:1940`) | Inner radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Inner 14 px on draft 516). |
| `annulus_outer_fwhm` | 5.2; range 1.5 .. 12 | Setting | config.json | PSF photometry (`psf_photometry.py:1941`) | Outer radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Outer 27 px on draft 516). |
| `aperture_fwhm_factor` | 1.35; range 0.5 .. 6 | Setting | config.json | per-frame catalogues (`photometry_shared.enhance_catalog_dataframe_aperture`) | Aperture radius of the per-frame catalogues: r = f x frame QC FWHM (APERTURE-01). The light curves use k* per target from the fixed grid 1.0-2.0 x FWHM (APERTURE-K-2026-10). |
| `epsf_auto_run` | False | Setting (config.json) | config.json | night-run orchestration (`night_run.py`) | When true, RUN VYVAR / night_run automatically runs the ePSF stage (**beta**) after aperture photometry. Default OFF. Aperture is validated end-to-end (AstroImageJ 4.86 mmag RMS, 134 epochs, BO CVn). ePSF is internally consistent with aperture on the reference night (era06); independent external and multi-rig validation pending. RUN VYVAR from the GUI is the night run (same rule); otherwise the ePSF tab runs the stage only when its RUN ePSF button is pressed. ePSF fits every target with an aperture light curve (`vsx_out_of_scope_types` excludes a target from both photometries; `epsf_targets` was removed 2026-10-31, S6-FIX EPSF-SCOPE-2026-10). |
| `epsf_export_auto` | true | Setting (config.json) | config.json | ePSF stage (`epsf_stage.py`, `epsf_export.py`) | Write the ePSF light-curve files for the user (`photometry/lightcurves_reports/epsf/<star>_<date>_<setup>_epsf.txt`: bjd_tdb, delta_mag, err with a descriptive header) automatically when RUN ePSF finishes. false: the ePSF tab's "Generate ePSF light-curve files" button writes them. Not an AAVSO / VarAstro submission (EPSF-EXPORT-2026-10). |
| `gs11_dilution_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:144`) | Enables catalog-based dilution estimation (how much neighbour light leaks into apertures). |
| `save_lightcurve_png` | False | Setting (config.json) | config.json | aperture photometry (`photometry_core.py:7386`) | Also saves per-target light-curve PNG previews during the run. |

## Comparison-star selection

Choosing the ensemble of constant stars against which the target is measured (differential photometry). Criteria: color match (Gaia BP-RP tiers), brightness difference, distance, stability (RMS), isolation and frame coverage.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `comp_max_delta_bprp` | 0.79; range 0 .. 5 | Setting | config.json | comparison-star selection (`photometry_comp.py`) | Upper colour limit of the comparison ladder (last rung after colour classes 1-3). Wider only through the T4 fallback when fewer than n_comp_min comps exist inside it. |
| `comp_max_slope_mmag_hr` | 5.0; range 0 .. 500 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:116`) | Maximum linear trend (mmag/hour) a comparison star may show before being rejected as drifting. |
| `comp_catalog_offset_max_mag` | 0.20; range 0 .. 2 (0 = off) | Setting (config.json) | config.json | aperture-photometry comparison selection (`comp_smart.smart_select`) | A comparison star whose catalogue minus measured magnitude differs from the other comparison stars by more than this (mag) is excluded and replaced from the reserve (wrong catalogue value, misidentification, unseen neighbour). |
| `comp_color_tiers` | [{bprp: 0.15}, {bprp: 0.3}, {bprp: 0.55}] | Setting (config.json) | config.json | comparison-star selection (`photometry_comp.py`) | Colour classes \|BP-RP(comp) - BP-RP(target)\| in row order. Selection uses the first class (rows 1-3, then comp_max_delta_bprp) holding at least n_comp_min comps and orders inside it by LOO RMS, colour, distance. Above comp_max_delta_bprp is the red-target T4 fallback. The ensemble weights are Broeg weights, not set per class. |
| `phase01_comparison_max_comp_rms` | 0.1; range 0.01 .. 0.5 | Setting | config.json | aperture photometry (`photometry_core.py:14750`) | Absolute upper bound on comparison-star LOO mag MAD (INV-COMP-RMS-01). The gate is min(this, k x photon). |
| `phase01_comparison_min_frames_frac` | 0.2; range 0.05 .. 0.95 | Setting (config.json) | config.json | aperture photometry (`photometry_core.py:14752`) | Minimum fraction of frames a comparison must be measured on. |
| `phase01_comparison_n_comp_max` | 8 | Setting (config.json) | config.json | check-star handling (`check_star_kmag.py:537`) | Maximum ensemble size; literature shows scintillation gains saturate around 6-8 comparisons. |
| `snr_min` | 10.0; range 3 .. 1000 | Setting (config.json) | config.json | Settings UI (`ui_settings.py`) | Faint limit for every star: aperture SNR below this -> star not measured; a frame where the target SNR falls below it is not exported. 10 = 0.11 mag (SNR-MIN-2026-10). |
| `comp_peak_sat_frac` | 0.80; range 0.5 .. 0.9 | Setting (config.json) | config.json | Settings UI (`ui_settings.py`) | Comparison stars: a star whose raw peak reaches this fraction of the measured saturation level on the MASTERSTAR or on any frame is not a comparison star (AAVSO CCD/CMOS guide 3.3.5 / 3.3.7; targets 0.90). COMP-SAT-ONE-2026-10. |
| `phase01_comparison_n_comp_min` | 3 | Setting | config.json | aperture photometry (`photometry_core.py:14748`) | Minimum number of comparison stars per target (the colour ladder widens until it is reached). |

## Trust & quality flags

How VYVAR grades its own results: GREEN/YELLOW/RED trust of the comparison ensemble and check star, minimum epochs and frames, and light-curve quality thresholds. Since 2026-06 the min-comps value is the GREEN threshold, not a hard floor - fewer comps degrade gracefully to YELLOW with errors scaled accordingly.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `comp_trust_min_comps` | 5 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1097`) | GREEN trust threshold of ensemble size (default 5 per literature; production runs 3 with sigma scaled by N - see DECISIONS). Fewer comps degrade to YELLOW, not RED. |
| `lc_quality_min_frames` | 20 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1094`) | Minimum frames for a full-quality light-curve verdict. |

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
| `hrd_nss_category_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:799`) | Adds the Gaia non-single-star category to HRD highlights. |
| `hrd_online_enrich_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:753`) | Online lookups for the highlighted HRD objects: Gaia archive (temperature, gravity) and SIMBAD object type (needs internet). |

## Export

Output for the outside world: AAVSO/VarAstro submission and TESS cross-analysis.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `tess_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:2100`) | Enables the TESS cross-analysis block (comparing your light curve against TESS data). |
| `transient_search_enabled` | False | Setting (config.json) | config.json | field variability search (`variability_field.py`, `transient_field.py`) | Also searches every frame for new sources (novae, supernovae, outbursts), lists stars brighter than Gaia predicts and names the minor planets / comets in the field (MPC files and OpenNGC in the catalogue folder). Off by default. |

## System & performance

Machine-level behavior: parallel workers and RAM reserves. Computed from your hardware; override only when you must.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
