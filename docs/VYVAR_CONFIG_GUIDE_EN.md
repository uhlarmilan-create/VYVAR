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
  hydrated from the LOCATION row selected by observer_location_id), the
  detector gain and read_noise (resolved DB/FITS-first), and the plate-scale
  labels (plate_scale_arcsec_per_px, phase01_plate_scale_arcsec_per_px,
  export_arcsec_per_px, derived from the WCS/optics).
- **Dynamic (FITS / runtime)** - values measured or computed for each run:
  detector gain, read noise, frame size, plate scale, binning, filter,
  exposure time. VYVAR reads them from the FITS file headers or derives
  them (e.g. from the astrometric solution). These no longer carry
  config.json fallbacks after WAVE-B.
- **Setting (config.json)** - the genuine user-tunable behavior of the
  pipeline. This is the majority of the 264 registered parameters (config.json
  persists 249 of them; the rest are database facts, FITS/runtime values, or
  internal plumbing). A few settings are marked "runtime auto-adjust": the
  configured value is a base that the pipeline may adapt to the field (for
  example star-density based loosening/tightening of comparison-star criteria).
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
| `aavso_observer_code` | UMIA | Static (database) | config.json | config assembly & validation (`config.py:1193`) | Your official AAVSO observer code (UMIA); stamped into every AAVSO submission so the observation is credited to you. |
| `observer_alt_m` | 275.0 | Static (database) | database (LOCATION) | main application UI (`app.py:2045`) | Altitude of the observing site above sea level in meters; part of the site definition used for airmass and time corrections. WAVE-B removed the config.json copy - hydrated from the LOCATION row. |
| `observer_code` | (empty) | Static (database) | config.json | photometry engine (Phase 2A) (`photometry_core.py:9690`) | Short observer identifier printed in reports and exports (separate from the AAVSO code). |
| `observer_lat` | 50.1121658 | Static (database) | database (LOCATION) | main application UI (`app.py:2043`) | Latitude of the observing site in degrees. Hydrated from the draft's LOCATION record; WAVE-B removed the config.json copy. |
| `observer_location_id` | 2 | Static (database) | config.json | main application UI (`app.py:1965`) | Database ID of the currently selected observing site (LOCATION table row). This id stays in config.json and drives the hydration of the coordinates. |
| `observer_location_name` | (empty) | Static (database) | database (LOCATION) | main application UI (`app.py:2046`) | Human-readable name of the selected observing site (e.g. Jirny, Dablice). WAVE-B removed the config.json copy - hydrated from the LOCATION row. |
| `observer_lon` | 14.6982547 | Static (database) | database (LOCATION) | main application UI (`app.py:2044`) | Longitude of the observing site in degrees; hydrated from the LOCATION row (WAVE-B removed the config.json copy). |
| `observer_name` | Unknown Observer | Static (database) | config.json | photometry engine (Phase 2A) (`photometry_core.py:9691`) | Full observer name printed in reports. |

## File & catalog paths

Where VYVAR finds its data on disk: the star catalogs (Gaia, VSX, exoplanets), the plate-solve indexes, the archive and the calibration library. Machine-specific plumbing - set once, then forget.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `archive_root` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:76`) | Root folder of the observation archive (raw and processed drafts). Resolved per machine at startup. |
| `blind_index_fine_path` | (empty) | Internal | config.json | Settings UI (`ui_settings.py:1065`) | Path to the fine-scale astrometric index used by the blind plate solver for narrow-field rigs. |
| `blind_index_path` | (empty) | Internal | code default only | Settings UI (`ui_settings.py:1067`) | Legacy single-index path for the blind solver; superseded by the fine/wide pair with automatic selection. |
| `blind_index_select_mode` | auto | Setting (config.json) | config.json | config assembly & validation (`config.py:868`) | How the blind solver picks its index: 'auto' chooses fine or wide by the rig's field of view; can be forced manually. |
| `blind_index_wide_path` | (empty) | Internal | config.json | Settings UI (`ui_settings.py:1066`) | Path to the wide-field astrometric index used by the blind plate solver for wide rigs (e.g. the 200mm Zeiss). |
| `calibration_library_root` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:233`) | Root folder of the calibration library (master darks/flats organized by camera, binning and temperature). Resolved per machine. |
| `database_path` | (resolved at runtime) | Internal | config.json | main application UI (`app.py:1918`) | Path to the main VYVAR SQLite database (drafts, equipment, locations, results). Resolved per machine. |
| `exoplanet_local_db_path` | exoplanets/vyvar_exoplanet_local.db | Internal | config.json | calibration & frame processing (`pipeline.py:4986`) | Path to the local NASA Exoplanet Archive database used to cross-match detected stars with known exoplanet hosts. |
| `gaia_db_path` | (empty) | Internal | config.json | comparison-star selection (`comp_selection_per_target.py:151`) | Path to the local full-sky Gaia DR3 SQLite database (40M+ stars) used for star identification and comparison selection. |
| `project_root` | VYVAR | Internal | code default only | main application UI (`app.py:2047`) | VYVAR installation root; derived from the code location, never edit. |
| `vsx_local_db_path` | (empty) | Internal | config.json | calibration & frame processing (`pipeline.py:8015`) | Path to the local AAVSO VSX database of known variable stars, used to identify variables in the field. |

## Calibration

Removing the camera signature from raw frames using master darks and flats, plus sanity gates that catch a wrong or stale master before it silently damages the science.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `bpm_dark_mad_sigma` | 5.0; range 2 .. 12 | Setting (config.json) | config.json | config assembly & validation (`config.py:1645`) | Sensitivity of bad-pixel detection on the master dark: pixels deviating more than this many robust sigmas are mapped as defective. |
| `calibration_library_native_binning` | 1; range 1 .. 16 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:694`) | Binning in which the calibration library masters were built; masters get resampled (with provenance flag) when a draft uses another binning. |
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:743`) | Maximum allowed CCD temperature difference (deg C) between a master dark and the light frames it calibrates. |
| `dao_qc_in_calibrate` | True | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14476`) | Runs the star-detection based QC directly during calibration so bad frames are flagged as early as possible. |
| `masterdark_validity_days` | 90 | Setting (config.json) | config.json | main application UI (`app.py:1853`) | How many days a master dark stays valid; older masters trigger a warning to shoot new darks (currently ~21.7. deadline). |
| `masterflat_validity_days` | 200 | Setting (config.json) | config.json | main application UI (`app.py:1854`) | How many days a master flat stays valid before VYVAR asks for a fresh one. |

## Frame quality control (QC)

Automatic checks of each frame: star sharpness (FWHM/HFR), elongation, background, minimum star counts, and optional gates that drop bad frames from photometry. Also includes the sky-surface background preprocess.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `auto_fwhm_enabled` | True | Setting (config.json) | config.json | main application UI (`app.py:408`) | Automatically derives the frame-quality FWHM limit from the night's own seeing statistics instead of a fixed number. |
| `auto_fwhm_k_factor` | 1.5 | Setting (config.json) | config.json | main application UI (`app.py:418`) | Multiplier applied to the night's median FWHM when deriving the automatic quality limit. |
| `auto_fwhm_k_max` | 4.0 | Setting (config.json) | config.json | Quality dashboard UI (`ui_quality_dashboard.py:586`) | Upper clamp of the automatic FWHM limit multiplier. |
| `auto_fwhm_k_min` | 1.0 | Setting (config.json) | config.json | Quality dashboard UI (`ui_quality_dashboard.py:585`) | Lower clamp of the automatic FWHM limit multiplier. |
| `frame_align_residual_gate_enabled` | False | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1105`) | Optional gate dropping frames whose alignment residuals are unusually large (poorly registered frames). |
| `frame_align_residual_max_frac` | 0.25; range 0.05 .. 1 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7321`) | Maximum fraction of frames the alignment-residual gate may drop; protects against discarding the night. |
| `frame_align_residual_min_keep_frames` | 10; range 3 .. 100000 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7324`) | Minimum number of frames that must survive the alignment-residual gate. |
| `preprocess_sky_surface_order` | 2; range 0 .. 2 | Setting (config.json) | config.json | config assembly & validation (`config.py:1330`) | Polynomial order of the flux-conserving sky-surface background model subtracted before photometry (2 = gentle 2D gradient removal; part of the draft_435 anchor). |
| `qc_after_calibrate_enabled` | True | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14449`) | Runs the full QC pass right after calibration and stores per-frame quality metrics. |
| `qc_dao_detection_sigma` | 5.0 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14450`) | Detection sensitivity (sigma above background) of the star finder used by QC to count stars per frame. |
| `qc_elong_limit` | 1.8 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:16575`) | Maximum acceptable star elongation; higher values indicate trailing (guiding/wind) and flag the frame. |
| `qc_fwhm_limit` | 8.0 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:16570`) | Absolute upper FWHM limit (px) a frame may have before being flagged as too soft (used when auto-FWHM is off). |
| `qc_max_background_rms` | None | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14470`) | Optional cap on background noise RMS per frame; None disables the check. |
| `qc_max_hfr` | 5.0 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14468`) | Maximum half-flux radius per frame - an alternative sharpness measure familiar from capture software. |
| `qc_min_stars` | 10 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14469`) | Minimum number of detected stars a frame needs to be considered usable. |

## Alignment

Registering all frames of a series onto a common pixel grid so that each star stays at the same coordinates throughout the night.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `alignment_detection_sigma` | 5.0 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:122`) | Detection sensitivity of the star finder used to pick alignment control stars. |
| `alignment_max_control_points` | 80 | Setting (config.json) | config.json | main application UI (`app.py:2163`) | Maximum number of control points the aligner may use per frame; more is robuster but slower. |
| `alignment_max_stars` | 160; range 10 .. 5000 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:122`) | Cap on stars considered by the alignment matcher. |

## Detection, plate solving & masterstar

Finding stars on the frames, solving the sky coordinates (plate solving, including the blind solver), building the masterstar reference catalog, matching against Gaia, and classifying field density and variability candidates.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `blind_img_select_mode` | per_cell | Setting (config.json) | config.json | config assembly & validation (`config.py:941`) | Strategy for picking which detected stars feed the blind solver ('per_cell' spreads them evenly across the frame). |
| `blind_img_star_budget` | 80 | Setting (config.json) | config.json | config assembly & validation (`config.py:935`) | Maximum number of image stars handed to the blind solver; a budget keeps solving fast. |
| `blind_use_rig_prior` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:945`) | Uses the known rig plate scale as a prior to reject implausible blind-solve candidates early. |
| `blind_verify_early_accept` | 30 | Setting (config.json) | config.json | config assembly & validation (`config.py:909`) | Number of verified star matches at which the blind solver stops early and accepts the solution. |
| `blind_verify_early_floor` | 0 | Setting (config.json) | config.json | config assembly & validation (`config.py:916`) | Minimum matches required before early acceptance may trigger at all. |
| `blind_verify_early_fraction` | 0.2; range 0 .. 0.95 | Setting (config.json) | config.json | config assembly & validation (`config.py:923`) | Fraction of catalog stars that must match for early acceptance. |
| `blind_verify_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:873`) | Verifies every blind-solve candidate against the Gaia catalog before trusting it - the safety net of blind solving. |
| `blind_verify_inmemory_catalog` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:900`) | Keeps the verification catalog in memory for speed instead of re-querying the database. |
| `blind_verify_match_tol_px` | 2.5; range 0.5 .. 20 | Setting (config.json) | config.json | blind plate-solve (`vyvar_blind_series.py:215`) | Pixel tolerance when matching image stars to catalog positions during verification. |
| `blind_verify_min_fraction` | 0.15; range 0.05 .. 0.95 | Setting (config.json) | config.json | config assembly & validation (`config.py:894`) | Minimum fraction of stars that must match for a blind solution to pass verification. |
| `blind_verify_min_matches` | 12 | Setting (config.json) | config.json | blind plate-solve (`vyvar_blind_series.py:217`) | Absolute minimum of matched stars for a verified blind solution. |
| `blind_verify_top_n` | 15 | Setting (config.json) | config.json | config assembly & validation (`config.py:875`) | How many best blind candidates get the full verification treatment. |
| `catalog_query_max_rows` | 15000; range 1000 .. 500000 | Setting (config.json) | config.json | config assembly & validation (`config.py:1021`) | Cap on rows returned by a single Gaia catalog query; protects memory on very dense fields. |
| `crowding_blend_tighten_threshold` | 0.04; range 0 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:2076`) | Blend-fraction level above which the (experimental) crowding classifier would tighten comparison criteria. |
| `crowding_classifier_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:2072`) | Experimental crowding classifier switch; OFF pending validation on Newton dense-field data. |
| `crowding_comp_availability_loosen_count` | 500.0; range 0 .. 1000000 | Setting (config.json) | config.json | config assembly & validation (`config.py:2083`) | Comparison-star availability level treated as plentiful when the crowding logic decides about loosening. |
| `crowding_tighten_min_fwhm_px` | 3.0; range 0 .. 30 | Setting (config.json) | config.json | config assembly & validation (`config.py:2092`) | Minimum FWHM (px) below which crowding-driven tightening is not applied (undersampled images). |
| `debug_platesolver` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1666`) | Verbose diagnostic logging of the plate solver; for troubleshooting only. |
| `epsf_min_stars` | 30 | Setting (config.json) | config.json | config assembly & validation (`config.py:1335`) | Minimum suitable stars required to build an empirical PSF model (ePSF) for PSF photometry. |
| `exoplanet_match_max_sep_arcsec` | 3.0; range 0.5 .. 30 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:5026`) | Maximum sky separation (arcsec) for identifying a detected star with a known exoplanet host. |
| `field_density_adaptive_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:2069`) | Master switch of density adaptation: sparse/normal/dense field profiles automatically loosen or tighten comparison-star criteria. |
| `field_density_dense_threshold` | 1000.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2060`) | Matched-star count above which the field is treated as dense (tighter criteria). |
| `field_density_sparse_threshold` | 300.0; range 1 .. 50000 | Setting (config.json) | config.json | config assembly & validation (`config.py:2053`) | Matched-star count below which the field is treated as sparse (looser criteria so an ensemble can still be formed). |
| `frame_height_px` | 1397 | Internal | FITS header | photometry engine (Phase 2A) (`photometry_core.py:14713`) | Frame height in pixels; measured from FITS NAXIS2 at run time. WAVE-B internalized it (no longer stored in config.json). |
| `frame_width_px` | 2082 | Internal | FITS header | photometry engine (Phase 2A) (`photometry_core.py:14712`) | Frame width in pixels; measured from FITS NAXIS1 at run time. WAVE-B internalized it (no longer stored in config.json). |
| `masterstar_accept_mode` | odds | Setting (config.json) | config.json | config assembly & validation (`config.py:1775`) | Acceptance strategy of the masterstar plate solution ('odds' = statistical odds-ratio test). |
| `masterstar_best_of_n` | 10; range 1 .. 25 | Setting (config.json) | config.json | config assembly & validation (`config.py:1476`) | How many best frames are stacked/considered when building the masterstar reference. |
| `masterstar_catalog_recovery_min` | 0.65; range 0.4 .. 0.95 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:353`) | Minimum fraction of catalog stars the masterstar must recover for the solution to be trusted. |
| `masterstar_centre_rms_max_px` | 1.2; range 0.5 .. 5 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:355`) | Maximum RMS (px) of star positions near the frame centre for an acceptable astrometric solution. |
| `masterstar_dao_pass2_sigma` | 1.9 | Setting (config.json) | code default only | calibration & frame processing (`pipeline.py:7411`) | Detection sigma of the second, deeper DAO pass on the masterstar stack. |
| `masterstar_dao_threshold_sigma` | 2.1; range 0.1 .. 6 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:13137`) | Primary DAO detection threshold (sigma) on the masterstar; lower finds fainter stars but more noise. |
| `masterstar_detection_cap_adaptive` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:1816`) | Adapts the detection cap to field density instead of a fixed count. |
| `masterstar_detection_cap_k` | 0.08; range 0.01 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:1837`) | Scaling constant of the adaptive detection cap. |
| `masterstar_detection_cap_max` | 800 | Setting (config.json) | config.json | config assembly & validation (`config.py:1827`) | Upper clamp of the adaptive detection cap. |
| `masterstar_detection_cap_min` | 250 | Setting (config.json) | config.json | config assembly & validation (`config.py:1820`) | Lower clamp of the adaptive detection cap. |
| `masterstar_distortion_benign_ratio_max` | 3.2; range 2 .. 5 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:356`) | Limit on the edge-to-centre distortion ratio still considered benign for the optics. |
| `masterstar_min_matched_floor` | 40 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:354`) | Absolute floor of matched stars a masterstar solution must reach. |
| `masterstar_platesolve_sip_max_order` | 4 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:351`) | Highest SIP distortion polynomial order the solver may fit. |
| `masterstar_platesolve_sip_min_order` | 3 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:352`) | Lowest SIP distortion order tried by the solver. |
| `masterstar_prematch_peak_sigma_floor` | 1.8; range 0.5 .. 6 | Setting (config.json) | config.json | calibration & frame processing (`pipeline.py:14006`) | Minimum peak significance of stars used in the pre-match stage. |
| `masterstar_quality_crowded_n_cat_min` | 800 | Setting (config.json) | config.json | config assembly & validation (`config.py:1808`) | Catalog-star count above which the masterstar quality checks switch to crowded-field mode. |
| `masterstar_sibling_min_matched` | 40 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:358`) | Minimum matches a sibling (recovery) solution needs. |
| `masterstar_sibling_min_quadrants` | 3 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:360`) | Quadrant coverage required from a sibling solution. |
| `masterstar_sibling_recovery_enabled` | True | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:357`) | Enables the sibling-stack recovery path when the primary masterstar solve fails. |
| `masterstar_sibling_rms_max_px` | 2.0; range 0.5 .. 10 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:359`) | RMS limit for accepting a sibling recovery solution. |
| `masterstar_sibling_stack_n` | 10 | Setting (config.json) | config.json | Masterstar/DAO UI (`ui_dao_stars.py:361`) | How many frames the sibling recovery stack combines. |
| `masterstar_use_best_frame_fwhm` | True | Setting (config.json) | code default only | calibration & frame processing (`pipeline.py:11889`) | Uses the best frame's FWHM for masterstar detection kernels instead of an average. |
| `phase01_chip_interior_margin_px` | 50 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14732`) | Margin (px) from the chip edge inside which stars are excluded from comparison selection (edge effects). |
| `phase01_plate_scale_arcsec_per_px` | 1.3; range 0 .. 30 | Dynamic (FITS / runtime) | computed from WCS | photometry engine (Phase 2A) (`photometry_core.py:10063`) | Phase-1 plate scale; resolved from the WCS solution at run time. WAVE-B removed its config.json fallback - the resolver is authoritative. |
| `plate_scale_arcsec_per_px` | 1.3; range 0.1 .. 30 | Dynamic (FITS / runtime) | computed from WCS | photometry engine (Phase 2A) (`photometry_core.py:9934`) | Global plate scale (arcsec/px); resolved from the WCS at run time - the number that converts pixels to sky angles. WAVE-B removed its config.json fallback. |
| `plate_solve_fov_deg` | 1.0 | Dynamic (FITS / runtime) | computed (FITS + DB optics) | main application UI (`app.py:2155`) | Field-of-view estimate (degrees) fed to the plate solver; computed from frame size and optics. |
| `saturate_limit_fraction` | 0.80 | Setting (config.json) | code default only | calibration & frame processing (`pipeline.py:6097`) | Fraction of the detector's saturation level above which a star is treated as saturated and excluded from photometry. INV-SAT-LIMIT authority is 0.80 x 65535 = 52428 ADU when the linearity knee is unmeasured. |
| `sips_dao_fwhm_px` | 2.5; range 1 .. 8 | Setting (config.json) | config.json | main application UI (`app.py:529`) | Assumed star FWHM (px) for the SIPS-style DAO detection preset. |
| `sips_dao_threshold_sigma` | 3.5 | Setting (config.json) | config.json | main application UI (`app.py:530`) | Detection threshold (sigma) of the SIPS-style DAO preset. |
| `variability_clip_ratio_min` | 0.8 | Setting (config.json) | config.json | config assembly & validation (`config.py:2420`) | Minimum surviving-points ratio after sigma clipping for a star to stay in variability analysis. |
| `variability_comp_floor_factor` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2415`) | How many times above the comparison-star noise floor a star must scatter to count as variable. |
| `variability_mag_limit` | 14.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2417`) | Faint magnitude limit of the variability search. |
| `variability_min_amplitude_mag` | 0.01 | Setting (config.json) | config.json | config assembly & validation (`config.py:2419`) | Minimum amplitude (mag) for a variability candidate. |
| `variability_min_frames` | 30 | Setting (config.json) | config.json | config assembly & validation (`config.py:2010`) | Minimum frames a star must appear on to be analysed for variability. |
| `variability_min_frames_frac` | 0.5; range 0.05 .. 0.99 | Setting (config.json) | config.json | config assembly & validation (`config.py:2016`) | Minimum fraction of the night's frames a star must cover for variability analysis. |
| `variability_min_points_rms` | 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:2422`) | Minimum points needed before an RMS-based variability statistic is computed. |
| `variability_min_rms_pct` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2418`) | Percentile floor of the RMS-vs-magnitude relation used to normalize variability scores. |
| `variability_p85_filter` | 85 | Setting (config.json) | config.json | config assembly & validation (`config.py:2412`) | Percentile filter removing the noisiest tail before variability statistics. |
| `variability_sigma_threshold` | 2.3 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:5519`) | Significance (sigma) a star's scatter must exceed to be flagged as a variability candidate. |
| `variability_slope_floor` | 0.02 | Setting (config.json) | config.json | config assembly & validation (`config.py:2413`) | Minimum slope of the excess-scatter trend considered meaningful in candidate scoring. |
| `variability_smoothness_max` | 0.8 | Setting (config.json) | config.json | config assembly & validation (`config.py:2416`) | Maximum smoothness score - very smooth curves are more likely trends/artifacts than stellar variability. |
| `variability_vdi_z_threshold` | 3.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2421`) | Z-score threshold of the variability detection index (VDI). |
| `verify_mag_limit` | 14.0; range 8 .. 18 | Setting (config.json) | config.json | config assembly & validation (`config.py:904`) | Faint limit of catalog stars used for blind-solve verification. |
## Photometry

Measuring star brightness: aperture sizing, sky annulus, error model, aperture correction, optional PSF photometry and neighbor subtraction, and optional detrending methods. The heart of the pipeline.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `annulus_inner_fwhm` | 2.7; range 1 .. 10 | Setting + runtime auto-adjust | config.json | PSF photometry (`psf_photometry.py:1940`) | Inner radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Inner 14 px on draft 516). Density adaptation may tighten it on dense fields. |
| `annulus_outer_fwhm` | 5.2; range 1.5 .. 12 | Setting + runtime auto-adjust | config.json | PSF photometry (`psf_photometry.py:1941`) | Outer radius of the sky annulus in FWHM units (APERTURE-01d; AIJ Sky_Outer 27 px on draft 516). |
| `aperture_comp_factor` | 1.1; range 0.25 .. 3 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7092`) | Aperture size multiplier applied to comparison stars. |
| `aperture_correction_enabled` | True | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8650`) | Enables aperture correction: transferring flux measured in a small aperture to the total-flux scale using bright reference stars. |
| `aperture_correction_max_contamination` | 0.15; range 0 .. 2 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8656`) | Maximum neighbour contamination a reference star may have to be used for aperture correction. |
| `aperture_correction_max_scatter_mag` | 0.03; range 0 .. 2 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8657`) | Maximum scatter (mag) allowed among aperture-correction reference stars. |
| `aperture_correction_min_ref_stars` | 3; range 1 .. 50 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8655`) | Minimum reference stars required to compute an aperture correction. |
| `aperture_fwhm_factor` | 1.9; range 0.5 .. 6 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:7393`) | Base aperture radius in FWHM units; the SNR-optimal sizing sweep may adapt the effective radius per star. |
| `aperture_snr_sizing` | {small: 1.5, large: 4.0} | Setting + runtime auto-adjust | config.json | pipeline (`pipeline.py`) | SNR-optimal aperture sizing sweep bounds as FWHM multiples: 'small' is the minimum radius (best for faint stars), 'large' the maximum. WAVE-B merged the former aperture_fwhm_factor_small/_large scalars into this mapping (there is no medium class). Consumed by the pipeline aperture-bounds path (`pipeline.py:187-188`); NOT by the SNR-optimal aperture sweep, which uses hardcoded 0.8/2.5 x FWHM in `compute_snr_optimal_aperture_table`. |
| `aperture_photometry_enabled` | True | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7899`) | Master switch of aperture photometry - the production measurement method of VYVAR. |
| `aperture_variable_factor` | 1.0; range 0.25 .. 3 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7091`) | Aperture size multiplier applied to the variable (target) star. |
| `cog_ac_factor_max` | 5.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1439`) | Upper clamp of the curve-of-growth aperture-correction factor. |
| `cog_aperture_correction_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1405`) | Enables the curve-of-growth variant of aperture correction (experimental alternative). |
| `cog_isolation_fwhm` | 6.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1419`) | Isolation radius (FWHM) a star needs to serve as a curve-of-growth reference. |
| `cog_ladder_step_px` | 0.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:1434`) | Radius step (px) of the curve-of-growth aperture ladder. |
| `cog_min_stars` | 8; range 1 .. 500 | Setting (config.json) | config.json | config assembly & validation (`config.py:1415`) | Minimum stars needed to fit a curve of growth. |
| `cog_ref_fwhm` | 4.5; range 1.5 .. 10 | Setting (config.json) | config.json | config assembly & validation (`config.py:1409`) | Reference FWHM used to normalize the curve of growth. |
| `cog_sat_frac` | 0.85 | Setting (config.json) | config.json | config assembly & validation (`config.py:1429`) | Saturation-fraction cutoff for curve-of-growth reference stars. |
| `cog_snr_min` | 50.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1424`) | Minimum SNR of a curve-of-growth reference star. |
| `democratic_detrend_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:323`) | Optional 'democratic' detrend of the light curve (median trend of the field); OFF by default - detrending can eat real variability. |
| `democratic_sg_window_frac` | 0.5; range 0.05 .. 0.95 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:329`) | Window width (fraction of the night) of the democratic detrend smoother. |
| `epsf_auto_run` | False | Setting (config.json) | config.json | night-run orchestration (`night_run.py`) | When true, RUN VYVAR / night_run automatically runs the ePSF stage after aperture photometry. Default OFF. Distinct from `psf_photometry_enabled` (Phase 2A psf_* columns). UI ePSF buttons still force the stage. |
| `err_empty_apertures_min` | 16 | Setting (config.json) | config.json | config assembly & validation (`config.py:1469`) | Minimum empty apertures required for a valid empirical background estimate. |
| `err_empty_apertures_n` | 64 | Setting (config.json) | config.json | config assembly & validation (`config.py:1462`) | How many empty apertures are placed per frame for the empirical background noise measurement. |
| `gain` | 1.0 | Dynamic (FITS / runtime) | FITS header | calibration & frame processing (`pipeline.py:309`) | Detector gain (e-/ADU) converting counts to electrons in the error model; resolved from the FITS header (cross-checked against the DB). WAVE-B removed its config.json fallback - the resolver is authoritative. |
| `gs11_comp_max_dilution` | 0.9; range 0.01 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:1944`) | Maximum dilution (flux contamination) a comparison star may have under the GS11 dilution model. |
| `gs11_comp_suspect_dilution` | 0.98; range 0.01 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:1944`) | Dilution level at which a comparison star is marked suspect. |
| `gs11_dilution_aperture_arcsec` | 0.0; range 0 .. 120 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:165`) | Aperture (arcsec) used when computing catalog-based dilution; 0 derives it from the photometric aperture. |
| `gs11_dilution_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:144`) | Enables catalog-based dilution estimation (how much neighbour light leaks into apertures). |
| `gs11_dilution_mag_limit_delta` | 5.0; range 0.5 .. 15 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:183`) | How many magnitudes fainter than the star the dilution census still counts neighbours. |
| `gs11_target_min_dilution` | 0.5; range 0.01 .. 1 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:6096`) | Minimum acceptable target dilution before the target is flagged as badly blended. |
| `neighbor_sub_centroid_max_fwhm` | 1.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2233`) | Maximum centroid shift (FWHM) allowed after neighbour subtraction. |
| `neighbor_sub_chi2_max` | 120.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2230`) | Chi-square cap of the neighbour-model fit. |
| `neighbor_sub_max_neighbor_overmag` | 0.3 | Setting (config.json) | config.json | config assembly & validation (`config.py:2235`) | Guard: the fitted neighbour must not come out brighter than expected by more than this (mag). |
| `neighbor_sub_max_target_undermag` | 0.2 | Setting (config.json) | config.json | config assembly & validation (`config.py:2236`) | Guard: the target must not lose more than this (mag) by the subtraction. |
| `neighbor_sub_min_recovered_snr` | 5.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2237`) | Minimum SNR the target must retain after neighbour subtraction. |
| `neighbor_sub_nn_contam_dmag` | 2.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2234`) | Brightness difference within which a nearest neighbour counts as contaminating. |
| `neighbor_sub_refuse_sep_fwhm` | 0.8 | Setting (config.json) | config.json | config assembly & validation (`config.py:2232`) | Below this separation (FWHM) subtraction is refused - the pair is too blended to fix. |
| `neighbor_sub_regime_dmag_min` | 2.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:2238`) | Brightness-difference bound defining the regime where neighbour subtraction applies. |
| `neighbor_sub_regime_sep_max` | 1.1 | Setting (config.json) | config.json | config assembly & validation (`config.py:2239`) | Separation bound (FWHM) of the neighbour-subtraction regime. |
| `neighbor_sub_residual_rms_max` | 150.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:2231`) | Residual RMS cap after subtraction for the result to be accepted. |
| `nonlinearity_fwhm_ratio` | 1.25; range 1.01 .. 3 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:713`) | FWHM ratio threshold of the detector nonlinearity diagnostic (bright stars growing fatter than faint ones). |
| `nonlinearity_peak_percentile` | 20.0; range 0 .. 50 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:706`) | Peak-brightness percentile at which the nonlinearity diagnostic samples star shapes. |
| `phase2a_airmass_before_outlier` | False | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:4411`) | Order switch: apply the airmass detrend before (True) or after (False) outlier rejection in Phase 2A. |
| `photometry_mode` | both | Setting (config.json) | config.json | Photometry UI (`ui_photometry.py:51`) | Which measurement families run: aperture, PSF, or both. |
| `psf_ac_policy` | p4_none | Setting (config.json) | config.json | F6 ePSF merge (`epsf_psf_merge.py`, `psf_photometry.py`) | F6 aperture-correction policy. `p4_none` stamps uncorrected fit flux (`psf_ac_factor=1`). `chi2_lt5_legacy` is the named fallback: median DAO/PSF among chi2<5 stars (EPSF-AC-01 A2 brightness-cut defect). |
| `psf_zp_membership` | fit_ok_for_zp | Setting (config.json) | config.json | internal PSF LC (`psf_internal_lc.py`) | INV-PSF-LC-PIN-01 ZP membership. `fit_ok_strict` uses stored `psf_fit_ok`. `fit_ok_for_zp` also admits epochs with finite `psf_flux>0` and finite `psf_chi2`. Production default is `fit_ok_for_zp` on validated rigs only. |
| `psf_zp_for_zp_validated_rigs` | ["1:1"] | Setting (config.json) | config.json | internal PSF LC (`psf_internal_lc.py`) | Rig identity keys `equipment_id:telescope_id` allowed to use `fit_ok_for_zp`. Draft 516/517 wide pair is `1:1`. Unlisted rigs stay `fit_ok_strict` (EPSF-ZP-OK-XRIG-01). |
| `psf_adaptive_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1310`) | Adaptive per-star routing between PSF and aperture measurement (PSF program is OFF pending the Newton dense-field gate). |
| `psf_adaptive_resolve_fwhm` | 2.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1312`) | Separation (FWHM) below which the adaptive router prefers PSF for blended pairs. |
| `psf_adaptive_snr_lo` | 15.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1318`) | SNR below which the adaptive router prefers PSF measurement. |
| `psf_chi2_threshold` | 50.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1264`) | Chi-square limit of an acceptable PSF fit. |
| `psf_group_sep_fwhm` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:1270`) | Separation (FWHM) within which stars are fitted together as a PSF group. |
| `psf_grouper_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1268`) | Enables simultaneous group fitting of close stars in PSF photometry. |
| `psf_neighbor_include_fwhm` | 3.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1275`) | Radius (FWHM) within which neighbours are included in a PSF fit. |
| `psf_neighbor_sub_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1279`) | Enables PSF-model neighbour subtraction before aperture measurement of blended targets. |
| `psf_photometry_enabled` | False | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8686`) | Master switch of PSF photometry (currently OFF; enablement gated on the Newton dense-field validation). |
| `psf_quality_fallback_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:1307`) | Falls back to the aperture measurement when a PSF fit fails quality checks. |
| `psf_spatial_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1299`) | Enables spatially varying PSF models across the frame. |
| `psf_spatial_grid` | 3x3 | Setting (config.json) | config.json | config assembly & validation (`config.py:1300`) | Grid of spatial PSF cells (e.g. 3x3). |
| `psf_spatial_min_stars_per_cell` | 25 | Setting (config.json) | config.json | config assembly & validation (`config.py:1303`) | Minimum stars per cell to build a local PSF model. |
| `psf_spatial_order` | 0; range 0 .. 2 | Setting (config.json) | config.json | config assembly & validation (`config.py:1259`) | Polynomial order of PSF spatial variation (0 = constant PSF). |
| `pytics_enabled` | True | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:125`) | Enables the PyTICS-style iterative comparison calibration step of light-curve construction. |
| `pytics_n_iter` | 5; range 1 .. 20 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:124`) | Iteration count of the PyTICS-style calibration. |
| `read_noise` | 10.0 | Dynamic (FITS / runtime) | database (EQUIPMENTS) | calibration & frame processing (`pipeline.py:310`) | Detector read noise (electrons) in the error model; resolved DB-first (equipment-intrinsic), then FITS. WAVE-B removed its config.json fallback. |
| `save_lightcurve_png` | False | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:7386`) | Also saves per-target light-curve PNG previews during the run. |
| `savgol_detrend_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:309`) | Optional Savitzky-Golay smoothing detrend; OFF by default for the same reason as other detrends. |
| `savgol_polyorder` | 2 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:315`) | Polynomial order of the Savitzky-Golay filter. |
| `savgol_window_frac` | 0.5 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:314`) | Window width (fraction of the series) of the Savitzky-Golay filter. |
| `sigma_sys_mag` | {} | Setting (config.json) | config.json | config assembly & validation (`config.py:1519`) | Per-band systematic error floor (mag) added in quadrature to statistical errors (e.g. {'4': 0.018} for band 4). |
| `sysrem_enabled` | False | Setting (config.json) | config.json | night-run orchestration (`night_run.py:527`) | Optional SysRem systematics removal (Tamuz+ 2005); OFF by default - validated as risky for preserving real variability. |
| `sysrem_n_iter` | 3 | Setting (config.json) | config.json | night-run orchestration (`night_run.py:529`) | Number of SysRem iterations when enabled. |
| `temporal_bin_window` | 0; range 0 .. 51 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:106`) | Time-bin width for temporal binning (0 = none). |
| `temporal_binning_enabled` | False | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:107`) | Temporal binning of light curves; OFF by design - injection testing proved it harmful (24/25 targets worse). |

## Comparison-star selection

Choosing the ensemble of constant stars against which the target is measured (differential photometry). Criteria: color match (Gaia BP-RP tiers), brightness difference, distance, stability (RMS), isolation and frame coverage. Several limits adapt automatically to field density.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `comp_contamination_penalty_k` | 3.0; range 0 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:2359`) | Weight penalty strength for contaminated comparison stars in ensemble weighting. |
| `comp_iterative_clip_enabled` | False | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1129`) | Iterative re-clipping of the comparison ensemble (drop-and-reweigh loop); ON in production since the Brno fix of 2026-06-14. |
| `comp_max_delta_bprp` | 0.79; range 0 .. 5 | Setting + runtime auto-adjust | config.json | comparison-star selection (`comp_selection_per_target.py:239`) | Maximum Gaia BP-RP color difference between a comparison star and the target - the primary defense against extinction systematics on unfiltered data; density adaptation adjusts it. |
| `comp_max_slope_mmag_hr` | 5.0; range 0 .. 500 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:116`) | Maximum linear trend (mmag/hour) a comparison star may show before being rejected as drifting. |
| `comp_select_rms_floor` | 1e-06 | Setting (config.json) | config.json | config assembly & validation (`config.py:2211`) | Numerical floor of comparison RMS in weighting (prevents division blow-ups). |
| `comp_slope_significance_k` | 3.0; range 0 .. 10 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1092`) | Statistical significance required before a comparison star's trend counts as real drift. |
| `comp_sparse_fallback_enabled` | True | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1128`) | Allows the sparse-field fallback path when strict criteria yield too few comparisons. |
| `comp_sparse_fallback_min` | 0 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1131`) | Minimum comparisons the sparse fallback aims for (0 = take what exists). |
| `comp_color_tiers` | [{bprp: 0.15, w: 1.0}, {bprp: 0.3, w: 0.85}, {bprp: 0.55, w: 0.5}, {bprp: 1.1, w: 0.25}] | Setting (config.json) | config.json | comparison-star selection (`comp_selection_per_target.py`) | Ordered color-match tiers for comparison weighting. Each entry is a BP-RP color-match limit ('bprp', tighter = better color match) and the ensemble weight ('w') for stars in that tier. WAVE-B merged the eight comp_tier{1..4}_bprp_limit / _weight scalars into this list of dicts. |
| `comp_pool_derived_admission` | True | Setting (config.json) | config.json | global comp pool (`photometry_core.build_global_comp_pool`) | COMP-POOL-01: admit pool via draft-derived noise/stability/dilution criteria. |
| `comparison_stars_pool_n` | 0 | Setting (config.json) | config.json | plan comparison pool (`pipeline.select_comparison_stars_spatial_grid`) | Plan/spatial pool size; 0 = uncapped (COMP-POOL-01). |
| `phase01_comparison_exclude_gaia_extobj` | True | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:12656`) | Excludes Gaia-flagged extended objects (galaxies) from comparison candidates. |
| `phase01_comparison_exclude_gaia_nss` | True | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:12654`) | Excludes Gaia non-single-star (binary) sources from comparison candidates. |
| `phase01_comparison_fov_fraction` | 0.75 | Setting (config.json) | code default only | photometry engine (Phase 2A) (`photometry_core.py:14739`) | Fraction of the field of view around the target within which comparisons are searched. |
| `phase01_comparison_isolation_radius_px` | 25.0; range 1 .. 200 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14762`) | Isolation radius (px): a comparison must have no bright neighbour inside it. |
| `phase01_comparison_mag_bright_threshold` | 12.75; range 6 .. 18 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14756`) | Brightness above which the stricter bright-star mag-difference floor applies. |
| `phase01_comparison_max_comp_rms` | 0.1; range 0.01 .. 0.5 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:14750`) | Maximum night RMS of a comparison-star series; density adaptation may tighten it. |
| `phase01_comparison_max_dist_deg` | 1.5; range 0.05 .. 10 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:14740`) | Maximum sky distance (deg) of a comparison from the target; density adaptation adds to the FOV-derived base. |
| `phase01_comparison_max_fwhm_factor` | 1.5; range 0.5 .. 5 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14761`) | Maximum FWHM ratio vs the field median a comparison may have (rejects defocused/blended shapes). |
| `phase01_comparison_max_mag_diff` | 1.5; range 0.05 .. 5 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:14742`) | Base maximum brightness difference (mag) between comparison and target; adapted by density profile. |
| `phase01_comparison_max_mag_diff_absolute` | 3.0; range 1 .. 10 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1113`) | Hard ceiling of the brightness difference no adaptation may exceed. |
| `phase01_comparison_max_mag_diff_bright_floor` | 1.5; range 0 .. 4 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14758`) | Mag-difference floor applied to bright stars regardless of adaptation. |
| `phase01_comparison_max_psf_chi2` | 50.0; range 1 .. 500 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14760`) | Maximum PSF-fit chi-square of a comparison star (shape sanity). |
| `phase01_comparison_min_dist_arcsec` | 60.0; range 0 .. 600 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:14751`) | Minimum distance (arcsec) between comparison and target to avoid mutual contamination; density-adapted. |
| `phase01_comparison_min_frames_frac` | 0.2; range 0.05 .. 0.95 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14752`) | Minimum fraction of frames a comparison must be measured on. |
| `phase01_comparison_n_comp_max` | 8 | Setting (config.json) | config.json | check-star handling (`check_star_kmag.py:537`) | Maximum ensemble size; literature shows scintillation gains saturate around 6-8 comparisons. |
| `phase01_comparison_n_comp_min` | 3 | Setting + runtime auto-adjust | config.json | photometry engine (Phase 2A) (`photometry_core.py:14748`) | Minimum comparisons the selector aims for; density adaptation may lower it on sparse fields. |
| `phase01_ct_extrapolation_tol` | 0.0 | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:8927`) | Allowed color-range extrapolation of the color-term relation (0 = no extrapolation). |
| `phase01_ct_min_comp` | 7; range 2 .. 30 | Setting (config.json) | config.json | light-curve construction (`method_lc_output.py:249`) | Minimum comparisons required to fit a color-term relation. |
| `phase01_flux_col` | dao_flux | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py:14763`) | Which flux column feeds Phase-1 comparison statistics (dao_flux = detection-stage flux). |
| `phase01_tiers` | [0.5, 1.0, 1.5, 2.0] | Setting (config.json) | config.json | photometry engine (Phase 2A) (`photometry_core.py`) | Brightness-difference bounds (mag) of the candidate magnitude tiers, ascending. WAVE-B merged the phase01_tier{1..4}_mag scalars into this list. |
| `phase01_use_bprp_primary` | True | Setting (config.json) | config.json | Aperture photometry UI (`ui_aperture_photometry.py:1701`) | Uses Gaia BP-RP directly as the primary color criterion (instead of a computed B-V) - the grounded design choice of VYVAR. |

## Trust & quality flags

How VYVAR grades its own results: GREEN/YELLOW/RED trust of the comparison ensemble and check star, minimum epochs and frames, and light-curve quality thresholds. Since 2026-06 the min-comps value is the GREEN threshold, not a hard floor - fewer comps degrade gracefully to YELLOW with errors scaled accordingly.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `check_select_rms_floor` | 0.0001; range 0 .. 0.01 | Setting (config.json) | config.json | config assembly & validation (`config.py:2210`) | RMS floor in check-star selection scoring. |
| `check_star_min_epochs` | 5 | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1100`) | Minimum epochs the check star must cover for its quality verdict to count. |
| `comp_qa_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:1182`) | Per-epoch quality assessment of the comparison ensemble feeding the trust verdict. |
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
| `apply_color_term` | off | Setting (config.json) | config.json | band classification (`band_classify.py:348`) | Whether a color-term transformation is applied to magnitudes ('off' keeps instrumental system). |
| `k2_ceiling` | 0.1 | Setting (config.json) | config.json | config assembly & validation (`config.py:1532`) | Upper cap of the fitted second-order extinction coefficient k2 (mag per airmass per color unit). |
| `k2_defaults_bprp` | {} | Setting (config.json) | config.json | config assembly & validation (`config.py:1508`) | Per-band literature defaults of k2 keyed by BP-RP, used when fitting is off or fails. |
| `k2_fit_consistency_sigma` | 2.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1542`) | Consistency requirement (sigma) between the fitted k2 and expectations before the fit is trusted. |
| `k2_fit_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:1530`) | Enables per-night fitting of the second-order extinction coefficient (v2 NIGHT_FIT; OFF until validated). |
| `k2_fit_lit_factor` | 4.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1548`) | Allowed multiple of the literature k2 value; fits outside it are rejected as unphysical. |
| `k2_fit_min_detectability` | 3.0 | Setting (config.json) | config.json | config assembly & validation (`config.py:1536`) | Minimum detectability (signal strength) the night must offer for a k2 fit to be attempted. |
| `k2_mode` | literature | Setting (config.json) | config.json | Settings UI (`ui_settings.py:1091`) | Source of k2: 'literature' (defaults) or fitted per night when the fit program is enabled. |

## Reports & HRD

What appears in the PDF summary report: the color HR diagram of the field, online enrichment of interesting objects (SIMBAD, Gaia), and its visual tuning.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `hrd_color_bg_box_px` | 96; range 32 .. 512 | Setting (config.json) | config.json | config assembly & validation (`config.py:838`) | Background sampling box (px) for star colors in the color HR diagram. |
| `hrd_color_chroma_boost` | 2.2; range 1 .. 3 | Setting (config.json) | config.json | config assembly & validation (`config.py:831`) | Chroma amplification of the HRD star colors for visual clarity. |
| `hrd_color_chroma_snr` | 3.0; range 0 .. 20 | Setting (config.json) | config.json | config assembly & validation (`config.py:822`) | Minimum color SNR before a star gets a saturated color in the HRD. |
| `hrd_color_field_enabled` | True | Setting (config.json) | config.json | config assembly & validation (`config.py:809`) | Renders the color HR diagram of the observed field in the report. |
| `hrd_color_highlight_mode` | soft | Setting (config.json) | config.json | config assembly & validation (`config.py:819`) | Highlight style of interesting objects in the HRD ('soft' = gentle emphasis). |
| `hrd_color_saturation` | 0.85; range 0 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:813`) | Base color saturation of HRD points. |
| `hrd_color_white_point` | field_median | Setting (config.json) | config.json | config assembly & validation (`config.py:828`) | White-point reference of HRD colors ('field_median' balances to the field's median color). |
| `hrd_dsc_confirm_prob` | 0.9; range 0.5 .. 1 | Setting (config.json) | config.json | config assembly & validation (`config.py:803`) | Probability threshold for confirming a candidate's HRD-based classification. |
| `hrd_enrich_max_candidates` | 20; range 1 .. 100 | Setting (config.json) | config.json | config assembly & validation (`config.py:760`) | Cap on objects sent to online enrichment services per report. |
| `hrd_enrich_tap_timeout_s` | 20.0; range 5 .. 120 | Setting (config.json) | config.json | config assembly & validation (`config.py:767`) | Timeout (s) of TAP queries during online enrichment. |
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
| `export_arcsec_per_px` | 1.3 | Dynamic (FITS / runtime) | computed from WCS / optics | main application UI | Plate-scale label written into export metadata; the science value comes from the WCS. WAVE-B removed its config.json fallback (derivable from WCS/optics). |
| `tess_enabled` | False | Setting (config.json) | config.json | config assembly & validation (`config.py:2100`) | Enables the TESS cross-analysis block (comparing your light curve against TESS data). |

## System & performance

Machine-level behavior: parallel workers and RAM reserves. Computed from your hardware; override only when you must.

| Parameter | Default | Type | Source | Used in | Explanation |
|---|---|---|---|---|---|
| `per_frame_mp_reserve_ram_gb` | 1.5 | Setting (config.json) | config.json | config assembly & validation (`config.py:1028`) | RAM (GB) kept free per worker when sizing per-frame parallelism. |
| `qc_preprocess_workers` | 1 | Internal | environment / machine | calibration & frame processing (`pipeline.py:15384`) | Number of parallel preprocessing workers; computed from CPU/RAM at startup, overridable by the VYVAR_PARALLEL_WORKERS environment variable. |