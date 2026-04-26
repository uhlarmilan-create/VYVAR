# VYVAR — Development State

Last updated: 2026-04-26 14:50:02 +02:00

## Project overview

VYVAR is a Python desktop pipeline and Streamlit control room for variable-star imaging workflows built around calibrated FITS archives, a local Gaia DR3 SQLite cone, and optional VSX subsets. The application ingests USB or disk-based observation folders, matches equipment and calibration masters from a calibration library, runs bias/dark/flat correction and optional cosmic-ray rejection, and writes calibrated and processed light frames with QC metadata into an archive layout under a configurable root. Stepwise UI drives preprocess (detrend, quality filters, optional pointing injection), plate solving on a chosen MASTERSTAR reference using VYVAR’s local Gaia solver and optional blind triangle index, WCS refinement, and per-frame aligned catalogs in `detrended_aligned/lights/`. Differential ensemble photometry (Phase 2A) builds per-target light curves, comparison-star tables, optional PNG exports, and summary CSVs. A variability layer stacks per-frame fluxes into a matrix, fits a field-wide “hockey stick” RMS-vs-magnitude envelope, flags RMS outliers, runs a complementary VDI (median-crossing) statistic, and surfaces edge-safe candidates with Vizier/SIMBAD cross-match and optional TESS cutout verification. Reporting includes a ReportLab PDF (English) with cover metrics, FITS QA plots, per-star pages, TESS panels when `_tess/{catalog_id}/result.json` exists, and a SQLite-backed observation log. The long-term vision is unattended night runs producing a morning PDF on modest hardware (Windows today; Raspberry Pi 5 + StellarMate later). The codebase favors explicit paths, draft-scoped working directories (`draft_XXXXXX`), and session state for UI continuity while keeping heavy computation in `pipeline.py` and `photometry_core.py`. Database tables store draft status, equipment, pointing, MASTERSTAR paths, and per-frame QC for dashboards. Alignment uses `astroalign` with logging; aperture photometry can use `photutils` circular apertures or legacy DAO flux columns from per-frame CSVs. PSF photometry is an optional parallel path with its own runner. Quality dashboards combine DB-backed `OBS_FILES` metrics with optional `qc_metrics.csv`. The UI mixes Slovak user-facing captions with English in generated reports and code identifiers. Configuration merges `config.json` with `AppConfig` dataclass defaults. Importer logic builds draft rows and archive folder trees. Several diagnostic scripts support regression and field troubleshooting without being part of the Streamlit app. Plate-solve quality gates SIP polynomial orders, anisotropy checks, and NN WCS refinement thresholds from `config.py`. Color-term photometry can be gated by filter when metadata supports it. Per-frame CSVs carry both DAO and photometric columns so downstream variability can choose `dao_flux` vs `psf_flux`. Ensemble photometry follows differential-instrumental magnitudes with AIJ-style total flux aggregation for comparison stars where configured. Master bias and master flat validity windows steer library selection. The observation hash ties preprocess outputs to a consistent bundle for reruns. Footers and infolog hooks expose long-running job progress without blocking the GIL on the main thread more than necessary. `VyvarDatabase` abstracts schema evolution behind helper methods so UI pages stay thin. Archive memory profiling warns before large plate-solve or alignment batches. Crowding and saturation flags propagate from per-frame catalogs into variability metadata. VSX names and types attach to targets via `variable_targets.csv` merges during RMS scoring. Gaia BP–RP informs color terms and B–V proxies in reports. Comparison-star tiers and RMS feed comp-quality JSON sidecars for the PDF COMP table coloring. TESS verification stores human-readable sector summaries as JSON next to PNGs for PDF embedding. Crossmatch bullets are plain text lines suitable for small-font PDF rows. Edge-safe masks remove chip-border false positives from candidate lists. Session reset on draft override prevents stale TESS or crossmatch state from leaking across nights. Settings persist Gaia DB paths needed before MAKE MASTERSTAR. The calibration library page validates master ages against configured day limits. Database explorer exposes maintenance counters to avoid accidental destructive deletes.

## Tech stack

| Component | Notes |
|-----------|--------|
| **Python** | 3.12.x observed on dev machine (`python --version`); not pinned in repo — use 3.10+ compatible with type hints in codebase. |
| **Streamlit** | Declared in `requirements.txt` without version pin — install current stable from PyPI. |
| **Core libs** | `pandas`, `numpy`, `psutil`, `scipy`, `astropy`, `photutils`, `astroscrappy`, `pillow`, `plotly`, `lightkurve`, `astroquery`, `astroalign`, `reproject`, `pytest`, `reportlab`. |

Full dependency list (from `requirements.txt`, unpinned unless edited locally):

`streamlit`, `pandas`, `numpy`, `psutil`, `scipy`, `astropy`, `photutils`, `astroscrappy`, `pillow`, `plotly`, `lightkurve`, `astroquery`, `astroalign`, `reproject`, `pytest`, `reportlab`.

## Architecture — všetky súbory

Below: every `*.py` file under the VYVAR project tree (including `GAIA_DR3/`, `VSX/`, `scripts/`). *VYVAR imports* lists same-repo modules only (stdlib/third-party omitted). *Session state* lists keys touched via `st.session_state` in that file, if any.

### app.py
- **Účel:** Streamlit `main()`: sidebar pages, draft override, tab host for VAR-STREM / QA / photometry / variability / infolog, preprocess & job orchestration, session wiring.
- **Hlavné funkcie/triedy:** `main`, `_vyvar_reset_variability_session_state`, `_vyvar_effective_draft_dir_override`, `_vyvar_execute_preprocess_pending`, `_vyvar_render_fixed_footer_into`, `render_live_view`, large inline job handlers.
- **Importuje z:** `config`, `database`, `infolog`, `importer`, `pipeline`, `ui_*`, `platesolve_ui_paths`, `utils`, `ui_aperture_photometry`, `ui_variability`, …
- **Session state keys:** Many `vyvar_*`, `center_ra`/`center_de`, draft center keys via `ui_components`, `fwhm_threshold`, `dao_*`, `var_*` reset helper, import/platesolve job outputs, footer state, `vyvar_draft_dir_override`, `vyvar_varstrem_session_id`, etc. (see consolidated list below).

### config.py
- **Účel:** `AppConfig` dataclass + JSON load/save + `recommended_vyvar_parallel_workers` RAM/CPU heuristic.
- **Hlavné funkcie/triedy:** `AppConfig`, `load_config_json`, `save_config_json`, `recommended_vyvar_parallel_workers`.
- **Importuje z:** *(žiadne iné VYVAR moduly)*
- **Session state keys:** —

### database.py
- **Účel:** SQLite access layer for drafts, equipments, OBS_FILES, technical metadata, Gaia path validation helpers.
- **Hlavné funkcie/triedy:** `VyvarDatabase`, draft/equipment fetchers, insert/update helpers (large module).
- **Importuje z:** `config` (types only / optional), project-local paths.
- **Session state keys:** —

### pipeline.py
- **Účel:** Core `AstroPipeline`: calibration, preprocess, platesolve, alignment, catalog enhancement, batch jobs, memory estimates, Streamlit progress hooks.
- **Hlavné funkcie/triedy:** `AstroPipeline`, preprocess/platesolve/alignment functions, `scan_usb_folder`, utilities referenced across UI.
- **Importuje z:** `config`, `database`, `calibration`, `importer`, `utils`, `vyvar_platesolver`, `photometry_core`, `ui_aperture_photometry` (lazy for FWHM), …
- **Session state keys:** `vyvar_footer_state`, `vyvar_ui_rerender_footer`; optional `st.session_state.update` from QC paths.

### importer.py
- **Účel:** Smart import / scan of source folders, draft creation, archive layout, BPM sidecars.
- **Hlavné funkcie/triedy:** `smart_import_session`, `smart_scan_source`, related helpers.
- **Importuje z:** `config`, `database`, `pipeline` utilities, …
- **Session state keys:** —

### calibration.py
- **Účel:** Master calibration application (dark/flat), cosmic ray options, library matching.
- **Hlavné funkcie/triedy:** calibration routines used by pipeline/importer.
- **Importuje z:** `config`, `utils`, …
- **Session state keys:** —

### utils.py
- **Účel:** Shared numeric/FITS/header helpers, observation hash, catalog cone radius from optics, misc pure functions.
- **Hlavné funkcie/triedy:** `generate_session_id`, `fits_binning_xy_from_header`, `catalog_cone_radius_deg_from_optics`, …
- **Importuje z:** *(minimal)*
- **Session state keys:** —

### variables.py
- **Účel:** Shared constants / variable definitions for pipeline or UI.
- **Hlavné funkcie/triedy:** module-level constants.
- **Importuje z:** *(check usages)*
- **Session state keys:** —

### infolog.py
- **Účel:** In-memory / session-coupled log lines for Infolog tab and `log_event`.
- **Hlavné funkcie/triedy:** `log_event`, `get_lines`, `clear_log`, `ensure_infolog_logging`, `last_job_snapshot`.
- **Importuje z:** —
- **Session state keys:** —

### vyvar_ui_status.py
- **Účel:** Footer status helpers for long-running jobs (`vyvar_footer_running`, `vyvar_footer_idle`).
- **Hlavné funkcie/triedy:** small functions mutating `vyvar_footer_state`.
- **Importuje z:** `streamlit`
- **Session state keys:** `vyvar_footer_state`, reads `vyvar_ui_rerender_footer`.

### platesolve_ui_paths.py
- **Účel:** Resolve draft directories from user string + `default_bundle_dir` helpers.
- **Hlavné funkcie/triedy:** `resolve_draft_directory`, `default_bundle_dir`.
- **Importuje z:** `pathlib`
- **Session state keys:** —

### jd_axis_format.py
- **Účel:** JD/BJD axis titles and relative JD series for plots.
- **Hlavné funkcie/triedy:** `jd_axis_title`, `jd_series_relative`.
- **Importuje z:** —
- **Session state keys:** —

### time_utils.py
- **Účel:** Time conversion helpers.
- **Hlavné funkcie/triedy:** small utilities.
- **Importuje z:** —
- **Session state keys:** —

### fits_suffixes.py
- **Účel:** FITS filename/extension normalization.
- **Hlavné funkcie/triedy:** helper functions.
- **Importuje z:** —
- **Session state keys:** —

### gaia_catalog_id.py
- **Účel:** Normalize / validate Gaia-style catalog IDs.
- **Hlavné funkcie/triedy:** ID parsing helpers.
- **Importuje z:** —
- **Session state keys:** —

### masterstar_context.py
- **Účel:** MASTERSTAR FITS context extraction for QA / downstream.
- **Hlavné funkcie/triedy:** context builders.
- **Importuje z:** `astropy` / local utils.
- **Session state keys:** —

### vyvar_platesolver.py
- **Účel:** Local Gaia cone plate solve, SIP orders, WCS quality checks, `VY_` header annotations.
- **Hlavné funkcie/triedy:** `solve_wcs_with_local_gaia`, related.
- **Importuje z:** `config`, `database`, `utils`, …
- **Session state keys:** —

### vyvar_blind_solver.py
- **Účel:** Blind triangle-hash solver fallback using `gaia_triangles.pkl`.
- **Hlavné funkcie/triedy:** blind match routines.
- **Importuje z:** `numpy`, local index paths from config.
- **Session state keys:** —

### astrometry_optimizer.py
- **Účel:** WCS refinement / mirror-orientation handling post-solve.
- **Hlavné funkcie/triedy:** optimizer entrypoints used by platesolver path.
- **Importuje z:** `numpy`, `astropy`
- **Session state keys:** —

### vyvar_alignment_frame.py
- **Účel:** Single-frame alignment helpers (astroalign glue).
- **Hlavné funkcie/triedy:** alignment helpers.
- **Importuje z:** `numpy`, `astropy`, `astroalign`
- **Session state keys:** —

### photometry.py
- **Účel:** Legacy or thin façade re-exporting photometry entrypoints (merged into `photometry_core` historically).
- **Hlavné funkcie/triedy:** re-exports / wrappers.
- **Importuje z:** `photometry_core` (pattern per file header in repo).
- **Session state keys:** —

### photometry_core.py
- **Účel:** Phase 2A pipeline: per-frame CSV enhancement, ensemble photometry, light curve CSV/PNG, comparison tables, field maps, summary rows.
- **Hlavné funkcie/triedy:** `run_full_photometry_pipeline`, large internal helpers, stress tests, catalog normalization.
- **Importuje z:** `utils`, `config`, `pipeline` types, …
- **Session state keys:** —

### photometry_phase2a.py
- **Účel:** Split Phase 2A helpers (plotting, color-term branches, setup-specific logic) consumed by `photometry_core` / UI to keep the monolith navigable.
- **Hlavné funkcie/triedy:** phase2a-specific functions (see file for current exports).
- **Importuje z:** `photometry_core`, `pandas`, `numpy`, plotting stack as needed.
- **Session state keys:** —

### photometry_report.py
- **Účel:** ReportLab landscape PDF: cover, FITS QA page, per-star pages (LC+field, TESS block, catalog crossmatch for candidates), compact sparse pages, summary table, VYVAR footer.
- **Hlavné funkcie/triedy:** `generate_photometry_report` (+ nested layout helpers).
- **Importuje z:** `config` (optional), `ui_variability.count_edge_safe_combined_candidates`, `ui_aperture_photometry._load_fwhm`, `numpy`, `pandas`, `reportlab`, …
- **Session state keys:** —

### pdf_report.py
- **Účel:** Thin wrapper: `generate_report` → `generate_photometry_report` with variability args; output `photometry/report_{setup}.pdf`.
- **Hlavné funkcie/triedy:** `generate_report`.
- **Importuje z:** `photometry_report`
- **Session state keys:** —

### variability_detector.py
- **Účel:** `load_field_flux_matrix`, `compute_rms_variability` (hockey stick + candidate filters), `compute_vdi` (median-crossing index + z-threshold).
- **Hlavné funkcie/triedy:** above + internal `_norm_cid`, `_mad_sigma`.
- **Importuje z:** `numpy`, `pandas`
- **Session state keys:** —

### catalog_crossmatch.py
- **Účel:** Parallel SIMBAD + Vizier catalog queries; `CatalogMatch`, `CrossmatchResult`, `check_candidate_in_catalogs`.
- **Hlavné funkcie/triedy:** `_CATALOG_WORKERS` order, per-catalog `_query_*`, bullet formatting.
- **Importuje z:** `astroquery`, `astropy`
- **Session state keys:** —

### tess_verify.py
- **Účel:** TESS cutout download via `lightkurve`, per-sector processing, phased plots, `result.json` under `photometry/_tess/{catalog_id}/`.
- **Hlavné funkcie/triedy:** `TessSectorResult`, `TessResult`, `run_tess_analysis`, `_get_aperture_params`, `_process_one_sector`.
- **Importuje z:** `lightkurve`, `matplotlib`, `numpy`, `astropy`
- **Session state keys:** — *(UI layer passes paths; this module is mostly pure)*

### psf_photometry.py
- **Účel:** ePSF / PSF-based photometry extensions and column contracts for per-frame tables.
- **Hlavné funkcie/triedy:** PSF fitting helpers, column requirements.
- **Importuje z:** `photutils`, `numpy`, …
- **Session state keys:** —

### psf_runner.py
- **Účel:** Batch PSF photometry runner over draft setups (CLI-oriented, used for experiments).
- **Hlavné funkcie/triedy:** runner `main` / orchestration functions.
- **Importuje z:** `photometry_core`, `config`, …
- **Session state keys:** —

### masterstar_qa_plot.py
- **Účel:** Matplotlib/Plotly helpers for MASTERSTAR QA visuals.
- **Hlavné funkcie/triedy:** plotting helpers.
- **Importuje z:** `pandas`, `plotly` / `matplotlib`
- **Session state keys:** —

### masterstar_wcs_dao_diagnostic.py
- **Účel:** Diagnostic CLI for WCS vs DAO catalog on MASTERSTAR.
- **Hlavné funkcie/triedy:** diagnostic `main`.
- **Importuje z:** `astropy`, local pipeline utils.
- **Session state keys:** —

### diagnose_crowding_filters.py
- **Účel:** Standalone diagnostic for crowding filters.
- **Hlavné funkcie/triedy:** script-style functions.
- **Importuje z:** `pandas`, `numpy`
- **Session state keys:** —

### diagnose_flux.py
- **Účel:** Flux matrix / per-frame diagnostics.
- **Hlavné funkcie/triedy:** diagnostic plots/stats.
- **Importuje z:** `pandas`, `numpy`
- **Session state keys:** —

### diagnose_ensemble.py
- **Účel:** Ensemble photometry diagnostic.
- **Hlavné funkcie/triedy:** diagnostic functions.
- **Importuje z:** `pandas`, `numpy`
- **Session state keys:** —

### debug_qc.py
- **Účel:** Debug QC CSV / metrics inspection.
- **Hlavné funkcie/triedy:** small CLI.
- **Importuje z:** `pandas`
- **Session state keys:** —

### run_smoothness_report.py
- **Účel:** CLI report on smoothness / variability columns from saved results.
- **Hlavné funkcie/triedy:** `main`.
- **Importuje z:** `pandas`
- **Session state keys:** —

### plot_top_candidates_lightcurves.py
- **Účel:** Plot top variability candidates from CSVs.
- **Hlavné funkcie/triedy:** plotting script.
- **Importuje z:** `matplotlib`, `pandas`
- **Session state keys:** —

### ui_components.py
- **Účel:** Reusable Streamlit widgets: draft center persistence, equipment/draft pickers, MASTERSTAR candidate listing.
- **Hlavné funkcie/triedy:** `persist_draft_center_on_change`, `DRAFT_*_STATE_KEY` constants, picker helpers.
- **Importuje z:** `streamlit`, `pipeline`, `infolog`, `plotly`
- **Session state keys:** `cur_draft_ra`, `cur_draft_de`, `cur_draft_focal_mm`, `cur_draft_pixel_um`, `center_ra`, `center_de`, `vyvar_last_saved_draft_center_sig`.

### ui_calibration.py
- **Účel:** Large “Step 2/3” calibration + preprocess + MAKE MASTERSTAR UI (Analyze, QC, detrend, platesolve triggers).
- **Hlavné funkcie/triedy:** `draft_runtime_status`, render functions for calibration workflow, progress integration.
- **Importuje z:** `streamlit`, `pipeline`, `database`, `config`, `infolog`, `ui_components`, …
- **Session state keys:** Many `vyvar_*` job outputs, `vyvar_pending_job`, preprocess staging, `vyvar_status_*`, `center_*`, `fwhm_threshold`, `dao_*`, `vyvar_observation_processing_hash`, `vyvar_memory_profile`, etc.

### ui_calibration_library.py
- **Účel:** “Calibration Library” page: master dark/flat inventory and validity.
- **Hlavné funkcie/triedy:** `render_calibration_library_dashboard`.
- **Importuje z:** `streamlit`, `database`, `config`
- **Session state keys:** —

### ui_quality_dashboard.py
- **Účel:** “FITS QA” tab: FWHM/sky Plotly, frame table, FWHM limit sync, MASTERSTAR candidate pick → session paths for platesolve.
- **Hlavné funkcie/triedy:** `render_quality_dashboard`, FWHM auto-limit from median×1.05 pattern, DB/`qc_metrics.csv` merge.
- **Importuje z:** `streamlit`, `database`, `plotly`, `pandas`, `ui_components` constants.
- **Session state keys:** `vyvar_pending_center_ra`, `vyvar_pending_center_de`, `center_ra`, `center_de`, `fwhm_threshold`, `fwhm_limit`, `vyvar_ui_reject_fwhm`, `vyvar_last_logged_fwhm_limit`, `vyvar_qdash_last_fwhm_nonzero`, `vyvar_ms_candidates`, `vyvar_ms_candidate_top1_path`, `vyvar_masterstar_candidate_paths`, `vyvar_last_import_result`, `vyvar_last_job_output`, `cur_draft_ra`, `cur_draft_de`, dynamic Plotly widget keys (`key_fwhm`, `key_sky`), `vyvar_flatfb_{group_key}` pattern for flat feedback.

### ui_masterstar_qa.py
- **Účel:** “MASTERSTAR QA” tab: inspect MASTERSTAR.fits, overlays, sanity vs DB path.
- **Hlavné funkcie/triedy:** `render_masterstar_qa`.
- **Importuje z:** `streamlit`, `astropy`, `config`, `pipeline`, …
- **Session state keys:** `vyvar_masterstar_qa_force_refresh`, `vyvar_last_import_result`.

### ui_aperture_photometry.py
- **Účel:** “Aperture Photometry” tab: Phase 2A run per setup, Plotly LCs, PDF generation hooks, **auto variability** after successful multi-setup run.
- **Hlavné funkcie/triedy:** `render_aperture_photometry`, `_latest_report_pdf`, cached CSV reads, JD axis helpers.
- **Importuje z:** `streamlit`, `photometry_core`, `photometry_report`, `pdf_report`, `ui_variability`, `config`, …
- **Session state keys:** `var_analysis_done`, `var_analysis_timestamp`, `pdf_ready`, `var_results`, `_var_run_sig`, `var_catalog_bullets`, `var_obs_group`, `var_candidate_count_autorun`, `phase2a_setup_select`, `var_flux_source`, `var_min_frames_pct`, `var_sigma_thr`, `var_mag_limit`, `vyvar_pdf_paths`.

### ui_variability.py
- **Účel:** “Variabilita” tab: hockey-stick plot, candidate table, export to `variable_targets.csv`, TESS run, crossmatch modal, accepted period UI.
- **Hlavné funkcie/triedy:** `render_variability_dashboard`, `run_variability_detection_session`, `_edge_ok_from_masterstar`, `count_edge_safe_combined_candidates`, `_variability_crossmatch_dialog`.
- **Importuje z:** `streamlit`, `variability_detector`, `tess_verify`, `catalog_crossmatch`, `config`, …
- **Session state keys:** `tess_results`, `accepted_period`, `accepted_period_msg`, `var_analysis_done`, `var_analysis_timestamp`, `pdf_ready`, `var_candidate_count_autorun`, `var_photometry_dir`, `var_results`, `_var_run_sig`, `var_catalog_bullets`, `var_candidates`, `crossmatch_result`, `selected_for_export`, `var_cm_*`, `var_cm_pick_id`, `var_cm_coord_by_cid`, `var_cm_do_open`, `tess_selected_sector`, widget keys `var_*` for sliders/radio.

### ui_settings.py
- **Účel:** Settings page: archive root, Gaia DB path, equipment, config JSON editor, validation.
- **Hlavné funkcie/triedy:** `render_settings_dashboard`.
- **Importuje z:** `streamlit`, `config`, `database`, …
- **Session state keys:** `vyvar_last_draft_id`, `GAIA_DB_PATH`.

### ui_database_explorer.py
- **Účel:** DB explorer / maintenance (draft rows, equipment, destructive actions guarded by counters).
- **Hlavné funkcie/triedy:** `render_database_explorer`.
- **Importuje z:** `streamlit`, `database`, `pipeline`
- **Session state keys:** `vyvar_maint_nuke_gen`.

### ui_photometry.py
- **Účel:** Legacy or alternate photometry UI surface (verify usage from `app.py` — may be imported elsewhere).
- **Hlavné funkcie/triedy:** photometry-related widgets.
- **Importuje z:** `streamlit`, …
- **Session state keys:** *(if used)*

### ui_photometry_quality.py
- **Účel:** Photometry quality sub-dashboards.
- **Hlavné funkcie/triedy:** render helpers.
- **Importuje z:** `streamlit`, `pandas`
- **Session state keys:** —

### ui_photometry_results.py
- **Účel:** Rich photometry results explorer (targets, comp stars, variability hints, apply lists to session for downstream detection).
- **Hlavné funkcie/triedy:** large `render_*` functions, merge of pipeline outputs.
- **Importuje z:** `streamlit`, `pandas`, `plotly`, …
- **Session state keys:** `vyvar_suspicious_comp_stars`, `vyvar_var_detection_exclusions`, `vyvar_newly_added_var_targets`.

### ui_finalization.py
- **Účel:** Finalization / approval UI (observer name, packaging steps as implemented).
- **Hlavné funkcie/triedy:** render finalization dashboard.
- **Importuje z:** `streamlit`, …
- **Session state keys:** `vyvar_observer_name`.

### ui_select_stars.py
- **Účel:** Star selection UI for pipeline steps.
- **Hlavné funkcie/triedy:** selection widgets.
- **Importuje z:** `streamlit`, …
- **Session state keys:** —

### ui_dao_stars.py
- **Účel:** DAO catalog inspection UI.
- **Hlavné funkcie/triedy:** DAO-related tables/plots.
- **Importuje z:** `streamlit`, …
- **Session state keys:** —

### ui_suspected_lightcurves.py
- **Účel:** UI for suspected problematic light curves.
- **Hlavné funkcie/triedy:** suspected LC review.
- **Importuje z:** `streamlit`, `pandas`, …
- **Session state keys:** —

### GAIA_DR3/build_gaia_blind_index.py
- **Účel:** Build `gaia_triangles.pkl` blind index from Gaia subset.
- **Hlavné funkcie/triedy:** index build CLI.
- **Importuje z:** `numpy`, local DB access patterns.
- **Session state keys:** —

### GAIA_DR3/gaia-dr3_make.py
- **Účel:** Legacy / primary ETL script variant for building Gaia DR3 SQLite from source tables.
- **Hlavné funkcie/triedy:** schema + bulk insert pipeline (see file).
- **Importuje z:** `sqlite3`, `pandas`, …
- **Session state keys:** —

### GAIA_DR3/gaia-dr3_make_v2.py
- **Účel:** Revised Gaia DB build (v2 schema / performance path).
- **Hlavné funkcie/triedy:** v2 ingest + indexing.
- **Importuje z:** `sqlite3`, `pandas`, …
- **Session state keys:** —

### GAIA_DR3/gaia_dr3_make_fast.py
- **Účel:** Fast-path Gaia ingest optimizations.
- **Hlavné funkcie/triedy:** accelerated copy/index routines.
- **Importuje z:** `sqlite3`, `pandas`, …
- **Session state keys:** —

### GAIA_DR3/gaia-dr3_index_solver.py
- **Účel:** Standalone index/solver experiments against Gaia DB.
- **Hlavné funkcie/triedy:** solver prototyping.
- **Importuje z:** local GAIA tooling, `numpy`.
- **Session state keys:** —

### VSX/vsx_make.py
- **Účel:** Build local VSX SQLite subset for `vsx_local_db_path`.
- **Hlavné funkcie/triedy:** VSX ingest.
- **Importuje z:** `sqlite3`, …
- **Session state keys:** —

### scripts/compare_psf_vs_dao_rms_comp.py
- **Účel:** Dev script — compare PSF vs DAO RMS on comparison stars.
- **Hlavné funkcie/triedy:** CLI / analysis main.
- **Importuje z:** `pandas`, `numpy`, project photometry modules.
- **Session state keys:** —

### scripts/diagnose_masterstar_wcs_dao.py
- **Účel:** Dev diagnostic — MASTERSTAR WCS vs DAO catalog offsets.
- **Hlavné funkcie/triedy:** diagnostic main.
- **Importuje z:** `astropy`, local paths.
- **Session state keys:** —

### scripts/fix_alignment_report_detected_stars.py
- **Účel:** Maintenance — fix alignment report detected-star lists.
- **Hlavné funkcie/triedy:** file rewrite helpers.
- **Importuje z:** `pandas`, `pathlib`.
- **Session state keys:** —

### scripts/plot_mn_boo_psf_lightcurve.py
- **Účel:** Example / regression plot for PSF light curve on MN Boo (or similar).
- **Hlavné funkcie/triedy:** matplotlib plotting.
- **Importuje z:** `matplotlib`, `pandas`.
- **Session state keys:** —

### scripts/run_draft_000244_align_and_photometry.py
- **Účel:** One-off batch runner for a specific draft ID (template for automation).
- **Hlavné funkcie/triedy:** scripted pipeline calls.
- **Importuje z:** `pipeline`, `pathlib`, …
- **Session state keys:** —

---

## Directory structure (repository)

```
VYVAR/
├── app.py                    # Streamlit entry
├── config.json               # User/machine overrides (not always in git)
├── config.py
├── requirements.txt
├── img/VYVAR_logo.png
├── GAIA_DR3/                 # Offline Gaia tools + local DB files (*.db*, scripts)
├── VSX/vsx_make.py
├── scripts/                  # Maintenance / diagnostics
├── [large Python modules as listed above]
```

**Archive layout (runtime, under `AppConfig.archive_root`):** typicky  
`<archive_root>/Drafts/draft_XXXXXX/`  
s podpriečinkmi ako `calibrated/`, `non_calibrated/`, `processed/`, `detrended_aligned/`, `platesolve/<filter>_<exptime>_<bin>/` (názov *observation group* / setup), `platesolve/.../photometry/` (Phase 2A outputs, `lightcurves/`, `photometry_summary.csv`, `_tess/`, `report_<setup>.pdf`), atď. Presný strom závisí od fázy spracovania a počtu filtrov.

## Key decisions & design rules

- **Variability / RMS:** Field stars (not VSX-known, not Gaia-varisum flagged) define a log-linear RMS vs magnitude envelope with robust MAD scatter `sigma_log`. A star is an RMS candidate if `rms_pct` exceeds `10^(log10(expected)+sigma_threshold*sigma_log)` plus filters: clip ratio, min RMS above floor, SNR, mean normalized flux, smoothness `< variability_smoothness_max`, amplitude, `mag <= variability_mag_limit`, enough frames (`variability_min_frames` / fraction). **Hockey-stick fit** optionally weights stars by inverse comp RMS variance (`comp_rms_map`). Top RMS tail (`variability_p85_filter`) excluded from fit sample.
- **Variability / VDI:** For each star, count median crossings of normalized flux; `vdi_score = crossings/sqrt(N)`; field MAD z-score; candidate if `vdi_z_score < -variability_vdi_z_threshold` (low crossings ⇒ more “variable” under this heuristic).
- **Combined candidates (UI/PDF):** `is_candidate_combined = RMS_flag | VDI_flag`, then exclude `vsx_known_variable`, require `edge_ok` from MASTERSTAR WCS annulus (`_edge_ok_from_masterstar` in `ui_variability.py`).
- **Auto-trigger variability:** After **RUN Aperture Photometry** completes **all** setups with **zero errors** (`ui_aperture_photometry.py`), `run_variability_detection_session` runs once using `var_obs_group` or `phase2a_setup_select` or current setup, flux column `var_flux_source` (default `dao_flux`), sliders `var_min_frames_pct`, `var_sigma_thr`, `var_mag_limit`. Results + timestamp stored; `var_catalog_bullets` cleared; `var_analysis_done=True`.
- **Catalogs / Vizier:** `_CATALOG_WORKERS` order: SIMBAD, VSX (`B/vsx/vsx`), ASAS-SN, ZTF, Gaia varisum, ATLAS, CSS (`J/AJ/147/119/table1`), KELT, VSBS, TESS-EB — queries run in parallel (`ThreadPoolExecutor`, max 6, 30s timeout). `CrossmatchResult.best_period()` prefers catalogs in order VSX, ASAS-SN, ZTF, Gaia varisum, ATLAS, CSS, KELT, VSBS, TESS-EB, SIMBAD.
- **TESS:** `lightkurve` TESScut search; aperture box size from `_get_aperture_params(mag)` (brighter ⇒ larger cutout and aperture); per-sector period search and phased PNGs; consensus period median across OK sectors written to `result.json` under `photometry/_tess/<catalog_id>/`.
- **PDF:** **ReportLab** canvas, **English** UI strings in report; primary output `photometry/report_{setup_name}.pdf` via `pdf_report.generate_report`; legacy auto-run after Phase 2A per setup may still emit `VYVAR_report_<setup>_YYYYMMDD.pdf` in `platesolve/<setup>/`. Cover includes variability counts when `var_results` present. Footer: `VYVAR — draft_… — <setup>    Page N`.
- **Session state:** kompletný katalóg kľúčov je nižšie (statické + vzory dynamických / widgetov).
- **Draft reset (`_vyvar_reset_variability_session_state` in `app.py`):** When loading a new draft path override: clears variability results, TESS store, accepted period maps, crossmatch result, export selection, PDF flags, candidate counts, `var_candidates`, `_var_run_sig`, `var_results`; sets `var_catalog_bullets` to `{}`, `var_analysis_done` False, timestamps None.

### Session state — kompletný zoznam kľúčov

Zdroj: grep `st.session_state[...]`, `.get(`, `.setdefault(`, `.pop(`, `.update(` a explicitné `key=` widgetov v `*.py` (2026-04). **Nezahŕňa** vnútorné kľúče dict hodnôt (napr. `tess_results[catalog_id]` sú pod-kľúče dictu, nie samostatné top-level session kľúče).

#### A–Z (statické top-level reťazce)

- **`_var_run_sig`** — `tuple` podpis parametrov variability runu; zmena invaliduje cache výsledkov.
- **`accepted_period`** — `dict[str, float]` (resp. numerické hodnoty): používateľom akceptovaná perióda podľa `catalog_id`.
- **`accepted_period_msg`** — `dict[str, str]`: správy / stav k akceptácii periódy.
- **`center_de`**, **`center_ra`** — `float`: zrkadlený stred pole (deg); synchronizované s `cur_draft_*` z UI / pipeline.
- **`crossmatch_result`** — `CrossmatchResult` z posledného modal crossmatchu (`ui_variability`); pri resete draftu `pop`.
- **`cur_draft_de`**, **`cur_draft_ra`** — `float`: draft center z `ui_components.DRAFT_CENTER_*` (literály `cur_draft_de`, `cur_draft_ra`).
- **`cur_draft_focal_mm`**, **`cur_draft_pixel_um`** — optional equipment metadata v session (čítané/zapisované cez `persist_draft_center_on_change`).
- **`dao_fwhm_px`**, **`dao_threshold_sigma`** — `float`: DAO parametre pre MASTERSTAR katalóg (Krok 3 / pending job).
- **`drift_limit_arcmin`** — `float`: limit driftu; nastavuje pipeline po QC.
- **`fwhm_limit`**, **`fwhm_limit_slider`**, **`fwhm_threshold`** — `float` / widget: FWHM limit pre filter snímok, posuvník a prah z QA.
- **`GAIA_DB_PATH`** — `str`: override cesty k Gaia SQLite (Settings).
- **`max_roundness_error`** — `float`: horná hranica roundness pre pending job / QC.
- **`pdf_ready`** — `bool`: po kliknutí „Generovať PDF“ — ďalší rerun spustí generovanie.
- **`phase2a_preload_all_curves`**, **`phase2a_run_full`**, **`phase2a_setup_select`**, **`phase2a_show_all_filters`**, **`phase2a_show_outliers`**, **`phase2a_target_select`** — stav Streamlit widgetov (Aperture Photometry).
- **`selected_for_export`** — `list[str]`: `catalog_id` označené exportom v tabuľke kandidátov.
- **`tess_results`** — `dict[str, TessResult]`: výsledky TESS analýzy podľa `catalog_id`.
- **`tess_selected_sector`** — `int`: vybraný sektor v TESS UI.
- **`toggle_am_detrend`**, **`toggle_show_airmass`** — widget stavy (Phase 2A plots).
- **`var_add_to_var2`**, **`var_cm_open_btn`**, **`var_export`** — tlačidlá (`key=`); držia bool/click stav podľa Streamlit.
- **`var_analysis_done`**, **`var_analysis_timestamp`** — bool + `str` časová pečiatka autorun / manuálnej variability.
- **`var_candidate_count_autorun`** — `int`: počet edge-safe kandidátov z posledného autorun.
- **`var_candidates`** — `list[str]`: zoznam `catalog_id` kandidátov pre PDF / session.
- **`var_candidates_editor`** — `st.data_editor` stav (tabuľka + checkbox export).
- **`var_catalog_bullets`** — `dict[str, str]`: textové zhrnutie crossmatchu podľa `catalog_id`.
- **`var_cm_cid`**, **`var_cm_dec`**, **`var_cm_do_open`**, **`var_cm_mag`**, **`var_cm_ra`** — crossmatch modal: aktuálny kandidát a súradnice / latch na otvorenie modalu.
- **`var_cm_coord_by_cid`** — `dict[str, dict]`: mapa `catalog_id` → `{ra, dec, mag}` pre crossmatch.
- **`var_cm_pick_id`** — `str`: vybraný riadok v selectboxe kandidátov pre crossmatch.
- **`var_flux_source`**, **`var_lc_select2`**, **`var_mag_limit`**, **`var_min_frames_pct`**, **`var_obs_group`**, **`var_sigma_thr`** — widgety variability (radio/slider/selectbox).
- **`var_photometry_dir`** — `str`: absolútna cesta k `.../photometry` pre aktuálny setup variability.
- **`var_results`** — `dict`: `rms_df`, `vdi_df`, flux matrix, metadata, … (`pop` pri resete draftu).
- **`vyvar_applied_analyze_token`** — token poslednej aplikácie Analyze výstupu (zabráni dvojitému seedovaniu).
- **`vyvar_draft_dir_override`** — `str`: absolútna cesta k `draft_XXXXXX` mimo archívu.
- **`vyvar_db_masterstar_path`** — `str`: cesta MASTERSTAR z DB.
- **`vyvar_enable_background_flattening`**, **`vyvar_enable_lacosmic`** — preprocess toggle hodnoty.
- **`vyvar_footer_state`** — `dict`: stav pätičky (progress, text, …); aktualizuje `pipeline` / `app` / `vyvar_ui_status`.
- **`vyvar_last_draft_id`** — `int`: aktívny draft PK.
- **`vyvar_last_import_equipment_id`**, **`vyvar_last_import_plan`**, **`vyvar_last_import_result`** — posledný smart import.
- **`vyvar_last_job_output`**, **`vyvar_last_job_summary`** — posledný job (analyze, quality, platesolve, …).
- **`vyvar_last_logged_fwhm_limit`** — posledná zalogovaná raw hodnota FWHM limitu (UI).
- **`vyvar_last_qc_csv`** — optional cesta k exportovanému QC CSV (`pop` pri novom analyze).
- **`vyvar_last_qc_suggestions`** — návrhy / metadáta z posledného QC jobu.
- **`vyvar_last_saved_draft_center_sig`** — `str`: podpis uloženého RA/Dec.
- **`vyvar_manual_flat_map`** — `dict`: manuálny výber flatov po skupinách.
- **`vyvar_masterstar_candidate_paths`**, **`vyvar_ms_candidate_top1_path`**, **`vyvar_ms_candidates`** — cesty kandidátov na MASTERSTAR z FITS QA.
- **`vyvar_masterstar_qa_force_refresh`** — bool: vynútiť obnovenie MASTERSTAR QA tabu.
- **`vyvar_maint_nuke_gen`** — `int`: generácia počítadla pre deštruktívne údržby v DB exploreri.
- **`vyvar_memory_profile`** — `dict`: odhad RAM pre QC / platesolve.
- **`vyvar_newly_added_var_targets`** — `list[str]`: `catalog_id` práve pridané do `variable_targets.csv` z výsledkov fotometrie.
- **`vyvar_observer_name`** — `str`: meno schvaľovateľa (finalizácia).
- **`vyvar_pending_center_de`**, **`vyvar_pending_center_ra`** — staging stredu z QA pred apply v `app`.
- **`vyvar_pending_job`** — `dict`: čakajúci platesolve / ONLY MASTER / preprocess job.
- **`vyvar_pdf_paths`** — `dict[str, str]`: setup → cesta k PDF po batch 2A.
- **`vyvar_pointing_scan_cache`** — cache z `vyvar_last_job_output["pointing_scan"]`.
- **`vyvar_post_cal_archive_path`**, **`vyvar_post_cal_plan_source`** — post-calibration paths.
- **`vyvar_qdash_confirm_masterstar`**, **`vyvar_qdash_fwhm_enable`**, **`vyvar_qdash_ms_pick_name`**, **`vyvar_set_masterstar_from_preview`** — FITS QA widget kľúče.
- **`vyvar_qdash_last_fwhm_nonzero`** — posledná nenulová hodnota FWHM prahu (návrat posuvníka).
- **`vyvar_smart_plan`** — smart import plán objekt.
- **`vyvar_suspicious_comp_stars`** — `list` / serializované ID podozrivých comp hviezd (`ui_photometry_results`).
- **`vyvar_staged_preprocess_job`**, **`vyvar_staged_processing_hash`** — staging preprocessu (`pop` pri apply).
- **`vyvar_status_analyzed`**, **`vyvar_status_calibrated`** — bool stavy draftu.
- **`vyvar_ui_max_align_stars`** — `int`: max hviezd pre align.
- **`vyvar_ui_reject_fwhm`** — `float`: UI FWHM reject (synchronizované s `fwhm_limit`).
- **`vyvar_var_detection_exclusions`** — `list[str]`: `catalog_id` vylúčené z variability / detekcie podľa UI.
- **`vyvar_ui_rerender_footer`** — **callable** (lambda): callback na prekreslenie pätičky.
- **`vyvar_varstrem_session_id`** — `str`: náhodné ID session pre VAR-STREM tab.

#### Dynamické / vzorové kľúče (presný tvar závisí od dát)

- **`vyvar_flatfb_{group_key}`** — `str` voľba flat fallbacku pre skupinu `group_key` z import plánu (`app._apply_flat_fallback_choices_from_session`).
- **`vyvar_qdash_editor_{draft_id}`** — stav `st.data_editor` pre tabuľku snímok (FITS QA), `draft_id` int.
- **`vyvar_qdash_fwhm_auto_seed_done_{draft_id}`** — bool „už sme raz nastavili auto FWHM“ pre daný draft.
- **`vyvar_qdash_plot_fwhm_{draft_id}`**, **`vyvar_qdash_plot_sky_{draft_id}`** — Plotly `st.plotly_chart` selection state (FWHM / sky graf).
- **`pdf_download_{setup_name}`**, **`pdf_dl_hdr_{selected_setup}`**, **`pdf_gen_dl_{selected_setup}`**, **`vyvar_gen_pdf_{selected_setup}`** — PDF tlačidlá / download (`ui_aperture_photometry`).
- **`tess_custom_num_{candidate_catalog_id}`**, **`tess_sector_pick_{candidate_catalog_id}`**, **`tess_use_2p_{candidate_catalog_id}`**, **`tess_use_custom_{candidate_catalog_id}`**, **`tess_use_p_{candidate_catalog_id}`**, **`var_tess_retry_{candidate_catalog_id}`**, **`var_tess_run_{candidate_catalog_id}`** — TESS widgety / tlačidlá per kandidát (`catalog_id` string).

#### Poznámky

- **`tess_results`**: hodnota je dict; kľúče vnútri sú `catalog_id` pre uložené `TessResult` — to sú **nie** samostatné `st.session_state` top-level kľúče.
- **`vyvar_footer_state`**: štruktúra dict závisí od aktuálneho jobu (percentá, text, detail).
- **`vyvar_pdf_paths`**: kľúče = názov setupu (`str`).

## Dataclasses & data structures

### `CatalogMatch` (`catalog_crossmatch.py`)
| Field | Type | Meaning |
|-------|------|---------|
| `catalog` | `str` | Catalog label (e.g. `SIMBAD`, `VSX`). |
| `name` | `str` | Primary identifier / designation. |
| `var_type` | `str` | Variable/object type string when available. |
| `period` | `float \| None` | Literature period (days) if known. |
| `amplitude` | `float \| None` | Approx amplitude if derivable. |
| `delta_r` | `float \| None` | Separation from query position (arcsec). |
| `mag` | `float \| None` | Representative magnitude. |
| `epoch` | `float \| None` | Epoch / reference time if present. |
| `extra` | `dict[str, Any]` | Raw/aux columns. |

### `CrossmatchResult` (`catalog_crossmatch.py`)
| Field | Type | Meaning |
|-------|------|---------|
| `ra`, `dec` | `float` | Query coordinates (deg). |
| `mag` | `float \| None` | Optional input magnitude. |
| `radius_arcsec` | `float` | Search radius. |
| `matches` | `dict[str, list[CatalogMatch]]` | Per-catalog hit lists. |
| `errors` | `dict[str, str]` | Per-catalog error messages (timeout, query failure). |

### `TessSectorResult` (`tess_verify.py`)
| Field | Type | Meaning |
|-------|------|---------|
| `sector` | `int` | TESS sector number. |
| `jd_start`, `jd_end` | `float` | Time span approximations for sector LC. |
| `period_found`, `period_2p` | `float \| None` | Detected period and 2× variant. |
| `lc_raw_path`, `plot_raw_path` | `str \| None` | Saved artifact paths. |
| `plot_phased_p_path`, `plot_phased_2p_path` | `str \| None` | Phased plot PNG paths. |
| `n_points` | `int` | Number of valid points used. |
| `error` | `str \| None` | Sector-level failure message. |

### `TessResult` (`tess_verify.py`)
| Field | Type | Meaning |
|-------|------|---------|
| `catalog_id` | `str` | Gaia / target id string. |
| `ra`, `dec` | `float` | Degrees. |
| `mag` | `float \| None` | Input magnitude. |
| `sectors` | `list[TessSectorResult]` | Per-sector outcomes. |
| `period_consensus`, `period_2p_consensus` | `float \| None` | Aggregated periods across sectors. |
| `output_dir` | `str` | Base folder for JSON/PNGs. |
| `total_sectors_found`, `total_sectors_ok` | `int` | Search vs successful processing counts. |
| `error_global` | `str \| None` | Fatal / global error (e.g. no TESScut). |

## UI structure

**Main navigation (`app.py` sidebar):** primary workflow page (“VAR-STREM” hub + draft loader), separate pages for **Calibration Library**, **Database Explorer**, **Settings**.

**Main workflow tabs (single page):**
1. **VAR-STREM** — `render_live_view`: equipment/telescope pickers, smart import, session id display, links to calibration path (placeholder-style “Session Upload Automation” block per current code).
2. **FITS QA** — `ui_quality_dashboard`: per-frame FWHM/sky plots, reject mask editor, FWHM limit, MASTERSTAR candidate promotion to session paths.
3. **MASTERSTAR QA** — `ui_masterstar_qa`: inspect active MASTERSTAR FITS, WCS/photometry sanity.
4. **Aperture Photometry** — `ui_aperture_photometry`: per-setup output status, RUN full Phase 2A, per-target Plotly light curves (from CSV), optional PNG paths, PDF download / generate, autorun variability banner when done.
5. **Variabilita** — `ui_variability`: sliders (sigma, mag limit, min frames %), flux source radio, hockey-stick Plotly figure, candidate `st.data_editor` with **export** checkbox column, “Open crossmatch” flow, TESS panel per candidate, “add selected to variable_targets.csv”.
6. **Infolog** — `render_infolog`: scrollable log from `infolog.log_event`.

**Crossmatch modal (within Variability):** `@st.dialog`-style flow (`_variability_crossmatch_dialog`): runs `check_candidate_in_catalogs`, shows match tables, writes bullet lines into `var_catalog_bullets[catalog_id]`, optional TESS sub-tab with sector picker / phased plots / accept period buttons.

**Candidate table columns (non-exhaustive):** `catalog_id`, `mag`, `bp_rp`, `rms_pct`, `smoothness_ratio`, `vdi_score`, `vdi_z_score`, `detection_method`, `variability_score`, `katalógy` (crossmatch summary), `zone`, `Vizier` link column, `export` checkbox.

## Current implementation status (high level)

| Area | Status |
|------|--------|
| Calibration + preprocess + detrend | [X] Implemented (`pipeline`, `ui_calibration`, `importer`) |
| Platesolve + alignment + per-frame CSV | [X] Implemented |
| Phase 2A aperture photometry | [X] Core complete (`photometry_core`) |
| PSF photometry path | [~] Optional; runner exists; UI integration partial |
| Variability RMS + VDI | [X] Implemented (`variability_detector`, `ui_variability`) |
| Catalog crossmatch + TESS verify | [X] Implemented (network-dependent) |
| PDF report (English, variability, TESS, compact sparse pages) | [X] Implemented (`photometry_report`, `pdf_report`) |
| AAVSO export | [ ] Not implemented |
| LC classification hint (shape + BP-RP + catalog type) | [ ] Roadmap only |
| TESS “lightkurve LAB” exploratory UI | [ ] Roadmap |
| Raspberry Pi / StellarMate deployment | [ ] Roadmap |

## Known issues & pending fixes

- **TODO/FIXME grep:** no classic `TODO`/`FIXME` tokens in tracked `.py` (only user-facing strings containing “XXXXXX” in path hints). Technical debt is mostly implicit (unpinned Streamlit, large `pipeline.py` / `app.py` surface).
- **Risk:** Network catalogs (Vizier/SIMBAD) and TESS downloads can time out — crossmatch sets per-catalog errors; TESS uses 30s futures timeout pattern in crossmatch, separate handling in `tess_verify`.
- **Session drift:** `var_results` from manual Variability tab vs autorun from Aperture Photometry can disagree until signatures match — UI shows info when `_var_run_sig` differs.

## Pending features (roadmap)

- [ ] AAVSO export
- [ ] LC classification hint (shape + bp_rp + catalog type)
- [ ] TESS lightkurve LAB
- [ ] Raspberry Pi 5 / StellarMate deployment

## Next session — start here

**Last completed work:** Photometry PDF pipeline wired through `pdf_report.generate_report`, `photometry_report.generate_photometry_report` extended (English copy, variability cover block, compact pages for stars without LC/COMP, TESS + catalog crossmatch sections, footer format, `report_{setup}.pdf` path). Aperture Photometry UI triggers PDF build; Variability autorun after full Phase 2A batch clears bullets and sets `var_candidates` / reset on empty RMS / draft override.

**Suggested next steps:**
1. Run end-to-end PDF on a real draft with TESS + crossmatch populated; tune `tess_extra_h` / layout if vertical overflow persists on dense pages.
2. Decide whether legacy `VYVAR_report_*.pdf` auto-emission after each setup should be migrated to `report_{setup}.pdf` only for consistency.
3. Pin `streamlit` and Python in `requirements.txt` / `runtime.txt` for reproducible deploy.

## Notes for AI assistant

- User = architect/designer; Cursor = builder; treat long chat instructions as specs, not as mandatory final code without verification.
- Prefer diagrams/mockups for large UI changes before coding.
- Conversation: **Slovak** acceptable; **code and PDF user-visible strings: English**.
- Platform: **Windows** dev now; target **Raspberry Pi 5 + StellarMate** later — avoid Windows-only assumptions in new code where possible.
- **VYVAR vision:** automatic pipeline — morning coffee + PDF summarizing the night’s photometry and variability context.

---

## Git history — recent design-relevant commits (abridged)

- `762278f` — Photometry/variability QA, Gaia fallbacks, comp tiers, PDF QA page, Masterstar paths, variability config + edge filters.
- `333c298` — Photometry PDF report, variability dashboard, GAIA script refresh.
- `1372a07` — Phase2A color-term gate; PSF runner work.
- `ab34a62` — PSF runner with COMP-calibrated LCs.
- `9755ea5` — BJD/JD CSV precision, TIME-OBS mid-time, AIJ-style plot axes.
- `854acf2` — Draft override, batch phases, suspected LCs, QA.
- `9aad807` — MAKE MASTERSTAR feature block, footer status plumbing.
- `40ead13` / `eb0d6ad` — Ensemble flux aggregation aligned with AIJ-style totals.
- Multiple commits — **Gaussian FWHM** for apertures from `VY_FWHM×0.667` (DAO→Gaussian) replacing older moment-based heuristics.

---

*End of VYVAR_STATE.md — regenerate when architecture or session keys change materially.*
