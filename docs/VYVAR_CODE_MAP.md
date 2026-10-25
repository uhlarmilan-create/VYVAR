# VYVAR - Code map (modul -> rola vo workflow)

**Datum:** 2026-06-08 . **Autor:** Claude . **Pripravil:** Cursor (z docstringov a importov)

Prehlad kazdeho VYVAR `.py` modulu: co robi v pipeline, klucove vazby, orientacny Cython status.
Zivy dokument - pri pridani/zmazani modulu aktualizovat tabulku a kontrolu uplnosti na konci.

**Legenda - Cython?:** `yes` = core compute (kandidat na kompilaciu); `no` = Streamlit/UI/config; `entry` = CLI/runner; `util` = pomocne, vacsinou zostava `.py`.

## ENTRY / RUNNERS

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `app.py` | entry | Streamlit entrypoint pre live aj archivne drafty; orchestruje UI zalozky a vola pipeline. | - | entry |
| `night_run.py` | entry | Headless VYVAR night pipeline runner. Extracted from ``app.py`` ``_run_vyvar_full_pipeline``. | scripts/archive/diag/_debug_fk343.py; scripts/archive/draft_runs/_simulate_night_run_draft343.py; scripts/chiandh_allfilters_overnight.py; scripts/chiandh_continue375_solve.py; ... | entry |
| `simulate_night_run.py` | entry | E2E simulacia nocneho behu (ekvivalent UI Session Upload -> RUN VYVAR); volitelny `--dry-run`. | - | entry |
| `xval_run.py` | entry | Offline cross-validation harness: nezavisla photutils/sep extrakcia vs VYVAR dao_flux a lc_rms (draft-agnostic). | `xval_harness_core` | entry |
| `psf_runner.py` | entry | Headless RUN ePSF na jeden draft / obs group (tenky CLI nad `epsf_stage.run_epsf_stage`). | `epsf_stage` | entry |
| `run_smoothness_report.py` | entry | Jednorazovy dev skript: RMS variabilita + smoothness ratio na hardcoded draft (``variability_detector``). | ``variability_detector`` | util |
| `comp_qa.py` | entry | Comp-star LOO QA CLI - delegates to comp_qa_core (same math as pipeline stage). | - | entry |
| `trust_flag.py` | entry | CLI for per-target trust flags (uses draft photometry_summary.csv). | - | entry |
| `orchestrator/vyvar_orchestrator.py` | entry | VYVAR Orchestrator - bridges Claude (claude.ai) and Cursor via files. Milan runs this script; it handles the Claude API loop automatically. | - | entry |

## PIPELINE CORE

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `pipeline.py` | core | Core processing pipeline for FITS observations. | photometry_report.py; photometry_core.py; night_run.py; importer.py; hrd_analysis.py; ... | yes |
| `photometry_core.py` | core | Photometry core - facade over the photometry modules. | method_lc_output.py; export_reports.py; comp_selection_per_target.py; check_star_kmag.py; ... | yes |
| `photometry.py` | core | Backward compatibility - povodny modul; implementacia je v ``photometry_core``. | ui_select_stars.py; pipeline.py; importer.py | yes |
| `aperture_photometry.py` | core | Aperture photometry and light curves (with `aperture_photometry_state.py`, `aperture_photometry_target.py`); target and comparison-star selection is `target_comp_selection.py`. | ui_aperture_photometry.py; pipeline.py | yes |
| `calibration.py` | core | Master calibration resampling (software binning) pre dark/flat/light FITS podla CalibrationLibrary. | `pipeline`; `ui_calibration`; `ui_calibration_library` | yes |
| `importer.py` | core | Session importer for VYVAR (file-first workflow). | night_run.py; draft_provenance.py; app.py; scripts/_platesolve_test_v842.py; ... | yes |
| `proc_frame_store.py` | core | ProcFrameStore - unified in-memory store for proc_*.csv frames. Replaces shared_csv_cache (comparison-star selection) and the aperture-photometry CSV cache | photometry_core.py; comp_qa_core.py; scripts/comp_qa_flagged_lcs.py; scripts/diagnose_epsf_quality_364.py; ... | yes |
| `draft_provenance.py` | core | Draft run provenance - calibration mode and manifest I/O. | pipeline.py; photometry_report.py; photometry_core.py; night_run.py; app.py; ... | yes |
| `masterstar_context.py` | core | Read-only summary from ``MASTERSTAR.fits`` for Settings / UI (WCS, scale, VY_* headers). | ui_settings.py; ui_dao_stars.py | yes |
| `masterstar_qa_plot.py` | core | MASTERSTAR QA: plate-solve facts, DAO <-> Gaia match counts, overlay map (MSQA-2026-10). | ui_photometry_quality.py; ui_masterstar_qa.py | yes |

## COMP / QA / TRUST

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `comp_selection_per_target.py` | core | Per-target comparison star selection (CQ-3 / PERF-4B / PERF-9). Extracted from ``photometry_core.select_comparison_stars_per_target``. | psf_photometry.py; photometry_core.py; scripts/archive/diag/_diag_catalog_only_sep.py; scripts/archive/fixes/patch_orchestrator.py; ... | yes |
| `comp_pool_rms.py` | core | Globalny RMS vypocet pre comp pool (jedna pass napriec framami). Oddelene od ``photometry_core`` kvoli objemu a cyklickym importom. | tests/test_comp_determinism_synthetic.py; photometry_core.py; comp_selection_per_target.py | yes |
| `comp_qa_core.py` | core | Comp-star LOO QA (Sokolovsky indices + magnitude locus) - shared core for pipeline and CLI. | tests/test_proc_csv_glob.py; photometry_core.py; comp_qa.py; scripts/diagnose_nclean_trust.py | yes |
| `trust_flag_core.py` | core | Per-target trust flag (GREEN / YELLOW / RED) - shared by pipeline and CLI. Uses draft ``photometry_summary.csv`` columns from comp QA | trust_flag.py; photometry_core.py; export_reports.py; scripts/diagnose_nclean_trust.py | yes |
| `dilution.py` | core | dilution.py - Flux dilution factor computation for VYVAR (TODO-GS11). Computes per-star dilution factor D = F_star / (F_star + SigmaF_neighbors) | tests/test_gs11_pipeline.py; tests/test_dilution.py; photometry_core.py; method_lc_output.py | yes |
| `validate_lc_crossval.py` | core | LC cross-validation: VYVAR photometry_summary lc_rms vs dao_flux differential LC. Reads dao_flux from proc_*.csv (same flux as VYVAR aperture photometry), applies the same | scripts/build_lc_from_fits_aperture.py; scripts/test_aperture_sweep.py | yes |
| `xval_harness_core.py` | core | Shared helpers for the offline cross-validation harness (``xval_run.py``). Confidence thresholds and LOO differential utilities - not used by the production pipeline. | xval_run.py | yes |

## CATALOG / CROSSMATCH

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `database.py` | core | SQLite database layer for the VYVAR project. | dilution.py; config.py; catalog_crossmatch.py; calibration.py; astrometry_optimizer.py; ... | yes |
| `catalog_crossmatch.py` | core | Standalone cross-match of variable-star candidates against public catalogs (SIMBAD, Vizier). | ui_variability.py; ui_aperture_photometry.py; crossmatch_runner.py; scripts/verify_vsx_local_db.py | yes |
| `crossmatch_runner.py` | core | Batch auto-crossmatch kandidatov: doplni stlpec 'katalogy' a cache JSON cez ``catalog_crossmatch``. | `photometry_core` (pipeline stage); `tess_runner` | util |
| `gaia_catalog_id.py` | core | Canonical Gaia DR3 source_id normalization utilities for VYVAR. All catalog_id normalization must route through normalize_gaia_source_id() | dilution.py; database.py; comp_selection_per_target.py; comp_qa_core.py; check_star_kmag.py; ... | yes |
| `check_star_kmag.py` | core | Check-star measured KMAG for AAVSO ensemble exports (additive reporting). | photometry_report.py; photometry_core.py; export_reports.py; scripts/backfill_check_kmag_sidecars.py | yes |

## PLATESOLVE / ASTROMETRY

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `vyvar_platesolver.py` | core | Plate-solve helpers kept after PLATESOLVE-2026-10: FITS pointing hints, Gaia proper motion, Gaia cone catalogue, sibling-filter MASTERSTAR WCS recovery. | `masterstar_build`; `pipeline_astrometry`; `dao_gaia_common` | yes |
| `astrosolve.py` | core | MASTERSTAR plate solver (PLATESOLVE-2026-10): 4-star pattern codes, hinted / blind hypotheses, Bayesian log-odds verification, TAN+SIP refinement (order by BIC), `confirm_fits` for handed-over WCS. | `masterstar_build`; `astro_index` | yes |
| `astro_index.py` | core | Blind plate-solve index from the local Gaia DB (multi-scale 4-star patterns, memmaps in `vyvar_astro_index/`); `python GAIA_DR3/build_astro_index.py`. | `astrosolve` | entry |
| `vyvar_alignment_frame.py` | core | Per-frame alignment worker (DAO + WCS reproject / astroalign) pre ProcessPoolExecutor. | `pipeline` | yes |
| `platesolve_ui_paths.py` | core | Resolve ``draft/.../platesolve`` artifacts for Streamlit (per-setup vs legacy root). | ui_photometry_quality.py; ui_masterstar_qa.py; ui_dao_stars.py; ui_aperture_photometry.py; masterstar_context.py; ... | yes |

## PSF

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `psf_photometry.py` | core | ePSF build-star selection on the MASTERSTAR catalogue (`select_epsf_build_stars`), ePSF builder helpers (photutils 3.1.1), gain/RN authority; the per-star cutout fitter was removed (EPSF-CLEAN-2026-10). | psf_runner.py; pipeline.py; app.py; ... | yes |
| `epsf_frame.py` | core | Per-frame ePSF + night spatial perturbation (3x3/2x2 GriddedPSFModel); frame-level PSF photometry fitted jointly with Gaia neighbours at fixed Gaia positions, unresolved fainter companions merged, aligned-footprint mask (EPSF-SPATIAL / EPSF-BLEND / EPSF-MERGE / EPSF-FOOTPRINT-2026-10). | epsf_psf_merge.py | no |
| `frame_footprint.py` | core | Data footprint of an aligned frame on the reference grid (VY_AM0..8): covered-pixel mask and the off-footprint test for apertures / PSF fit boxes and their sky annuli (EPSF-FOOTPRINT / AP-FOOTPRINT-2026-10). | epsf_frame.py, aperture_pertarget.py, photometry_shared.py | no |
| `epsf_export.py` | core | ePSF light curve export for the user: bjd_tdb, delta_mag, err with a descriptive header, one file per target and setup (EPSF-EXPORT-2026-10). | epsf_stage.py; ui_epsf_dashboard.py | yes |
| `epsf_lc_compare.py` | core | Aperture vs ePSF differential light curves of one draft: p2p scatter, p2p/err, PSF-AP difference vs frame FWHM (EPSF-LC-COMPARE-2026-10). | ui_epsf_dashboard.py; star_page_m3.py | yes |

## VARIABILITY / HRD

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `variability_detector.py` | core | RMS a VDI variabilita z flux matice pola; detekcia kandidatov premennych hviezd. | `photometry_core`; `ui_variability`; `run_smoothness_report` | yes |
| `hrd_analysis.py` | core | Hertzsprung-Russell diagram helpers from MASTERSTAR field catalog + local Gaia DR3 SQLite. | ui_hrd.py; photometry_report.py | yes |

## REPORTY / VYSTUP

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `photometry_report.py` | core | PDF Summary Measure Report pre draft (lightcurves, ePSF page after the aperture page (EPSF-PDF-2026-10), comp QA, variabilita, TESS, overflow guard). | `app.py`; `night_run.py`; `pdf_report` | yes |
| `export_reports.py` | core | Generuje AAVSO a VarAstro lightcurve exporty (ensemble mag, GS11 poznamky, citacie, filter map). | `photometry_core`; `photometry_report`; `trust_flag_core` | yes |
| `pdf_report.py` | core | PDF variability-aware photometry report (wrapper around photometry_report). | ui_aperture_photometry.py | yes |
| `report_methods.py` | core | Method-keyed report layout for AAVSO / VarAstro / PDF exports. | tests/test_report_methods.py; photometry_report.py; photometry_core.py; method_lc_output.py; export_reports.py | yes |

## TESS

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `tess_runner.py` | core | TESS for one star (UI button) and for the night's candidates when tess_enabled (night run) into `photometry/TESS/<gaia id>/`; analysis in `tess_analysis.py`. | aperture_photometry, ui_variability |
| `transient_field.py` | core | TRANSIENT-2026-10: new sources on the night's frames (per-frame DAO vs MASTERSTAR, linking, forced photometry, empty-sky noise, detector-pixel test, Gaia / OpenNGC checks), moving objects, minor bodies; run by the variability search when transient_search_enabled. | variability_field, ui_variability, photometry_report |
| `minor_bodies.py` | core | Minor planets / comets in the field from MPCORB.DAT(.gz) / CometEls.txt in the catalogue folder: two-body positions, H-G / comet magnitudes, light time, topocentric. | transient_field |
| `variability_store.py` | core | Files the Variability tab and the PDF share: report include list, observer-adopted periods, TESS result reader, Peranso exports, phase / PDM Theta helpers (VAR-TAB-2026-10). | ui_variability, photometry_report |

## CONFIG / PARAMS

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `config.py` | core | Project configuration for the variable-star processing system. | importer.py; export_reports.py; comp_selection_per_target.py; citations.py; check_star_kmag.py; ... | no |
| `param_resolver.py` | core | Unified per-parameter provenance resolver (VYVAR). ONE place that decides, per physical parameter, which source wins: | psf_photometry.py; pipeline.py; photometry_core.py; ... | no |

## UTIL

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `utils.py` | core | Utility helpers shared across the project. | importer.py; database.py; crossmatch_runner.py; calibration.py; astrometry_optimizer.py; ... | util |
| `time_utils.py` | core | JD / HJD / BJD helpers for per-frame catalog metadata (mid-exposure times). | tests/test_geo.py; pipeline.py | util |
| `jd_axis_format.py` | core | Formatovanie JD osi (AIJ): relativne ticky = JD - floor(min) pre prehladne grafy. | `photometry_core`; UI LC grafy | util |
| `infolog.py` | core | The run's single draft infolog `infolog_<ts>.txt` (append, flushed per line) + logging handler (INFOLOG-ONE-2026-10). | hrd_analysis.py; database.py; comp_selection_per_target.py; calibration.py; astrometry_optimizer.py; ... | util |
| `fits_suffixes.py` | core | Shared FITS filename suffix rules (case-insensitive on disk). | utils.py; pipeline.py; importer.py | util |
| `lunar_context.py` | core | lunar_context.py - Lunar observing conditions for a VYVAR session. Uses astropy ephemeris (built-in, no internet required). | tests/test_lunar_context.py; photometry_core.py | util |
| `optics_autodetect.py` | core | Scan Source auto-detect (VYVAR Phase 3). Fingerprint the camera / telescope / observer site from a representative FITS | app.py | util |
| `optics_selection.py` | core | Active camera + telescope selection - one source of truth for VYVAR. Priority when resolving optics for platesolve / calibration / masters: | tests/test_optics_resolve.py; pipeline.py; app.py | util |
| `citations.py` | core | VYVAR run citation emitter - single source: CITATIONS.bib. | tests/test_export_citations.py; photometry_report.py; photometry_core.py; export_reports.py; scripts/reexport_draft_aavso.py; ... | util |

## STATUS

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `vyvar_ui_status.py` | core | Spolocna aktualizacia spodneho stavoveho riadku (Streamlit session + rerender). | ui_select_stars.py; ui_aperture_photometry.py | no |

## UI (Streamlit)

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `ui_aperture_photometry.py` | UI | Aperture Photometry Lightcurves - aperture photometry UI. | photometry_report.py; night_run.py; app.py; scripts/_rerun_phase01_draft365.py; ... | no |
| `ui_calibration.py` | UI | Streamlit UI helpers for calibration / smart-import (equipment header, master binning, multi-obs status). | app.py | no |
| `ui_calibration_library.py` | UI | Streamlit dashboard for CalibrationLibrary validity overview, delete, and master generation. | app.py | no |
| `ui_components.py` | UI | Reusable Streamlit UI components. | ui_quality_dashboard.py; app.py | no |
| `ui_dao_stars.py` | UI | DAO-STARS: uprava hlavnych MASTERSTAR parametrov detekcie / SIP (config.json). | ui_settings.py | no |
| `ui_database_explorer.py` | UI | Database Explorer tab: table browser + staging maintenance (OBS_FILES / OBS_DRAFT only). | app.py | no |
| `ui_epsf_dashboard.py` | UI | ePSF tab: target selection (ALL / VSX types / names, live preview), RUN ePSF over every setup of the night (EPSF-MULTISETUP-2026-10), ePSF light curves with the aperture curve as an option (EPSF-TARGETS-2026-10). | app.py | no |
| `ui_finalization.py` | UI | UI: final approval step - persist OBSERVATION and archive key artifacts under ``finalized/``. | app.py | no |
| `ui_hrd.py` | UI | Field HRD view (HRD-TAB-2026-10): three diagram variants, one notable-star table, interactive field image. | ui_aperture_photometry.py | no |
| `hrd_field_render.py` | core | Field images of the HRD view: inverted, catalogue colour (vivid, Lupton 2004 hue-preserving) and physical blackbody; BP-RP palette and Teff. | ui_hrd.py | yes |
| `ui_masterstar_qa.py` | UI | MASTERSTAR QA tab: informative - plate solving, DAO <-> Gaia matching, map (Gaia / matches). | app.py | no |
| `ui_photometry.py` | UI | Streamlit dashboard: photometry-related ``AppConfig`` fields (vystupy a prepinace). | ui_settings.py | no |
| `ui_photometry_quality.py` | UI | Streamlit dashboard: Photometry Quality Diagnostic (MASTERSTAR + per-frame catalogs). | ui_settings.py | no |
| `ui_photometry_results.py` | UI | Streamlit tabs: aperture / PSF time series, comp validation, variable detection (per-frame sidecars). **Odpojene od `app.py`** (ROADMAP - zamer vs regresia). | - | no |
| `ui_quality_dashboard.py` | UI | Streamlit Quality Dashboard: OBS_FILES metrics, Plotly, FITS preview, data editor. | app.py | no |
| `ui_select_stars.py` | UI | Select Stars dashboard - target and comparison-star selection. | ui_suspected_lightcurves.py | no |
| `ui_settings.py` | UI | Unified Settings dashboard: paths, QC, photometry, phase 0+1, alignment, tools + rich help. | app.py | no |
| `ui_suspected_lightcurves.py` | UI | Light curves for suspected variables - instrumentalna mag z per-frame CSV. **Odpojene od `app.py`** (ROADMAP - zamer vs regresia). | - | no |
| `ui_variability.py` | UI | Variability tab (VAR-TAB-2026-10): interactive hockey-stick, one table of new candidates + recovered VSX variables, night LC, TESS download and Peranso-like period / phase view (all plotly). | `app.py` | no |

## GAIA build skripty (GAIA_DR3/)

| Subor | Typ | Rola vo workflow (1-2 vety) | Vola / volany kym | Cython? |
|---|---|---|---|---|
| `GAIA_DR3/build_gaia_catalog.py` | build | Script #1: stiahne Gaia DR3 do lokalnej SQLite (`gaia_dr3`) s resume a batch zapisom. | pilot/build skripty | entry |

---

## Kontrola uplnosti

- Root `.py` modulov: **78**
- + orchestrator + 2 GAIA build skripty: **81**
- Riadkov v tabulke: **81**
- Status: **OK** (tabulka == root + 3)

## Poznamky

- `ui_photometry_results.py` a `ui_suspected_lightcurves.py` nie su importovane z `app.py` (legacy/alternativne UI; otvorena otazka v ROADMAP).
- `photometry.py` is a thin re-export of `photometry_core.py`.
- Offline harness: `xval_run.py` + `xval_harness_core.py` - mimo produkcneho pipeline.
- GAIA: `build_gaia_catalog.py` (SQLite), blind plate-solve index `GAIA_DR3/build_astro_index.py` (folder `vyvar_astro_index/`).

