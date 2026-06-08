# VYVAR — Code map (modul → rola vo workflow)

**Dátum:** 2026-06-08 · **Autor:** Claude · **Pripravil:** Cursor (z docstringov a importov)

Prehľad každého VYVAR `.py` modulu: čo robí v pipeline, kľúčové väzby, orientačný Cython status.
Živý dokument — pri pridaní/zmazaní modulu aktualizovať tabuľku a kontrolu úplnosti na konci.

**Legenda — Cython?:** `yes` = core compute (kandidát na kompiláciu); `no` = Streamlit/UI/config; `entry` = CLI/runner; `util` = pomocné, väčšinou zostáva `.py`.

## ENTRY / RUNNERS

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `app.py` | entry | Streamlit entrypoint pre live aj archívne drafty; orchestruje UI záložky a volá pipeline. | — | entry |
| `night_run.py` | entry | Headless VYVAR night pipeline runner. Extracted from ``app.py`` ``_run_vyvar_full_pipeline``. | scripts/archive/diag/_debug_fk343.py; scripts/archive/draft_runs/_simulate_night_run_draft343.py; scripts/chiandh_allfilters_overnight.py; scripts/chiandh_continue375_solve.py; … | entry |
| `simulate_night_run.py` | entry | E2E simulácia nočného behu (ekvivalent UI Session Upload → RUN VYVAR); voliteľný `--dry-run`. | — | entry |
| `xval_run.py` | entry | Offline cross-validation harness: nezávislá photutils/sep extrakcia vs VYVAR dao_flux a lc_rms (draft-agnostic). | `xval_harness_core` | entry |
| `psf_runner.py` | entry | Standalone CLI pre ePSF build + PSF fotometriu na draft (dev/regresia, volá ``psf_photometry``). | `psf_photometry`; `psf_runner` | entry |
| `run_crowding_index.py` | entry | Runner pre depth-aware crowding index — zapisuje ``crowding_index.json`` per draft (read-only voči pipeline). | `crowding_index` | entry |
| `run_smoothness_report.py` | entry | Jednorazový dev skript: RMS variabilita + smoothness ratio na hardcoded draft (``variability_detector``). | ``variability_detector`` | util |
| `comp_qa.py` | entry | Comp-star LOO QA CLI — delegates to comp_qa_core (same math as pipeline stage). | — | entry |
| `trust_flag.py` | entry | CLI for per-target trust flags (uses draft photometry_summary.csv). | — | entry |
| `orchestrator/vyvar_orchestrator.py` | entry | VYVAR Orchestrator — bridges Claude (claude.ai) and Cursor via files. Milan runs this script; it handles the Claude API loop automatically. | — | entry |

## PIPELINE CORE

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `pipeline.py` | core | Core processing pipeline for FITS observations. | photometry_report.py; photometry_core.py; night_run.py; importer.py; hrd_analysis.py; … | yes |
| `photometry_core.py` | core | Photometry core — zlúčený modul (photometry + photometry_phase2a). | method_lc_output.py; export_reports.py; crowding_index.py; comp_selection_per_target.py; check_star_kmag.py; … | yes |
| `photometry.py` | core | Backward compatibility — pôvodný modul; implementácia je v ``photometry_core``. | ui_select_stars.py; pipeline.py; importer.py | yes |
| `photometry_phase2a.py` | core | Backward compatibility — pôvodný modul; implementácia je v ``photometry_core``. | ui_aperture_photometry.py; pipeline.py | yes |
| `calibration.py` | core | Master calibration resampling (software binning) pre dark/flat/light FITS podľa CalibrationLibrary. | `pipeline`; `ui_calibration`; `ui_calibration_library` | yes |
| `importer.py` | core | Session importer for VYVAR (file-first workflow). | night_run.py; draft_provenance.py; app.py; scripts/_platesolve_test_v842.py; … | yes |
| `proc_frame_store.py` | core | ProcFrameStore — unified in-memory store for proc_*.csv frames. Replaces shared_csv_cache (Phase 1) and _phase2a_csv_cache (Phase 2A) | photometry_core.py; comp_qa_core.py; scripts/comp_qa_flagged_lcs.py; scripts/diagnose_epsf_quality_364.py; … | yes |
| `draft_provenance.py` | core | Draft run provenance — calibration mode and manifest I/O. | pipeline.py; photometry_report.py; photometry_core.py; night_run.py; app.py; … | yes |
| `masterstar_context.py` | core | Read-only summary from ``MASTERSTAR.fits`` for Settings / UI (WCS, scale, VY_* headers). | ui_settings.py; ui_dao_stars.py | yes |
| `masterstar_qa_plot.py` | core | MASTERSTAR QA: downsampling + vrstvy DAO / MATCH / Gaia kužeľ na PNG (Streamlit). | ui_photometry_quality.py; ui_masterstar_qa.py | yes |

## COMP / QA / TRUST

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `comp_selection_per_target.py` | core | Per-target comparison star selection (CQ-3 / PERF-4B / PERF-9). Extracted from ``photometry_core.select_comparison_stars_per_target``. | psf_photometry.py; photometry_core.py; scripts/archive/diag/_diag_catalog_only_sep.py; scripts/archive/fixes/patch_orchestrator.py; … | yes |
| `comp_pool_rms.py` | core | Globálny RMS výpočet pre comp pool (jedna pass naprieč framami). Oddelené od ``photometry_core`` kvôli objemu a cyklickým importom. | tests/test_comp_determinism_synthetic.py; photometry_core.py; comp_selection_per_target.py | yes |
| `comp_qa_core.py` | core | Comp-star LOO QA (Sokolovsky indices + magnitude locus) — shared core for pipeline and CLI. | tests/test_proc_csv_glob.py; photometry_core.py; comp_qa.py; scripts/diagnose_nclean_trust.py | yes |
| `trust_flag_core.py` | core | Per-target trust flag (GREEN / YELLOW / RED) — shared by pipeline and CLI. Uses draft ``photometry_summary.csv`` columns from comp QA | trust_flag.py; photometry_core.py; export_reports.py; scripts/diagnose_nclean_trust.py | yes |
| `crowding_index.py` | core | Depth-aware crowding index nezávislý od detekcie (paralelná diagnostika pre adaptive PSF). | `run_crowding_index`; `photometry_core`; adaptive verify skripty | yes |
| `dilution.py` | core | dilution.py — Flux dilution factor computation for VYVAR (TODO-GS11). Computes per-star dilution factor D = F_star / (F_star + ΣF_neighbors) | tests/test_gs11_pipeline.py; tests/test_dilution.py; photometry_core.py; method_lc_output.py | yes |
| `validate_lc_crossval.py` | core | LC cross-validation: VYVAR photometry_summary lc_rms vs dao_flux differential LC. Reads dao_flux from proc_*.csv (same flux as VYVAR Phase 2A), applies the same | scripts/build_lc_from_fits_aperture.py; scripts/test_aperture_sweep.py | yes |
| `xval_harness_core.py` | core | Shared helpers for the offline cross-validation harness (``xval_run.py``). Confidence thresholds and LOO differential utilities — not used by the production pipeline. | xval_run.py | yes |

## CATALOG / CROSSMATCH

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `database.py` | core | SQLite database layer for the VYVAR project. | dilution.py; config.py; catalog_crossmatch.py; calibration.py; astrometry_optimizer.py; … | yes |
| `catalog_crossmatch.py` | core | Standalone cross-match of variable-star candidates against public catalogs (SIMBAD, Vizier). | ui_variability.py; ui_aperture_photometry.py; crossmatch_runner.py; scripts/verify_vsx_local_db.py | yes |
| `crossmatch_runner.py` | core | Batch auto-crossmatch kandidátov: doplní stĺpec „katalogy“ a cache JSON cez ``catalog_crossmatch``. | `photometry_core` (pipeline stage); `tess_runner` | util |
| `gaia_catalog_id.py` | core | Canonical Gaia DR3 source_id normalization utilities for VYVAR. All catalog_id normalization must route through normalize_gaia_source_id() | dilution.py; database.py; comp_selection_per_target.py; comp_qa_core.py; check_star_kmag.py; … | yes |
| `check_star_kmag.py` | core | Check-star measured KMAG for AAVSO ensemble exports (additive reporting). | photometry_report.py; photometry_core.py; export_reports.py; scripts/backfill_check_kmag_sidecars.py | yes |

## PLATESOLVE / ASTROMETRY

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `vyvar_platesolver.py` | core | Plate solving: ICRS hint z FITS/UI hlavičiek; fallback na blind solver (lokálna Gaia DR3). | `pipeline`; blind/verify skripty | yes |
| `vyvar_blind_solver.py` | core | VYVAR Blind Plate Solver — Triangle Hash Matching. Nájde aproximatívne RA/Dec stredu snímky bez akéhokoľvek hintu | tests/test_blind_knn_construction.py; scripts/blind_density_runbook.py; scripts/blind_index_regression.py; scripts/diagnose_blind_solver_380.py; scripts/diagnose_blind_solver_wide.py; … | yes |
| `vyvar_blind_series.py` | core | Blind index series: config tier paths, scale-aware order, try-in-order verify. | vyvar_platesolver.py; tests/test_blind_series.py; scripts/blind_solve_rate.py | yes |
| `astrometry_optimizer.py` | core | MASTERSTAR astrometry optimizer: displacement model, SIP refit a re-match s širším Gaia radiusom. | `pipeline`; `vyvar_platesolver` | yes |
| `vyvar_alignment_frame.py` | core | Per-frame alignment worker (DAO + WCS reproject / astroalign) pre ProcessPoolExecutor. | `pipeline` | yes |
| `platesolve_ui_paths.py` | core | Resolve ``draft/.../platesolve`` artifacts for Streamlit (per-setup vs legacy root). | ui_photometry_quality.py; ui_masterstar_qa.py; ui_dao_stars.py; ui_aperture_photometry.py; masterstar_context.py; … | yes |

## PSF

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `psf_photometry.py` | core | Effective PSF (ePSF) construction on MASTERSTAR and per-star PSF photometry. Uses Photutils EPSFBuilder / PSFPhotometry. Does not import ``pipeline`` (avoid cycles). | psf_runner.py; pipeline.py; app.py; scripts/analyze_prepost_alignment_elong.py; … | yes |

## VARIABILITY / HRD

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `variability_detector.py` | core | RMS a VDI variabilita z flux matice poľa; detekcia kandidátov premenných hviezd. | `photometry_core`; `ui_variability`; `run_smoothness_report` | yes |
| `hrd_analysis.py` | core | Hertzsprung–Russell diagram helpers from MASTERSTAR field catalog + local Gaia DR3 SQLite. | ui_hrd.py; photometry_report.py | yes |

## REPORTY / VÝSTUP

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `photometry_report.py` | core | PDF Summary Measure Report pre draft (lightcurves, comp QA, variabilita, TESS, overflow guard). | `app.py`; `night_run.py`; `pdf_report` | yes |
| `export_reports.py` | core | Generuje AAVSO a VarAstro lightcurve exporty (ensemble mag, GS11 poznámky, citácie, filter map). | `photometry_core`; `photometry_report`; `trust_flag_core` | yes |
| `pdf_report.py` | core | PDF variability-aware photometry report (wrapper around photometry_report). | ui_aperture_photometry.py | yes |
| `report_methods.py` | core | Method-keyed report layout for AAVSO / VarAstro / PDF exports. | tests/test_report_methods.py; photometry_report.py; photometry_core.py; method_lc_output.py; export_reports.py | yes |
| `method_lc_output.py` | core | Write alternate-method lightcurve CSVs (PSF / adaptive) for method-keyed reports. | photometry_core.py; scripts/verify_method_report_separation.py | yes |

## TESS

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `tess_runner.py` | core | Batch TESS overenie kandidátov: pre každý catalog_id zapíše ``result.json`` + ``result.txt``. | `photometry_core` (pipeline); `tess_verify` | util |
| `tess_verify.py` | core | Download and analyze TESS cutout light curves for variable-star candidates (lightkurve). | ui_variability.py; ui_aperture_photometry.py; tess_runner.py | util |

## CONFIG / PARAMS

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `config.py` | core | Project configuration for the variable-star processing system. | importer.py; export_reports.py; comp_selection_per_target.py; citations.py; check_star_kmag.py; … | no |
| `param_resolver.py` | core | Unified per-parameter provenance resolver (VYVAR Phase 1). ONE place that decides, per physical parameter, which source wins: | psf_photometry.py; pipeline.py; photometry_core.py; crowding_index.py; … | no |

## UTIL

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `utils.py` | core | Utility helpers shared across the project. | importer.py; database.py; crossmatch_runner.py; calibration.py; astrometry_optimizer.py; … | util |
| `time_utils.py` | core | JD / HJD / BJD helpers for per-frame catalog metadata (mid-exposure times). | tests/test_geo.py; pipeline.py | util |
| `jd_axis_format.py` | core | Formátovanie JD osi (AIJ): relatívne ticky = JD − floor(min) pre prehľadné grafy. | `photometry_core`; UI LC grafy | util |
| `infolog.py` | core | Ring buffer + logging handler for the Streamlit «Infolog» tab (session-global in-process). | hrd_analysis.py; database.py; comp_selection_per_target.py; calibration.py; astrometry_optimizer.py; … | util |
| `fits_suffixes.py` | core | Shared FITS filename suffix rules (case-insensitive on disk). | utils.py; pipeline.py; importer.py | util |
| `lunar_context.py` | core | lunar_context.py — Lunar observing conditions for a VYVAR session. Uses astropy ephemeris (built-in, no internet required). | tests/test_lunar_context.py; photometry_core.py | util |
| `optics_autodetect.py` | core | Scan Source auto-detect (VYVAR Phase 3). Fingerprint the camera / telescope / observer site from a representative FITS | app.py | util |
| `optics_selection.py` | core | Active camera + telescope selection — one source of truth for VYVAR. Priority when resolving optics for platesolve / calibration / masters: | tests/test_optics_resolve.py; pipeline.py; app.py | util |
| `citations.py` | core | VYVAR run citation emitter — single source: CITATIONS.bib. | tests/test_export_citations.py; photometry_report.py; photometry_core.py; export_reports.py; scripts/reexport_draft_aavso.py; … | util |

## STATUS

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `vyvar_ui_status.py` | core | Spoločná aktualizácia spodného stavového riadku (Streamlit session + rerender). | ui_select_stars.py; ui_aperture_photometry.py | no |

## UI (Streamlit)

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `ui_aperture_photometry.py` | UI | Aperture Photometry Lightcurves — Fáza 2A UI. | photometry_report.py; night_run.py; app.py; scripts/_rerun_phase01_draft365.py; … | no |
| `ui_calibration.py` | UI | Streamlit UI helpers for calibration / smart-import (equipment header, master binning, multi-obs status). | app.py | no |
| `ui_calibration_library.py` | UI | Streamlit dashboard for CalibrationLibrary validity overview, delete, and master generation. | app.py | no |
| `ui_components.py` | UI | Reusable Streamlit UI components. | ui_quality_dashboard.py; app.py | no |
| `ui_dao_stars.py` | UI | DAO-STARS: úprava hlavných MASTERSTAR parametrov detekcie / SIP (config.json). | ui_settings.py | no |
| `ui_database_explorer.py` | UI | Database Explorer tab: table browser + staging maintenance (OBS_FILES / OBS_DRAFT only). | app.py | no |
| `ui_epsf_dashboard.py` | UI | Standalone ePSF photometry dashboard tab. | app.py | no |
| `ui_finalization.py` | UI | UI: final approval step — persist OBSERVATION and archive key artifacts under ``finalized/``. | app.py | no |
| `ui_hrd.py` | UI | Streamlit tab: field Hertzsprung–Russell diagram (HRD) from masterstars + Gaia. | ui_aperture_photometry.py | no |
| `ui_masterstar_qa.py` | UI | MASTERSTAR QA: projekcia masterstars cez WCS, metriky a náhľad mapy (DAO / MATCH / Gaia). | app.py | no |
| `ui_photometry.py` | UI | Streamlit dashboard: photometry-related ``AppConfig`` fields (výstupy a prepínače). | ui_settings.py | no |
| `ui_photometry_quality.py` | UI | Streamlit dashboard: Photometry Quality Diagnostic (MASTERSTAR + per-frame catalogs). | ui_settings.py | no |
| `ui_photometry_results.py` | UI | Streamlit tabs: aperture / PSF time series, comp validation, variable detection (per-frame sidecars). **Odpojené od `app.py`** (ROADMAP — zámer vs regresia). | — | no |
| `ui_quality_dashboard.py` | UI | Streamlit Quality Dashboard: OBS_FILES metrics, Plotly, FITS preview, data editor. | app.py | no |
| `ui_select_stars.py` | UI | Select Stars dashboard — Fáza 0+1: výber aktívnych premenných a porovnávacích hviezd. | ui_suspected_lightcurves.py | no |
| `ui_settings.py` | UI | Unified Settings dashboard: paths, QC, photometry, phase 0+1, alignment, tools + rich help. | app.py | no |
| `ui_suspected_lightcurves.py` | UI | Light curves for suspected variables — inštrumentálna mag z per-frame CSV. **Odpojené od `app.py`** (ROADMAP — zámer vs regresia). | — | no |
| `ui_variability.py` | UI | Streamlit záložka Variability Detection: RMS/VDI analýza, VSX, crossmatch, TESS, period acceptance. | `app.py`; `ui_aperture_photometry`; `variability_detector` | no |

## GAIA build skripty (GAIA_DR3/)

| Súbor | Typ | Rola vo workflow (1–2 vety) | Volá / volaný kým | Cython? |
|---|---|---|---|---|
| `GAIA_DR3/build_gaia_catalog.py` | build | Script #1: stiahne Gaia DR3 do lokálnej SQLite (`gaia_dr3`) s resume a batch zápisom. | pilot/build skripty | entry |
| `GAIA_DR3/build_blind_index.py` | build | VYVAR script #2 — Gaia blind triangle index (fine + wide PKL tiers). Builds density-matched triangle hash indexes from local ``gaia_dr3`` SQLite. | tests/test_blind_knn_construction.py | entry |

---

## Kontrola úplnosti

- Root `.py` modulov: **78**
- + orchestrator + 2 GAIA build skripty: **81**
- Riadkov v tabuľke: **81**
- Status: **OK** (tabuľka == root + 3)

## Poznámky

- `ui_photometry_results.py` a `ui_suspected_lightcurves.py` nie sú importované z `app.py` (legacy/alternatívne UI; otvorená otázka v ROADMAP).
- `photometry.py` / `photometry_phase2a.py` sú tenké re-exporty; implementácia je v `photometry_core.py`.
- Offline harness: `xval_run.py` + `xval_harness_core.py` — mimo produkčného pipeline.
- GAIA: script #1 `build_gaia_catalog.py` (SQLite), script #2 `build_blind_index.py` (fine+wide PKL).
