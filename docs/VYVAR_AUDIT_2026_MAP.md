# VYVAR audit 2026 ù Wave 1: capability map (A?Z)

**Date:** 2026-08-13  
**Scope:** Every user-invokable capability, full raw?export flow, reachability, parameters, duplication.  
**Method:** Static trace of `app.py`, `night_run.py`, CLI entry points, `ui_*.py`, `params_registry.json`, prior `VYVAR_CODE_MAP.md` (2026-06-08) updated against current tree (119 `src_py/*.py` modules).

---

## 1. Capability inventory

Legend: **Reach** = production UI | headless (`night_run`) | CLI only | dev/tools | unwired.  
**Last exercised:** from on-disk draft timestamps in `Archive/Drafts/` unless noted UNVERIFIED.

### 1.1 Session import and archive

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Scan Source (smart import plan) | `importer.py:1202` `smart_scan_source`; UI `app.py:2190` | Walks USB tree; matches CalibrationLibrary masters; builds `SmartImportPlan` | production | draft_510 2026-08-13 |
| Auto-detect optics from FITS | `optics_autodetect.py`; UI `app.py:2001` | Fingerprint camera/telescope/location from headers | production | UNVERIFIED recent |
| Create Archive & Do Calibration | `importer.py:1712` `smart_import_session`; UI `app.py:2499` | Copies FITS to draft; DB manifest; optional cal queue | production | draft_510 |
| RUN VYVAR (full chain) | `app.py:119` `_run_vyvar_full_pipeline` ? `night_run.py:497` | Scan?import?cal?QC?preprocess?platesolve?photometry?PDF | production + headless | draft_510 2026-08-13 |
| RUN VYVAR (non-cal / pre-cal) | `app.py:2238`; `draft_provenance.apply_pre_calibrated_import_plan` | Import into `non_calibrated/lights`; skip cal step | production | F-B01 path UNVERIFIED |
| Load existing draft | `app.py:2786` | Opens archived draft by path/ID | production | daily |
| Simulate night run | `simulate_night_run.py:47` | E2E dry-run wrapper | CLI | dev |
| Draft manifest I/O | `draft_provenance.py` | `draft_manifest.json`, calibration provenance | pipeline | draft_510 |
| Known-field library banner | `ui_finalization.py:405`; partial wire `app.py:2366` | Shows field comp-star library | production (banner only) | conditional |
| Use these comp stars (library) | `ui_finalization.py:481` | Copy library CSV into draft | production | conditional |
| Observation finalization panel | `ui_finalization.py:132` `render_finalization` | Persist OBSERVATION + archive | **unwired** | never in UI |

### 1.2 Calibration library and masters

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Calibration Library dashboard | `ui_calibration_library.py:259`; sidebar `app.py:2867` | Validity overview, delete masters | production | UNVERIFIED |
| Generate Master Dark | `ui_calibration_library.py:381`; `calibration.py` | Median stack darks ? library | production | UNVERIFIED |
| Generate Master Flat | `ui_calibration_library.py:413`; `calibration.py` | Median stack raw flats ? library | production | UNVERIFIED |
| Software-bin master resample | `calibration.py`; `pipeline.get_processed_master` | Dark/flat resample to light binning | pipeline | draft_510 |
| CAL-DIAG v2 (INV-CAL-01) | `cal_diag.py:727` `run_cal_diag_pregate` | SUM vs MEAN dark resample gate | pipeline (always on) | draft_510 cal_diag.json |
| CAL stage stamp (INV-CAL-02) | `cal_stage.py`; `pipeline._qc_enrich_one_frame` | `VY_CALSTAGE` + datasum on pixel flush | pipeline | draft_510 |
| OSC channel extraction | `osc_extract.py`; `pipeline.run_osc_channel_extraction_for_archive` | oneRGGB/R/G/B from Bayer mosaic | pipeline (OSC equip) | UNVERIFIED on home rig |
| BPM dark MAD mask | `importer.py:1139` (`bpm_dark_mad_sigma`) | Bad-pixel map from dark stack | import path | UNVERIFIED (no sidecars on disk) |

### 1.3 Calibration (per draft)

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Quick calibrate lights | `pipeline.py:18843` `quick_calibrate_last_import` | Dark/flat/bias apply ? `calibrated/lights` | pipeline | draft_510 |
| RAM QC ? manifest (PERF-10 alt) | `pipeline.py:2079` | FWHM/sky/stars without writing calibrated FITS | pipeline when not PERF-10 | skipped draft_510 |
| PERF-10 DAO QC in calibrate | `pipeline.py:1973` | QC columns during cal | pipeline default | draft_510 |
| Preprocess / QC enrich in-place | `pipeline.py:18027` `qc_enrich_calibrated_lights_in_place` | Sky surface, FWHM, elongation on **calibrated/** | pipeline | draft_510 |
| SAT-DIAG | `sat_diag.py:741` `run_sat_diag` | Raw placed-aperture saturation/linearity | pipeline | draft_510 sat_diag.json |
| Cosmic-ray rejection | **ABSENT** (removed `0ab686f` 2026-08-12) | Deliberately no CR cleaning on science pixels (iron rule 2); departs from van Dokkum (2001) field standard | pipeline | N/A ù decision documented |

### 1.4 QC, MASTERSTAR, alignment

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| FITS QA dashboard | `ui_quality_dashboard.py:349` | FWHM/sky plots; frame reject; MASTERSTAR pick | production | UNVERIFIED |
| Confirm MASTERSTAR / Use as MASTERSTAR | `ui_quality_dashboard.py:956,1068` | DB masterstar source path | production | UNVERIFIED |
| Auto FWHM limit | `night_run.py:796`; `photometry_core.compute_auto_fwhm_limit` | MAD-based QC ceiling | headless + UI | draft_510 |
| Plate solve + align + MASTERSTAR | `pipeline.py:15298` `astrometry_align_and_build_masterstar` | WCS, SIP, astroalign, DAO catalog | pipeline | draft_510 |
| Astrometry optimizer (Grip+SIP) | `astrometry_optimizer.py` | Iterative rematch + SIP refit | pipeline | draft_510 |
| Blind plate solve | `vyvar_blind_solver.py`; `vyvar_blind_series.py` | Triangle hash Gaia match | config-gated | UNVERIFIED recent |
| MASTERSTAR QA tab | `ui_masterstar_qa.py:244` | WCS/DAO/Gaia diagnostic overlay | production | UNVERIFIED |
| DAO-STARS settings | `ui_dao_stars.py:87` | Detection/SIP tuning | Settings/Tools | UNVERIFIED |
| Photometry quality diagnostic | `ui_photometry_quality.py:72` | Per-frame catalog QA plots | Settings/Tools | UNVERIFIED |
| Crowding index | `run_crowding_index.py:23`; `crowding_index.py` | Depth-aware crowding JSON | CLI | dev |
| Invariants end-of-run | `invariants_runtime.py:930` | FAIL/WARN registry checks | pipeline | draft_510 |

### 1.5 Detection and catalogue joins

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| DAO detection (MASTERSTAR) | `pipeline.py` (align body) | DAOStarFinder on reference frame | pipeline | draft_510 |
| Per-frame DAO + match | `vyvar_alignment_frame.py`; pipeline align | Match MASTERSTAR catalog per frame | pipeline | draft_510 |
| Gaia DR3 local SQLite | `GAIA_DR3/build_gaia_catalog.py`; `database.py` | Field cone queries | pipeline + UI | continuous |
| VSX local DB | `database.py`; UI test `ui_settings.py:293` | Variable star types/names | variability UI | UNVERIFIED |
| Exoplanet archive DB | `database.py`; UI test `ui_settings.py:332` | Exoplanet cross-ref | variability UI | UNVERIFIED |
| VSX?Gaia plan matcher | `vsx_gaia_crossmatch.py` | Two-step mixture fit at plan time | pipeline export plan | draft_510 |
| Field catalog cone | `pipeline.py` (catalog build) | `field_catalog_cone.csv` | pipeline | draft_510 |
| Variability detection (RMS/VDI) | `variability_detector.py`; `ui_variability.py:528` | Field RMS matrix; candidates | production | auto on tab |
| Crossmatch dialog (SIMBAD/Vizier) | `catalog_crossmatch.py`; `ui_variability.py:847` | External catalog lookup | production | manual |
| TESS cutout analysis | `tess_verify.py:1343`; `tess_runner.py` | lightkurve sector LC + period | production + batch | UNVERIFIED |
| Target scoping / VSX type filter | `vsx_type_scope.py`; INV-CFG-01 | Out-of-scope VSX types skip | config-gated | UNVERIFIED |
| HRD field diagram | `hrd_analysis.py`; `ui_hrd.py:15` | Gaia CMD from MASTERSTAR field | production sub-tab | UNVERIFIED |

### 1.6 Photometry paths

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Phase 0+1 (comp selection per target) | `photometry_core.py:14545` | Tiers, colour, RMS gates, variable_targets | pipeline | draft_510 |
| Phase 2A (aperture ensemble LC) | `photometry_core.py:10771` | Aperture photometry, ensemble ZP, differential mags | pipeline | draft_510 |
| SNR-optimal aperture table | `photometry_core.py` (aperture_snr) | Draft-constant radii from MASTERSTAR FWHM | pipeline | draft_510 aperture_snr_table.json |
| Comparison stability (p2p/slope) | `photometry_core.py:2941` | Comp quality good/suspect/excluded | pipeline | draft_510 |
| Ensemble normalize (Broeg weights) | `photometry_core.py` `ensemble_normalize` | Weighted ZP; ZP MAD clip **removed** 2026-08-12 | pipeline | draft_510 |
| Colour term fit | `photometry_core.py` (color term block) | Gaia BP-RP linear term; gated | pipeline | UNVERIFIED |
| K2 extinction | `k2_extinction.py` | Literature K2(BP-RP) | config `k2_mode` | UNVERIFIED anchor |
| Dilution factor (GS11) | `dilution.py` | Neighbor flux dilution | pipeline | UNVERIFIED |
| Airmass / extinction detrend | `photometry_core.py` | Optional LC detrend in UI | UI display + optional | UNVERIFIED |
| ePSF build + PSF photometry | `psf_photometry.py`; `ui_epsf_dashboard.py:307` | Photutils EPSFBuilder path | production (tab) | UNVERIFIED recent |
| PSF runner CLI | `psf_runner.py:1499` | Standalone ePSF dev/regression | CLI | dev |
| SysRem field detrend | `photometry_core.py:15529` `run_sysrem_field` | Post-2A Tamuz et al. detrend; called from `run_full_photometry_pipeline` when `sysrem_enabled=True` | config-gated (default off), reachable via `night_run` / UI | UNVERIFIED on anchor |
| Adaptive / method LC outputs | `method_lc_output.py` | Alternate method CSVs | pipeline when modes on | UNVERIFIED |
| Per-frame saturation mask | `photometry_core.py`; INV-CFG-01 | Mask saturated epochs | config default on | draft_510 |

### 1.7 Time systems

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| JD mid-exposure | `time_utils.py:62` `mid_exposure_jd` | From FITS DATE-* headers | pipeline | draft_510 |
| HJD / BJD (scalar) | `time_utils.py:141` `compute_hjd_bjd` | astropy `light_travel_time` | pipeline | draft_510 |
| HJD / BJD (batch) | `photometry_core.py:8913` | Vectorized equivalent to scalar | pipeline Phase 2A | draft_510 |
| Time columns export | `time_utils.py:271` `compute_time_columns` | JD/HJD/BJD_TDB columns in proc CSV | pipeline | draft_510 |

### 1.8 Exports and reports

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| AAVSO ensemble export | `export_reports.py:904` | Requires BJD_TDB; filter map; observer code | UI + manual CLI | UNVERIFIED recent |
| VarAstro export | `export_reports.py` | Same family as AAVSO | UI | UNVERIFIED |
| Export all methods | `export_reports.py:1406` | Multi-method LC reports | UI | UNVERIFIED |
| PDF photometry report | `photometry_report.py:6383`; UI `ui_aperture_photometry.py:1236` | Summary PDF with LC, comp QA, TESS | production + headless Step 15 | draft_510 |
| Trust flags (GREEN/YELLOW/RED) | `trust_flag_core.py`; CLI `trust_flag.py` | Check-star scatter thresholds | pipeline + CLI | draft_510 GREEN |
| Comp-star LOO QA | `comp_qa_core.py`; CLI `comp_qa.py` | Sokolovsky indices | pipeline + CLI | UNVERIFIED |
| Citations emitter | `citations.py` | CITATIONS.bib keyed strings in exports | export path | UNVERIFIED |
| Variability candidates CSV | `ui_variability.py:1800` | Export detection table | production | UNVERIFIED |
| Check-star KMAG (AAVSO) | `check_star_kmag.py` | Measured K for ensemble reporting | export when enabled | UNVERIFIED |

### 1.9 Database, equipment, diagnostics

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Database Explorer | `ui_database_explorer.py:118` | Edit TELESCOPE/EQUIPMENTS/LOCATION | production | UNVERIFIED |
| Per-camera gain/RN save | `ui_settings.py:563` | EQUIPMENTS table | Settings | UNVERIFIED |
| Parameters dashboard (272 params) | `ui_params_dashboard.py:303` | Registry-backed config editor | Settings | UNVERIFIED |
| Settings save ? config.json | `ui_settings.py:1013` | Persist AppConfig subset | production | continuous |
| Infolog ring buffer | `infolog.py`; tab `app.py:2661` | Session log + save to disk | production | draft_510 |
| Session baseline / anchor check | `dev/scripts/session_baseline_check.py` | Photometry SHA regression | dev/CI | P1 stale per STATE |
| Offline xval harness | `xval_run.py:88` | photutils/sep vs VYVAR | dev | **2026-08-13 draft_510** |
| INV-CAL validators | `dev/tools/inv_cal01_validate.py`, `inv_cal02_validate.py` | Cal gate regression | dev | 2026-08-13 |
| Inspect drafts CLI | `inspect_drafts.py:120` | Draft folder audit | CLI | dev |
| Repair catalog IDs | `repair_catalog_ids.py:232` | Fix Gaia id formatting | CLI maintenance | dev |
| Lunar context | `lunar_context.py` | Moon distance/illum for session | pipeline metadata | UNVERIFIED |
| Orchestrator (Claude?Cursor) | `dev/orchestrator/vyvar_orchestrator.py` | File-based task bridge | Milan workflow | continuous |

### 1.10 OSC path (colour cameras)

| Capability | Entry point | What it does | Reach | Last exercised |
|------------|-------------|--------------|-------|----------------|
| Bayer ? oneRGGB/R/G/B extract | `osc_extract.py` | Half-size mono planes + optional bin | OSC equipment only | UNVERIFIED home rig (mono) |
| OSC unified frame sets (OSC-02) | `osc_align.py`; `invariants_runtime.py:281` | Four channels same frame IDs | OSC pipeline | UNVERIFIED |
| OSC export filter codes TR/TG/TB | `invariants_runtime.py:307` `check_osc03_export_eligibility` | Block oneRGGB from AAVSO | export | UNVERIFIED |
| Gaia?Johnson OSC transforms | `gaia_johnson.py:173` | Per-channel comp transforms | OSC photometry | UNVERIFIED |

---

## 2. Data flow: raw pixel ? exported magnitude

```
USB FITS
  ? [Scan] smart_scan_source (importer.py:1202)
  ? [Import] Raw/ + non_calibrated/ + DB manifest (importer.py:1712)
  ? [Calibrate] calibrated/lights/** (pipeline.py:16515) + cal_diag.json
       ? CONTRACT: output dir named "calibrated" but may receive second-stage preprocess
  ? [RAM QC] manifest FWHM/sky (optional skip if PERF-10) (pipeline.py:2079)
  ? [Preprocess IN-PLACE] qc_enrich on calibrated/lights (pipeline.py:18027)
       writes qc_metrics.csv; stamps VY_SKYSF, VY_CALSTAGE, VY_FWHM
       ? NAMING: Step label "calibrated?processed"; no processed/lights population
  ? [Plate solve + Align] detrended_aligned/lights/{setup}/ (pipeline.py:15298)
       MASTERSTAR.fits, masterstars_full_match.csv, proc_*.csv sidecars
  ? [Phase 0+1] comparison_stars_per_target.csv, active_targets (photometry_core.py:14545)
  ? [Phase 2A] photometry_summary.csv, lightcurve_*.csv (photometry_core.py:10771)
  ? [Trust/Comp QA] trust columns, comp QA sidecars (trust_flag_core, comp_qa_core)
  ? [Export optional] AAVSO/VarAstro (export_reports.py) ù NOT in headless night_run
  ? [PDF] generate_all_method_photometry_reports (photometry_report.py:6383)
```

### In-place mutations and naming mismatches (C-class loci)

| Location | Issue | Evidence |
|----------|-------|----------|
| `calibrated/lights/` | Preprocess mutates same files cal stage wrote | `pipeline.py:18027`; INV-CAL-02 mitigates with stage stamp |
| `preprocess_calibrated_to_processed` | Deprecated alias; name implies copy to processed/ | `pipeline.py:18101` |
| `resolve_obs_file_to_processed_fits` | Name says processed; resolves calibrated path | `pipeline.py:3165` |
| `SATURATE_ADU` in EQUIPMENTS | Was binned ADU stored as bin1 authority | `database.py:2854` migration |
| `archive_path` vs draft root | Import may return `ù/non_calibrated` subpath | `night_run.py:646` normalizes |
| `detrended` vs `detrended_aligned` | Legacy docstring paths | `pipeline.py:15328` vs runtime |
| `manifest files[]` vs "obs files" | Same DB rows, two names | UI vs pipeline logs |
| Header `OFFSET=0` vs measured pedestal | CAL-DIAG derives ~24.5 ADU/bin1 | register U-PED-01 |

---

## 3. Reachability (exact)

See **`docs/VYVAR_AUDIT_2026_REACHABILITY.md`** for per-module classification (119 modules).

| Class | Count |
|-------|------:|
| production_reachable | 88 |
| cli_entry | 12 |
| unwired_ui | 4 |
| not_statically_reachable | 15 |

**Method:** AST import closure from entry modules + explicit lazy Streamlit tab set.  
**Limit:** dynamic imports inside render callbacks ù 11 lazy UI modules listed in reachability doc.

**Config-gated untested (default off):** `frame_align_residual_gate_enabled`, `sysrem_enabled`, blind solve tiers, `cog_aperture_correction_enabled`, OSC subtree on mono rig.

---

## 4. Parameters (272 registered)

Full trace: **`docs/VYVAR_AUDIT_2026_PARAMS.md`**.

| Class | Count |
|-------|------:|
| OBSERVED (3+ non-config files) | 84 |
| UI_OR_CONFIG (1ù2 non-config files) | 183 |
| CONFIG_ONLY | 6 |
| UNREAD | 0 |

**Flagged ignored/misread (fixed 2026-08-13):**
- `alignment_detection_sigma` ù wired in pipeline align path (C-ALIGN-01 FIXED)
- `apply_color_term` ù trust path uses `resolve_apply_color_term` (C-TRUST-01 FIXED)
- `aavso_observer_code` ù mirror of `observer_code`, not independent

---

## 5. Duplication

| Functionality | Locations | Numerical agreement |
|---------------|-----------|---------------------|
| HJD/BJD | `time_utils.compute_hjd_bjd` vs batch in `photometry_core.py:8913` | **Agree by design** (batch documents scalar equivalence) |
| BJD in pipeline per-frame headers | `pipeline.py:10144`, `10982` via `compute_time_columns` | Same astropy path |
| Comp selection RMS | `comp_pool_rms.py` vs `photometry_core.select_comparison_stars_per_target` | Shared call; defaults differ 0.05 vs cfg 0.1 |
| Comp QA | `comp_qa_core.py` vs `comp_qa.py` CLI | Same math (explicit delegate) |
| Trust flag | `trust_flag_core.py` vs `trust_flag.py` CLI | Same core |
| Photometry | `photometry.py` / `photometry_phase2a.py` | Re-exports of `photometry_core` |
| Aperture photometry math | `photometry_core` vs `xval_run` photutils | **draft_510: median \|phot?VYVAR\| target 3 mmag; comp RMS 7.8 mmag phot vs 10.5 mmag dao** |
| Background estimation | pipeline annulus vs photutils vs sep in xval | comp RMS sep 7.6 mmag vs phot 7.8 mmag (draft_510) |
| DAO detection threshold | `masterstar_dao_threshold_sigma` vs `sips_dao_threshold_sigma` vs UI session override | **Intentionally separate** roles |
| WCS SIP fit | `vyvar_platesolver.py` vs `astrometry_optimizer.py` | Optimizer refines after initial solve |
| Flat normalize | `invariants_runtime.check_flat_median` vs `calibration.normalize_flat_master` | Same 1e-3 tolerance |
| Gaia id normalize | `gaia_catalog_id.normalize_gaia_source_id` | Single canonical function (good) |

---

## Wave 1 closing

### Inventory verification (Part A, 2026-08-13)

**Method:** For each section-1 capability row, verify cited `.py` exists under `src_py/` or `dev/` and key function names resolve; grep `src_py/` for `astroscrappy` (must be absent post-`0ab686f`).

| Outcome | Count (section 1.1ñ1.10, 91 rows) |
|---------|-------------------------------------|
| Verified present | 88 |
| Corrected path / entry | 2 (`session_baseline_check` ? `dev/scripts/`; OSC-03 ? `check_osc03_export_eligibility`) |
| Deliberately absent (CR) | 1 |
| **Wrong from recall/docs** | **1** (astroscrappy listed as live) |

**Conclusion:** Single slip from July closure carry-forward, not systemic. Inventory method is sound after correction.

### SAT-DIAG `raw_peaks_used: false` (draft 510)

`sat_diag.json` is written at **align start** by `run_sat_diag()` (pile-up + limit derivation only) ó `pipeline.py:10641`. **Placed-aperture raw peaks** run later per frame in `apply_raw_peaks_to_proc_df` (`sat_diag.py:939`, `pipeline.py:8348`) and set `meta["raw_peaks_used"]=True` on proc sidecars, but **`sat_diag.json` is not updated**. Yesterday's validation used per-frame proc columns, not the stale JSON flag. **C-class stale provenance** (same family as VY_QCBG vs pixels).

**Surprised:** Headless `night_run` does not call AAVSO/VarAstro exporters.

**Could not determine:** Last UI exercise date for most Settings/Tools tabs; OSC path on real Bayer data in this workspace.

**Next (Wave 2):** Enumerate every gate with boundary measurements on drafts 435/509/510; verify CAL-DIAG and SAT-DIAG fire history on disk.
