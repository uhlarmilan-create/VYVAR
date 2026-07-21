OSC GAP INVENTORY - IMX533 / Bayer OSC discovery (2026-07-21)

Discovery-only run of the CURRENT pipeline on OSC data. No src_py changes.
Headless import + night run via simulate_night_run.py (subset, naive defaults).

Data note
---------
IMX533 RGGB FITS were NOT found on this machine (searched C:\ASTRO, Downloads,
Desktop; 0 files with BAYERPAT=RGGB or QHY533C/ASI533MC headers).

Proxy dataset used (only Bayer OSC lights available):
  Path: C:\ASTRO\pozorovania\pixin_tmp\SLCHBT
  Camera: Seestar S50, BAYERPAT=GRBG, FILTER=LP, 1080x1920, 20s, CCD-TEMP ~26C
  Run: draft_000437, 12 lights, equipment_id=1 (QHY294MM mono - DB default)

IMX533-specific deltas vs this proxy: pattern RGGB vs GRBG (flat tile order in
normalize_flat_master only); sensor size/gain/RN differ. All other gaps apply to
any OSC Bayer mosaic run today.

Run conditions (naive user)
---------------------------
- Source folder: lights only (no session darks/flats).
- Equipment: default ID=1 (monochrome QHY294MM), not an OSC profile.
- No manual flat map, no pre-calibrated mode, no config overrides.
- SysRem disabled (--no-sysrem) to isolate OSC gaps.

Stage inventory
---------------

| Stage | Verdict | Evidence |
|-------|---------|----------|
| Import + master pairing | SUSPECT | smart_scan: Darks missing, Flats missing, quick_look=true. Warnings: "Observation LP-20s has no Master Flat..."; "No suitable MasterDark found -> Quick Look Mode". OBS_DRAFT.IS_CALIBRATED=0. BAYERPAT=GRBG in FITS primary header but extract_fits_metadata / scan plan metadata_sample.bayerpat=null (not surfaced at import). CCD-TEMP present (~26.2C) but no dark to pair. ID_EQUIPMENTS=1 (mono); SENSORTYPE=IMX492 does not trigger OSC flat path. |
| Calibration + CAL-DIAG | SUSPECT | Quick-look draft: lights copied under non_calibrated/lights/LP_20_1; cal_path=draft_non_calibrated. Night run "calibration" step 0.4s (passthrough). No master dark subtract, no flat divide, no normalize_flat_master call (no LP flat in library). Raw median ~560 ADU; checkerboard column delta mean ~68 ADU (strong Bayer mosaic). CAL-DIAG pregate not blocking (no masters to mismatch). |
| QC (RAM, DAO on mosaic) | SUSPECT | OBS_FILES 12 frames: FWHM median 5.06 px (range 4.78-5.36); ELONGATION_MEAN median 1.97 (1.60-2.20). Star counts 142-204. Values plausible for detection but elongated/oversized vs mono expectation (~2-3 px FWHM on processed stage). Artifact: Bayer cells read as asymmetric sources. |
| Preprocess / sky-surface | SUSPECT | qc_metrics.csv: all 10 preprocessed frames status=ok; sky_surface_applied=True; residual_flatness_p99 9.7-21.7 ADU. Processed FITS: checkerboard column delta still ~68 ADU (unchanged vs raw). Example: proc_Light_...230220.fit elongation=1.83, fwhm_px=2.57 (DAO on mosaic). Sky plane fits smooth function over checkerboard -> structured residual, not flat field. |
| Plate solve + DAO catalogs | FAIL | simulate_night_run FAILED at Step 12. Log: "MASTERSTAR plate-solve zlyhal... match_rate=78.0%, rms=19.55px, hint_sep=0.542deg, catalog_recovery_tight=0.3%, n_matched_tight=3". Pipeline aborts before photometry. Bayer grid likely degrades star centroids / WCS fit on this field. |
| Per-frame catalogs + Phase 0/1 | NOT REACHED | Blocked by platesolve FAIL. Code expectation: DAO catalogs built on mosaic pixels; comp selection assumes mono aperture flux. |
| Phase 2A | NOT REACHED | Blocked. Would integrate mixed R/G/B pixel values in aperture -> non-physical "magnitude" vs Gaia G/bp-rp. |
| Exports (code probe) | SUSPECT | band_classify("LP") -> PhotometricBand.UNKNOWN (not CLEAR). _resolve_aavso_filter("LP") -> UNKN + warning. TG/TB/TR map to AAVSO codes in export_reports.py builtin table but there is no debayer/split path to produce them from a single OSC frame. GREEN/RED/BLUE -> STANDARD_FILTER in band_classify but AAVSO UNKN (no builtin mapping). No TG/TB/TR emission from an LP mosaic run. |

Screenshot-worthy artifacts (under Archive/Drafts/draft_000437/)
-----------------------------------------------------------------
- Raw mosaic: non_calibrated/lights/LP_20_1/Light_LDN 1093_20.0s_LP_20260709-230220.fit
  (visible RGGB/GRBG checkerboard in any FITS viewer).
- Preprocess residual: processed/lights/LP_20_1/proc_Light_...230220.fit (checkerboard
  persists; qc_metrics residual_flatness_p99=16.04 ADU).
- QC table: processed/lights/qc_metrics.csv (elongation up to 2.14).
- Failure log: simulate_night_run.log / tmp/osc_discovery/night_run_stdout.txt
  (platesolve invalid solution line).

Code-only findings (not exercised on data)
------------------------------------------
- normalize_flat_master (calibration.py): only production Bayer hook. Uses BAYERPAT
  header or assumes RGGB when EQUIPMENTS.SENSORTYPE hints OSC. Per-tile median norm
  for 2x2 Bayer tiles; never debayers lights. Not invoked this run (no flat).
- No debayer/cvtColor path anywhere in src_py (grep: BAYERPAT only in calibration.py).
- k2_extinction: TG/TB/TR/BLUE/GREEN/RED in k2_none_tokens (no k'' fit) - policy only;
  photometry still runs on mosaic if upstream passes.

Gap list
--------

### MUST (blocks any OSC science use)

1. **No debayer step** - entire pipeline treats OSC FITS as a mono 2D array. All
   photometry, QC, and WCS see a checkerboard.
2. **Plate solve / MASTERSTAR failure on mosaic** - observed hard FAIL on proxy data;
   chain stops before catalogs/LCs.
3. **No OSC calibration path without mono masters** - quick-look import skips dark/flat;
   no guidance to build OSC-specific master flat (BAYERPAT-aware) or G-channel flat.
4. **DAO source detection on Bayer grid** - centroids, FWHM, ellipticity not meaningful
   for mono pipeline assumptions; breaks downstream matching.

Effort sketch (MUST): ~2-3 weeks for Phase-1 scope per TODO-45/JOURNAL - debayer to
single G (or luminance) FITS at import or preprocess boundary, BAYERPAT from header,
thread through calibration flat norm, re-validate platesolve+QO on one OSC night.
Without this arc, OSC data must not ship as supported.

### SHOULD (wrong science silently if user pushes past failures)

1. **Default mono equipment profile** - gain/RN/saturate from QHY294MM applied to OSC
   frames; no IMX533/OSC row in EQUIPMENTS.
2. **BAYERPAT invisible in import metadata** - operator cannot confirm pattern in UI/logs.
3. **Sky-surface fit on mosaic** - runs and reports ok while leaving ~68 ADU checkerboard
   (residual_flatness_p99 10-22 ADU); flat-field error enters photometry.
4. **band_classify LP -> UNKNOWN** - not mapped to CLEAR/CV; k''/colour-term policy
   ambiguous; export emits UNKN not CV.
5. **AAVSO filter codes** - LP unmapped; TG/TB/TR exist in lookup but no pipeline stage
   emits them; GREEN/RED/BLUE classify as standard but export UNKN.
6. **Mixed-channel aperture flux** - if platesolve were forced, Phase 2A magnitudes would
   blend Bayer pixels with no Gaia band correspondence (colour mess, not guarded).

Effort sketch (SHOULD): ~1 week after MUST - equipment catalog entry, filter synonym
table (LP/L-Pro/L-eXtreme -> policy band), import warnings when BAYERPAT present,
fail-closed gate "OSC detected, debayer required", export map docs + UNKN guards.

### NICE (quality / UX)

1. **Per-tile flat norm validation UI** - show VYFLTPAT/VYFLTNRM when OSC flat used.
2. **Per-channel R/G/B LC export** (TODO-45 Phase 2).
3. **Pre-filled IMX533 in camera catalog** (ROADMAP parked catalog arc).
4. **Checkerboard residual QC plot** - auto-flag mosaic residual above threshold.

Effort sketch (NICE): multi-week Phase 2 (per-channel photometry, colour indices);
catalog/onboarding tied to separate PARKED arc.

Recommendation
--------------
Park OSC for **release 1.1** unless Milan accepts TODO-45 Phase-1 (G-channel debayer
only) as a 1.0 blocker. Current tree: **NO-GO** for IMX533/OSC science.

Revisit trigger: Milan 1.0 vs 1.1 decision + IMX533 RGGB dataset on disk for
byte-identity regression of debayer boundary.

Docs impact: ROADMAP OSC-SUPPORT arc; STATE one-liner.
Recurrence: n/a - discovery.

Run artifacts (tmp, not committed)
----------------------------------
- tmp/osc_discovery/import_source/ (12 lights)
- tmp/osc_discovery/scan_plan.json
- tmp/osc_discovery/night_run_stdout.txt
- tmp/osc_discovery/draft437_evidence.json
- Archive/Drafts/draft_000437/ (DB draft from discovery run)
