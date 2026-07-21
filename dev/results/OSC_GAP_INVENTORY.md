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

---

IMX533 M71 re-run (real target data, full calib) - 2026-07-21
------------------------------------------------------------

Data path: C:\ASTRO\python\VYVAR\Archive\M71\
Equipment: ASI533MC Pro (ZWO), BAYERPAT=RGGB, EQUIPMENTS id=5 (created for run).

Header / frame summary
----------------------
| Set | Count | Key header values (sample) |
|-----|------:|----------------------------|
| Lights | 255 | INSTRUME=ZWO CCD ASI533MC Pro; BAYERPAT=RGGB; 3008x3008 bin1; EXPTIME=15s; GAIN=100; CCD-TEMP -10.0 to -10.2C (279 frames at -10.0C); FILTER absent (import -> NoFilter) |
| Darks | 40 | Same camera/pattern; EXPTIME=15s; CCD-TEMP=-10.0C; median ~2040 ADU; checkerboard col-delta ~935 ADU |
| Flats | 30 | Same camera/pattern; EXPTIME~0.0034s; CCD-TEMP=-9.9C; median ~23020 ADU; checkerboard ~10982 ADU; RGGB tile medians before norm [13682, 23058, 23058, 36863] |

Pipeline walk: 15 lights subset, simulate_night_run.py, equipment_id=5, sysrem off.
Run: draft_000438 (SUCCESS, 2215 s, 11 frames after QC rejections, 74 LCs).

Master build (naive path + harness note)
----------------------------------------
- **Naive UI/library path FAIL:** `generate_master_dark_from_source_dir` /
  `generate_master_flat_from_source_dir` reject all ZWO frames because
  `_looks_like_master()` treats filenames `Dark_*.fits` / `Flat_*.fits` as
  already-combined masters (0 raw frames left). **New SHOULD gap.**
- **Harness (discovery only):** stacked 40 darks + 30 flats via
  `_write_master_to_library` -> CalibrationLibrary:
  - Dark_15s_Dark_100G_-10deg_Bin1_20260721.fits (checkerboard ~935 ADU on master dark)
  - Flat_0s_NoFilter_100G_-9.9deg_Bin1_20260721.fits (RGGB tile structure preserved in stack)
- **normalize_flat_master at calibrate (FLOW test):** `get_processed_master(flat)` sets
  `flat_normalized_at_calibrate=True`; per-tile BAYER4 norm -> all four RGGB quadrant
  medians 1.0, global median 1.0, checkerboard col-delta ~0.007 ADU post-norm.
  Header at calibrate: VYFLTNRM=BAYER4, VYFLTPAT=RGGB (via explicit probe). Claim
  **partially validated** (norm runs at calibrate, not at library stack time).
- **Scan pairing SUSPECT:** import scan status Darks=master points at session
  `Dark_081.fits` (single raw frame), while library master also linked via
  `dark_by_obs_key`. CCD_TEMP pairing accepted (-10.1C lights vs -10.0C master).

Stage table (M71)
-----------------

| Stage | Verdict | Evidence |
|-------|---------|----------|
| Import + master pairing | SUSPECT | quick_look=false; IS_CALIBRATED=1; library flat OK; dark scan confusion (see above); no FILTER -> NoFilter_15_1 group |
| Calibration + CAL-DIAG | SUSPECT | Real dark subtract + Bayer flat divide ran (25.7 s). Raw light checkerboard ~936 ADU -> calibrated ~21 ADU (large reduction, Bayer residual remains). CAL-DIAG passed |
| QC (RAM) | SUSPECT | FWHM median ~5.95 px; elongation median ~1.36 (mosaic DAO artifacts) |
| Preprocess / sky-surface | SUSPECT | residual_flatness_p99 median ~3.86 ADU (better than Seestar 10-22); processed checkerboard still ~21 ADU; fwhm_px median ~5.07, elong ~1.08 |
| Plate solve + catalogs | SUSPECT | **Completed** (vs Seestar FAIL). DAO pass1 **29292 detections** on mosaic; initial catalog match ~2.7%; WCS refine rejected rms=187 px repeatedly; pipeline continued on M71 globular. Plate scale solved ~0.55 arcsec/px |
| Phase 0/1 + Phase 2A | SUSPECT | Full photometry ran: 74 targets, lc_rms median **0.026 mag** (looks healthy). Aperture photometry on mixed Bayer flux |
| Exports | SUSPECT | AAVSO files emit **FILT=CV** (NoFilter fallback); example ASAS J195403 ~9.5 mag CV. No TG/TB/TR. 8 export failures (empty LC points). Not OSC-aware band coding |

Deltas vs Seestar proxy pass (draft_437)
----------------------------------------
| Aspect | Seestar (no calib) | IMX533 M71 (full calib) |
|--------|--------------------|-------------------------|
| Pattern / camera | GRBG, Seestar S50 | **RGGB**, ASI533MC Pro |
| Calibration | Quick-look, passthrough | **Real dark+flat**, IS_CALIBRATED=1 |
| Checkerboard post-cal | ~68 ADU (unchanged) | **~21 ADU** (reduced, not removed) |
| Flat Bayer norm | Not exercised | **BAYER4 per-tile at calibrate** |
| Plate solve | FAIL (MASTERSTAR abort) | **Completed** (dense GC field; 29k spurious DAO) |
| Phase 2A | NOT REACHED | **74 LCs produced** (silent wrong-band flux) |
| AAVSO export | UNKN (LP filter) | **CV** (NoFilter; still not OSC) |

Gap list movement
-----------------
- **MUST #2 refined:** plate solve is not guaranteed FAIL; on M71 it **passes** while
  DAO/WCS quality is **SUSPECT** (spurious detections, low initial match). Still MUST
  because centroids are mosaic-based and WCS/photometry are not trustworthy.
- **New SHOULD:** ZWO `Dark_`/`Flat_` filenames break master generation via
  `_looks_like_master`; naive user cannot build masters from typical ASI exports.
- **New SHOULD:** import scan labels session darks as "master" when names match
  `Dark_*` pattern.
- **SHOULD #3 (sky-surface):** improved residual with calib (p99 ~3.9 vs 10-22) but
  Bayer pattern persists in calibrated image (~21 ADU).
- **MUST #1, #3, #4, export/band gaps:** unchanged.

Conclusion
----------
**Overall NO-GO unchanged** for release 1.0 OSC science. M71 shows the pipeline can
**complete end-to-end** and produce plausible LC/export output on a globular cluster
without debayering — which **increases silent wrong-science risk** vs the Seestar run
that failed closed at plate solve. Debayer boundary remains mandatory before OSC support.

Revisit trigger unchanged: Milan 1.0 vs 1.1 + debayer regression on this M71 set.

Run artifacts (M71, not committed)
----------------------------------
- CalibrationLibrary: Dark_15s_Dark_100G_-10deg_Bin1_20260721.fits,
  Flat_0s_NoFilter_100G_-9.9deg_Bin1_20260721.fits
- EQUIPMENTS id=5 (ASI533MC Pro / IMX533 OSC)
- tmp/osc_discovery/m71_run/import_source/ (15 lights + darks/flats copies)
- Archive/Drafts/draft_000438/
- tmp/osc_discovery/m71_night_run.txt

---

M71 channel-extraction experiment (L/B/G/R, superpixel 2x2) - 2026-07-21
-------------------------------------------------------------------------

**Method (AAVSO grounding, data-prep only — zero src_py changes):** channel
**separation**, never interpolated demosaic. RGGB superpixel extraction via
`dev/scripts/osc_extract_channels.py`:

| Channel | Formula | AAVSO role |
|---------|---------|------------|
| L | (R+G1+G2+B)/4 | CV-equivalent luminance (differential detection) |
| G | (G1+G2)/2 | TG ~ V |
| B | native B cell | TB |
| R | native R cell | TR |

2x2 binning is the extraction step (3008^2 -> 1504^2 float32). Headers: copy
DATE-OBS/EXPTIME/GAIN/CCD-TEMP/telescope keys; XBINNING=YBINNING=2;
XPIXSZ=YPIXSZ=7.52; FILTER=L|G|B|R; strip BAYERPAT; add OSC-MODE + OSC-SRC.

**Input:** `Archive/M71/` (255 lights, 40 darks, 30 flats, ASI533MC RGGB).
**Extracted trees (local):** `tmp/m71_extract/{L,G,B,R}/{Lights,Darks,Flats}/`.
**Equipment:** id=5 (ASI533MC Pro); telescope 1480 mm (~1.05 arcsec/px at bin2).
**Runs:** 12 lights per channel, `simulate_night_run.py --no-sysrem`, per-channel
master flats; master dark rebuilt per channel when rerunning (see gaps below).

Per-channel results
-------------------

| Ch | Draft | Status | Gaia DAO % | n matched | WCS id p95 px | FWHM med px | comp stars | lc_rms med | G_lim_50 | MASTERSTAR max ADU | sky p2p med ADU |
|----|-------|--------|------------|-----------|---------------|-------------|------------|------------|----------|-------------------|-----------------|
| **L** | 439 | **OK** | 97.6 | 8253 | 7.31 | 3.88 | 150 | 0.025 | 17.5 | 60380 | 1.7 |
| **G** | 440/443/445 | **FAIL** | — | — | — | — | — | — | — | — | cal abort |
| **B** | 441 | **OK** | 86.5 | 868 | 0.40 | 4.57 | 145 | 0.048 | 14.4 | 64909 | 232 |
| **R** | 444 | **OK** | 97.4 | 8324 | 6.98 | 4.07 | 150 | 0.048 | 17.5 | 59506 | 1.3 |

**G failure modes (all runs):**

1. **First pass (draft 440):** reused L-stack master dark
   (`Dark_15s_Dark_100G_-10deg_Bin2_20260721.fits`, median ~2464 ADU) on G lights
   (median ~2028 ADU) -> CAL-DIAG **ABORT** (convention mismatch; no calibrated
   lights written -> preprocess "no frames IS_REJECTED=0").
2. **Rerun with G-specific dark (draft 443):** dark pairing fixed, but every frame
   hit **INV-FLUX-02 FAIL** (post-flat mean=1.001413, tol=0.001) during
   calibrate; chain aborts before photometry.
3. Flat rebuild did not relieve INV-FLUX-02 (draft 445, identical mean).

**R** succeeded only after channel-specific master dark (median ~2073 ADU matching
R lights ~2090 ADU). **L** passed CAL-DIAG with MEAN_AUTOCORRECT on shared dark.

Color-slope diagnostic (Gaia-matched stars)
-------------------------------------------

Proxy for `(m_inst - Gaia G)` vs `BP-RP` using per-frame DAO flux, per-frame
zero-point from 9<G<14 stars, median per `catalog_id` across frames (n shown).
Not `mag_calib` (pipeline ties catalog `mag` to Gaia G in proc CSVs).

| Ch | slope (mmag / bp-rp) | n stars | Sign / ordering |
|----|----------------------|---------|-----------------|
| B | **+863** | 5938 | steepest (blue channel vs Gaia G) |
| L | **+625** | 4702 | intermediate (luminance) |
| R | **+234** | 4191 | flattest (red channel) |
| G | n/a | — | run did not reach photometry |

**Interpretation:** clear **physical ordering B > L > R** in color sensitivity
(not noise — thousands of stars). All slopes positive in this sign convention
(bluer Gaia stars brighter relative to G in narrower blue-ish bands). **G
(smallest expected)** could not be measured. Opposite B/R signs in a strict
TG/TB/TR calibration sense still need G channel + proper band zeropoints.

**Saturation (cluster core):** MASTERSTAR peak **59-65k ADU** (~90-99% of 65535)
on L/B/R — M71 core saturated on 15s bin2 extracted channels; differential
work should exclude core or shorten exposure.

Conclusions for OSC-SUPPORT Phase-1 design
------------------------------------------

1. **Extraction path is viable** — mono FILTER=L|G|B|R frames run through the
   existing pipeline without src_py changes; plate solve + photometry complete
   on L/B/R.
2. **First in-pipeline mode:** prefer **L (CV/luminance superpixel)** for
   differential/time-series (best depth, lowest lc_rms, 97%+ Gaia recovery).
   **G (TG)** is the AAVSO target band but blocked today by flat-field invariant
   INV-FLUX-02 on this dataset (+ shared master-dark naming — see gap).
3. **New MUST (multi-channel prep):** `_write_master_to_library` names darks
   `..._Dark_...` regardless of OSC FILTER header -> only one bin2 dark per
   night; G/R need channel-specific stacks or autocorrect policy extension.
4. **B channel:** usable for bluest science but **shallow** (G_lim_50~14.4),
   high sky residual (~232 ADU p2p) — poor for faint-variable work on M71.
5. **Debayer-in-pipeline still required** for naive OSC import; this experiment
   proves channel separation **offline** is the safer Phase-1 bridge (matches
   AAVSO practice) rather than forcing interpolated demosaic.
6. **Overall NO-GO unchanged** for release 1.0 naive OSC — but Phase-1 design
   should lead with **superpixel L or G extraction at ingest**, not raw mosaic.

**Surprises:** (a) M71 mosaic run (draft 438) could complete on checkerboard,
   while extracted **G** fails closed on a 0.14% flat norm deviation; (b) B
   completeness collapses vs L/R despite "successful" run; (c) color slope
   ordering visible even without band_classify TG/TB/TR mapping.

Run artifacts (channel experiment, not committed)
-------------------------------------------------
- `dev/scripts/osc_extract_channels.py`
- `tmp/m71_extract/` (derived FITS)
- `tmp/m71_channel_experiment/` (import trees, logs, `channel_compare.json`)
- `CalibrationLibrary/Flat_0s_{L,G,B,R}_100G_-9.9deg_Bin2_20260721.fits`
- `CalibrationLibrary/Dark_15s_Dark_100G_-10deg_Bin2_20260721.fits` (last build: R)
- `Archive/Drafts/draft_000439` (L), `441` (B), `444` (R); G attempts `440/443/445`
