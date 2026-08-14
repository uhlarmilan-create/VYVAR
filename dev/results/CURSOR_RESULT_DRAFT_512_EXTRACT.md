CURSOR RESULT - DRAFT-512-EXTRACT (2026-08-14)

Register ID: **DRAFT-512-EXTRACT**
Draft: **000512** (BO CVn, NoFilter_60_2)
Comparison draft: **000510** (same raw frames per E2)
Extraction only -- no fixes, no re-runs, no interpretation.

**Comparability (task section 1):** Draft 512 is a fresh full-pipeline run. Any 512-vs-510
column difference is a joint effect of all changes between runs (X-R0). No column difference
is attributed to any single cause in this memo.

Machine-readable outputs:
- `dev/results/draft512_draft_level.json` (draft-level fields E0-E7)
- `dev/results/draft512_star_by_frame.csv` (804 rows: 6 stars x 134 frames)
- `dev/results/draft512_per_frame_fwhm.csv` (134 QC-ok frames)

---

## Pre-registered rules

| Rule | Applies? | Quote |
|------|----------|-------|
| **X-R0** | Yes | Any 512-vs-510 difference is joint effect; report differences only |
| **X-R1** | No (E2 matched) | Raw sets identical; 512-vs-510 columns are descriptive joint-effect pairs |
| **X-R2** | Partial | Absences noted where stages recorded nothing |
| **X-R3** | **Yes** | Run tree was uncommitted (`git_dirty: true`); trial run, not recoverable reference standard |

---

## E0 -- Run provenance

| Field | Value | Source |
|-------|-------|--------|
| Run infolog | `Archive/Drafts/draft_000512/infolog_20260814_092155.txt` | draft tree |
| Run start (UTC) | 2026-08-14 07:21:53 | infolog milestone |
| Run end (UTC) | 2026-08-14 07:39:45 | infolog last timestamp |
| Git SHA recorded at photometry | `4a3e855f6cd48eb66d8df1254ab2356727c95a69` | `pipeline_meta.json` -> `provenance.git_hash` |
| Git dirty at run | **true** | `pipeline_meta.json` -> `provenance.git_dirty` |
| Git SHA in `draft_manifest.json` | **absent** | `draft_manifest.json` (no commit field) |

**Uncommitted tree evidence (from run artifacts, not memory):**
- `pipeline_meta.json` -> `provenance.git_dirty_files` lists modified/untracked paths including
  `src_py/app.py`, `src_py/night_run.py`, `src_py/photometry_core.py`, `src_py/invariants_runtime.py`,
  iron-gate test files, CLOSE-IRON-GATES / PP-KWARG memos, etc.
- Content SHA256 at run for `src_py/photometry_core.py`:
  `252f49c5ed141ac2459e49dd957f2e071dfd082a38268cdc92e99230e670f71f` (matches current working tree file)
- Content SHA256 at run for `src_py/app.py`:
  `1fc4558846c24d32fc47af5072c256b3f82cbf0188e79d6cef684b6250c275ac` (matches current working tree file)

**SKY-CLIP-01 in tree for this run (artifact evidence):**
- `photometry_core.py` run hash matches post-SKY-CLIP-01 working-tree hash (above).
- Target median `sky_adu_per_px_annulus` (proc CSVs): **1564.683 ADU/px** (512) vs **1562.803 ADU/px** (510).
  Source: `draft512_star_by_frame.csv` / recomputation from proc CSVs.
- No FITS header or `pipeline_meta` key names the sky estimator; column is `sky_adu_per_px_annulus`.

**PP-KWARG-01 in tree for this run (artifact evidence):**
- Preprocess and MASTERSTAR phases completed (infolog milestones 07:24:30 - 07:37:04).
- `app.py` run hash matches current file (PP-KWARG fix removes dead kwarg from `_pp_kw`).
- Draft 511 failed at preprocess with PP-KWARG TypeError on base `4a3e855` without fix; 512 completed.

**`draft_manifest.json` (schema v3):**

| Field | Draft 512 | Draft 510 |
|-------|-----------|-----------|
| `schema_version` | 3 | 3 |
| `draft_id` | 512 | 510 |
| `status` | INGESTED | INGESTED |
| `calibration_mode` | vyvar_calibrated | vyvar_calibrated |
| `updated_utc` | 2026-08-14T07:39:46+00:00 | (see draft 510 manifest) |
| Top-level keys | calibration_mode, center, draft_id, files, final_observation_id, is_calibrated, observation_start_jd, observer_location, paths, rig, schema_version, status, updated_utc | same set |
| `rig` | equipment_id=1, telescope_id=1, location_id=2, scanning_id=109043809 | **identical** |
| `paths.masterstar` | BO_CVn_Light_109.fits | BO_CVn_Light_109.fits |

**Config snapshot diff** (`pipeline_meta.json` -> `provenance.config_snapshot`, 512 vs 510):

| Key | 510 | 512 |
|-----|-----|-----|
| `comp_max_delta_bprp` | 0.79 | 0.99 |
| `phase01_comparison_max_mag_diff` | 1.5 | 2.0 |
| `phase01_comparison_n_comp_min` | 3 | 2 |

All other config_snapshot keys matched between 512 and 510 at extraction time.

---

## E1 -- Frame accounting

| Stage | Count | Source |
|-------|-------|--------|
| Raw lights | 150 | `Raw/lights/NoFilter_60_2/*.fits` |
| Calibrated lights | 150 | `calibrated/lights/NoFilter_60_2/*.fits` |
| QC rows | 150 | `calibrated/lights/qc_metrics.csv` |
| QC status `ok` | 134 | qc_metrics.csv |
| QC status `rejected_prefilter_fwhm` | 16 | qc_metrics.csv |
| Aligned FITS (detrended_aligned) | 135 | glob (134 science + MASTERSTAR reference) |
| proc CSVs | 134 | `detrended_aligned/lights/NoFilter_60_2/proc_*.csv` |
| BO CVn LC points | 134 | `lightcurve_1498613634033133184.csv` |

**Loss at each step:**

| Transition | Lost | Reason |
|------------|------|--------|
| Raw -> calibrated | 0 | all 150 calibrated |
| Calibrated -> QC ok | 16 | `rejected_prefilter_fwhm` (prefilter after auto FWHM limit) |
| QC ok -> photometry | 0 | 134 proc CSVs for 134 ok frames |

**Auto FWHM limit (512 vs draft 511 reference):**

| Quantity | Draft 512 | Draft 511 (architect report) |
|----------|-----------|------------------------------|
| Limit (px) | **5.362** | 5.362 |
| k | **1.50** | 1.50 |
| Median FWHM input | **5.311** (infolog JSON block) | 5.311 |
| Median VY_FWHM (150 processed FITS) | **5.195 px** (infolog) | (not extracted here) |
| Rejected | **16 / 150** | 16 / 150 |

Source infolog: `Auto FWHM limit=5.362 px (k=1.50)`; `"median_fwhm": 5.3106824933043715`.

**Per-frame FWHM (QC ok frames only):**

| Stat | px | Source |
|------|-----|--------|
| min | 5.138 | `draft512_per_frame_fwhm.csv` |
| median | 5.192 | same |
| max | 5.305 | same |

Draft 510 QC-ok FWHM series: min/med/max **identical** to 512 (same numeric values at extraction).

**Rejected frames (16, all `rejected_prefilter_fwhm`):**

BO_CVn_Light_002, 007, 009, 049, 056, 058, 066, 074, 111, 122, 131, 141, 142, 147, 149, 150.

Source: `calibrated/lights/qc_metrics.csv`.

---

## E2 -- Raw-data identity vs draft 510

| Check | Result | Source |
|-------|--------|--------|
| Same 150 filenames | **yes** | Raw/lights glob both drafts |
| SHA256 all 150 common files | **all match** | pairwise hash |
| Matched-pair prerequisite for E7 columns | **satisfied** | E2 |

---

## E3 -- Stages outside anchor (`INV-ANCHOR-00`)

### Calibration

| Item | Value | Source |
|------|-------|--------|
| Master dark | `CalibrationLibrary/Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits` | infolog / cal_diag |
| Master flat | `CalibrationLibrary/Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits` | infolog / cal_diag |
| CAL-DIAG v2 | PASS, convention=SUM, src=DERIVED, R=101.76, pedestal_p=24.47 | `cal_diag.json` |
| Sample header (Light_001) VY_DKRSMP | SUM | FITS header |
| VY_DKRSMP_SRC | DERIVED | FITS header |
| VY_CALSTAGE | SKYSF_2 | FITS header |
| VY_CALDATASUM | 641664276 | FITS header |
| INV-CAL-01 (infolog) | PASS | infolog 07:21:57 |

### Preprocess

| Item | Value | Source |
|------|-------|--------|
| Sky surface order | 2 | infolog `[PREPROCESS] start in-place QC sky_order=2` |
| Frames with sky surface applied | 150 / 150 | `pipeline_meta.json` sky_surface_n_applied |
| Sky surface p2p median | 135.736 ADU | `pipeline_meta.json` sky_surface_p2p_median_adu |
| Sky surface skip count | 0 (no skip lines in infolog) | infolog / pipeline_meta |
| INV-PREP-01 | large_small_ratio=**0.01x** (warn>10) | infolog 07:25:04 |
| INV-PREP-01 in pipeline_meta | **not recorded** | pipeline_meta invariants list |
| INV-FLAT-01 | ok=true, max residual_flatness_p99=**58.6 ADU** (band=400; n=150) | pipeline_meta invariants |

### Alignment

| Item | Value | Source |
|------|-------|--------|
| alignment_report rows | 135 | `alignment_report.csv` |
| align_residual_px (science frames) | min=0, median=0, max=0 | alignment_report.csv |
| rotation (MASTERSTAR) | 179.683 deg | alignment_report.csv row 1 |
| detected_stars per frame | 200 (typical science row) | alignment_report.csv sample |

### MASTERSTAR

| Item | Value | Source |
|------|-------|--------|
| Reference frame | BO_CVn_Light_109.fits | `draft_manifest.json` paths.masterstar |
| MASTERSTAR.fits | platesolve/NoFilter_60_2/MASTERSTAR.fits | draft tree |
| masterstars_full_match rows | 735 | `masterstars_full_match.csv` |
| INV-WCS-01 | ok=true, p95=**1.265 px** (warn<2) | pipeline_meta invariants |
| wcs_roundtrip_p99_px | 1.07e-11 | pipeline_meta.json |

### DAO detection

| Item | Value | Source |
|------|-------|--------|
| qc_dao_detection_sigma (config) | 5.0 | pipeline_meta config_snapshot |
| Detections per frame (n_detected) | ~726-792 | `per_frame_catalog_index.csv` |
| Matched per frame (n_matched) | ~699-706 | same |
| aperture_snr_table fwhm_px | 3.389 | `aperture_snr_table.json` |
| fwhm_estimator | dao_moment_median | aperture_snr_table.json |
| fwhm_px_scope (SNR table) | per_draft_median_frame_dao_moment | aperture_snr_table.json |
| vy_fwhm_dao_px | 5.19465 | aperture_snr_table.json |

---

## E4 -- Photometry outputs (BO CVn target `1498613634033133184`)

### Side-by-side headline fields (joint-effect label per X-R0; raw matched per E2)

| Field | Draft 512 | Draft 510 | Source |
|-------|-----------|-----------|--------|
| aperture_px | **4.211** | **4.261** | photometry_summary.csv |
| check_scatter (**S1**) | **0.009300** | **0.008638** | trust_1498613634033133184.json |
| n LC points | **134** | **134** | photometry_summary.csv / lightcurve CSV |
| n comparison stars | **5** | **5** | photometry_summary.csv |
| trust | **GREEN** | **GREEN** | photometry_summary.csv |
| lc_rms (**S6**) | **0.145389** | **0.145359** | photometry_summary.csv |
| lc_rms_ooe (**S7**) | **0.046659** | **0.046644** | photometry_summary.csv |
| ac_scatter (epoch field in LC) | **0.013283** | **0.009283** | lightcurve CSV first epoch |

### Comparison star IDs (512)

1497368849430107904, 1497771992240531712, 1497974027502858240, 1499053747922698240, 1499200223486564608.

**Same set as draft 510:** yes.

**S4 comp_rms per comp star (512):** 0.0206, 0.0112, 0.0140, 0.0146, 0.0118 (catalog IDs above respectively).
Source: `comparison_stars_per_target.csv` rows for target 1498613634033133184.

### Per-star / per-frame geometry (machine-readable)

`dev/results/draft512_star_by_frame.csv` columns include:
`frame_proc_csv`, `catalog_id`, `aperture_r_px`, `r_in_derived`, `sky_annulus_r_out_px`,
`fwhm_px_for_aperture`, `fwhm_px_scope`, `sky_adu_per_px_annulus`, `dao_flux`.

Note: `r_in` is **not stored** in proc CSV; `r_in_derived = max(aperture_r_px + 0.5, 4.75 * fwhm_px_for_aperture)`.

### fwhm_px_scope inconsistency (observation)

| Location | Value |
|----------|-------|
| proc CSV `fwhm_px_scope` | `per_draft_gaussian_override` (all rows) |
| `aperture_snr_table.json` | `per_draft_median_frame_dao_moment`, estimator `dao_moment_median` |

### Sky estimator

- Stored column: `sky_adu_per_px_annulus` (proc CSV).
- No proc column names the estimator function.
- Run-tree `photometry_core.py` hash matches SKY-CLIP-01 implementation state (E0).

### Saturation

| Item | Value | Source |
|------|-------|--------|
| sat_limit_source (target, all frames) | DERIVED | proc CSV |
| target likely_saturated flags | 0 / 134 | proc CSV |
| sat_diag sat_source | DERIVED | sat_diag.json |
| sat_peak_source | PLACED_APERTURE | sat_diag.json |
| INV-SAT-01 | ok=true, detail="sat_diag not stamped..." | pipeline_meta invariants |

---

## E5 -- Gates and invariants (runtime recorded)

From `pipeline_meta.json` -> `invariants` (512 run):

| ID | ok | policy | measured detail |
|----|-----|--------|-----------------|
| INV-WCS-01 | true | WARN | matched_world2pix_identity_p95_px=1.265 |
| INV-DAG-01 | true | FAIL | stamped masterstar/phase01/phase2a/postprocess (4 entries) |
| INV-FLAT-01 | true | WARN | max residual_flatness_p99=58.6 ADU (band=400; n=150) |
| INV-PROV-01 | true | FAIL | prov_schema_version=1 ok |
| INV-CFG-01 | true | FAIL | config<->behavior markers clean |
| INV-SAT-01 | true | WARN | sat_diag not stamped |
| INV-CAL-01 | true | FAIL | cal_diag keys=1 spec_version='CAL-DIAG-v2' |
| INV-CAL-02 | true | WARN | cal_stage not stamped |

**Iron-rule gates (INV-NOCLIP-01 etc.):** not present in `pipeline_meta.invariants` for this run.
Iron-gate test files appear in run dirty-file list; no runtime iron-gate verdict recorded.

**INV-PREP-01:** recorded in infolog only (0.01x); not in pipeline_meta invariants.

---

## E6 -- Log extraction

**Infolog:** `Archive/Drafts/draft_000512/infolog_20260814_092155.txt`

### WARNING lines (11 unique COMP warnings)

Color-window / sparse-pool warnings for targets 1497169940906156032, 1496795041799526400,
1496998382733052928, 1497007144465726080, 1497245497969274240, 1497683722074089728,
1500549977088828160, 1498278351706325248 (see infolog lines 07:37:14 - 07:38:17).

### ERROR lines

- `07:25:47` `[PLATE-SOLVE] _plate_solve_input_bundle failed: database is locked`
- `07:26:57` same database locked error

(WCS refined message also tagged ERROR in log parser: `Mean residual error = 0.61 pixels`.)

### Skip summary

- `[AC] run summary: applied=8 skipped=37` with reasons:
  `no_comp_rms=4`, `insufficient_ref_stars=4`, `scatter_too_high=1`, `unknown=28`.
  Source: infolog 07:39:03.

### except_fix_counters

All zero in `pipeline_meta.json` -> `except_fix_summary` (40+ counters listed; all 0).

### Wall time per phase (UTC, infolog timestamps)

| Phase | Start | End | Duration |
|-------|-------|-----|----------|
| Calibration | 07:21:55 | 07:23:28 | ~1m 33s |
| Analyze QC | 07:23:28 | 07:24:30 | ~1m 02s |
| MASTERSTAR (incl. preprocess) | 07:24:30 | 07:37:04 | ~12m 34s |
| Photometry (Phase 0+1 + 2A) | 07:37:04 | 07:39:45 | ~2m 41s |
| **Total run** | 07:21:53 | 07:39:45 | **~17m 52s** |

---

## E7 -- Light curve products

### BO CVn light curve (`lightcurve_1498613634033133184.csv`)

| Quantity | Value |
|----------|-------|
| Points | 134 |
| BJD span | 2461154.320827 - 2461154.526693 |
| mag_calib_final min | 9.401 |
| mag_calib_final max | 9.863 |
| mag_calib_final std (**reported LC scatter, not S1-S14 label**) | 0.146 |

### Scatter values with S1-S14 labels

| ID | Quantity | Value (512) | Source |
|----|----------|-------------|--------|
| **S1** | check_scatter | 0.009300 | trust JSON |
| **S4** | comp_rms (per comp, 5 values) | see E4 | comparison_stars_per_target.csv |
| **S6** | lc_rms | 0.145389 | photometry_summary.csv |
| **S7** | lc_rms_ooe | 0.046659 | photometry_summary.csv |
| **S8** | stability p2p per comp | not bulk-extracted | (comp QA JSON exists per target) |
| AC epoch scatter | ac_scatter in LC | 0.013283 (constant across epochs in file) | lightcurve CSV |

### Exported product set

| Product | Present? | Path / count |
|---------|----------|--------------|
| Light curve CSV | yes | `photometry/lightcurves/lightcurve_1498613634033133184.csv` |
| Trust JSON | yes | `photometry/lightcurves/trust_1498613634033133184.json` |
| AAVSO export (BO CVn) | yes | `lightcurves_reports/aavso/BO_CVn_20260423.txt` |
| AAVSO exports (all targets) | 17 files | `lightcurves_reports/aavso/` |
| VarAstro exports | yes | `lightcurves_reports/varastro/` (companion set) |
| PDF report | yes | `VYVAR_report_NoFilter_60_2_20260814.pdf` |

Headless vs UI export gap (C-EXPORT-GAP): this run produced AAVSO files under
`lightcurves_reports/aavso/` (UI/export path present in tree).

---

## Could not extract

| Item | Reason |
|------|--------|
| Run-time `git status` / `git diff --stat` at 07:21 UTC | Not stored; only `pipeline_meta` dirty file list + hashes |
| Iron-rule gate runtime verdicts | Not written to pipeline_meta |
| INV-PREP-01 in invariant block | Infolog only |
| S8 / S11 per-comp scatter bulk table | Would require parsing all comp_qa JSON files |
| Explicit sky-estimator name in proc outputs | Column values only; no metadata field |
| proc CSV `mag` column populated | Not in export (dao_flux present) |

---

## Register diff for authorization

See `dev/results/REGISTER_DIFF_DRAFT_512_EXTRACT.md`.
