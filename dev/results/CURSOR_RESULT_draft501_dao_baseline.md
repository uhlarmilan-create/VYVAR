CURSOR RESULT - 2026-08-05 (DRAFT-501 DAO baseline vs prior Newton runs)

What I did
Read-only scan of Archive/Drafts for Newton (V filter) drafts; computed
dao_only_fraction from masterstars_full_match.csv; git blame on INV-MS-01;
config_snapshot comparison; FITS header/pixel stats on draft_501. No pipeline
runs. No config changes.

## 1 -- Prior Newton drafts on disk

Search scope: Archive/Drafts/ (all draft_* and _quarantine).

Criterion: setup subfolder V_* or V_60*, or resolved_facts.filter = V.

Result: **one Newton draft on disk -- draft_000501 only.**

All other drafts use NoFilter_60_2 (wide rig, plate_scale ~9.77 arcsec/px).
No pipeline_meta equipment_id field present in any draft examined.

Chronological table (all drafts with pipeline_meta + masterstars):

| draft | setup | stamped_at_utc | git_hash | dirty | ms_rows | phase2a | lc_good | lc_no_data | lc_total |
|-------|-------|----------------|----------|-------|---------|---------|---------|------------|----------|
| draft_000435 | NoFilter_60_2 | 2026-07-16T13:35:35Z | 10d610c0e79d | true | 2951 | yes | 116 | 1 | 169 |
| draft_000435_snapshot_skysurface_20260716 | NoFilter_60_2 | 2026-07-16T13:35:35Z | 10d610c0e79d | true | 2951 | yes | 116 | 1 | 169 |
| draft_000435_p1mini | NoFilter_60_2 | 2026-07-29T13:31:04Z | 226d269f8648 | true | 2951 | yes | 0 | 1 | 165 |
| draft_000499 | NoFilter_60_2 | 2026-07-30T19:43:03Z | dd8a2d02ca03 | true | 3639 | yes | 110 | 1 | 233 |
| draft_000500 | NoFilter_60_2 | 2026-08-04T09:25:48Z | 20dde2bcbaca | true | 4122 | yes | 113 | 0 | 249 |
| draft_000501 | V_60_2 | 2026-08-05T09:49:12Z | 2c964cb660e8 | true | 1668 | yes | 0 | 22 | 22 |

draft_000501 resolved_facts: filter=V, binning=2x2, exptime_s=60, plate_scale
1.30 arcsec/px, calibration_mode=pre_calibrated.

No prior Newton draft in Archive/Drafts. Journal references draft 422 V_60_2
(2026-04 era) but that tree is not present under Archive/Drafts/ (only a stub
path under tmp/ with no masterstars).

## 2 -- dao_only_fraction by draft

Method: source_type==DAO_ONLY if column present; else empty catalog_id / total.

| draft_id | date (stamped_at) | n_total | dao_only | dao_only_fraction | INV-MS-01 |
|----------|-------------------|---------|----------|-------------------|-----------|
| 435 | 2026-07-16 | 2951 | 109 | 0.037 | pass |
| 435_snapshot | 2026-07-16 | 2951 | 109 | 0.037 | pass |
| 435_p1mini | 2026-07-29 | 2951 | 109 | 0.037 | pass |
| 499 | 2026-07-30 | 3639 | 143 | 0.039 | pass |
| 500 | 2026-08-04 | 4122 | 561 | 0.136 | warn (>0.10), pass (<0.25) |
| 501 | 2026-08-05 | 1668 | 696 | 0.417 | **FAIL** (>0.25) |

draft_501: 972 Gaia-matched, 696 DAO-only (empty catalog_id); no source_type
column (INV-MS-01 blocked CSV write before source_type stamp).

Thresholds (invariants_runtime.py): WARN=0.10, FAIL=0.25.

Git blame -- INV-MS-01 introduction:

| item | value |
|------|-------|
| commit | 119157936097a01b6a4d5653afa5c4e03dadb191 |
| date | 2026-07-27 11:47:51 +0200 |
| message | feat(invariants): add preprocess gradient and DAO_ONLY export guards |
| functions | dao_only_fraction_from_masterstars (:454), check_dao_only_fraction (:470) |
| pipeline wiring | same commit added check_dao_only_fraction call in pipeline.py |

Timeline vs drafts:
  - draft_435 (2026-07-16): predates INV-MS-01 -- guard would NOT have run
  - draft_499/500/501 (2026-07-30 onward): post INV-MS-01 -- guard active
  - Wide-rig 499/500 pass or warn-only; Newton 501 fails at 0.417

No prior Newton draft on disk to test whether dao_only_fraction was historically
low on Newton. Milan's prior Newton success (if any) is not recoverable from
Archive/Drafts.

Interpretation of (a) vs (b):
  (a) INV-MS-01 introduced 2026-07-27 -- any Newton run BEFORE that date would
      not have hit the guard regardless of dao_only_fraction.
  (b) For draft_501 specifically, dao_only_fraction=0.417 is intrinsic to this
      field/rig run, not caused by the guard being new; wide-rig runs on the
      same code pass with 3.7-13.6%.

## 3 -- Config comparison

No prior Newton draft available. Closest chronological reference: draft_500
(2026-08-04, wide rig, phase2a succeeded). Side-by-side vs draft_501:

| parameter | draft_501 (Newton V_60_2) | draft_500 (wide NoFilter_60_2) | differs? |
|-----------|---------------------------|--------------------------------|----------|
| masterstar_dao_threshold_sigma | 3.8 | 3.8 | no |
| sips_dao_threshold_sigma | 3.5 | 3.8 | **yes** |
| sips_dao_fwhm_px | 2.5 | 2.5 | no |
| masterstar_prematch_peak_sigma_floor | 1.8 | 1.8 | no |
| dao_detection_n_equiv | 3.78 | 3.78 | no |
| dao_centroid_max_shift_fwhm | 1.0 | 1.0 | no |
| masterstar_dao_pass2_sigma | 1.9 | 1.9 | no |
| qc_dao_detection_sigma | 5.0 | 5.0 | no |
| verify_mag_limit | 14.0 | 14.0 | no |
| match_depth (pipeline_meta) | 18.0 | 18.0 | no |
| gaia_db_path | vyvar_gaia_dr3.db | vyvar_gaia_dr3.db | no |
| gaia max_g_mag (provenance) | 17.5 | 17.5 | no |
| plate_solve_fov_deg | 1.0 | 1.0 | no |
| export_arcsec_per_px (config) | 1.3 | 1.3 | no |
| phase01_plate_scale_arcsec_per_px | 1.3 | 1.3 | no |
| masterstar_detection_cap min/max/k | 250/800/0.08 | 250/800/0.08 | no |
| filter (resolved_facts) | V | NoFilter\|60\|2 | rig (expected) |
| plate_scale (resolved_facts) | 1.30 arcsec/px | 9.77 arcsec/px | rig (expected) |
| calibration_mode | pre_calibrated | vyvar_calibrated | **yes** |

faintest_mag_limit: not in config_snapshot for either draft; match_depth=18.0
stamped in pipeline_meta for both.

vs anchor draft_435 (2026-07-16, pre-INV-MS-01):
  masterstar_dao_threshold_sigma 2.1 (435) vs 3.8 (501/500) -- changed since anchor
  dao_detection_n_equiv / dao_centroid_max_shift_fwhm absent in 435 snapshot

DAO tuning params are effectively identical between draft_501 and draft_500 except
sips_dao_threshold_sigma (3.5 vs 3.8). Rig-specific differences (filter, plate
scale, calibration_mode) are expected. Config alone does not explain why
dao_only_fraction is 41.7% on Newton vs 13.6% on wide rig with nearly the same
DAO knobs.

## 4 -- Data-side check

No prior Newton FITS on disk under Archive/Drafts. draft_501 sample vs
draft_500 calibrated wide-rig reference (best available non-501 comparison).

draft_501: non_calibrated/lights/V_60_2/TOI-1131.01.b_2025-04-22_23-05-09_V.fits
  (Milan: externally master-dark/flat corrected; draft_manifest calibration_mode
  = pre_calibrated)

| stat | draft_501 (Newton V) | draft_500 ref (wide, vyvar_cal) |
|------|----------------------|----------------------------------|
| OBJECT | (empty) | BO_CVn |
| DATE-OBS | 2025-04-22T23:04:09 | 2026-04-23T19:35:20 |
| EXPTIME | 60.0 s | 60.0 s |
| IMAGETYP | OBJECT | Light Frame |
| XBINNING | 2 | 2 |
| GAIN | 3.12 | 0.0 |
| FILTER | V | (not in header) |
| VY_FWHM | 2.48 px | 3.52 px |
| shape | 2088 x 3126 | (not repeated) |
| median | 33487 ADU | 2416 ADU |
| std (sigma-clipped) | 8.65 ADU | 55.94 ADU |
| p05 / p95 | 33473 / 33502 | 2325 / 2521 |
| pixels > 50000 ADU | 69 | 132 |
| max ADU | 98232 | 68569 |

Qualitative differences:
  - draft_501 median ~33.5k ADU (high pedestal from external pre-calibration);
    draft_500 median ~2.4k ADU (VYVAR internal calibration scale).
  - draft_501 sky RMS very low (std 8.7 ADU on ~33k pedestal); consistent with
    pre-reduced frames already near saturation proxy (MASTERSTAR log:
    saturation_proxy max=98300, noise_floor~33494 ADU).
  - Both have bright peaks near 65-98k ADU and hot pixels >50k.
  - No VY_CAL / CALSTAT header stamp on draft_501 sample; VY_QC/VY_FWHM from
    in-place QC pass only.

Pre-calibrated input path (draft_501) vs vyvar_calibrated (draft_500) is the
dominant data-side difference observable on disk. Cannot compare to a prior
Newton run (none archived).

## Files changed

None (read-only).

DAO-BASELINE-INDETERMINATE -- no prior Newton run found, or numbers do not point clearly
