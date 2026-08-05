CURSOR RESULT - 2026-08-04 (WIDE-ERR A2b)

What I did
Measured effective camera gain from raw calibration and science data (read-only).
No DB writes, no config/param_resolver changes. Harness: dev/tools/wide_err_a2b.py.
Output: tmp/wide_err_a2b/wide_err_a2b.json.

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

## M0 -- Run parameters and scaling audit

**pipeline_meta.json** (draft_000435 / NoFilter_60_2 photometry):

| parameter | value | source (resolver) |
|-----------|-------|-------------------|
| gain | 3.17 e-/ADU | header_index_mapped (GAIN=0 -> map) |
| read_noise | 15.2 e- | db (7.6 bin1 x 2 for bin2) |

**Raw light header** (BO_CVn_Light_001.fits, NOT detrended_aligned):

| keyword | value |
|---------|-------|
| GAIN | 0.0 (setting index, not e-/ADU) |
| EGAIN | absent |
| XBINNING / YBINNING | 2 / 2 |
| XPIXSZ | 9.26 um |
| READMODE | 0 |
| OFFSET | 0 |
| EXPTIME | 60 s |
| DATE-OBS | 2026-04-23T19:35:20.355 |
| APTDIA | 70.0 mm |
| FOCALLEN | 200.0 mm |

**EQUIPMENTS ID=1 (QHY294MM):**

| field | value | bin1 intrinsic? |
|-------|-------|-----------------|
| GAIN_ADU | 3.17 e-/ADU | yes (documented camera table) |
| READNOISE_E | 7.6 e- | yes |

**cal_diag dark resampling:** convention **SUM**, block_factor=2, status PASS.

**Did `_scale_bin1_db_for_header` fire for gain?** **NO.** Gain resolved via
`header_index_mapped` (param_resolver.py GAIN_SETTING_INDEX_MAP {0: 3.17}) before any
DB bin-scaling path. Read noise DID use DB with bin2 scaling (7.6 -> 15.2 e-).

## M1 -- Bin1 gain from raw flat pairs (PTC)

**Sources:**
- Flats: D:\FLAT56\Flat\FLAT56_Flat_*.fits (bin1, EXPTIME=0.15 s)
- Bias: CalibrationLibrary\Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits
  (D:\DARKS had no 0.15 s frame accessible; 60 s master bias level ~25 ADU, negligible
  dark current at 0.15 s)

**Pairs used (3):**

| pair | DATE-OBS A | DATE-OBS B | mean level (ADU) | g1 median (e-/ADU) |
|------|------------|------------|------------------|---------------------|
| 001/002 | 2026-04-22T18:14:30 | 2026-04-22T18:14:31 | 34246 / 34131 | 1.002 |
| 003/004 | 2026-04-22T18:14:33 | 2026-04-22T18:14:35 | 34021 / 33914 | 1.011 |
| 005/006 | 2026-04-22T18:14:36 | 2026-04-22T18:14:38 | 33807 / 33700 | 1.014 |

**All regions (15):** g1 median = **1.011 e-/ADU** (naive; absolute value RETRACTED in E1.1 --
60 s dark subtracted from 0.15 s flats is a pedestal mismatch).

**Var vs signal regression** (all region/pair points): slope = 2.006 ADU (implied g ~ 0.50),
intercept = -35736 ADU^2, R^2 = 0.83. Level lever arm across pairs is only ~1.6% (~34k to
~33.7k ADU); intercept dominates -- regression slope is NOT used as primary g1 estimate.

**Read noise at bin1** (std of bias master x g1): **17.95 e-** (includes FPN structure, not
pure RN).

## M2 -- Effective gain at bin2 from RAW science frames (RETRACTED absolute g)

**Path used:** Archive\Drafts\draft_000435_snapshot_skysurface_20260716\Raw\lights\NoFilter_60_2\
(D:\BO_CVn empty; matched by DATE-OBS 2026-04-23 and setup name from snapshot).

**NOT detrended_aligned/lights/** (alignment resampling would suppress pixel variance and bias
g_eff high).

**Method attempted:** Sky-background PTC on 150 raw bin2 frames (unflat-fielded); bin1 dark
master resampled SUM to bin2; 3 star-free 200x200 regions per frame; sigma-clip |z|>3.

| metric | value (raw run) |
|--------|-----------------|
| n_frames | 150 |
| sky mean p05 / p50 / p95 (ADU) | 1368 / 1624 / 2030 |
| fit slope (1/g_eff) | 1.042 ADU |
| fit R^2 | 0.882 |
| g_eff (naive) | 0.960 e-/ADU |

**RETRACTION (WIDE-ERR E1.1):** Absolute g_eff is **NOT valid**. Sky PTC ran on raw frames
**without flat-fielding**; PRNU (variance proportional to sky^2) and vignetting gradients
inside 200x200 regions inflate pixel variance, inflate the fitted slope, and **deflate** the
recovered gain. Independent bound from faint5 science data: measured scatter 1.119 x 201 =
225 mmag; g = 0.96 would imply photon floor 200*sqrt(3.17/0.96) = 364 mmag -- a star cannot
scatter below its photon floor. Inverting: 200*sqrt(3.17/g) <= 225 gives **g >= 2.50 e-/ADU**.
The naive g_eff = 0.96 is excluded. **gain_used = 3.17 e-/ADU is consistent with this bound
and is not implicated.**

## M3 -- Summing or averaging? (relative result only)

| quantity | value | status |
|----------|-------|--------|
| g1 (bin1, M1 naive) | 1.011 e-/ADU | absolute RETRACTED (M1 pedestal mismatch) |
| g_eff (bin2, M2 naive) | 0.960 e-/ADU | absolute RETRACTED (unflat-fielded sky PTC) |
| **g_eff / g1** | **0.949** | **survives: ~1, consistent with SUM binning** |
| gain_used (M0) | 3.17 e-/ADU | consistent with g >= 2.50 bound; not implicated |
| gain_used / g_eff (naive) | 3.304 | **invalid** (numerator/denominator not comparable) |

**Reading:** Only the **relative** ratio g_eff/g1 ~ 1 supports SUM binning (e-/ADU unchanged
across bin step). Absolute gain comparison to gain_used is retracted. The bin^2
param_resolver scaling path did not fire on this run (header_index_mapped).

## M4 -- Does measured gain mismatch account for k? (RETRACTED)

Photon-term rescale test used retracted g_eff; **not valid**. Faint5 median ratio_orig = 1.119
(photon-dominated subset k ~ 1.12 per WIDE-ERR E1); gain correction overshot (ratio_corr 0.62)
because g_eff was wrong.

**Check star -- single T3 bright-representative field 1485540612577549568** (NOT W1 median
over 163 fields): ensemble-dominated (phot 2.5 mmag, ens 58.1 mmag); ratio_orig = **0.515**
(chi2-friendly sparse-comp field). W1 median sigma_total_robust/err over 163 fields = **1.83**.
Gain rescale unchanged: ratio_corr = 0.514.

## Combined line

**WIDE-ERR-A2B-UNDECIDED (revised E1.1)** -- absolute gain NOT measured (M1 pedestal mismatch,
M2 unflat-fielded sky PTC); g_eff/g1 ~ 1 supports SUM binning only; gain_used 3.17 consistent
with science lower bound g >= 2.50 e-/ADU and is not implicated; photon rescale test invalid.

## Files created

- dev/results/CURSOR_RESULT_wide_err_a2b.md (this file)
- dev/tools/wide_err_a2b.py
- tmp/wide_err_a2b/wide_err_a2b.json

## Errors

None blocking. D:\BO_CVn and D:\DARKS short-exposure paths unavailable; snapshot Raw/lights
and CalibrationLibrary bias used instead.
