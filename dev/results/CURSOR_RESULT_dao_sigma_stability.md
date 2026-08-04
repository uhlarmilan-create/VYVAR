CURSOR RESULT - DAO-SIGMA-STABILITY (2026-07-30)

Read-only measurement. No code changes. No commits.

---

## Provenance

| Item | Value |
|------|-------|
| Draft | `draft_000435` |
| Input | `Archive/Drafts/draft_000435/calibrated/lights/NoFilter_60_2/` (150 lights) |
| Code | Local AUDIT-T3 bundle (P-10 sign fix + `sigma_pp` DAO estimator + threshold 3.8) |
| Git (uncommitted bundle base) | `06ed950` |
| Preprocess | Fresh `_fit_subtract_preprocess_sky_surface` order=2 on calibrated data (P-10 corrected) |
| Align | Production `_alignment_compute_one_frame` (astroalign) vs ref `BO_CVn_Light_001.fits` |
| **Not used** | Archived `proc_*` on disk (pre-P-10 doubled gradient) |
| Background2D | `box_size ~ 8x FWHM` (odd, 16-256), `filter_size=3`, `MADStdBackgroundRMS` |
| Script | `tmp/dao_sigma_stability_measure.py` ? `tmp/dao_sigma_stability.json` |

**Note:** Calibrated FITS lack `VY_FWHM`; top-5 M2/M3 candidates were the first five preprocessed files (001-005), which overlap the true archived top-FWHM set (002, 001, 003, 007, 004) except 005 (7th by FWHM). M2 spread numbers are representative but should be re-checked after wiring FWHM from QC/detrended headers.

---

## M1 - Estimator comparison (10 frames, 3 stages)

### Summary medians (ADU) and ratio to `actual_scatter`

| Stage | sigma_pp med | bkg2d_rms med | actual_scatter med | ratio sigma_pp | ratio bkg2d |
|-------|-------------:|--------------:|-------------------:|-----------:|------------:|
| calibrated | 42.52 | 42.42 | 41.59 | **1.023** | 1.019 |
| preprocessed (P-10) | 42.52 | 42.41 | 41.17 | **1.033** | 1.029 |
| detrended_aligned | **30.64** | 34.31 | 32.42 | **0.943** | 1.055 |

**Finding:** P-10 preprocess barely moves estimators (cal -> pre). **Align/resample drops `sigma_pp` ~28%** (42.5 -> 30.6 ADU) while `actual_scatter` drops ~22%. After align, **`sigma_pp` under-estimates** ground truth (ratio median 0.94 vs 1.03 pre-align). The simulated resampling concern **materialised on real frames**.

---

## M2 - Top-5 candidate stability (post-align)

| frame | VY_FWHM* | align | dx_frac | dy_frac | sigma_pp | bkg2d_rms | thr 3.8*sigma_pp | thr 3.8-bkg2d |
|-------|---------:|-------|--------:|--------:|---------:|----------:|-------------:|--------------:|
| Light_001 | - | astroalign | 0.0009 | 0.0006 | 51.94 | 51.70 | 197.4 | 196.5 |
| Light_002 | - | astroalign | 0.0241 | 0.0313 | 50.17 | 50.18 | 190.6 | 190.7 |
| Light_003 | - | astroalign | 0.1100 | 0.0348 | 47.39 | 47.91 | 180.1 | 182.0 |
| Light_004 | - | astroalign | 0.9710 | 0.1281 | 47.97 | 48.02 | 182.3 | 182.5 |
| Light_005 | - | astroalign | 0.8166 | 0.1849 | 44.93 | 45.47 | 170.7 | 172.8 |

*VY_FWHM absent on calibrated headers in this run.

### Spread (max/min ~ 1)

| Estimator | Relative spread |
|-----------|----------------:|
| **sigma_pp** | **15.6%** |
| **bkg2d_rms_median** | **13.7%** |
| Implied threshold (3.8*sigma_pp) | **15.6%** |
| Implied threshold (3.8-bkg2d) | **13.7%** |

Subpixel shifts **are recoverable** from astroalign matrix (fractional tx/ty reported).

---

## M3 - Convolved background vs pixel estimators (5 candidates)

| frame | convolved_std | sigma_pp | bkg2d | conv/sigma_pp | conv/bkg2d |
|-------|-------------:|-----:|------:|----------:|-----------:|
| Light_001 | 23.15 | 51.94 | 51.70 | 0.446 | 0.448 |
| Light_002 | 21.36 | 50.17 | 50.18 | 0.426 | 0.426 |
| Light_003 | 19.62 | 47.39 | 47.91 | 0.414 | 0.410 |
| Light_004 | 18.82 | 47.97 | 48.02 | 0.392 | 0.392 |
| Light_005 | 18.08 | 44.93 | 45.47 | 0.402 | 0.398 |

**Ratio spread:** conv/sigma_pp **13.6%**; conv/bkg2d **14.3%** - neither ratio is stable across candidates. Per-pixel estimators do not track convolved DAO significance uniformly (~0.39-0.45 of pixel ?).

---

## M4 - Pass-1 DAO counts (extreme sigma_pp frames)

| frame | sigma_pp | mode | threshold ADU | pass-1 DAO |
|-------|-----:|------|-------------:|-----------:|
| Light_005 (low sigma_pp) | 44.93 | 3.8*sigma_pp | 170.7 | **2489** |
| Light_005 | 44.93 | 3.8-bkg2d | 172.8 | 2417 |
| Light_005 | 44.93 | anchor abs | 175.4 | 2340 |
| Light_003 (mid-low) | 47.39 | 3.8*sigma_pp | 180.1 | 3004 |
| Light_003 | 47.39 | anchor abs | 175.4 | 3241 |
| Light_001 (high sigma_pp) | 51.94 | 3.8*sigma_pp | 197.4 | 2426 |
| Light_001 | 51.94 | anchor abs | 175.4 | **3294** |

At fixed 3.8*sigma_pp: **2489-3004** (~21% spread). Background2D thresholds move counts ~3% vs sigma_pp on same frame. Anchor absolute 175.4 ADU is **not** equivalent to 3.8*sigma_pp on these post-align frames (sigma_pp ~45-52 ? nominal threshold 171-197).

---

## VERDICT: **S3** *(superseded - see Tranche 4)*

> **ERRATA (2026-07-30, Audit Tranche 4).** Background2D switch recommendation **withdrawn**.
> M3 used wrong kernel; DAOFIND nominal N-sigma_pixel = N sigma_conv (verified rel_err~1.36).
> Real defect: correlated noise on aligned frames. M1/M2 used different frame lists (overlap: Light_001 only).
> See `dev/results/CURSOR_RESULT_audit_t4.md`.

**Original (historical):** sigma_pp spread 15.6%; do not re-cut on sigma_pp alone without addressing resampling correlation.

---

## Files

| Path | Role |
|------|------|
| `tmp/dao_sigma_stability_measure.py` | Measurement harness |
| `tmp/dao_sigma_stability.json` | Full numeric output |
| `tmp/dao_sigma_stability_work/preprocessed/` | Fresh P-10 preprocessed FITS (150) |
