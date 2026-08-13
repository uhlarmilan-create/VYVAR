# A-1 growth-curve closure measurement

**Date:** 2026-08-13  
**Method:** Read-only. 134 frames draft_510, 139 frames draft_435. Same annulus sky as production (`photometry_core._annulus_sky_subtracted_flux`, r_in=10, r_out=15 px). Radii 0.5-15 px step 0.25. Eight comparison stars spanning mag 8-11 per draft. Raw JSON: `tmp/a1_growth_curve_results.json` (gitignored).

---

## 1. Growth curves

### 1.1-1.2 Per-star metrics (draft 510 exemplar)

| catalog_id | mag | prod r [px] | EE@prod | r50 | r90 | r95 | r99 | asym flux |
|------------|-----|-------------|---------|-----|-----|-----|-----|-----------|
| 1498428263244809344 | 8.53 | 4.54 | **0.880** | 1.75 | 5.0 | 6.5 | 9.0 | 342465 |
| 1499200223486564608 | 9.68 | 4.14 | 0.844 | 1.75 | 5.25 | 6.5 | 8.75 | 117452 |
| BO CVn 1498613634033133184 | - | 4.14 | **0.858** | 1.75 | 5.0 | 6.5 | 9.0 | 121493 |

Median EE@production over eight comps: **0.815**. BO CVn: **0.858**.

### 1.3 PSF model fit

Growth curves fit Moffat better than Gaussian (lower SSE; beta~1.5 typical for ground PSF wings).

| estimator | draft 510 | draft 435 |
|-----------|-----------|-----------|
| r50 -> Gaussian FWHM | ~3.0-3.4 px | ~3.0-3.8 px |
| r50 -> Moffat FWHM (beta~1.5) | ~2.7-3.1 px | ~2.7-3.8 px |
| MASTERSTAR VY_FWHM_GAUSS (SNR authority) | 3.30 | 2.40 |
| per-frame proc `fwhm_estimate_px` median | 3.16 | 3.32 |
| QC moment `fwhm_px` median | 5.19 | 3.21 |

**VY_FWHM_GAUSS tracks per-frame DAO FWHM and growth-curve r50, not QC moment.**

### 1.4 Enclosed fraction at production radius

| draft | BO CVn r [px] | EE@prod | comp median EE@prod |
|-------|---------------|---------|---------------------|
| **510** | 4.14 | **85.8%** | **81.5%** |
| **435** | 2.72 | **73.1%** | **66.8%** |

**90% EE requires r ~5.0-5.75 px** on draft 510 for typical comps (vs production ~4.1 px).

### 1.5 Draft 435 confirmation

At production r=2.716 px (star 1502232642196131072): **EE=61.0%** - refutes literal 54% at *current* SNR-sized radii but confirms **material undersizing** vs 90% target. Smaller SNR-table FWHM (2.40) on 435 drove smaller apertures than per-frame PSF (~3.3 px) would imply.

---

## 2. Why estimators disagree

| estimator | measures | draft 510 | draft 435 |
|-----------|----------|-----------|-----------|
| MASTERSTAR VY_FWHM_GAUSS | stacked Gaussian fit | 3.30 | 2.40 |
| MASTERSTAR VY_FWHM / QC moment | wider moment on stack/QC crop | 5.19 | 3.21 |
| proc `fwhm_estimate_px` | per-frame DAO on science frames | 3.16 | 3.32 |
| xval harness (prior) | photutils on aligned frame | 2.96 | - |

**2.2 Stacking:** 435 SNR FWHM (2.40) << per-frame DAO (3.32) - stack product is **narrower** than single-frame PSF used for photometry. Bicubic alignment resampling adds wing flux but does not explain this gap; likely different fit domain (stack vs frame).

**2.3 Moment bias:** QC moment FWHM (~5.2 px on 510) includes background/neighbour wings - not the core PSF used for aperture photometry.

**2.4 Verdict:** Each estimator is correct for its definition. Defect is **using VY_FWHM_GAUSS for SNR sizing when per-frame DAO FWHM is the operative PSF width**, compounded on 435 by an underestimated stack Gaussian.

---

## 3. What to do

**3.1 FWHM for SNR table:** Recommend **per-frame DAO/`fwhm_estimate_px` median** (or growth-curve r50) over stack VY_FWHM_GAUSS. Evidence: matches EE curves; 435 stack Gaussian is inconsistent with frame PSF.

**3.2 COG correction:** At EE~0.82-0.86 (510), Stetson-style correction ~0.15-0.20 mag in absolute flux; **mostly cancels in differential** if comps share similar EE. Enabling `cog_aperture_correction_enabled` would matter most for unequal-EE comp sets and publication absolute calibration.

**3.3 Fixed enclosed fraction:** DAOPHOT/AIJ often size to empirical curve; SExtractor `MAG_AUTO` is Kron-like (~90%+ intent); photutils docs recommend verifying curve of growth. Fixed 90% EE sizing would remove estimator argument; requires per-star or per-field r90 from growth curve.

**3.4 If radius increases (~5.5 px for 90% EE on 510):** Re-cut SNR table; draft 510 anchor LC/trust re-verify; saturation admission uses same r - fewer rejects; aperture correction reference stars change. **Do not re-cut without Milan authorization.**

---

## 4. Pre-registered case

**Neither pure case applies alone:**

- **Not case 1** (>90% EE): production EE is **81-86%** (510) and **67-73%** (435).
- **Case 2 partial:** materially below 90%; not as extreme as 54% at *actual* production radii on 510, but 435 approaches it.
- **Case 3 applies:** **draft 435 systematically lower EE** than 510 at production radii - sizing stability issue (SNR FWHM authority vs frame PSF).

**A-1 status:** Real undersizing vs 90% EE target; definitional FWHM spread explains mechanism; **not closed as cosmetic**.

---

## DECISION REQUIRED

1. **Accept 82-86% EE on anchor 510 as sufficient for differential work**, document in Wave 7, defer radius change.
2. **Re-author SNR FWHM to per-frame DAO/growth-curve r50** and re-cut anchor (510 first).
3. **Enable COG correction** for absolute-calibration claims only.
4. **Adopt fixed 90% EE sizing** (larger architectural change).

**Recommendation:** (2) then re-verify 510 anchor - fixes 435/510 inconsistency at root without jumping to fixed-EE architecture. Differential scatter may barely move (comps already similar EE); absolute and publication claims benefit most.
