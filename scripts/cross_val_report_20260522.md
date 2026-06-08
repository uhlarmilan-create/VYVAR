# VYVAR draft_342 Cross-Validation Report
Generated: 2026-05-22 12:09 UTC  |  VYVAR commit: 4701def

## Summary
| Test | Name | Result | Metric | Threshold | N compared |
|------|------|--------|--------|-----------|------------|
| T01 | Astrometry | PASS | 2.956e-13 | 2 | 135 |
| T02 | Aperture flux | PASS | 0.03228 | 0.035 | 90 |
| T03 | Sky background | PASS | 11.88 | 20 | 60 |
| T04 | FWHM / seeing | PASS | 0.7574 | 1.5 | 3 |
| T05 | Differential LC | PASS | 0.003665 | 0.06 | 3 |
| T06 | BJD/HJD | PASS | 4.023e-05 | 1 | 10 |
| T07 | Airmass | PASS | 0.01134 | 0.05 | 120 |
| T08 | Plate scale | PASS | 0 | 2 | 1 |
| T09 | Comp stability | PASS | 0.01836 | 0.055 | 10 |
| T10 | Variability agreement | PASS | 0.75 | 0.7 | 4 |

**Overall:** 10/10 passed

## Details
### T01 - Astrometry
- **Result:** PASS
- **Metric:** 2.95586e-13 (threshold 2)
- **N compared:** 135
- **Details:** astroalign round-trip median 0.000 arcsec (0.000 px)

### T02 - Aperture flux
- **Result:** PASS
- **Metric:** 0.0322823 (threshold 0.035)
- **N compared:** 90
- **Details:** median |d_mag| = 32.28 mmag

### T03 - Sky background
- **Result:** PASS
- **Metric:** 11.8828 (threshold 20)
- **N compared:** 60
- **Details:** median |d_sky| = 11.88 ADU (VYVAR annulus vs noise_floor_adu)

### T04 - FWHM / seeing
- **Result:** PASS
- **Metric:** 0.757371 (threshold 1.5)
- **N compared:** 3
- **Details:** median |d_FWHM| = 0.757 px

### T05 - Differential LC
- **Result:** PASS
- **Metric:** 0.00366546 (threshold 0.06)
- **N compared:** 3
- **Details:** median |d_lc_rms| = 3.67 mmag (10 frames)

### T06 - BJD/HJD
- **Result:** PASS
- **Metric:** 4.02331e-05 (threshold 1)
- **N compared:** 10
- **Details:** frame-level BJD: max |d_BJD| = 0.0000 s (10 frames)
- **Warning:** TODO-BJD-PERTARGET: VYVAR BJD uses field-center coords, not per-target RA/Dec. Max LTT error ~12s for 2-deg field radius. Negligible for periods >0.01d.
- **Warning:** Measured median per-star LTT offset vs frame BJD: 0.8 s

### T07 - Airmass
- **Result:** PASS
- **Metric:** 0.0113375 (threshold 0.05)
- **N compared:** 120
- **Details:** median |d_airmass| = 0.0113

### T08 - Plate scale
- **Result:** PASS
- **Metric:** 0 (threshold 2)
- **N compared:** 1
- **Details:** VYVAR config=1.3000 arcsec/px; WCS PC/CD=9.7681 (ignored, matches _resolve_plate_scale_arcsec_per_px)
- **Warning:** WCS PC/CD scale 9.768 arcsec/px outside 0.3-5.0; VYVAR uses config override

### T09 - Comp stability
- **Result:** PASS
- **Metric:** 0.0183603 (threshold 0.055)
- **N compared:** 10
- **Details:** median |d_comp_rms| = 18.36 mmag

### T10 - Variability agreement
- **Result:** PASS
- **Metric:** 0.75 (threshold 0.7)
- **N compared:** 4
- **Details:** agreement 75.0% (3/4)
- **Warning:** ROT/noisy stars may disagree by design

