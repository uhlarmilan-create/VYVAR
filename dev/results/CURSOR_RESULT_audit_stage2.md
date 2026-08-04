CURSOR RESULT - 2026-07-30 AUDIT REMEDIATION STAGE 2 (measurement only)

Source JSON: `tmp/audit_stage2_measure.json`. Anchor: `draft_000435_snapshot_skysurface_20260716`.

## 2.1 D5-1 Aperture radius behaviour

### Code answers (a/b)

**(a) FWHM for global aperture:** `compute_fwhm_gaussian_for_aperture_catalog`
(`photometry_core.py:11657+`) uses **per-frame** `VY_FWHM` from each aligned FITS header
(DAO scaled x0.667), else moment median. **Not** a single draft-global constant.

**(b) SNR aperture mode on anchor:** **OFF** - no `snr_aperture_table.json` on snapshot.
`global_fixed` with `aperture_fwhm_factor` x per-frame FWHM.

### Measurements

| Quantity | Value |
|----------|-------|
| Comp `aperture_r_px` median | **2.416 px** (range 1.916-3.166) |
| Per-frame FWHM median spread | **10.8%** (5.92-6.57 px) |
| BO CVn LC `aperture_r_px` | **2.716 px** constant all epochs |
| LC mag vs FWHM slope (40 targets) | median **-3.5 mmag/px** (IQR 84; inconclusive) |

**Literature (R1):** Stetson (1990) PASP 102, 992; Howell (1989) PASP 101, 616 - differential
photometry cancels aperture differences when **same aperture** applied to target and comps on each
frame. VYVAR uses **fixed global radius per frame** (not per-star SNR mode on anchor), so predicted
FWHM-tracking bias slope ~ **0** at first order.

**Verdict:** Measured LC slope **not consistent with zero** at median but highly scattered;
dominant anchor behaviour is **constant target aperture 2.716 px** vs comp **2.416 px** (role factor
1.1 x FWHM implied). Full residual test needs Stage 1.2 provenance on regen.

## 2.2 D10-1 CV vs CR - DECISION REQUIRED

148 comparison stars (Gaia G + BP-RP -> Johnson):

| Band | slope (mag vs BP-RP) | scatter | n |
|------|----------------------|---------|---|
| **V** | **-0.386 +/- 0.008** | 0.129 mag | 146 |
| **R** | **+0.107 +/- 0.007** | **0.043 mag** | 146 |

**R band flatter** (lower colour slope and scatter). Instrumental passband closer to Cousins R than
Johnson V for unfiltered QHY294.

**Literature (R1):** AAVSO CCD Manual / Extended Format - CV and CR are **calibrated Johnson/Cousins
comparison star magnitudes** in V and R respectively; unfiltered hardware must be mapped to one.

**DECISION REQUIRED (Milan):** do not change band mapping in this task.

## 2.3 D1-2 Detector non-linearity

| Peak ADU bin | n | median G |
|--------------|---|----------|
| 0-15k | 13948 | 10.74 |
| 15-30k | 3807 | 9.24 |
| 30-45k | 1300 | 8.38 |
| 45-60k | 186 | 8.35 |

Strong G vs peak correlation (r=-0.85) - **dominated by magnitude**, not linearity defect.
Brightest comp peak **61404 ADU** (G~8.35). No monotonic residual trend isolated without
instrumental-minus-catalogue on linearized scale.

**Literature:** CMOS linearity typically tested via flat-field ratio vs exposure; manufacturer
spec **UNVERIFIED** this session (IMX294 datasheet not fetched).

## 2.4 P-02 Variance budget - DECISION REQUIRED

LC export **lacks** per-term `err_photon`, `sem_rel`, `sigma_sys_rel` columns.

Proxy: `lc_rms / lc_rms_ooe` median **2.08**; **98%** of targets >1 -> predominantly
**under-quoted** vs OOE scatter on this anchor (consistent with audit correction).

**Scintillation:** Osborn et al. (2015) MNRAS 452, 1707 - implemented in `sigma_budget`; **not
wired** to production err (per task).

**DECISION REQUIRED:** do not enable scintillation in production err without Milan approval.

## 2.5 U-09 DATE-OBS convention

Rig **NoFilter_60_2 / QHY294PROM** (5 sampled lights):

| Keyword | Present |
|---------|---------|
| DATE-OBS | **yes** (ISO start timestamp) |
| EXPTIME | 60 s |
| DATE-END | **no** |
| MIDPOINT / EXPMID | **no** |

**Literature (R1):** FITS Standard 4.0 / Rots et al. (2015) A&A 574, A36 - `DATE-OBS` is the
**start** of the observation unless `DATE-BEG`/`DATE-END` pair defined.

**Risk:** mid-exposure time = DATE-OBS + EXPTIME/2 (**30 s** bias). **UNVERIFIED** whether QHY
driver stamps shutter-open vs readout start.

## 2.6 T4-1 Detection noise on resampled frames - DECISION REQUIRED

**Literature (R1):** Fruchter & Hook (2002) PASP 114, 144 - resampling correlates pixels;
noise on output pixel scale **underestimates** noise on detection scales. Casertano et al. (2000)
AJ 120, 2747 Appendix A - same.

**Tranche 4 verified (white noise, FWHM 3.2, photutils rel_err=1.360):**

| Case | nominal 3.8 sigma | effective sigma |
|------|-------------------|-----------------|
| unresampled | 3.80 | 3.80 |
| shift 0.25 px | 3.80 | ~3.58 |
| shift 0.50 px | 3.80 | ~3.30 |

**Options (DECISION REQUIRED):**

| Opt | Cost | Survives stacked MASTERSTAR? |
|-----|------|------------------------------|
| A detect pre-align | medium | yes (recommended with TODO-A) |
| B convolved RMS threshold | low | yes |
| C MC drizzle factor | medium | yes (calibration per setup) |
| D document only | none | yes |

## DECISION REQUIRED summary

1. **Stage 0.1:** keep or drop sigma_pp (median ratio 1.034; frame 001 at 1.083).
2. **Stage 2.2:** CV vs CR band mapping for unfiltered.
3. **Stage 2.4:** scintillation in production err.
4. **Stage 2.6:** detection noise strategy A/B/C/D before anchor re-cut.
