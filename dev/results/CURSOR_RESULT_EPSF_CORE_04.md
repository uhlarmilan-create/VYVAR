CURSOR RESULT - 2026-09-15 EPSF-CORE-04

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: e970354 (HEAD at start of task
was already `0528518`, the CORE-03 summary.json regenerate).
Class: MEASUREMENT + dev-only harness. No production code change.
Live 516/517 and Archive read-only. a2/ never staged.

## Housekeeping

`session_20260914_epsf_core_03/summary.json` was regenerated from
the delivered integer-peak-init CSVs at `0528518` (T1 ptp 72.45,
live-window slope 8.384). Matches `CURSOR_RESULT_EPSF_CORE_03.md`.

## Architect error 25 (self-reported)

CORE-03 R-Q4 "slope x observed phase spread" yields a
peak-to-peak quantity, not an RMS. "8.56 of 10.48" over-states
the phase share. Correct accounting: ptp over the ~0.095 px live
window ~8 mmag; RMS (uniform phase) = ptp/sqrt(12) ~2.3 mmag;
empirical share from CORE-01 (target phase rank R^2 0.221)
~4.9 mmag in quadrature. Phase is a co-driver of the target
residual, not 82% of it. R-Q3 stands unchanged.

## Definitions

- RMS_med(d) = sqrt(mean((d - median(d))^2)) over the 134
  identical-ensemble epochs, mmag.
- Phase RMS from a bias grid: bilinear-interpolate the noise-off
  bias surface over the live window, sample uniformly, take
  sqrt(var). Report ptp alongside.
- Live windows: target fracX 0.587-0.682, fracY 0.402-0.504;
  check fracX 0.539-0.669, fracY 0.829-0.935.
- Floor (harness vs production): 1.396 mmag (CORE-03 B1).

## Part A - reference arbitration (zero new photometry)

Aperture product: live `proc_*.csv` column `dao_flux`
(`Archive/.../detrended_aligned/.../NoFilter_60_2/`). Stamped
`aperture_factor_applied=snr_table`; median `aperture_r_px` =
5.749; `fwhm_px_for_aperture` = 3.3014 (per-draft gaussian
override). Annulus APERTURE-01d 2.7/5.2 FWHM. Diff LC rebuilt
with the SAME pinned 4-star AIJ flux-sum ensemble as A2-COMPARE.
Production `mag_calib` LCs use a larger comp pool and are NOT
used. All 6 M2 stars present on all 134 epochs (0 lost).

| id | comparison | target | check | n |
|---|---|---:|---:|---:|
| A1 | PSFEx cat vs aperture | **12.442** | **15.691** | 134 |
| A2 | VYVAR psf_flux vs aperture | 8.495 | 14.073 | 134 |
| A3 | VYVAR vs PSFEx (restate A2-COMPARE) | 10.478 | 21.413 | 134 |
| A4 | aperture check-star RMS_med (arbiter floor) | -- | **8.234** | 134 |

A3 restates 10.48 / 21.41. Arbiter noise floor on the check is
8.23 mmag; both PSF methods sit well above it against aperture.

## Part B - osamp phase probe (Q-PHASE)

Builder: `psf_photometry.build_epsf_model`
(`psf_photometry.py:1357`) offline with
`sandbox_output_dir` under the session dir. Same MASTERSTAR +
`masterstars_full_match.csv` + draft 516. Live
`masterstar_epsf.fits` untouched. Truth T1 = PSFEx deg2 of
Light_076 at the target (block-sum rendering; never ImagePSF).
Fit: `psf_photometry_stars`, CORE-02 kwargs, integer-peak init.
Noise-off 9x9; noise-on n=30 on the 6 target live-window cells.

| osamp | source | ptp full | slope (mmag/0.1px) | phase RMS live | ptp live | mean bias full |
|---:|---|---:|---:|---:|---:|---:|
| 2 | CORE-03 T1 | 72.452 | 8.384 | **2.674** | 10.649 | -169.2 |
| 3 | sandbox | 1053.9 | -152.3 | 42.03 | 186.1 | +713.3 |
| 4 | sandbox | -- | -- | -- | -- | -- |

osamp=3: same 67-star funnel; FITS shape 51x51, sum=9. Probe
cell `fit_ok=False`, chi2~186; flux-scale collapses
(mean_bias +713 mmag). Phase surface is worse than osamp=2.

osamp=4: FITS shape 69x69, sum=16, but ePSF QC FWHM null and
ringing (min/max ~ +/-0.5). All 81 noise-off cells yield
non-finite bias (negative / failed fluxes). Probe:
`psf_flux=-76`, `fit_ok=False`, chi2~4663. Sampling-alone does
not collapse the phase component.

## Part C - machinery knobs (134 epochs, 6 M2 stars)

Live ePSF, live aligned lights, CORE-02 baseline. Floor 1.396
mmag on every row. RMS_med vs PSFEx catalog AND vs Part A
aperture.

| knob | tgt vs PSFEx | chk vs PSFEx | tgt vs ap | chk vs ap | Delta chk vs ap |
|---|---:|---:|---:|---:|---:|
| baseline | 10.461 | 21.271 | 8.794 | 13.936 | 0 |
| K1 gain/RN g_pt | 10.225 | 20.722 | 8.865 | 13.587 | +0.35 |
| K2 fit_shape 13x13 | 11.997 | 24.009 | 11.476 | 16.938 | -3.00 |
| K3 uniform weights | **4.923** | **9.928** | 12.651 | 13.347 | +0.59 |
| K4 wider annulus 4/8 | 10.491 | 21.423 | 8.554 | 14.081 | -0.14 |
| K5 K1+K2 | 11.444 | 22.965 | 11.146 | 15.940 | -2.00 |

No knob lowers B1 vs aperture by >= 3.0 mmag on either star.
K3 moves the harness toward PSFEx (check 21.27 -> 9.93) but
does not improve vs the aperture arbiter (check 13.94 -> 13.35;
target worsens).

Cites: K1 `_psf_resolve_gain_read_noise`; K2
`_fit_shape_for_cutout` (prod ceil(2*fwhm+1) -> 9x9); K3
`_psf_fit_error_cutout_full_ccd`; K4 `_psf_annulus_radii_px`
inner/outer 4.0/8.0 FWHM (prod residual annulus 2.7/5.2).

## Readings (binding)

R-R2: A1 >= 10.0 mmag on EITHER star (target 12.44, check
15.69). No PSF method reaches 3 mmag against the aperture
arbiter on this rig. DoD reference re-decision is Milan's
(not a code verdict).

R-P2: osamp=4 does not collapse the phase component (pathological
rebuild / non-finite bias surface). Dithered build is the
remaining direction for the phase share (TODO-A scope).

R-K0: no listed knob moves B1 vs aperture by >= 3.0 mmag.

No fix. Any fix that moves `psf_flux` belongs to the 520 era
re-cut (D-EPSF-XVAL-DOD-01). Sequencing is Milan's.

## G4 / gates

Live 516 unchanged: csv `bfa24039` / fits `13e77cf8` / epsf
`172f9540` (all PASS). Rebuilt osamp 3/4 FITS live under
`session_20260915_epsf_core_04/` only. `--fast --clean`
OVERALL PASS (1629 passed, 34 skipped; clean-tree PASS).
`a2/` never staged.

## Files

- `dev/xval_psfex/epsf_core_04.py` (zero `src_py` imports of it)
- `dev/results/context/session_20260915_epsf_core_04/`
  arbitration_lc.csv, arbitration_summary.csv,
  phase_bias_osamp3.csv, phase_bias_osamp4.csv,
  phase_window_summary.csv, knobs.csv, summary.json,
  epsf_osamp3/, epsf_osamp4/
- this file

## STOP

Measurement only. No production change, no fix, no era re-cut.
Sequencing is Milan's on these numbers.
