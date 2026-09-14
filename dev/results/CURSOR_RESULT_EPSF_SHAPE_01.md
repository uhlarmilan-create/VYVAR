CURSOR RESULT - 2026-09-14 EPSF-SHAPE-01

Date: 2026-09-14. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: fe6a92d (verified HEAD at start).
Class: MEASUREMENT + dev-only script. No production code change.
Live 516/517 and Archive read-only. Closure of anything is Milan's.

## What I did

Root-cause measurement for R-A2-3 (VYVAR ePSF vs PSFEx differential
LC: 10.48 mmag target / 21.41 mmag check RMS on 134 identical
epochs). Dev-only script `dev/xval_psfex/epsf_shape_01.py` (zero
`src_py` imports of it). Linux `a2/` read, never staged.

## Premise checks

P1 CONFIRMED -- 516 used a single global ePSF. Do not STOP.

- Live `masterstar_epsf.fits` sha prefix `172f9540`, shape (35, 35),
  oversampling=2, n_stars_used=67, spatial_order=0.
- `epsf_sum_native=1.0` (sum/osamp^2 convention;
  `psf_photometry.py:649-661`). Production FWHM self-check
  `_epsf_fwhm_native_from_profile` (`psf_photometry.py:527-569`) on
  this array: 2.36385 px (meta qc 2.364).
- Read path: `fits.getdata` + meta osamp
  (`psf_photometry.py:2767-2818`). Not `get_epsf_fwhm_from_context`.
- `pipeline_meta.json`: `psf_spatial_enabled=false`,
  `psf_spatial_order=0`. `psf_spatial_grid="3x3"` is persist-only
  (`src_py/config.py`), never consumed. `photometry_plan.json`
  `grid_nx=0` is the comparison-star grid, not ePSF. No `*epsf*grid*`
  files. `build_epsf_grid_model` did not run for 516.

P2 CONFIRMED -- one `.psf` per frame, POLDEG1=2 under
`a2/out/work/` (134) and deg3 under `a2/out/deg3/work/`.

P3 CONFIRMED -- `a2_compare/m2_epochs_{target,check}.csv` carry
`resid_after_median` for target `1498613634033133184` and check
`1497613731286514432`.

Position source: local `a2_compare/match_rows_{deg2,deg3}.csv`
`XPSF_IMAGE`/`YPSF_IMAGE` (gitignored; frames aligned, positions
near-constant). qc FWHM: `vyvar_reference/qc_metrics.csv` `fwhm_px`
joined by `BO_CVn_Light_NNN`.

fitrad = 3.3014 px from live meta `fwhm_px` (A1 fitrad).

## Reader validation (5 frames 001/037/076/109/148)

Pure-Python PSFEx reader (FITS `PSF_DATA`; PolOrder, POLZERO1/2,
POLSCAL1/2, PSF_SAMP; total-degree basis 1,u,v,u^2,uv,v^2). No new
dependency.

Integrity PASS:

- reconstruct(POLZERO) == component 0 (maxabs=0) on all five.
- Degree 2 is not constant: corner-to-corner max |delta| 0.037-0.054
  (~30% of c0 peak).
- Header identity: PSF_FWHM / PSF_SAMP = 4.7 exactly on all five.

The task 5% reconstructed-FWHM vs header PSF_FWHM gate does not
measure the reader. See architect error 21. `psfex.stdout` is empty
on all five; the console diagnostic FWHM is on `psfex.stderr`
(2.03-2.21). Production radial FWHM of component 0 is ~24% below
header and ~8-12% above that stderr diagnostic. Metrics use the
production radial method on both models, as specified in step 4.

## Headline shape numbers (deg2 primary; 134 frames)

Population: target `1498613634033133184` and check
`1497613731286514432`; FWHM via `_epsf_fwhm_native_from_profile` on
the VYVAR oversampled array (osamp=2) and on the PSFEx
reconstruction at (XPSF,YPSF) times PSF_SAMP. VYVAR is constant.

| quantity | value |
|---|---|
| FWHM_vyvar | 2.364 px |
| median FWHM_psfex at target | 2.378 px |
| median FWHM_psfex at check | 2.319 px |
| (vyvar-psfex)/psfex at target | -0.59% |
| (vyvar-psfex)/psfex at check | +1.93% |
| spatial FWHM_psfex (check-target), median | -0.0395 px |
| qc fwhm_px std (134 identical-ensemble frames) | 0.0286 px |
| qc fwhm_px range (same 134) | 5.138-5.305 px |

Headline FWHM difference does not exceed 5%.

Standalone (pre-registered): |spatial FWHM spread| 0.0395 px exceeds
frame-to-frame seeing std 0.0286 px.

Static EE mismatch (not a correlation): median
MISMATCH = -2.5 log10(EE_vyvar/EE_psfex) is -190.0 mmag (target) /
-184.9 mmag (check) inside fitrad 3.3014 px. VYVAR puts more energy
in the fit radius than PSFEx. That offset is nearly constant across
frames; see C2.

deg3 sensitivity (same 134 x 6 stars): FWHM_psfex median 2.379
(target) / 2.379 (check); spatial check-target -0.005 px (does not
exceed seeing std). EE mismatch medians -189.0 / -175.5 mmag.

n_shape_deg2=804, n_shape_deg3=804 (6 stars x 134 frames).

## C1-C4 (deg2; Spearman + Theil-Sen; rank R^2 = rho^2)

| test | star | n | rho | p | slope | R^2 |
|---|---|---:|---:|---:|---:|---:|
| C1 M-SEEING resid vs qc fwhm_px | target | 134 | -0.150 | 0.083 | -55.0 mmag/px | 0.023 |
| C1 M-SEEING resid vs qc fwhm_px | check | 134 | -0.089 | 0.307 | -70.8 mmag/px | 0.008 |
| C2 M-SPATIAL resid vs MISMATCH(f,s) | target | 134 | 0.095 | 0.276 | 0.096 | 0.009 |
| C2 M-SPATIAL resid vs MISMATCH(f,s) | check | 134 | 0.005 | 0.953 | -0.014 | 0.000 |
| C4 common mode target resid vs check | both | 134 | 0.096 | 0.271 | 0.230 | 0.009 |

C1 population: 134 identical-ensemble epochs; `resid_after_median`
from `m2_epochs_{target,check}.csv` (mmag) vs qc `fwhm_px`. Seeing
lever on this set is 0.17 px peak-to-peak.

C2 population: same 134 epochs; MISMATCH is `ee_ratio_mmag` at that
star's (XPSF,YPSF) on that frame.

C3 population: `m1_per_star.csv` stars with n_ok >= 100 and finite G
(n=34). M1 is the 65-star diagnostic, not the M2 product metric.
IRLS plane `d_mmag ~ a + b x + c y` on live `masterstars_full_match.csv`
x,y (pixels). a=-219.78 mmag, b=0.0109 mmag/px, c=-0.0376 mmag/px.
Amplitude across those 34 stars: 46.54 mmag. Predicted
check-minus-target -16.61 mmag; observed M1 median_d difference
-68.76 mmag (target -247.84, check -316.61). Sign matches; scale
does not.

C4: top-8 |resid| overlap is Light_032 / 053 / 076 (as A2 flagged).
Common-mode R^2=0.009 -- do not attribute the series to a shared
temporal mode.

## Reading (binding)

R-SH3: nothing reaches 0.2 on either star -> shape unsupported as
the LC-level driver; escalate suspect to EPSF-CORE-01 (fit
machinery). M-SEEING target=0.023 check=0.008; M-SPATIAL
target=0.009 check=0.000.

Standalone (also fires): spatial FWHM spread exceeds frame-to-frame
seeing std. Headline FWHM % difference does not exceed 5%.

No fix. No production change. No grid enablement.

## Architect error 21

Task step 3 required reconstructed FWHM within 5% of header
`PSF_FWHM` (or `psfex.stdout`). Header `PSF_FWHM` is 4.7*PSF_SAMP
on every probe frame -- PSFEx sampling-design FWHM, not a measured
width of `PSF_MASK`. `psfex.stdout` is empty; stderr diagnostic
FWHM uses a different estimator than
`_epsf_fwhm_native_from_profile`. Reader validated by algebra
(center==c0, corners vary, 4.7 identity). Metrics used the
production FWHM method on both models (step 4).

## Errors on the record

None in the script run. Architect error 21 above.

## G4 / gates

Live 516 unchanged after the run: csv `bfa24039` / fits `13e77cf8` /
epsf `172f9540` (all PASS). `--fast --clean` OVERALL PASS
(1629 passed, 34 skipped; clean-tree PASS).
`git check-ignore` covers `a2/`; that tree was never staged.

## Files

- `dev/xval_psfex/epsf_shape_01.py` (zero `src_py` imports of it)
- `dev/results/context/session_20260914_epsf_shape_01/`
  `shape_metrics_deg2.csv`, `shape_metrics_deg3.csv`,
  `correlations.csv`, `headline.json`
- this file

## STOP

No reland, no grid build, no EPSF-CORE-01 work. Sequencing is
Milan's on these numbers.
