CURSOR RESULT - 2026-08-24T12:40:00Z (EPSF-SHAPE-01-M)

**H5 SUPERSEDED by `CURSOR_RESULT_EPSF_SHAPE_01_F.md` (F2: IterativePSFPhotometry ==
PSFPhotometry with AC off, iter/single = 1.000; the M3 0.667 vs 1.357 split was
production F6 global `psf_ac_factor` ~0.528, not a fitter-class scale). F2b
pedestal mechanism retracted (outer-annulus median 0; ringing, not a DC
pedestal).**

What I did
Measurement-only diagnosis of the bright-star PSF shape mismatch that survives
FD-A. Sandbox artifacts under `dev/results/session_20260824_epsf_shape_01_m/`.
Production models, per-frame catalogs, aperture LC files, and AAVSO/VarAstro
exports were not written. Not pushed; Milan authorizes.

Premise (0.1): post-FD-A BO CVn chi2 median 22.6 (expected ~1-3) and PSF/DAO
droop 0.671 are a shape/scale problem, not a variance-model problem (faint-end
chi2 ~1.5 is sane) and not saturation (BO CVn at 28 percent of knee). This task
ranks H1-H4 by measured contribution so SHAPE-01-F can be scoped against
evidence.

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G1 git pull / tip | PASS | `b1af0493dde53590850541376dceb233e5da0f46` (SESSION-CLOSE-20260823); already up to date |
| G2 `--fast` OVERALL | PASS | end-of-session `--fast` OVERALL PASS (1518 passed, 32 skipped); db-quick-check WARN via committed waiver |
| G3 anchor era | PASS | checker constants unchanged: core `9902d918` n=121, extended `472bc9e4` n=179 on frozen `draft_000516_snapshot_era03_20260820` |
| G4 production ePSF | PASS | live 516 `masterstar_epsf.fits` is the gated 67-star model (`sandbox_output=false`); SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20`; meta n_stars_used=67, oversampling=2, quadratic, cutout=17, created 2026-08-22T19:11:31Z; epsf_fwhm_native_px=2.364 (0.716 x input FWHM 3.301) |

Production-hash guard (22 watched files: ePSF FITS+meta, sample LCs including
additive `lightcurve_*_psf.csv`, AAVSO, VarAstro, proc CSVs):
`hashes_before.json` == `hashes_after.json` (n_diff=0). Positive control
`positive_control.txt` changed (`before` -> `after`). Guards that compare
nothing always pass -- this one compared 22 real production bytes.

## M1 - residual QA at production fit coordinates

20 frames spread over the night (`frame_subset.txt`). Science set n=333.
5179 residual cutouts. Model evaluated at stored catalog `(x,y)` + `psf_flux`
via `_psf_model_prediction_cutout` (sidecars do not persist `x_fit`).

Mag bins (bright -> faint): 5.94-10.55 / 10.55-12.39 / 12.39-13.25 / 13.25-14.06 / 14.06-15.56.

Fractional residual (data-model)/model_peak vs radius:

| bin | r=0.25 | r=1.25 | r=4.25 | r=8.25 |
|-----|--------|--------|--------|--------|
| 0 bright | 0.754 | 0.284 | 0.033 | 0.007 |
| 4 faint  | 1.051 | 0.273 | 0.027 | -0.002 |

Departures vs faint baseline with |delta|>=0.05: n=6, all at r=0.25 or r=1.25.
Zero wing departures. Primary discriminator = **core-shaped** (H1/H2, not H3).

Caveat: the core residual is large in EVERY bin (faint |core| 0.662 > bright
0.519). M1 therefore does not say "bright stars uniquely mismatch"; it says
the mismatch that differs from the faint baseline lives in the core, not the
wing. Catalog `(x,y)` may be DAO rather than PSF-fit, so some core residual
can be centering. That caveat does not revive H3.

Artifacts: `m1_summary.json`, `m1_radial_frac_resid.csv`, `m1_depart_vs_faint.csv`,
`m1_resid_bin{0-4}.png` + `.fits`, `m1_stack_bin{0-4}.npy`, `m1_star_frame_rows.csv`.

## M2 - curve of growth / truncation

Gaussian enclosed-energy at r=8.5 px, FWHM=3.3 px: 0.99999999; truncation
deficit 1.04e-8. That is 3e-8 of the observed 0.329 droop (1-0.671).
`trunc_explains_half_of_droop` = false.

Empirical bright-star EE at 8.5 px median = 0.949 (about 5 percent of flux
between 8.5 and 16 px). Model circular EE at 8.5 native px = 0.698 -- this is
a square-vs-circle integral on the oversampled array, not missing wings.
Production ePSF is flux-normalized inside the 17 px cutout, so model EE at the
cutout edge cannot explain PSF/DAO = 0.671 by construction.

**H3 is out for the ratio droop.** Truncation is also rig-leaning (undersampled
wide PSF is compact), but the measured contribution is negligible.

Artifacts: `m2_summary.json`, `m2_model_growth.csv`, `m2_empirical_growth.csv`.

## M3 - oversampling x smoothing (sandbox only)

Same gated 67-star pool. Six combos. Production ePSF SHA unchanged through
builds. Sandbox photometry used `use_iterative=False` (single-pass) so the
grid would finish; production remains iterative. Determinism: EPSFBuilder
maxiters=15, no global RNG; `sandbox_output_dir` isolates writes.

os2_quadratic sandbox FITS SHA256 equals production
`172f95403beae36d...` (bit-identical ePSF array; meta timestamp differs).

| tag | build | n_stars | n_fail last iter | fit_ok frac | ratio median | chi2 bright / faint | ePSF peak / min |
|-----|-------|---------|------------------|-------------|--------------|---------------------|-----------------|
| os1_quadratic | FAIL all stars | - | - | - | - | - | - |
| os1_quartic | FAIL all stars | - | - | - | - | - | - |
| os2_quadratic | OK | 67 | 0 | 0.812 | 1.746 | 41.8 / 0.78 | 0.095 / -0.055 |
| os2_quartic | OK file, bad PSF | 65 | 57 | 0.118 | -0.009 | 3430 / 1.47 | 4.61 / -7.02 |
| os4_quadratic | OK file, bad PSF | 66 | 65 | 0.093 | -0.001 | 2145 / 1.41 | 0.414 / -0.598 |
| os4_quartic | OK file, bad PSF | 67 | 39 | 0.023 | 0.041 | 2191 / 1.40 | 0.497 / -0.519 |

os1: `ValueError: The ePSF fitting failed for all stars.` Undersampled wide
rig (9.77 arcsec/px, FWHM 3.3 px) cannot build a native-sampled ePSF.
Anderson & King 2000 core-sampling requirement is confirmed as a **build
constraint**. It is rig-specific.

os2_quadratic is the only photometrically usable sandbox model. Quartic at
osamp=2 produces a ringing ePSF (peak 48x the quadratic peak). osamp=4
diverges in EPSFBuilder (almost every star fails the last iteration; QC FWHM
null). Raising oversampling to 4 is not an available fix until the builder
path is repaired.

Fitter-scale split (same production ePSF, 20-frame subset, BO CVn
`1498613634033133184`):
- production iterative PSF/DAO median = 0.667 (matches the 0.671 baseline)
- sandbox single-pass PSF/DAO median = 1.357
- sandbox BO CVn chi2 median = 22.2 (matches production ~22.6)

M3 sandbox ratios are therefore NOT a re-measure of the 0.671 droop. They
rank builder settings. The droop lives in the iterative production product.

Artifacts: `m3_summary.json`, `m3_sensitivity.csv`, `fits_os{1,2,4}_{quadratic,quartic}.csv`,
`models/<tag>/masterstar_epsf.fits` + meta (os1 has no FITS).

## M4 - two-pass F_model

Brightest 30 science-set stars, 20 frames, production ePSF. Second pass seeds
FD-A variance from first-pass `psf_flux` instead of DAO flux.

| metric | value |
|--------|-------|
| n_stars / n_rows | 30 / 600 |
| chi2 median one-pass | 68.44 |
| chi2 median two-pass | 62.34 |
| d_chi2 median | -5.33 |
| d_flux_frac median | +0.0022 (0.22 percent) |
| d_flux_frac p95 abs | 0.016 |

Bright chi2 drops a little toward the faint baseline but stays ~62 vs faint
~1. Flux barely moves. **H4 is a small chi2-statistic artifact; it does not
explain the ratio droop.** Keep the two findings separate.

Artifacts: `m4_summary.json`, `m4_two_pass_compare.csv`, `fits_m4_onepass.csv`,
`fits_m4_twopass.csv`.

## M5 - ranking

Literature (report only; not wired into CITATIONS.bib): Anderson & King 2000
(HST ePSF / core sampling); Stetson DAOPHOT II (quadratic ePSF smoothing).

| ID | Mechanism | Ratio droop | Bright chi2 | Rig-specific? | Rank droop | Rank chi2 |
|----|-----------|-------------|-------------|---------------|------------|-----------|
| H1 | osamp=2 insufficient at FWHM 3.3 px | osamp=1 cannot build (constraint). osamp=4 builder diverges, so "need more osamp" is untested as a photometry fix. Production ePSF FWHM 2.36 vs data 3.3 (ratio 0.716) is a too-narrow core, consistent with undersampled reconstruction. | same core mismatch | YES (9.77 arcsec/px undersampled) | 1 (lead, unresolved) | 1 |
| H2 | quadratic smoothing core bias | Quartic destroys the ePSF; quadratic is load-bearing, not the bug. Droop exists WITH quadratic. | quartic chi2 3430 vs quadratic 42 | kernel is generic; failure mode amplified when undersampled | 4 (rejected as cause) | 4 |
| H3 | wing truncation cutout 17 px | 1e-8 of 0.329 droop | wing |resid| ~0.01 | compact undersampled PSF (rig-leaning) | 3 (ruled out) | 3 |
| H4 | one-pass F_model init (FD-A) | 0.22 percent flux shift | chi2 68.4 -> 62.3, still >> faint | generic FD-A | 3 (not droop) | 2 (minor) |
| H5 (new) | IterativePSFPhotometry flux scale vs single-pass on the SAME ePSF | BO CVn 0.667 iterative vs 1.357 single-pass | chi2 stays ~22 | generic photutils, measured on this rig | 2 (fitter scale) | 2 |

**H5 row SUPERSEDED (SHAPE-01-F F2):** iterative == single-pass; production
droop is the chi2<5 AC gate. Keep this table as the M3-era ranking; do not
re-derive H5 from M3 numbers.

M1 core-shaped residual + production ePSF too narrow (FWHM 0.72x) is the
shape story. The M3-era H5 fitter-scale story is superseded (see banner).

## SHAPE-01-F recommendation

Do not ship osamp=4 or quartic. Do not enlarge cutout=17 for droop (H3 out).
Do not expect two-pass F_model to restore flux (H4 out for droop).

Fix-task scope, in order:

1. Builder: why EPSFBuilder at oversampling=4 drops 65/66 stars (null QC FWHM).
   If that path can be made to converge on the same 67-star pool, remeasure
   ePSF FWHM vs 3.3 px and PSF/DAO vs mag. That is the only honest test of H1
   as "osamp=2 is insufficient" rather than "osamp=1 is impossible".
2. Fitter H5: on the production ePSF, compare IterativePSFPhotometry vs
   PSFPhotometry flux (and vs DAO) for the brightest 30 science stars. If
   iterative flux is systematically low, the 0.671 droop may be a photutils
   iterative-scale issue, not an ePSF-array issue. Keep chi2 and flux separate.
3. Persist `x_fit`,`y_fit` through the F6 merge so residual stacks are not
   coordinate-ambiguous (supersedes the S5 sandbox caveat for good).
4. Only after (1)-(3): consider an Anderson & King style core-sampled ePSF
   (higher osamp once the builder works, or a constrained-core model) aimed at
   bringing `epsf_fwhm_native_px` from 2.36 up toward 3.3.

Out of scope for SHAPE-01-F: AAVSO/VarAstro, aperture LC rewrite, changing
the production 67-star gate composition.

## Files touched

Production (byte-identical, hash-verified): none written.

Sandbox / harness:
- `dev/scripts/epsf_shape_01_m.py` (new measurement harness)
- `dev/results/session_20260824_epsf_shape_01_m/**` (M1-M5 artifacts)
- `src_py/psf_photometry.py` -- optional `smoothing_kernel=` on
  `build_epsf_model` / `_epsf_build_imagepsf_from_stars`; default None keeps
  the old osamp rule (quadratic if osamp<=2 else quartic). Production RUN ePSF
  path does not pass the new kwarg.

Not pushed.
