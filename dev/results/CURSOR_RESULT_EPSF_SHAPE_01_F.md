CURSOR RESULT - 2026-08-24T13:15:00Z (EPSF-SHAPE-01-F)

What I did
Fix-phase measurements F0-F3 on the SHAPE-01-M diagnosis. No production ePSF
swap (no Milan GO). Candidate models stay in sandbox. Chi2 and flux reported
separately. Not pushed; Milan authorizes.

Premise (0.1): SHAPE-01-M compared production iterative PSF/DAO = 0.667 on
BO CVn with sandbox single-pass = 1.357 on the SAME ePSF array, and called
that H5 (fitter-scale). F2 repeats the comparison with identical init,
weights, grouping, and AC-off. Those two numbers are not comparable to each
other as a pure fitter split: production F6 merge applies
apply_aperture_correction=True (psf_ac_factor=0.528 on frame 001).

## Gates

| Gate | Status | Detail |
|------|--------|--------|
| G0 commit hygiene | PASS | LC-LOG-01 `dbb6967`; SHAPE-01-M `1f9f921`. DECISIONS: (a) Phase 2A no longer writes `lightcurve_*_psf.csv`; (b) INV-PSF-SUBMIT-01 unconditional. F-work started on that tree. |
| G1 tip | PASS | G0 tip `1f9f921516b0b7f38575e61628bff13532f2e77e` is a descendant of `b1af049`. |
| G2 `--fast` | PASS | End-of-task: 1520 passed, 32 skipped, OVERALL PASS. git-origin-main WARN is local G0 commits ahead of origin (do not pull/push). db-quick-check WARN via committed waiver. |
| G3 era | PASS | checker constants unchanged: core `9902d918` n=121 / ext `472bc9e4` n=179 |
| G4 production ePSF | PASS | SHA256 `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged through F1-F3 |

Hash guard (aperture LCs, AAVSO, VarAstro, production ePSF+meta):
`hashes_before.json` == `hashes_after.json`. Positive control changed.
Proc sidecars were not in the watch set; F3 re-merged a COPY under
`session_20260824_epsf_shape_01_f/f3_remerge/` only.

## F1 - osamp=4 EPSFBuilder divergence

Mechanism (measured, not assumed): `_fit_error_status == 3` cascade
("fitted position is outside the data cutout"). photutils 3.0.0
`epsf_builder.py`:

- status 1: `overlap_slices` PartialOverlap/NoOverlap (L1770-1772)
- status 2: fitter `ierr` not in 1..4 (L1811-1813) -- ZERO in the os4 curve
- status 3: fitted center outside cutout (L1821-1824)
- after iter>3, failed stars are permanently excluded (L1887-1906)
- all-fail raise: L1881-1883 `ValueError: The ePSF fitting failed for all stars.`
- recenter box is `box_size * oversampling` (L1471-1474)
- norm: sum(ePSF) = product of oversampling (L1360-1394)

M3/F1 maxiters=15 quadratic: iters 1-6 all 66 stars status 0; iter 7 first
status 3; then 1,3,14,41,52,57,59 status-3 by iter 15 (65/66 fail). Not
maxiters-too-low (it ran 15). Not fitter non-convergence (status 2 = 0).
Not photutils 3.0 deprecation of EPSFFitter (unused on this path).

Builder-parameter attempts (same 67-star gated pool; never widened):

| tag | last n_fail | FWHM native px | vs 3.301 | note |
|-----|-------------|----------------|----------|------|
| maxiters=15 | 65 | null | - | status-3 cascade |
| maxiters=6 | 0 | 1.401 | 0.425x | stops before cascade; core TOO NARROW |
| recentering_maxiters=1 | all-fail raise | - | - | L1883 at iter ~13 |
| recentering_boxsize=(3,3) | 9 | 0.82 | 0.25x | cascade delayed; even narrower |

Stopping early is not a higher-osamp fix: the pre-cascade ePSF is 1.4 px
FWHM vs data 3.3. Raising osamp is closed as a fix avenue on this rig with
stock photutils EPSFBuilder.

Rig tag: status-3 walk-off is **rig-amplified** (17 px cutout, FWHM 3.3 px,
9.77 arcsec/px undersampled). The exclude-after-iter-3 rule is **generic**
photutils.

Artifacts: `f1_summary.json`, `models/os4_quad_*/star_iter_status.csv`,
`recenter_deltas.csv`, `first_exceptions.json`.

## F2 - H5 fitter-scale split (production ePSF)

Brightest 30 science-set stars, 20-frame subset, AC off, identical init.

| quantity | median | RMS | RMS ~= |median| |
|----------|--------|-----|----------------|
| iter/DAO | 1.218 | 1.190 | yes (offset, not scatter) |
| single/DAO | 1.218 | 1.190 | yes |
| iter/single | 1.000 | 1.000 | yes |
| d chi2 (iter-single) | 0.0 | 0.0 | yes |

IterativePSFPhotometry with the production noop finder is **identical** to
plain PSFPhotometry on this set (600/600 flux match). The SHAPE-01-M 0.667
vs 1.357 split was not a fitter-class effect.

Flux finding (production path): F6 merge uses
`apply_aperture_correction=True`. Frame 001 `psf_ac_factor=0.528206`
(`_compute_aperture_correction`: median DAO/PSF among chi2<5 stars;
`psf_photometry.py` L460-491). Uncorrected F2 ratio 1.218 * 0.528 = 0.643,
vs BO CVn night droop 0.671. Direction and size match.

Chi2 finding: AC is a scalar on flux; it does not change chi2. Bright-30
chi2 median remains 68.4 (same as SHAPE-01-M M4 one-pass). Keep separate.

H5 pure-fitter term: **~0**. The production deficit is the **global AC**
trained on low-chi2 (mostly fainter) stars, then applied to bright stars
whose uncorrected PSF/DAO is smaller. Subtract-and-refit (iterative) does
not add a second flux scale when AC is held off.

Mag/group/frame: corr(iter/single, mag) undefined (zero variance);
frame-median std of iter/single = 0.

Rig tag: AC-from-chi2<5 is **generic**. The chi2-vs-mag slope that empties
bright stars from the AC ensemble is **rig-amplified** by the too-narrow
ePSF (H1 leftover).

Artifacts: `f2_summary.json`, `f2_three_way.csv`, `fits_f2_single.csv`,
`fits_f2_iter.csv`.

## F2b - ePSF array norm audit

Production 35x35, sum=4.0 (osamp^2). peak=0.095, min=-0.055
(negative lobes 0.57x the peak).

| metric | value |
|--------|-------|
| circular EE at 8.5 px | 0.698 |
| signed frac r>8.5 | +0.302 |
| signed frac in negative pixels | -0.302 |
| |model| frac in negative pixels | 0.188 |
| outer-annulus median (pedestal) | **0.0** |
| geom frac pixels r>8.5 | 0.264 |

The 0.698 EE is **not** a DC pedestal and **not** square-vs-circle geometry
alone. It is **ringing**: a negative ring at r=7.5-8.5 (bin 8.0-8.5 sum
-0.279) and positive lobes in the corners (r>8.5 sum +0.273). Outer-annulus
median is exactly 0, so pedestal-subtraction is a no-op.

F2b sandbox copy (subtract 0, renormalize) + both fitters: ratio and chi2
**identical** to F2 (1.218 / 68.4). Pedestal does not explain H5 or the
droop. The ringing is a BUILD/kernel property (quadratic 5x5 kernel in
photutils `_SmoothingKernel.QUADRATIC_KERNEL`, `epsf_builder.py` L52-57),
not a fitter property.

Predicted LS vs DAO from a DC pedestal: n/a (pedestal=0). Ringing makes
full-CCD weights see negative model pixels; that can bias flux vs a DAO
aperture that ignores those pixels -- consistent with uncorrected PSF/DAO
> 1, but the dominant production *0.528 is still the AC step.

Rig tag: quadratic-kernel ringing is **generic** (Stetson DAOPHOT II kernel
in photutils). Undersampling makes the negative ring a large fraction of
the signed sum (**rig-amplified**).

Artifacts: `f2b_norm_audit.json`, `f2b_radial_norm.csv`,
`f2b_refit_summary.json`, `models/prod_pedestal_sub/`.

## F3 - x_fit / y_fit / psf_group_n persistence

Production code: `psf_photometry_stars` now emits `x_fit`,`y_fit` (full-frame
pixels) and already had `psf_group_n`. `pipeline._fill_psf_catalog_columns`
maps all three. `epsf_psf_merge.is_psf_column` treats `x_fit`/`y_fit` as
PSF columns so INV-PSF-ADDITIVE-01 still holds by construction.
`psf_internal_lc.py` already reads `psf_group_n`.

Tests: `test_epsf_psf_merge.py` (additivity with x_fit present; x_fit is a
PSF column) + synthetic LC `n_group==2` in `test_psf_internal_lc.py`.
13 passed in 9.7 s.

Sandbox re-merge of draft 516 frame 001 (copy, not live write): BO CVn
`x_fit=957.597`, columns present, additive_ok True.
`f3_remerge.json`. Live 516 proc CSVs untouched.

## F4 - GATED, not unlocked

Precondition failed: F1 did not deliver a converging higher-osamp model
with FWHM moving toward 3.3 (best "stable" os4 FWHM is 1.40). F2b pedestal
correction is a no-op. Too-narrow core remains, but the honest next lever
is **AC policy / chi2-gated ensemble**, not a model swap.

No split-half certificate. No swap candidate. STOP for Milan.

## What changes on a well-sampled rig (Newton 0.65 arcsec/px)

If seeing ~3 arcsec, Newton FWHM ~4.6 px (critically to well sampled).

- F1: osamp=4 + 17 px cutout is photutils' own default; status-3 walk-off
  should be rarer when the core is several pixels. Still do not raise
  maxiters blindly -- the exclude-after-3 rule is generic.
- F2: AC-from-chi2<5 remains generic. If bright chi2 stays ~1, bright stars
  stay in the AC ensemble and the 0.67 droop should shrink.
- F2b: quadratic ringing remains generic; the negative ring is a smaller
  fraction of the signed sum when the core is well sampled.
- F3: fit-coordinate persistence is generic bookkeeping.
- F4: a higher-osamp rebuild is more likely to be legal on Newton; this
  wide-rig campaign must not be the template for swapping osamp=4 in.

**2026-08-24 measurement (EPSF-NEWTON-518-01):** the on-disk Newton draft
is 518 (TOI-1131, V 60 s, **bin2 1.30 arcsec/px**, not unbinned 0.65).
Part C gated pool 26 < `epsf_min_stars=30` (science_scope choke). ePSF
not built. P-A..P-E from this paragraph remain unmeasured. Evidence:
`CURSOR_RESULT_EPSF_NEWTON_518_01.md`.

## Files touched

Production code (no model/catalog/export byte change):
- `src_py/psf_photometry.py` -- `x_fit`/`y_fit` on fit rows; optional
  `builder_kwargs` for sandbox builds (default maxiters=15 unchanged)
- `src_py/pipeline.py` -- map `x_fit`,`y_fit`,`psf_group_n` in F6 fill
- `src_py/epsf_psf_merge.py` -- `x_fit`/`y_fit` classified as PSF columns
- `dev/tests/test_epsf_psf_merge.py`, `dev/tests/test_psf_internal_lc.py`

Sandbox:
- `dev/scripts/epsf_shape_01_f.py`
- `dev/results/session_20260824_epsf_shape_01_f/**`
- `dev/results/context/session_20260824_epsf_shape_01_f/` (CSV/JSON copy)

G0 (already committed, not this working tree):
- `dbb6967` EPSF-LC-LOG-01
- `1f9f921` EPSF-SHAPE-01-M

Not pushed.

`--fast` OVERALL PASS at end: 1520 passed, 32 skipped (two new F3 tests vs the LC-LOG 1518). Git origin WARN is expected: tip `1f9f921` is two local commits ahead of `origin/main` (`b1af049`). Do not pull; Milan authorizes push.
