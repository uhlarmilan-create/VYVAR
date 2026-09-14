CURSOR RESULT - 2026-09-14 EPSF-CORE-02

Date: 2026-09-14. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 8ef83bd (verified HEAD at start).
Class: MEASUREMENT + dev-only harness. No production code change.
Live 516/517 and Archive read-only. a2/ never staged.

## Premise (Rule 0.1)

Compared: H-PEAK (check-star peak ADU vs 60000 / 52428 on the
CALIBRATED pre-resample grid) against CORE-01 residuals
(resid_after_median, n=134). They are not the same quantity.
The hypothesis predicted ~59.8 kADU from aligned-grid FWHM
2.364 px and aligned psf_flux 3.785e5; that mixes grids
(architect error 22 family). Measured calibrated-grid check
p95 = 45260 ADU (aligned-grid 7x7 lower bound median 34144).

CORE-01 product-level machinery as tested is ~1 mmag dither RMS
against observed 10.48 / 21.41 mmag. Attribution remains
incomplete after this task (R-P0).

## Architect error 23 (self-reported)

CORE-01 Part A injected the live ePSF with the same ImagePSF
evaluation used by the fit, so interpolation error cancels by
construction; the noise-off 1.3e-7 mmag was guaranteed, not
measured. "Interpolation is not the driver" is NOT established.
Part C real-data phase correlations (target rho=0.470 p=1e-8;
check rho=-0.386 p=4.1e-6, n=134) remain the strongest live
signal on an axis Part A could not see. Root class: harness
validated against itself.

## Defects carried (record only; do not fix; next DOCS task)

GAIN-FALSY-01: PSF error-map gain/RN resolve to (1.0, 10.0)
via `value or 1.0` when aligned GAIN=0.0
(`psf_photometry.py:2268-2270`). Weights and chi2 scale on
config defaults, not the equipment authority. Aperture/LC
authority on this rig is g_pt = **0.637067** e-/ADU_container
(`gain_photon_transfer.json` source=g_pt) and RN = **15.2** e-
(`pipeline_meta.json` resolved_facts.read_noise source=db).

FIXPOS-NOOP-01: `_apply_psf_fixed_position` is a no-op on
`IterativePSFPhotometry` (EXC-0454). Latent: flag False on 516.

## Part A - peak census (CALIBRATED grid)

Position method (804/804 astroalign; 0 NCC fallback):
per-frame `astroalign.find_transform(aligned, calibrated)`
maps proc integer (x,y) onto the calibrated grid. That is the
inverse of the in-memory alignment (`VYALGM=astroalign`; no
persisted matrix). Peak = max in 7x7 (half=3), same footprint
as `sat_diag.box_peak_max` / `_box_peak_max_adu`. Uncertainty:
similarity-transform residual typically <1 px; once the seed
is inside the 7x7 the peak pixel is exact.

Do not use proc `peak_max_adu_raw` as authority: sat_diag
places aligned x,y on the calibrated array (`sat_diag.py:585-608`)
and missed the check on some frames (Light_001 sat_raw=2816
vs mapped 33825).

Aligned-grid 7x7 peaks are a stated **lower bound** (resample
smear). Every peak number below is calibrated-grid unless
labelled lower-bound.

Population: 6 M2 stars x 134 epochs. Runtime 70 s.

| role | catalog_id | median | p95 | max | n>60000 | n>52428 | aln median (lower bound) |
|---|---|---:|---:|---:|---:|---:|---:|
| target | 1498613634033133184 | 16913 | 21302 | 23377 | 0 | 0 | 14733 |
| check | 1497613731286514432 | 37469 | 45260 | 47975 | 0 | 0 | 34144 |
| ens1 | 1497771992240531712 | 18551 | 21914 | 24166 | 0 | 0 | 16156 |
| ens2 | 1499200223486564608 | 16472 | 19447 | 20946 | 0 | 0 | 13848 |
| ens3 | 1497974027502858240 | 5472 | 6322 | 7401 | 0 | 0 | 5211 |
| ens4 | 1497368849430107904 | 6643 | 7364 | 8033 | 0 | 0 | 6008 |

Check p95 45260 < 52428 and < 60000. 0/134 frames above either
threshold.

Correlations (n=134, Spearman + Theil-Sen; resid in mmag):

| id | population | rho | p | rank R^2 | Theil-Sen |
|---|---|---:|---:|---:|---:|
| A1 | check resid vs check peak ADU (cal) | 0.074 | 0.40 | 0.005 | +4.4e-4 mmag/ADU |
| A2 | target resid vs target peak ADU (cal) | 0.018 | 0.83 | 0.000 | +8.3e-5 |
| A3 | check psf_chi2 vs check peak ADU (cal) | 0.061 | 0.48 | 0.004 | +6.2e-4 |

## Part B - regressor race

Population: 134 identical-ensemble epochs. Rank R^2 = Spearman
rho^2. No multivariate fit.

| star | best | rank R^2 | peak | fwhm_psfex | r_phase | psf_chi2 | qc fwhm_px |
|---|---|---:|---:|---:|---:|---:|---:|
| target | r_phase | 0.221 | 0.000 | 0.002 | **0.221** | 0.005 | 0.023 |
| check | psf_chi2 | 0.214 | 0.005 | 0.004 | 0.149 | **0.214** | 0.008 |

fwhm_psfex-at-star does not win. Target's winner is |phase-0.5|
(the CORE-01 Part C live signal). Check's winner is psf_chi2.

## Part C - harness delta hunt; B2 STOP

Probe: 5 stars (target, check, ens1, ens2, ens3) x 5 frames
(001 / 037 / 076 / 109 / 148). Inst-mag RMS after median.
Runtime 650 s including the full 804-row rebuild.

| config | RMS vs frozen (mmag) | RMS vs baseline (mmag) |
|---|---:|---:|
| baseline_core01 (production-matching) | 1.326 | 0 |
| (i) no residual-annulus refine | 1.374 | 0.346 |
| (ii) clipped-sum init (no dao_flux) | 1.431 | 1.037 |
| (iii) gain=g_pt 0.637 / RN=15.2 | 2.167 | 1.589 |
| (iv) iterative off | 1.326 | 0 |
| (iv) maxiters=1 | 1.326 | 0 |

(iv) is bit-identical to baseline on this 25-row set
(`_epsf_noop_finder`; IterativePSFPhotometry does not add
sources). None of (i)-(iv) closes the 1.3 mmag gap.

Full 804-row B1 vs frozen = **1.396 mmag** (n=804, same as
CORE-01). Gate 1.0 not reached. STOP Part C. B2 not run.
Unexplained remainder sits on the production-matching
baseline itself (not on a named toggle). Gate not relaxed.

## Readings (binding)

R-P0: none of R-P1 / R-P2 / R-P3; full decomposition table,
no attribution claim.

R-P1 does not fire (A1 rank R^2=0.005; check p95 45260 below
both thresholds).
R-P2 does not fire (fwhm_psfex does not win the race).
R-P3 does not fire (B2 void).

No fix. Any fix that moves psf_flux belongs to the 520 era
re-cut (D-EPSF-XVAL-DOD-01). Sequencing is Milan's.

## Errors on the record

None in the scripted measurements. H-PEAK Gaussian ~59.8 kADU
used aligned-grid FWHM on aligned flux; that prediction is
not comparable to calibrated-grid peaks (grid named above).

## G4 / gates

Live 516 unchanged after the run: csv `bfa24039` / fits
`13e77cf8` / epsf `172f9540` (all PASS). `--fast --clean`
OVERALL PASS (1629 passed, 34 skipped; clean-tree PASS).
HEAD `a1dc85a`. `a2/` never staged.

## Files

- `dev/xval_psfex/epsf_core_02.py` (zero `src_py` imports of it)
- `dev/results/context/session_20260914_epsf_core_02/`
  peak_census.csv, regressor_race.csv, harness_toggles.csv,
  summary.json (no modelswap_lc.csv: B2 not run)
- this file

## STOP

Measurement only. No production change, no fix, no era re-cut,
no admission-policy change. Sequencing is Milan's on these
numbers.
