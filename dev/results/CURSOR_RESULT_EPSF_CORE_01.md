CURSOR RESULT - 2026-09-14 EPSF-CORE-01

Date: 2026-09-14. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 1680eeb (verified HEAD at start).
Class: MEASUREMENT + dev-only harness. No production code change.
Live 516/517 and Archive read-only. a2/ never staged.

## Premise (Rule 0.1)

Compared: live 516 fit machinery (`psf_photometry_stars`,
`psf_photometry.py:2723`) against (A) known injected ePSF flux
and (B) PSFEx-catalog differential LCs on the same 134-epoch
M2 set that produced R-A2-3. Frozen `proc_psf_flux.csv` is
the VYVAR side of A2. Snapshot aligned Light_001 sha12
`9fab4073c780` != live `2e085929a8a9`; B1 used live lights
because frozen proc is live-derived. Using snapshot would
void the reproduce gate.

## F1-F5 (live tree)

F1 CONFIRMED. 516 used FREE x_0/y_0. `psf_fix_position_enabled`
is absent from repo `config.json`, from `src_py/config.py`, and
from persisted `pipeline_meta.json`. Call site
`psf_photometry.py:2868-2869` getattr default False.
Live proc CSV has integer `x`,`y` (all six M2 stars, 134
frames) and no `x_fit`/`y_fit` columns. Fitted centroids are
computed (`psf_photometry.py:3245-3246`) but not persisted
in the 516 proc product. `pipeline_catalog.py:381-382` would
map them if present; 516 files do not carry them.

F2 CONFIRMED. Per-cutout sky estimated and subtracted before
the fit (`cut_sky_sub`, `:3092-3093`). Fit has no sky parameter.
One residual-annulus refine pass may re-subtract and refit
(`:3142-3185`).

F3 CONFIRMED. Error map is model-based
`_psf_fit_error_cutout_full_ccd` (`:3117-3127`) from
`flux_init` (DAO `dao_flux` if finite, else clipped-sum)
with the model at the INIT position (cutout-relative xc,yc
from the integer peak).

F4 CONFIRMED. `ImagePSF(osamp=2)`. `fit_shape` for 516 is
(9,9): meta `fit_shape [9,9]` and
`_fit_shape_for_cutout(17, fwhm_px=3.3014)` -> ceil(2*3.3014+1)
even -> 9 (`psf_photometry.py:337-351`, meta `:7-9`).
`psf_grouper_enabled=false`, `psf_chi2_threshold=50`,
`psf_quality_fallback_enabled=true` in pipeline_meta.

F5 CONFIRMED on frozen proc_psf_flux.csv AND live 516 proc
(identical counts; population: 6 M2 stars x 134 frames).

| role | catalog_id | n | n_ok | chi2 median | flux median |
|---|---|---:|---:|---:|---:|
| target | 1498613634033133184 | 134 | 133 | 22.64 | 1.559e5 |
| check | 1497613731286514432 | 134 | 0 | 196.26 | 3.785e5 |
| ens1 | 1497771992240531712 | 134 | 118 | 34.70 | 1.592e5 |
| ens2 | 1499200223486564608 | 134 | 27 | 58.58 | 1.478e5 |
| ens3 | 1497974027502858240 | 134 | 134 | 3.02 | 3.769e4 |
| ens4 | 1497368849430107904 | 134 | 134 | 5.69 | 5.457e4 |

chi2 grows with brightness except the check (brightest, 0/134 ok).

Downstream before the PSF LC is built:

- `psf_internal_lc.py:124-134` `psf_fit_ok_for_zp_mask` =
  stored `psf_fit_ok` OR (finite flux>0 AND finite chi2).
  `:490-509` applies that mask to target and ensemble.
  516 sidecar: `psf_zp_membership_effective=fit_ok_for_zp`,
  `psf_zp_membership_rig_validated=true`.
- A2-COMPARE (`a2_compare.py:508`) uses `psf_flux` with no
  `psf_fit_ok` filter.
- `photometry_lightcurve.py:2393` does filter `psf_fit_ok`;
  that is the aperture-vs-psf science-method picker, not the
  internal PSF LC.

Check-star 0/134 `psf_fit_ok` with finite chi2 ~196: all 134
fluxes enter the PSF LC under `fit_ok_for_zp`.

## Part A (injection-recovery)

Production `psf_photometry_stars` on synthetic frames of the
live ePSF (`172f9540`). Init at even integer peaks; true
position = integer + (dx,dy). 25 phase cells
{0, 0.125, 0.25, 0.375, 0.5}^2. Fluxes = frozen median
psf_flux (check / target / ens3). Sky = qc bg_median min/max
on the 134 M2 frames (1330.09 / 2413.17 ADU). n=50
realizations noise-on; n=1 noise-off. Fitter-resolved
gain,RN = (1.0, 10.0): aligned header GAIN=0.0, production
`value or 1.0` (`psf_photometry.py:2268`). Not g_pt 0.637067
(that is the SExtractor/A2 path). Injection noise used the
same (1.0, 10.0).

Noise-off (pure interpolation): max |median bias| =
1.3e-7 mmag (all 25 x 3 x 2 cells).

Noise-on free position (150 cells = 25 phases x 3 fluxes x
2 skies; n=50 each):

- worst cell |median bias| = **7.061 mmag** (ens3, sky=2413,
  dx=0.375, dy=0.0, n=50, scatter 11.25 mmag).
- check dither RMS 0.57 / 0.76 mmag (lo/hi sky).
- target dither RMS 0.69 / 0.92 mmag.
- ens3 dither RMS 2.50 / 2.87 mmag.

Free-vs-fixed: registered one-flux (target) sweep was run
(50 noise-off + 50 noise-on cells) but is not a contrast.
Production `_apply_psf_fixed_position` (`:2431-2440`) sets
`phot.psf_model.x_0.fixed`; `IterativePSFPhotometry` has no
`psf_model` attribute (EXC-0454). 516 uses iterative=True.
The flag is a no-op. Noise-off free vs claimed-fixed bias
diff = 0. Noise-on differences are independent realizations.

## Part B (model swap)

B1 sanity: rebuilt vs frozen instrumental mag RMS =
**1.396 mmag** (6 stars x 134 live frames, n=804, median
removed). Gate was 1.0 mmag. STOP B1: harness does not
replicate production to the pre-registered tolerance.
B2 not run. Part B model-swap readings VOID.

B1 vs PSFEx-catalog (still computed from B1 fluxes; not a
swap reading): target 10.46 mmag / check 21.27 mmag on 134
identical-ensemble epochs, AIJ flux-sum. Same scale as A2
R-A2-3 (10.48 / 21.41).

## Part C (correlations; frozen + a2_compare; n=134 each)

Population: 134 identical-ensemble epochs; resid_after_median
from `m2_epochs_{target,check}.csv`; phase from deg2
`XPSF_IMAGE`/`YPSF_IMAGE`; chi2 from frozen proc.

Frames aligned: target frac(X) 0.587-0.682, frac(Y) 0.402-0.504;
check frac(X) 0.539-0.669, frac(Y) 0.829-0.935. 2D 4x4 map is
degenerate (epochs sit in 1-2 bins). Spearman uses continuous
|phase-0.5|.

| star | Spearman r_phase vs resid | p | Spearman chi2 vs resid | p |
|---|---:|---:|---:|---:|
| target | 0.470 | 1.0e-8 | 0.072 | 0.41 |
| check | -0.386 | 4.1e-6 | 0.463 | 1.8e-8 |

C is not a numbered reading. B2-void does not erase these
numbers; they are not R-C2.

## Readings (binding)

R-C0: F5 confirmed AND no PSF-LC consumer filters on
psf_fit_ok (fit_ok_for_zp admits finite flux+chi2);
flag-consumed-silently.

R-C1: Part A worst-cell |bias|=7.061 mmag (>=3) on 150
noise-on free-position cells -> MACHINERY defect. Axis:
weights/sky at faint flux (ens3 G~11.2, high sky); pure
interpolation (noise=0) is not the driver. Free-vs-fixed
not measured (hook no-op).

R-C2 does not fire (Part A not clean; B2 void).
R-C3 does not fire (R-C1 fired).

No fix. Any fix that moves psf_flux belongs to the 520 era
re-cut (D-EPSF-XVAL-DOD-01).

## Errors on the record

None in the scripted measurements. B1 1.396 mmag vs 1.0 mmag
gate is a harness-replicate STOP, not an architect error.
Production `_apply_psf_fixed_position` / IterativePSFPhotometry
mismatch is a live-tree fact.

## G4 / gates

Live 516 unchanged after the run: csv `bfa24039` / fits
`13e77cf8` / epsf `172f9540` (all PASS). `--fast --clean`
OVERALL PASS (1629 passed, 34 skipped; clean-tree PASS).
`a2/` never staged.

## Files

- `dev/xval_psfex/epsf_core_01.py` (zero `src_py` imports of it)
- `dev/results/context/session_20260914_epsf_core_01/`
  injection_bias.csv, modelswap_lc.csv, phase_corr.csv,
  fitok_census.csv, summary.json
- this file

## STOP

Measurement only. No production change, no fix, no era re-cut.
Sequencing is Milan's on these numbers.
