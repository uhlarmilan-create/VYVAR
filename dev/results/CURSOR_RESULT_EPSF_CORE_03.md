CURSOR RESULT - 2026-09-15 EPSF-CORE-03

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: a39134e (verified HEAD at start).
Class: MEASUREMENT + dev-only harness. No production code change.
Live 516/517 and Archive read-only. a2/ never staged.

## D-EPSF-SWAP-DIFF-01 (Milan, 2026-09-14)

The model swap is a DIFFERENTIAL measurement inside the
harness: B1(harness, live ePSF) vs B2(harness, PSFEx model) vs
PSFEx catalog, same call path. The harness-vs-production floor
is the resolution limit. CORE-01 gate 1.0 mmag is NOT relaxed.
Part B did not pass it; Part C ran under this decision with
achieved B1 = **1.396 mmag** (n=804) as the floor.

## Architect error 24 (self-reported)

CORE-02 H-PEAK ~59.8 kADU was a Gaussian idealization presented
with too much confidence and grid-inconsistent; measured
aligned-grid peak ~34k (calibrated p95 45260). Root class:
prediction not calibrated against a measured quantity before
use.

Carried, record only: error 23 (harness validated against its
own ImagePSF), GAIN-FALSY-01, FIXPOS-NOOP-01.

## Call-site (production `psf_photometry_stars` on 516)

Entry: `pipeline_catalog.py:355-364` via
`_fill_psf_catalog_columns`; same fill at
`frame_export.py:792`. Signature defaults:
`psf_photometry.py:2723-2743`. Targeted IDs:
`pipeline_catalog.py:3637-3652`.

| argument | 516 value | cite |
|---|---|---|
| cutout_size | OMITTED -> None -> meta 17 | `psf_photometry.py:2774-2777`; `masterstar_epsf_meta.json` |
| error | full-frame Poisson float32 `sqrt(\|data\|/gain+(rn/gain)^2)` from `st.gain`/`st.read_noise` after `value or 1.0` (resolves 1.0, 10.0). Then `_psf_fit_error_cutout_full_ccd` rebuilds a MODEL-BASED map; the passed map does not replace it | `pipeline_catalog.py:299-306`; `psf_photometry.py:3117-3127` |
| use_iterative | OMITTED -> True | `psf_photometry.py:2731` |
| max_fit_iters | OMITTED -> 3 | `psf_photometry.py:2732` |
| ref_fluxes | `dao_flux` from `_fit_df`, float, same length as positions | `pipeline_catalog.py:347-353` |
| apply_aperture_correction | False | `pipeline_catalog.py:362` |
| psf_ac_policy | `"p4_none"` | `pipeline_catalog.py:363` |
| grouper_enabled | OMITTED -> AppConfig; 516 `pipeline_meta` false. `neighbor_catalog` None so inactive | `psf_photometry.py:2894-2918` |
| neighbor_catalog | None | omitted |
| nn_dist_fwhm_map | OMITTED -> `{}` | `psf_photometry.py:2971` |
| nn_delta_mag_map | OMITTED -> None | signature default |
| quality_fallback_enabled | OMITTED -> AppConfig default True; 516 meta true | `psf_photometry.py:2966-2969` |
| star_positions | columns `catalog_id, x, y, name`; Light_001 `catalog_id` int64, `x`/`y` float64; 225 rows with `psf_flux>0`. M2 `x,y` are integer-valued floats | `pipeline_catalog.py:341-345` |
| frame_data | `np.asarray(data, dtype=np.float32)` of the in-memory aligned array | `pipeline_catalog.py:305` |

CORE-02 baseline extra: `cutout_size=17` explicit, only the 6 M2 stars.

## Part A - phase bias, model != truth (repairs error 23)

Truth is NEVER rendered with ImagePSF.

- T1: PSFEx deg2 of `BO_CVn_Light_076` at the target XPSF/YPSF.
  `reconstruct` at PSF_SAMP=0.6619128; zoom to 1/8 native px;
  `nd_shift` in that domain; 8x8 block-sum; scale stamp sum =
  flux (1.559e5).
- T2: analytic Moffat beta=2.5, FWHM=2.364 px, same fine-grid
  block-sum.
- Fit: live 516 ePSF `172f9540` via `psf_photometry_stars`,
  CORE-02 baseline kwargs (Part B gate not passed).
- Init: integer peak `x_int` (proc convention). True =
  `x_int + (dx, dy)`.
- Sky: qc `bg_median` median 1549.725 ADU. Gain/RN (1.0, 10.0).
- Grid: 9x9, dx,dy in `{0, 0.125, ..., 1.0}`. Noise-off n=1;
  noise-on n=30. Live target phase spread 0.102 px
  (max of fracX/fracY span; CORE-01 `phase_corr.csv`).

Slope = bias change on the 0.1 px window covering live fracX
(dx=0.5 -> 0.625 at dy=0.5), stated as mmag per 0.1 px.
Predicted RMS = |slope| x (spread / 0.1). Observed target RMS
= 10.48 mmag.

| truth | noise-off ptp (mmag) | slope (mmag / 0.1 px) | predicted RMS (mmag) | vs 10.48 |
|---|---:|---:|---:|---|
| T1 PSFEx | 72.452 | **8.384** | **8.561** | 82% of observed |
| T2 Moffat | 48.987 | -0.166 | 0.170 | ~0 |

T1 window cells (noise-off): dx=0.5 -> -203.86 mmag; dx=0.625
-> -193.38 mmag. Mean bias is a flux-scale from model
mismatch; R-Q4 uses the phase slope, not the mean.

Full-grid Theil-Sen at the same dy (context only): T1 0.950 /
T2 0.191 mmag per 0.1 px.

## Part B - call-site replication; 1.0 gate FAIL

Probe: 6 M2 stars x 5 frames (001 / 037 / 076 / 109 / 148).
Inst-mag RMS after median vs frozen.

| config | star set | n | RMS vs frozen (mmag) |
|---|---|---:|---:|
| core02_baseline | m2 | 30 | 1.482 |
| cutout_size_None | m2 | 30 | 1.482 |
| error_omitted | m2 | 30 | 1.482 |
| quality_fallback_False | m2 | 30 | 1.482 |
| grouper_False_explicit | m2 | 30 | 1.482 |
| fit_all_psf_rows | all (225-273) | 30 | 1.482 |
| production_all_together | m2 | 30 | 1.482 |

Every listed kwarg is bit-identical to baseline on this probe
(1.482 mmag). Call-site arguments are not the 1.396 mmag gap.

Full 804-row B1 vs frozen = **1.396 mmag** (n=804). Gate 1.0
NOT passed. Largest remaining contributor: the
harness-vs-production floor itself (FITS reload vs in-memory
aligned array; no named kwarg moves it). Part C proceeds under
D-EPSF-SWAP-DIFF-01 with floor = 1.396 mmag.

## Part C - model swap

Same harness call, same live aligned cutouts, same sky / error
map / integer init as B1. 134 identical-ensemble epochs, pinned
4-star AIJ flux-sum, median-removed RMS. Floor **1.396 mmag**
on every row.

B2 wrap: PSFEx `reconstruct` at that star's XPSF/YPSF;
bilinear resample to 35x35 `dest_scale=0.5` (osamp=2);
normalize `sum = osamp^2` (`psf_photometry.py:649-661`);
temp FITS + meta so the fitter is ImagePSF.
B3: same wrap of frame 076's PSFEx for every epoch.

| comparison | target (mmag) | check (mmag) | floor |
|---|---:|---:|---:|
| RMS(B1 vs PSFEx cat) | 10.461 | 21.271 | 1.396 |
| RMS(B2 vs PSFEx cat) | 12.982 | 23.695 | 1.396 |
| RMS(B3 vs PSFEx cat) | 10.020 | 15.139 | 1.396 |
| RMS(B1 vs B2) | 10.537 | 23.761 | 1.396 |
| RMS(B1 vs B3) | 6.362 | 14.936 | 1.396 |
| RMS(B2 vs B3) | 9.723 | 16.888 | 1.396 |

R-Q1 threshold = max(3.0, 2x floor) = 3.0 mmag. B2 vs cat is
not under it. B3 does not achieve a catalog match either; it
is closer than B2 but still >= 10 on both stars (target 10.02).

B2 residual race (B2 minus PSFEx-cat, median removed, n=134).
CORE-02 B1 winners were target r_phase R^2=0.221 and check
psf_chi2 R^2=0.214.

| star | best | R^2 | r_phase | psf_chi2 | fwhm_psfex | qc fwhm |
|---|---|---:|---:|---:|---:|---:|
| target | r_phase | 0.075 | **0.075** | 0.011 | 0.000 | 0.004 |
| check | r_phase | 0.071 | **0.071** | 0.002 | 0.013 | 0.005 |

Phase R^2 dropped (0.221 -> 0.075) but did not vanish. Check
chi2 winner is gone.

## Readings (binding)

R-Q3: MACHINERY/SKY; B2 and B3 stay >= 10 mmag on the check
(23.70 / 15.14) and B2 stays >= 10 on the target (12.98).

R-Q4: PHASE; T1 slope x observed phase spread = 8.56 mmag
(>= 5) of the target RMS 10.48. Independent of R-Q3; co-fires.
T2 (Moffat control) predicts 0.17 mmag in the same window.

R-Q1 does not fire (B2 vs cat 12.98 / 23.70, threshold 3.0).
R-Q2 does not fire (B3 does not achieve a catalog match).

No fix. Any fix that moves `psf_flux` belongs to the 520 era
re-cut (D-EPSF-XVAL-DOD-01). Sequencing is Milan's.

## Errors on the record

None in the scripted B/C measurements. Part A was re-run after
two harness defects in the first draft: (1) stamp pasted at
`round(true)` (1 px shift for dx>0.5); (2) init at
`round(true)` instead of the integer proc peak. Delivered T1/T2
CSVs are the integer-peak-init run.

## G4 / gates

Live 516 unchanged after the run: csv `bfa24039` / fits
`13e77cf8` / epsf `172f9540` (all PASS). `--fast --clean`
OVERALL PASS (1629 passed, 34 skipped; clean-tree PASS).
`a2/` never staged.

## Files

- `dev/xval_psfex/epsf_core_03.py` (zero `src_py` imports of it)
- `dev/results/context/session_20260914_epsf_core_03/`
  phase_bias_T1.csv, phase_bias_T2.csv, callsite_toggles.csv,
  modelswap_lc.csv, modelswap_race.csv, summary.json
- this file

## STOP

Measurement only. No production change, no fix, no era re-cut.
Sequencing is Milan's on these numbers.
