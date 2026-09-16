CURSOR RESULT - 2026-09-16 EPSF-VAL-03

Date: 2026-09-16. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 17cac43 (LEDGER-EPSF-DOD-04).
Class: MEASUREMENT + dev-only photometry harness. No production
change. Live 516/517 and Archive read-only. a2/ never staged.

## Criterion quoted from D-EPSF-XVAL-DOD-04 at HEAD

Source: `docs/VYVAR_DECISIONS.md` lines 27-35 at commit `17cac43`.

2. ACCURACY (curve-of-growth-tied large aperture). On isolated,
   unsaturated, constant bright stars, d(s) =
   median_epochs(m_psf_inst - m_L_inst), where m_L is a
   large-aperture (r_L ~ 4 x FWHM) instrumental magnitude tied by a
   measured growth curve. Fit d = a + b*(G - 10): |b| <= 5.0
   mmag/mag and robust scatter (1.4826 * MAD) <= 15 mmag. The colour
   slope c of d vs BP-RP is RECORDED (expected ~0; a nonzero value
   would indicate PSF colour dependence), not judged. No catalogue
   transformation and no blended star enters the reference.

Numeric: T2b=5.0 mmag/mag, T2s=15 mmag.

## What I did

Ran `dev/xval_psfex/epsf_val_03.py` on the 134 identical epochs.
Large-aperture fluxes via photutils exact CircularAperture /
CircularAnnulus on live aligned lights (never written back). PSF
from frozen proc `psf_flux`. Isolation via local Gaia DR3 sqlite
box + haversine. Peaks: CORE-02 method (astroalign inverse, 7x7
on calibrated grid).

## Isolated-star selection

FWHM authority (isolation): 2.364 px (SHAPE-01). Plate scale
9.774"/px. Primary R_iso = 8 x FWHM = 18.912 px = 184.85"; only 8
pre-peak survivors -> RELAXED to 6 x FWHM = 14.184 px = 138.63".
VSX-clean as VAL-01 (`vsx_known_variable`, 5.0"). Neighbour query:
`GAIA_DR3/vyvar_gaia_dr3.db` table `gaia_dr3`; no G_n<=G+5 within
R_iso; no G_n<=G+2 within 2*R_iso.

Pass: n=18 (after relax). ENS4-BLEND-01 suspects excluded; metrics:

| catalog_id | G | isol | n5 | n2 | peak_p95 | min_sep" |
|---|---:|---|---:|---:|---:|---:|
| 1497368849430107904 | 11.52 | N | 2 | 2 | 7364 | 1.04 |
| 1496804834326599424 | 10.62 | N | 2 | 4 | 12947 | 83.9 |

Full table: `isolated_candidates.csv`.

## Growth curve and r_L

Radius grid: r = 1.0..5.0 x FWHM (0.5 steps). Photometry FWHM =
per-frame `qc_metrics.csv` `fwhm_px` (live calibrated lights;
typical ~5.3 px). Annulus [6,8] x FWHM; sky = sigma-clipped median
(harness). Production sky estimator name for comparison:
`sky_median_mask`.

No tabulated radius met the 1 mmag increment on >=90% of epochs.
Used r_L = 4.0 x FWHM (fallback). Plateau residual = 7.463 mmag.
Residual slope 4.0->4.5 = 1.810 mmag (limitation). Products:
`cog_table.csv`, `large_aperture_lc.csv`.

## Criterion 2 (PSF vs m_L)

Population: n=18 isolated unsaturated constants. Fit method:
Theil-Sen + 2000-sample bootstrap std on b.

| quantity | value |
|---|---:|
| a (mmag) | -123.10 |
| b (mmag/mag) | -4.934 |
| b_boot_std | 27.460 |
| robust scatter 1.4826*MAD (mmag) | 58.581 |

## Colour RECORD and D5-1

| quantity | value |
|---|---:|
| c (mmag/mag vs BP-RP-1) | +133.924 |
| c_boot_std | 44.707 |
| per-epoch PSF-vs-m_L median std (mmag) | 31.657 |
| b_ap (prod aperture vs m_L; D5-1) | +74.162 |
| b_ap_boot_std | 27.919 |
| n (ap) | 18 |

## Pre-registered readings

- R-X1 FAIL: |b|=4.934 (lim 5.0) OK, robust scatter=58.581 (lim
  15.0) FAIL (n=18). Plateau residual 7.463 mmag cannot account
  for the scatter alone.
- R-X2 RECORD: c=+133.924; per-epoch median std=31.657 mmag;
  b_ap=+74.162 (D5-1).
- R-X3: FAIL on R-X1; sequencing is Milan's.

No fix in this task.

## Output / findings

- Session: `dev/results/context/session_20260916_epsf_val_03/`
- Harness: `dev/xval_psfex/epsf_val_03.py` (zero src_py imports)

## Errors (if any)

None new. Architect errors 27-29 remain as previously ledgered.

## Files changed

- `dev/xval_psfex/epsf_val_03.py`
- `dev/results/context/session_20260916_epsf_val_03/*`
- `dev/results/CURSOR_RESULT_EPSF_VAL_03.md`
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` /
  `docs/VYVAR_ROADMAP.md` (status sync)

## Gates

G4 live 516: csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`
(all PASS). `--fast --clean` OVERALL PASS on `d484ec1` (1629
passed, 34 skipped; clean-tree PASS). a2/ never staged.

## STOP

Measurement only. No production change, no fix, no era re-cut.
Sequencing is Milan's on these numbers.
