CURSOR RESULT - 2026-09-15 EPSF-VAL-01

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 6693537 (LEDGER-EPSF-DOD-02).
Class: MEASUREMENT. Zero new photometry. No production change.
Live 516/517 and Archive read-only. a2/ never staged.

## Thresholds quoted from D-EPSF-XVAL-DOD-02 at HEAD

Source: `docs/VYVAR_DECISIONS.md` lines 27-34 at commit `6693537`.

1. PRECISION: on a set of >= 4 bright constant stars (check
   class; VSX-clean; not ensemble members),
   RMS_med(PSF diff LC) / RMS_med(aperture diff LC) has median
   <= 1.25 and no star > 1.50.
2. ACCURACY: per-star median(PSF_inst - aperture_inst) over the
   epochs, across stars with n_ok >= 100 and 8.5 <= G <= 12.5,
   fits d = a + b*(G - 10) with |b| <= 5.0 mmag/mag and residual
   RMS <= 10 mmag; and vs BP-RP with |c| <= 10 mmag/mag.

Numeric constants used: T1a=1.25, T1b=1.50, T2b=5.0,
T2r=10.0, T2c=10.0 mmag. Criterion 3 (CODE / fix list) is out
of scope for this task.

## What I did

Ran `dev/xval_psfex/epsf_val_01.py` against live 516
`proc_*.csv` + `masterstars_full_match.csv` on the 134-epoch
identical set (A2-COMPARE). Diff LCs rebuilt with the pinned
4-star AIJ flux-sum ensemble. Wrote session products under
`dev/results/context/session_20260915_epsf_val_01/`.

## Constant-star set (criterion 1 selection)

VSX product: `masterstars_full_match.csv` /
`vsx_known_variable` from
`catalog_match.detect_stars_and_match_catalog`
(`vsx_match_max_sep_arcsec=5.0`; `catalog_match.py:89`).

Selection: G <= 11.5; present on >= 130 of 134 epochs on BOTH
paths (dao and V-prod PSF); not ensemble; not target; no VSX
match; n_sat == 0. G cut NOT relaxed (51 survivors).

Historical check `1497613731286514432`: PASSES selection;
n_fit_ok=0 (fits under V-prod admission via flux/chi2 only).

| catalog_id | G | BP-RP | n_ap | n_prod | n_fit_ok | pass | hist |
|---|---:|---:|---:|---:|---:|---|---|
| 1500296402219939584 | 8.243 | 1.533 | 134 | 134 | 0 | Y | |
| 1497613731286514432 | 8.450 | 1.340 | 134 | 134 | 0 | Y | Y |
| 1499906247391001088 | 8.743 | 0.792 | 134 | 134 | 0 | Y | |
| 1497442379271632384 | 8.851 | 0.680 | 134 | 134 | 4 | Y | |
| 1498735778606786816 | 9.113 | 1.195 | 134 | 134 | 49 | Y | |
| 1497119157212720896 | 9.131 | 0.046 | 134 | 134 | 24 | Y | |
| 1497528072458898432 | 9.218 | 1.088 | 134 | 134 | 24 | Y | |
| 1500727513856914944 | 9.237 | 1.240 | 134 | 134 | 0 | Y | |
| 1497674651102612992 | 9.287 | 0.679 | 134 | 134 | 13 | Y | |
| 1497203132413443328 | 9.307 | 0.973 | 134 | 134 | 35 | Y | |
| 1497370563121917952 | 9.347 | 0.880 | 134 | 134 | 110 | Y | |
| 1498326455340079616 | 9.407 | 1.097 | 134 | 134 | 4 | Y | |

Full 51-row table:
`session_20260915_epsf_val_01/constant_star_candidates.csv`.

## Criterion 1 - precision

Population: 51 selected constant stars; pinned ensemble
1497771992240531712 / 1499200223486564608 /
1497974027502858240 / 1497368849430107904; 134 identical
epochs; RMS_med as CORE-04 (mmag).

V-prod (fit_ok_for_zp: finite flux > 0 AND finite chi2):

| metric | value |
|---|---:|
| median r | 1.119 |
| max r | 2.330 |
| n_stars | 51 |

Offenders r > 1.50 (V-prod):

| catalog_id | RMS_med_psf | RMS_med_ap | r |
|---|---:|---:|---:|
| 1500727513856914944 | 23.859 | 10.241 | 2.330 |
| 1497613731286514432 | 18.657 | 8.234 | 2.266 |
| 1497442379271632384 | 12.695 | 6.505 | 1.952 |
| 1500296402219939584 | 12.889 | 7.180 | 1.795 |
| 1498326455340079616 | 13.594 | 7.634 | 1.781 |
| 1499906247391001088 | 12.275 | 7.080 | 1.734 |
| 1497528072458898432 | 12.304 | 8.003 | 1.537 |

V-strict (psf_fit_ok == True only; informational):

| metric | value |
|---|---:|
| median r | 0.941 |
| max r | 2.398 |
| n_stars with finite ratio | 37 |
| n_below_100 | 18 |

Strict admission lowers the median below T1a but does not
bring max r under T1b; does not change the FAIL verdict.
Per-star rows (both variants):
`precision_per_star.csv`.

## Criterion 2 - accuracy

Population: n_ok >= 100 on both paths AND 8.5 <= G <= 12.5
(n=104); includes target, check, and ensemble members.

Conventions (for interpreting constant a; not a criterion):
- aperture: `dao_flux`; `aperture_factor_applied=snr_table`;
  median `aperture_r_px` ~ 2.0 px across all proc rows;
  annulus APERTURE-01d 2.7/5.2 FWHM.
- PSF: live ePSF ImagePSF `psf_flux`; production normalize
  sum=osamp^2 so native sum=1 (`psf_photometry.py:649-661`);
  `epsf_sum_native=1.0`.

Theil-Sen fits on d_median_mmag:

| axis | n | slope | intercept | resid RMS |
|---|---:|---:|---:|---:|
| G-10 | 104 | b=-111.394 mmag/mag | a=-276.526 | 72.989 |
| BP-RP-1.0 | 104 | c=+176.946 mmag/mag | a'=-340.950 | 139.317 |

G bins (0.5 mag; d_median_mmag):

| G_bin | n | d_median_mmag |
|---|---:|---:|
| [8.5, 9.0) | 3 | -236.5 |
| [9.0, 9.5) | 13 | -217.2 |
| [9.5, 10.0) | 15 | -255.5 |
| [10.0, 10.5) | 12 | -279.8 |
| [10.5, 11.0) | 11 | -324.8 |
| [11.0, 11.5) | 13 | -439.6 |
| [11.5, 12.0) | 8 | -495.0 |
| [12.0, 12.5) | 29 | -548.3 |

Products: `accuracy_per_star.csv`, `accuracy_fits.csv`,
`accuracy_g_bins.csv`.

## Pre-registered readings

- R-V1 FAIL (V-prod): median r=1.119 max r=2.330 (limits
  1.25/1.5); offenders>1.5:
  1500296402219939584, 1497613731286514432,
  1499906247391001088, 1497442379271632384,
  1497528072458898432, 1500727513856914944,
  1498326455340079616.
  V-strict informational: median r=0.941 max r=2.398;
  does not change the verdict.
- R-V2 FAIL: |b|=111.394 (lim 5.0), resid RMS=72.989
  (lim 10.0), |c|=176.946 (lim 10.0).
- R-V3: FAIL on criterion 1 (precision), criterion 2
  (accuracy); sequencing is Milan's.

No fix in this task.

## Output / findings

- Session:
  `dev/results/context/session_20260915_epsf_val_01/`
  (`constant_star_candidates.csv`, `precision_per_star.csv`,
  `accuracy_per_star.csv`, `accuracy_fits.csv`,
  `accuracy_g_bins.csv`, `summary.json`).
- Harness: `dev/xval_psfex/epsf_val_01.py` (dev-only; no
  src_py imports in this module).

## Errors (if any)

None on the record for this run. Architect errors 25-26 remain
as previously ledgered (not re-fired here).

## Files changed

- `dev/xval_psfex/epsf_val_01.py`
- `dev/results/context/session_20260915_epsf_val_01/*`
- `dev/results/CURSOR_RESULT_EPSF_VAL_01.md`
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` /
  `docs/VYVAR_ROADMAP.md` (status sync)

## Gates

G4 live 516 read-only after run: csv `bfa24039` / fits
`13e77cf8` / epsf `172f9540` (all PASS). `--fast --clean`
OVERALL PASS on `5c484e2` (1629 passed, 34 skipped;
clean-tree PASS). a2/ never staged.

## STOP

Measurement only. No production change, no fix, no era
re-cut. Sequencing is Milan's on these numbers.
