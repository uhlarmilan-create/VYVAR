CURSOR RESULT - 2026-09-15 EPSF-VAL-02

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 23e2680 (LEDGER-EPSF-DOD-03).
Class: MEASUREMENT. Zero new photometry. No production change.
Live 516/517 and Archive read-only. a2/ never staged.

## Thresholds quoted from D-EPSF-XVAL-DOD-03 at HEAD

Source: `docs/VYVAR_DECISIONS.md` lines 23-45 at commit `23e2680`.

1. PRECISION (per G bin). On constant stars selected as in VAL-01,
   r(s) = RMS_med(PSF diff LC) / RMS_med(aperture diff LC), V-prod
   admission:
   - Domain D = stars with G >= 9.5 (the PSF path's justified
     domain): median r <= 1.25 and max r <= 1.50.
   - Bright end G < 9.5: no ratio criterion. Instead an ADMISSION
     test: epochs with psf_fit_ok == False must be routed to the
     aperture method by the science-method picker
     (photometry_lightcurve.py:2393 region), verified in code and
     end-to-end on >= 2 bright stars in the 516 products. Any
     psf_fit_ok == False epoch that reaches a science LC as PSF is a
     FAIL.
2. ACCURACY (common-scale reference). For the PSF path, per-star
   offset d(s) = median_epochs(m_psf_inst - m_cat), where m_cat is
   the production Gaia-transformed catalogue magnitude for the
   system's declared band (cite the transform and band mapping,
   D10-1 open), over stars with n_ok >= 100 and 8.5 <= G <= 12.5.
   Simultaneous robust fit d = a + b*(G - 10) + c*(BP-RP - 1.0).
   Criterion on the flux-scale linearity only: |b| <= 5.0 mmag/mag
   and residual RMS <= 25 mmag. The colour term c is RECORDED (it is
   the system's colour response; feeds D10-1), not a criterion. The
   same fit on the aperture path (raw per-star apertures) is RECORDED
   against D5-1, not a criterion.

Numeric: T1a=1.25, T1b=1.50, T2b=5.0, T2r=25.0. Criterion 3 out of
scope.

## What I did

Ran `dev/xval_psfex/epsf_val_02.py`. Precision reused VAL-01
`precision_per_star.csv` (V-prod) + `constant_star_candidates.csv`
(51 constant stars; 134 epochs; pinned ensemble; RMS_med). Accuracy
vs GDR3 Table 5.9 Johnson V (embedded coeffs; no src_py import).
Admission: code census + science LC `method` column on 516.

## Criterion 1 - precision per G bin (from VAL-01)

| G_bin | n | median r | max r | med fit_ok frac |
|---|---:|---:|---:|---:|
| [8.0, 8.5) | 2 | 2.031 | 2.266 | 0.00 |
| [8.5, 9.0) | 2 | 1.843 | 1.952 | 0.01 |
| [9.0, 9.5) | 10 | 1.445 | 2.330 | 0.18 |
| [9.5, 10.0) | 11 | 1.134 | 1.328 | 0.90 |
| [10.0, 10.5) | 9 | 1.119 | 1.361 | 1.00 |
| [10.5, 11.0) | 9 | 0.949 | 1.061 | 1.00 |
| [11.0, 11.5) | 8 | 1.041 | 1.415 | 1.00 |

Domain D (G >= 9.5): n=37, median r=1.049, max r=1.415,
offenders >1.50: none.

Bright end G < 9.5 (record only): n=14, median r=1.497,
max r=2.330 (no ratio criterion).

## Criterion 1 - ADMISSION

### Code consumers

| consumer | location | honours psf_fit_ok |
|---|---|---|
| compute_lc_flux_method | photometry_lightcurve.py:2355-2398 | YES (requires fit_ok + ac + quality) |
| Phase2A primary science LC | phase2a_target.py:681-694,1442 | YES (hard aperture; adaptive off on 516) |
| _get_lc_psf_strict | photometry_exports.py:99-117 | YES |
| psf_internal_lc ZP mask | psf_internal_lc.py:124-134 | NO (FIT-OK-ADMISSION-01; diagnostic only) |
| AAVSO/VarAstro writers | export_reports.py INV-PSF-SUBMIT-01 | YES (refuse psf/adaptive) |

### End-to-end on 516

Science LC products persist per-epoch `method`
(`lightcurves/lightcurve_*.csv`). Census: 60 science LCs, 8040
epochs, n_method_psf=0 (all aperture). config
`psf_adaptive_enabled=false`.

Named stars:
- 1497613731286514432 (check, fit_ok 0/134): no science LC file
  (not in active_targets); covered by aperture-primary path.
- 1498735778606786816 (49/134 fit_ok): same - no science LC file.

Zero psf_fit_ok==False epochs reach a science LC as PSF.

## Criterion 2 - accuracy vs catalogue

m_cat: Johnson V from Gaia G + BP-RP via GDR3 CU5 Table 5.9 G-V
polynomial (coeffs as in `gaia_johnson.GDR3_TABLE59_COEFFS['V']`).
NoFilter -> AAVSO CV (`export_reports.py`); D10-1-CLOSE: CV uses
Johnson V comparison magnitudes. D10-2 guard: BP-RP [-0.5, 5.1],
G [8, 16]. Population BP-RP span on PSF set: -0.039 .. 2.249.

Fit method: alternating Theil-Sen (4 rounds) on
d_mmag = a + b*(G-10) + c*(BP-RP-1.0).

| path | n | a (mmag) | b | c | resid RMS |
|---|---:|---:|---:|---:|---:|
| PSF (criterion) | 104 | -22675.2 | +6.394 | +140.323 | 518.944 |
| aperture (D5-1 record) | 637 | -22431.8 | +122.512 | -21.256 | 535.823 |

Constant a is the instrumental ZP (ADU mag vs V); not a criterion.
Colour term c recorded for D10-1. Aperture |b| large as expected
for D5-1 (raw per-star apertures).

G-bin table (PSF; d after removing c term): see
`accuracy_vs_catalog_psf.csv` / summary.json
`g_bins_after_removing_c`.

## Pre-registered readings

- R-W1 PASS (domain D): median r=1.049, max r=1.415 (n=37).
- R-W2 PASS: science consumers honour psf_fit_ok; e2e
  n_method_psf=0; diagnostic fit_ok_for_zp out of science scope.
- R-W3 FAIL: |b|=6.394 (lim 5.0), resid RMS=518.944 (lim 25.0);
  c=+140.323 recorded.
- R-W4: FAIL on R-W3; sequencing is Milan's.

No fix in this task.

## Output / findings

- Session: `dev/results/context/session_20260915_epsf_val_02/`
- Harness: `dev/xval_psfex/epsf_val_02.py` (zero src_py imports)

## Errors (if any)

None new. Architect errors 27-28 remain as ledgered under DOD-03.

## Files changed

- `dev/xval_psfex/epsf_val_02.py`
- `dev/results/context/session_20260915_epsf_val_02/*`
- `dev/results/CURSOR_RESULT_EPSF_VAL_02.md`
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)

## Gates

G4 live 516: csv `bfa24039` / fits `13e77cf8` / epsf `172f9540`
(all PASS). `--fast --clean` PENDING then stamped. a2/ never staged.

## STOP

Measurement only. No production change, no fix, no era re-cut.
Sequencing is Milan's on these numbers.
