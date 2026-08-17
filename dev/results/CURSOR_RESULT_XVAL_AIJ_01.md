# CURSOR RESULT - XVAL-AIJ-01

Date: 2026-08-17
Draft 515 photometry SHA: da9cce4
Push: NOT authorized

Premise: epoch-by-epoch relative flux of BO CVn, AstroImageJ 6.0.10 vs VYVAR,
identical five-star comparison ensemble (RA/DEC match). Difference is AIJ minus
VYVAR in millimag after both series are normalized by their own median.
Numbers below are the architect's measured values; a sanity read of the
committed CSV confirms them (task MAD 3.24 mmag is 1.4826 x unscaled MAD).

## Verdict

EXTERNAL-XVAL / independent-tool cross-check **CLOSED**. VYVAR and AstroImageJ
agree to 3.3 mmag RMS per epoch over 134 frames of a 0.47-mag-amplitude
eclipser using identical comparison ensembles.

## Sanity read (CSV, not a re-derivation)

File: `dev/results/XVAL_AIJ_01_bo_compare.csv`
Columns: Label, rel_flux_T1, rel_flux_err_T1, FWHM_Mean, AIRMASS, vy_rel, diff_mmag
N = 134/134 finite epochs.

| quantity | domain | value |
|----------|--------|------:|
| median (AIJ - VYVAR) | mmag, per epoch | +0.151 |
| RMS (AIJ - VYVAR) | mmag, per epoch | 3.268 |
| MAD unscaled | mmag, per epoch | 2.188 |
| MAD x 1.4826 | mmag, per epoch | 3.243 |
| max \|diff\| | mmag, per epoch | 9.996 |
| AIJ amplitude ptp | mmag, -2.5 log10(rel_flux) | 474.04 |
| VYVAR amplitude ptp | mmag, -2.5 log10(vy_rel) | 474.82 |
| corr(diff, FWHM) | Pearson r | +0.080 |
| corr(diff, airmass) | Pearson r | -0.118 |
| AIJ CCD-eq median err | mmag, 2.5/ln(10) * err/flux | 6.181 |

Architect/task figures (0.15, 3.28, 3.24, 10.0, 474 vs 475, r=+0.08/-0.12,
AIJ err ~6.2 vs VYVAR exported 9.1) match this sanity read. RMS reported as
3.3 mmag in the JAAVSO sentence (3.268 rounded).

VYVAR exported median err on this LC is 9.14 mmag (WIDE-ERR-04 identity
sidecar, SHA da9cce4) vs AIJ CCD-equation 6.18 mmag - consistent with
WIDE-ERR-04 conservative closure (CT+SEM+scint vs CCD equation).

## Ensemble (architect, RA/DEC match)

T1 = BO CVn `1498613634033133184`
AIJ C2..C6 = VYVAR comps `1500748301498613248`, `1497771992240531712`,
`1499200223486564608`, `1497974027502858240`, `1497368849430107904`.

AIJ used a fixed 7 px aperture; VYVAR used per-magnitude 3.5-8.5 px. No
FWHM or airmass correlation of the difference.

## Spec defect

Milan's AIJ `Table.tbl` was not in the supplied folder (CSV + task only).
The epoch table is committed; the raw AIJ table is not. If it arrives, place
it next to the CSV. Not blocking: the compare CSV is the architect's
epoch-matched product.

## Cross-validation chain (see DECISIONS)

photutils/sep ~3 mmag RMS (library) -> architect independent reconstruction
0.0001 mmag (product formula) -> AIJ 3.3 mmag RMS (full chain, external tool).

## Files

- `dev/results/XVAL_AIJ_01_bo_compare.csv`
- `dev/results/CURSOR_TASK_XVAL_AIJ_SAT_LIMIT.md` (task copy)
- this file

## Errors

None blocking.

## Verify

Recorded with SAT-LIMIT-01. `session_baseline_check.py --fast`: OVERALL PASS.
