# CURSOR RESULT - XVAL-AIJ-02

Date: 2026-08-17 (evening)
Draft 515 photometry SHA: de6f7c8
Baseline: origin/main dde6ce0 (PUSH-AUTH #2 complete). No science-code
change. Push: authorized this close.

Premise (0.1): epoch-by-epoch BO CVn 2026-04-23 photometry, AstroImageJ
6.0.10 vs VYVAR de6f7c8, production clean 4-comp ensemble (saturated C2
absent). Compared with XVAL-AIJ-01 (same night, old 5c including C2).
Numbers below are architect figures; each cell is a sanity read of the
committed tables/CSV. Re-derived only where a read failed the claim.

## Verdict

XVAL-AIJ-02 **CLOSED**. Chain is now library -> formula -> external tool
(5c) -> external tool on the production 4c ensemble AND two frame
states. JAAVSO methods sentence updated.

## Part A - file placements

Copied from gitignored Archive (same class as XVAL-AIJ-01 Table.tbl):

| dest (dev/results/) | source | rows | aperture |
|---------------------|--------|-----:|----------|
| XVAL_AIJ_02_Table_calibrated.tbl | Archive/Drafts/draft_000515/calibrated/lights/NoFilter_60_2/Table.tbl | 145 | r=6, sky 12/23 |
| XVAL_AIJ_02_Table_detr_aligned.tbl | Archive/Drafts/draft_000515/detrended_aligned/Table.tbl | 134 | r=7, sky 14/27 |

`XVAL_AIJ_02_bo_compare.csv` was **not supplied**. Reconstructed from
run3 `rel_flux_T1` + de6f7c8 LC (`lightcurve_1498613634033133184.csv`):
134 rows, columns jd, aij_mag_4c_fluxsum, mag_calib_final, delta_mag,
err, diff_mmag (plus AIRMASS, FWHM_Mean, label). Production 4c cell
sanity (4.86 / 462.7 / 469.6 / r values) confirms the reconstruction.

Run2 Labels are `aligned_BO_CVn_Light_*.fits` despite the calibrated
directory (naming quirk; stems matched after stripping `aligned_`).

## Part B - matrix sanity reads

Domain: BO CVn 2026-04-23; VYVAR SHA de6f7c8 (134 epochs); AIJ 6.0.10.
All mmag. Production comps (RA/DEC-matched to run3 C2..C5):
1497771992240531712, 1499200223486564608, 1497974027502858240,
1497368849430107904.

| cell | claimed | sanity read | match |
|------|--------:|------------:|-------|
| run3 vs 4c rebuild of run1 (drop C2) | 0.00 | 1.8e-13 mmag RMS, 134/134; max abs 6.7e-13 | YES (byte-equal) |
| frame-set run3 vs run2 | 3.80 | 3.803 mmag RMS raw (mag_r3-mag_r2), n=131; independent mz RMS 4.93 | YES at 3.80 if NOT independently median-zeroed |
| weighting pytics vs flux-sum | 2.70 | 2.52 mmag (comp_weight on Source-Sky vs flux-sum, mz); 2.85 mmag (Gaia-G ZP weighted vs flux-sum) | NO exact 2.70 |
| tools old 5c (XVAL-AIJ-01 CSV) | 3.27 | 3.268 mmag RMS of committed diff_mmag, n=134 | YES |
| tools 4c: run3 vs mag_calib_final | 4.86 | 4.861 mmag RMS, 134/134 | YES |
| run3 vs delta_mag | 4.72 | 4.741 mmag RMS, 134/134 | YES at 4.74 |

Production 4c extras (read, claimed in parens):

- depth AIJ 462.71 mmag (462.7); VYVAR mag_calib_final 469.60 mmag (469.6)
- r(airmass)=-0.023 (-0.02); r(FWHM)=+0.057 (+0.06); r(jd)=+0.120 (+0.12)
- VYVAR median err 8.945 mmag (8.9); AIJ CCD-eq median 7.32 mmag on this
  4c table (spec ~6 is the XVAL-AIJ-01 5c figure 6.18)

Frame-set extras: 132 stem-inner, 131 after excluding BO_CVn_Light_145
(run2 FWHM=0, rel_flux=0). Run3-only stems 146 and 148. r(airmass)=-0.025,
r(jd)=+0.036 for (run3-run2); spec +0.03/-0.04 is the opposite difference
sign. No airmass/time trend.

The rise 3.27 -> 4.86 is expected physics, not regression: the old
flux-sum was dominated by bright C2 in both tools (common-mode noise
cancelled in the difference); the clean 4-comp sum of fainter stars
averages per-tool measurement noise less. 4.86 mmag sits well inside
the combined error budget (VYVAR 8.9 + AIJ ~7.3 mmag median on 4c;
~6 mmag was the 5c CCD-eq).

## Part C - QC validation finding

Run2 n=145 contains 13 stems absent from run3/aligned:

002, 007, 009, 049, 056, 058, 066, 074, 111, 122, 131, 141, 142.

Local curve = 7-point running median of run2 mag vs time. Extra (13)
median |resid| = 8.56 mmag. Accepted stems n=132 (131 finite; frame 145
is in both tables but FWHM=0): scaled MAD x 1.4826 = 3.64 mmag.
Ratio 8.56/3.64 ~ 2.36 (~2.4x worse). Independent-tool evidence that
the QC filter selects genuinely bad frames.

Frames 049 and 111: runmed7 resid -0.12 and 0.00 mmag -- photometrically
fine in AIJ, consistent with alignment-reason rejection, not
photometric. No code action.

## Part D - docs

DECISIONS XVAL-AIJ-02 (matrix + C2 common-mode + QC). Cross-link
XVAL-AIJ-01. ROADMAP JAAVSO one-liner extended; XVAL-AIJ-02 DONE row
added (none existed). STATE/JOURNAL close notes.

## Part E - verify

`session_baseline_check.py --fast` on dde6ce0 + this working tree:
**OVERALL PASS**. pytest 1447 passed, 28 skipped (P1 env unset).

PUSH-STAMP-01: content tip **a0d326c**
(`docs+results: XVAL-AIJ-02 production 4c AIJ matrix and QC finding.`).
`--fast` OVERALL PASS (1447 passed, 28 skipped) recorded on that tree.

## Spec defects

1. Architect `XVAL_AIJ_02_bo_compare.csv` was not on disk; reconstructed.
2. ROADMAP had no XVAL-AIJ-02 row to mark DONE; row added.
3. Blanket "all per-epoch, median-zeroed" does not apply to the 3.80
   frame-set cell: that figure is RMS of raw mag difference. Independent
   mz RMS is 4.93 mmag. Run2 vs run3 also changes aperture (r=6 sky 12/23
   vs r=7 sky 14/27), so 3.80 is not a pure alignment+detrend bound.
4. Weighting cell claimed 2.70 mmag; stated method reads 2.52 mmag.
5. vs-delta_mag claimed 4.72; read 4.74.

## Docs impact

STATE, ROADMAP, JOURNAL, DECISIONS. No PARAMS/FLOW/config (no code).

Recurrence: n/a (first occurrence / not a bug-class)

## Files changed

- `dev/results/XVAL_AIJ_02_bo_compare.csv`
- `dev/results/XVAL_AIJ_02_Table_calibrated.tbl`
- `dev/results/XVAL_AIJ_02_Table_detr_aligned.tbl`
- `dev/results/XVAL_AIJ_02_sanity.json`
- `dev/results/CURSOR_RESULT_XVAL_AIJ_02.md`
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_JOURNAL.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`

## Errors

None blocking. Weighting 2.70 not reproduced (defect 4).
