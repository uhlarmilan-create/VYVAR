CURSOR RESULT - 2026-09-29 EPSF-PIN-FITOK-01

Date: 2026-09-29. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: bb1ef13 (ERA-520-RECUT-01 close).
Class: MEASURE ONLY. No src_py / anchor / meter changes.
Push: origin consolidate-01 result+artifacts only.

## What I did

Verified tip bb1ef13. Ran G-FAST `--fast --clean` FIRST (missing
ERA-520 stamp). Measured M1-M4 on era05 / work-copy
20260928T183918Z and era04 snapshot (read-only). Wrote artifacts
under `dev/results/context/session_20260929_epsf_pin_fitok_01/`.
Stopped with fix menu for Milan. No production edits.

## G0 / G-FAST / G4

| gate | result |
|---|---|
| G0 tip | PASS `bb1ef13` (consolidate-01; dirt = declared untracked context only) |
| G-FAST | PASS OVERALL; 1643 passed, 34 skipped; clean-tree PASS. Log: `dev/results/context/session_20260929_epsf_pin_fitok_01/g_fast_bb1ef13.txt` |
| G4 start | PASS csv bfa24039 / fits 13e77cf8 / epsf 172f9540 |
| G4 end | PASS csv bfa24039 / fits 13e77cf8 / epsf 172f9540 (unchanged) |
| era05 aperture | PASS 87197716 n=53 / dd92e99d n=157 (checked, not regenerated) |
| end `--fast --clean` | PASS OVERALL; 1643 passed, 34 skipped; clean-tree PASS. Log: `dev/results/context/session_20260929_epsf_pin_fitok_01/g_fast_end.txt` |

G-FAST is the missing stamp for ERA-520-RECUT-01 at bb1ef13.

## Code cites (bb1ef13) - architect mechanism confirmed

- Admission: `src_py/psf_internal_lc.py:124-139`
  `psf_fit_ok_for_zp_mask` = stored fit_ok AND finite flux>0
  (FIT-OK-ADMISSION-01; chi2 OR-gate removed).
- Pin NaN: `src_py/psf_internal_lc.py:528-542` INV-PSF-LC-PIN-01.
- Ensemble: `src_py/psf_internal_lc.py:290-318`
  `resolve_ensemble_ids` -> pinned_ensembles.csv first.
- **SET** (not consume) of psf_fit_ok:
  - iterative: `src_py/psf_photometry.py:3350-3371`
    `converged=(flags&8)==0`; `chi2_ok=isfinite(chi2)&chi2<psf_chi2_threshold`
    (default 50); `fit_ok=converged and chi2_ok`.
  - grouped: `src_py/psf_photometry.py:2788-2794`
    same converged; chi2_ok allows nonfinite chi2.

Spec matches code. No refute.

## M1 - census of 53 era05 internal PSF LCs

Artifact: `m1_psf_lc_census.csv`.

Coverage bins (n_full/n_epochs):

| bin | era05 | era04 snapshot |
|---|---:|---:|
| >= 0.9 | 1 | 0 |
| 0.5-0.9 | 0 | 0 |
| (0, 0.5) | 15 | 0 |
| == 0 | 37 | 53 |
| n | 53 | 53 |

G3 targets era05: BO n_full=1 cov=0.00746; FW n_full=0 cov=0.
Pinned ensemble_source: 43/53; comparison_stars: 10/53.

Reading: collapse is PRODUCT-WIDE (37/53 empty; only 1 with
cov>=0.9), not limited to BO/FW.

era04 snapshot PSF LCs also show n_full=0 for all 53: those
snapshot procs have PSF stubs (`psf_flux` finite=0,
`psf_fit_ok` all False on sample frames). Healthy era04 G3
(n_full=134, 12.505/4.629) lived on `--full-epsf` work-copies
under pre-FIT-OK admission, not in the era04 freeze PSF LC bytes
as currently stored.

## M2 - comps that kill epochs

Artifact: `m2_pinned_comps.csv`, `m2_fit_ok_criteria.csv`,
`m2_fit_ok_frac_hist.png`, `m2_fit_ok_frac_all_stars.csv`.

Union of pinned comps across 53 targets: n=53 unique comps.

BO pins (4) fit_ok_frac_era05:
| comp | G | fit_ok_frac | fail (False epochs) |
|---|---:|---:|---|
| 1499200223486564608 | 9.68 | 0.007 | chi2_ge_50: 133 |
| 1497771992240531712 | 9.75 | 0.769 | chi2_ge_50: 31 |
| 1497974027502858240 | 11.17 | 1.000 | - |
| 1497368849430107904 | 11.52 | 1.000 | - |

FW pins (8) fit_ok_frac_era05 (killers first):
| comp | G | fit_ok_frac | fail |
|---|---:|---:|---|
| 1499906247391001088 | 8.74 | 0.000 | chi2_ge_50: 134 |
| 1497442379271632384 | 8.85 | 0.022 | chi2_ge_50: 131 |
| 1497674651102612992 | 9.29 | 0.097 | chi2_ge_50: 121 |
| others | ~9.3-10.3 | 0.89-1.00 | few/none |

Fail-criterion totals across pinned comps False epochs:
chi2_ge_50 = 1882; nonfinite_chi2 = 268; nonfinite_or_nonpos_flux = 268.
Dominant SET failure: reduced chi2 >= 50 (`psf_photometry.py:3370-3371`).

peak_over_sat_frac: NaN for these rows (saturate_limit / peak not
joined cleanly from comparison_stars for all pins); not used as a
claim.

fit_ok_frac_era04 from era04 proc: **n/a**  procs carry stub PSF
columns (psf_flux all nonfinite; fit_ok all False). Cannot attribute
rate change to FIXPOS-NOOP-01 (9e8dc91) or GAIN-FALSY-01
(1efc08c/0708abe) via era04-vs-era05 proc comparison.

CANDIDATE split (largest gap in sorted unique pinned fit_ok_frac,
not wired): threshold = 0.66045, gap = 0.2165, n_below = 17,
n_at_or_above = 36. Derivation in `summary.json` /
`m2.candidate_split`.

## M3 - is fit_ok=False honest?

Artifact: `m3_fitok_honesty.csv`.
Metric: per star, r = 2.5*log10(F_psf/F_ap) - median(r); RMS and
median of r on fit_ok=True vs fit_ok=False-but-finite epochs.

Pooled (60 mixed stars, n_true>=5 and n_false>=5):
- median RMS True = 445.8 mmag; False = 863.9 mmag
- median (RMS_false - RMS_true) = **+314.0 mmag**
- median (med_false - med_true) = **+17.3 mmag**

By G bin (mixed only):
| G | n | med delta_rms mmag | med delta_med mmag |
|---|---:|---:|---:|
| [8, 9.5) | 7 | +3.64 | +1.50 |
| [9.5, 11) | 5 | -2.04 | +6.47 |
| [11, 13) | 0 | - | - |
| [13, 16) | 27 | +316.7 | +47.2 |

Explicit reading:
- Pooled / faint: **fit_ok=False fits are measurably worse**
  (median delta_rms = 314 mmag).
- Bright G<9.5 (the pin-killer domain): delta_rms = 3.64 mmag,
  delta_med = 1.50 mmag  within a 5 mmag indifference band on
  this proxy; chi2 still fails the SET threshold (>=50; medians
  on False epochs ~70-161). Do **not** reopen the OR-gate: M3
  does not support "indistinguishable" pooled, and even bright-end
  closeness is only on the flux-ratio proxy, not a license to
  admit chi2 failures into INV-PSF-LC-PIN-01 membership.

M4-C (pre-fix comps OR-gate) SKIPPED (M3 not indistinguishable).

## M4 - sandbox G3 residual replay (meter unchanged)

Artifact: `m4_replay.json`.
Statistic: demeaned RMS of (psf_delta - ap_delta) on pin-ok epochs
(`epsf_zp_ok.residual_stats`; same as G3).

| target | variant | n_comps | n_full | cov | dem RMS mmag | delta vs era04 ref |
|---|---|---:|---:|---:|---:|---:|
| BO | A as-is | 4 | 1 | 0.007 | 0.0 | n/a (n_full!=134) |
| BO | B filter frac>=0.66045 | 3 | 100 | 0.746 | **12.785** | +0.280 vs 12.505 |
| FW | A as-is | 8 | 0 | 0.0 | nan | n/a |
| FW | B filter | 5 | 110 | 0.821 | **7.757** | +3.128 vs 4.629 |

Three worst non-G3 (cov=0): B restores partial/full coverage
(n_full 52 / 62 / 134) with dem RMS 29 / 56 / 119 mmag (not G3
targets; refs N/A).

Reading: filtering the aperture pin set to fit_ok-stable comps
(CANDIDATE threshold from M2) restores a usable residual for BO
near the era04 ref; FW improves but sits ~3 mmag above 4.629.
INV-PSF-LC-PIN-01 kept on the reduced set. Not wired.

## Readings

1. Collapse is product-wide (37/53 empty), explained by FIT-OK +
   INV-PSF-LC-PIN-01 on aperture-era pins that fail chi2>=50.
2. fit_ok SET criterion is chi2-dominated; False fits are worse
   pooled (314 mmag); do not restore the OR-gate.
3. A PSF-specific pin filter (M4-B) restores BO dem ~12.8 mmag
   without touching the meter.
4. epsf01 552ace75 locks largely empty internal PSF LCs; G3 stays
   red until the product (membership) is fixed and refs re-cut
   with the SAME statistic.

## Errors on the record

None blocking measurement. peak_over_sat_frac incomplete (NaN)
for pinned comps  not claimed. era04 proc PSF stubbed  fit_ok
rate vs FIXPOS/GAIN not measurable from snapshot procs.

## STOP - fix menu for Milan (nothing executed)

Evidence-ranked:

(a) **PSF-specific membership (preferred by M4-B):** PSF pin set =
    aperture pin set minus comps below a data-derived fit_ok_frac
    split (CANDIDATE 0.66045 shown; not wired). Own provenance
    file; do not edit aperture `pinned_ensembles.csv`.
    INV-PSF-LC-PIN-01 unchanged in spirit on the reduced set.

(b) **Layer-1 pin rule for PSF consumers:** add PSF fit stability
    (fit_ok_frac / chi2) as a pool criterion when selecting pins
    for PSF LCs (not aperture LCs).

(c) **If tightening SET of fit_ok:** only with a pre-fail/post-pass
    unit test at `psf_photometry.py:3350-3371` / `:2788-2794`.
    M3 does **not** support reopening the OR-gate. Bright-end
    flux-ratio closeness (~3.6 mmag) is not enough to admit
    chi2>=50 epochs into pin membership.

(d) **Anchor / closure status (Milan decide):**
    - Mark epsf01 `552ace75` **PROVISIONAL** until re-cut after
      product fix.
    - Whether CLOSE-TEXT-02 stays enacted or returns to pending
      until internal PSF LCs are full again.

(e) **G3 refs:** re-cut only AFTER the product fix, SAME statistic
    (demeaned RMS of psf_delta-ap_delta), as a new era. **No meter
    rewrite.**

## Files changed (this commit only)

- `dev/results/CURSOR_RESULT_EPSF_PIN_FITOK_01.md`
- `dev/results/context/session_20260929_epsf_pin_fitok_01/*`

## Docs impact

none (measure-only; Milan decides d/e before doc enact).

## Recurrence

n/a (measurement / product-membership class; not a new unit test
in this commit).

## Gates (end)

`--fast --clean` OVERALL PASS (2026-09-29T12:53:29Z).
1643 passed, 34 skipped; clean-tree PASS (worktree=b1b_clean_93febf1d).
Log: `dev/results/context/session_20260929_epsf_pin_fitok_01/g_fast_end.txt`.
G4 end unchanged (bfa24039 / 13e77cf8 / 172f9540).

STOP after M5.
