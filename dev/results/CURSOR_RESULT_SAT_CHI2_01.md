CURSOR RESULT - 2026-09-29 SAT-CHI2-01

Date: 2026-09-29. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: fdc17b5 (EPSF-PIN-FITOK-01).
Class: MEASURE ONLY. No src_py / config / anchor / meter / pin-file changes.
Push: origin consolidate-01 result+artifacts only.

## What I did

Verified tip fdc17b5. Measured M0-M4 on era05 work-copy
`tmp/session_baseline/20260928T183918Z` (read-only) and live 516
headers/DB. Wrote artifacts under
`dev/results/context/session_20260929_sat_chi2_01/`. STOP with
branch-specific fix menu. No production edits.

## Gates

| gate | result |
|---|---|
| G0 tip | PASS `fdc17b5` (consolidate-01) |
| G4 start | PASS csv bfa24039 / fits 13e77cf8 / epsf 172f9540 |
| G4 end | PASS unchanged (same prefixes) |
| era05 aperture | PASS checked n=53 core LCs on snapshot; constants 87197716 / dd92e99d n=157 (not regenerated) |
| end `--fast --clean` | PASS OVERALL; 1643 passed, 34 skipped; clean-tree PASS. Log: `dev/results/context/session_20260929_sat_chi2_01/g_fast_end.txt` |

## M0 - saturation limit on 516

Artifact: `m0_saturation_resolution.json`.

Resolved ceiling for NoFilter_60_2:
- sat_adu = 65535.0, source = DERIVED (SAT-DIAG pile-up)
- lin_adu = 52428.0 = 0.80 * sat_adu, source = DEFAULT_FRAC
- equipment ID=1 (QHY294): SATURATE_ADU = NULL after
  `database.py:2679-2700` migration (was wrong binned 16384)
- header SATURATE/DATAMAX/MAXADU: absent on raw lights

Resolver chain (code wins):
- docstring pointer `catalog_match.py:146`
- `_effective_saturation_limit` `pipeline_calibrate.py:1263-1312`
  (header -> equipment -> DATAMAX/MAXPIX -> BITPIX -> fallback ->
  container clip 65535)
- peak-test `_resolve_peak_saturation_limit_adu`
  `pipeline_catalog.py:1789-1833` applies saturate_fraction
- column fill: `sat_diag.py:159-162` and
  `pipeline_catalog.py:1974` write `saturate_limit_adu_85pct`
  = sat_adu * SATURATE_LIMIT_FRACTION (**0.80**, not 0.85)

Camera facts (raw Light_001):
- BITPIX=16, BZERO=32768, BSCALE=1 -> unsigned 16-bit container
- XBINNING=YBINNING=2, INSTRUME=QHY CCD QHY294PROM, GAIN=0.0,
  READMODE=0.0, no SATURATE card
- Native ADC: 14-bit sensor commonly scaled x4 into 16-bit; sat_diag
  records sat_adu_native=32767.5, lin_adu_native=26214.0
- Calibrated/detrended frames: BITPIX=-32 float

Comp-selection saturation gate:
- `comp_selection_per_target.py:763-807`: over if
  peak > limit * (admission_sat_peak_frac / saturate_limit_fraction)
  = peak > 0.70 * sat_adu (with limit = 0.80*sat_adu)
- `comp_pool_rms.py:152-164`: over if peak > limit (0.80*sat_adu),
  no admission remult
- Config: `saturate_limit_fraction=0.80` (`config.py:787`),
  `admission_sat_peak_frac=0.70` (`config.py:802`)
- Column name "85pct" is a misnomer; value on 516 = 52428 = 0.80*65535

Why EPSF-PIN-FITOK-01 M2 max_peak_adu was NaN:
- Peak lives in per-frame `proc_*.csv` column `peak_max_adu`
  (also catalog-level in comparison_stars / masterstars)
- Prior join used comparison_stars only: **27/53** pins present there;
  26 left empty. Did not fall back to proc CSV max.

## M1 - direct saturation evidence (53 pinned comps)

Artifact: `m1_pin_peaks.csv`, `m1_pileup.csv`, `m1_peak_vs_G.png`.
Bias state: `peak_max_adu` = calibrated/aligned float;
`peak_max_adu_raw` = raw-placed ADU. No `saturated_plateau` column
in procs.

| descriptor | n pins |
|---|---:|
| any frame ratio(peak/limit) >= 1.0 | 0 |
| any frame frac_raw_ge_ceiling > 0 | 0 |
| n_is_saturated_raw > 0 | 0 |
| max_core5_near_max > 0 (independent pile-up) | 0 |

Brightest low-fit_ok pins (median / max peak ADU vs ceiling 65535,
limit 52428, admission 45874.5):

| comp_id | G | fit_ok_frac | med peak | max peak | max ratio vs limit |
|---|---:|---:|---:|---:|---:|
| 1500296402219939584 | 8.24 | 0.000 | 31532 | 39528 | 0.754 |
| 1497613731286514432 | 8.45 | 0.000 | 33910 | 47176 | 0.900 |
| 1499906247391001088 | 8.74 | 0.000 | 31300 | 42044 | 0.802 |
| 1497442379271632384 | 8.85 | 0.022 | 30548 | 38640 | 0.737 |

Only 1497613731286514432 exceeds admission on **4/134** frames
(below DRAFT_SAT_EXCLUDE_FRAME_FRAC=0.50). Frame global pile-up
exists at 65535 (SAT-DIAG), but pin cores do not sit on it.

Reading: **no hard saturation / clip evidence for the pinned comps.**

## M2 - field linearity

Artifacts: `m2_linearity.csv`, `m2_linearity.png`, `m2_fwhm_vs_peak.csv`.

- Colour fit on peak < 0.30*ceiling: inst = a + b*G + c*(BP-RP)
  with b~1.037, c~0.427
- Bin medians of residual vs peak stay within 3*MAD of the faint
  residual across the whole peak range
- **CANDIDATE linearity limit peak ADU: null** (no departure bin
  found by the stated method)
- Field FWHM top-20% / median ratio never >= 1.25 on any frame
  (config check `config.py:778-780`); pinned bright comps have
  median fwhm_ratio_to_field ~0.9-1.0 (not inflated)
- No separate linearity / PTC series for this night in
  CalibrationLibrary; M2.1 is the only linearity evidence

Reading: **no measured bright-end nonlinearity knee on this field.**

## M3 - chi2 locus vs flux (H-MODEL)

Artifacts: `m3_chi2_locus.json`, `m3_chi2_locus.png`,
`m3_radial_residuals.png`.

Fit log10(chi2) = a + k*log10(psf_flux) on peak < 0.70*ceiling
(n=30076):
- **k = 1.060 +/- 0.003** (scatter_log10 = 0.337)
- Expectation under pure model error bright-end ~2; measured k~1
  is consistent with photon-noise-dominated residual scaling, still
  a smooth flux locus
- Locus crosses chi2=50 at psf_flux ~ 1.94e5 -> nearest star
  **G ~ 9.01**, peak ~ 22212 ADU
- Bright pin fit_ok=False epochs sit **on the locus**
  (|offset| typically 0.1-1.4 sigma), not as outliers above it
- Radial: peak-normalized profiles (3 frames); bright pins vs mid
  fit_ok stars - no flat-topped clip signature in cores
  (see PNG)

chi2 SET paths at fdc17b5:
- iterative `psf_photometry.py:3348-3371`: reduced_chi2;
  chi2_ok = isfinite(chi2) AND chi2 < threshold (default 50);
  nonfinite FAILS
- grouped `psf_photometry.py:2788-2794`: chi2_ok =
  (NOT isfinite(chi2)) OR chi2 < limit; **nonfinite PASSES**
- procs lack `psf_iterative`; `psf_group_n` present (grouped when >1)

## M4 - classification

Artifact: `m4_classification.csv`.

| class | n (all 53) | n among 17 low-fit_ok (<0.66045) |
|---|---:|---:|
| SAT | 0 | 0 |
| NONLIN | 0 | 0 |
| MODEL | 49 | 13 |
| OTHER | 4 | 4 |

(Two auto-NONLIN labels with |lin_resid| > 1000 mmag reclassed
OTHER: catastrophic photometry outliers at G>11, not a bright knee.)

Brightness boundary (data): locus crosses chi2=50 near **G~9.0 /
peak~2.2e4 ADU**. Below that G, fit_ok_frac collapses along the
locus; above it, pins are mostly fit_ok stable. Not a saturation
boundary (peaks still << 0.70*ceiling).

### Readings

**R1:** H-SAT holds for **0 of 17** low-fit_ok pins. Of the 17:
13 MODEL (on locus, no M1/M2 sat/nonlin evidence), 4 OTHER
(catastrophic resid / missing locus). Bright killers
(G 8.24-9.68, including FW/BO pin killers) are MODEL.

**R2:** No pinned APERTURE comp is SAT or NONLIN after measurement.
Aperture gate let them through because calibrated peaks stay below
0.70*65535 on nearly all frames (`comp_selection_per_target.py:763-807`);
numbers above. BO CVn target_id=1498613634033133184 and
FW CVn=1497343732462852864 pin MODEL-class bright comps, not
saturated ones. No aperture re-cut implied by H-SAT.

**R3:** k=1.060+/-0.003. The fixed threshold 50 **coincides with a
brightness boundary** at G~9.0 (locus crossing). Same class as a
magic brightness cut.

## Errors on the record

- Column name saturate_limit_adu_85pct vs actual 0.80 fraction
  (pre-existing naming debt; measured, not fixed here).
- M2 CANDIDATE linearity limit null (method found no knee).
- Field G for M1 plot from proc phot_g_mean_mag often empty;
  pin G from EPSF-PIN M2 / Gaia join.
- First end `--fast --clean` FAIL on ascii_policy: UTF-16 BOM in
  prior EPSF-PIN-FITOK-01 tee logs + smart dash in that RESULT md
  (already on tip fdc17b5). Scrubbed to ASCII in this commit
  (result tree only) so the gate can pass; no src_py change.

## STOP - branch-specific fix menu (nothing executed)

Verdict: **Branch M (MODEL)**. Mixed boundary not required for the
bright pin-killer set; no SAT/NONLIN among them.

Branch S: N/A for the low-fit_ok pin set (no SAT/NONLIN evidence).
Do not open an aperture saturation rewrite on this measurement.

Branch M (execute next, Milan chooses form):
(m1) Replace fixed `psf_chi2_threshold=50` with a data-derived
     criterion: per-frame outlier from the chi2(flux) locus, or
     chi2 with an explicit flux-proportional model-error term.
     Pre-fail / post-pass unit test at
     `psf_photometry.py:3350-3371` / `:2788-2794`. Then re-run
     EPSF-PIN-FITOK-01 M1/M4 **without** pin filtering.
(m2 optional companion) Keep INV-PSF-LC-PIN-01; do not filter pins
     for fit_ok_frac once (m1) stops treating the locus as failure.

In all branches:
- epsf01 552ace75 stays PROVISIONAL (Milan)
- G3 refs re-cut only AFTER the product fix, SAME statistic
  (demeaned RMS of psf_delta-ap_delta); **no meter rewrite**

## Files changed (this commit only)

- `dev/results/CURSOR_RESULT_SAT_CHI2_01.md`
- `dev/results/context/session_20260929_sat_chi2_01/*`
- ASCII scrub (gate unblock): `dev/results/CURSOR_RESULT_EPSF_PIN_FITOK_01.md`
  and tee logs under `session_20260929_epsf_pin_fitok_01/`

## Docs impact

none (measure-only).

## Recurrence

n/a (measurement).

## Gates (end)

`--fast --clean` OVERALL PASS (2026-09-29T14:49:50Z).
1643 passed, 34 skipped; clean-tree PASS.
Log: `dev/results/context/session_20260929_sat_chi2_01/g_fast_end.txt`.
G4 end unchanged (bfa24039 / 13e77cf8 / 172f9540).

STOP after M5.
