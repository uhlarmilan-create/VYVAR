CURSOR RESULT -- 2026-07-14 (WSN-FIX)

What I did
Retracted prior EXCESS_UNATTRIBUTED verdict (inverted tertile labels + WLS residual SE).
Flipped spec to APPROVED (Milan, 2026-07-14). Fixed analytic slope SE, tertile orientation,
detector-frame drift (path c), P2 partial-SS scaling, removed SD/5 se floor, added CV shares
and P4 consistency check. Re-ran draft_424 wide_CLEAR; updated STATE/ROADMAP/JOURNAL.

## Output / findings

**Prior verdict:** SUPERSEDED -- see CURSOR_RESULT_wide_slope_noise.md header.

**Root causes fixed:**
1. Tertile labels inverted in `excess_variance_by_tertile` (bright/faint swapped vs mag_g).
2. `se_use` used WLS residual SE (`b_X_se`) ~100x too small for faint stars; now analytic
   sqrt(1/sxx_w) with bootstrap max.
3. H1 used aligned proc x/y (p90 span 0.21 px); detector drift on calibrated pre-align lights
   gives p90 span 13.6 px (path c; no alignment shift columns archived).
4. P2 Type-II partial SS compared unscaled vs scaled RSS (all shares falsely 0 before fix).
5. Ad-hoc `se_floor = SD/5` removed (not in spec).

**SE audit (step-by-step, one star per tertile):**

| Tertile | N | SD(X) | med err | hand SE | analytic SE | WLS resid SE | bootstrap SE | se_use |
|---------|---:|------:|--------:|--------:|------------:|-------------:|-------------:|-------:|
| bright | 139 | 0.0560 | 0.0125 | 0.0189 | 0.0194 | 0.00029 | 0.0231 | 0.0231 |
| mid | 139 | 0.0560 | 0.0209 | 0.0316 | 0.0324 | 0.00259 | 0.1301 | 0.1301 |
| faint | 139 | 0.0560 | 0.0589 | 0.0892 | 0.0910 | 0.00542 | 0.1012 | 0.1012 |

Defect: WLS residual SE on mean-detrended mags; runner previously took `se_propagated` only.

**Tertile label verdict (corrected: lower mag_g = brighter):**

| Tertile | mag_min | mag_max |
|---------|--------:|--------:|
| bright | 8.74 | 11.37 |
| mid | 11.44 | 12.58 |
| faint | 12.60 | 15.15 |

**Prior tables with inverted labels (retraction list):**
- `CURSOR_RESULT_wide_slope_noise.md` P1 table (bright/faint rows swapped).
- `tmp/wide_slope_noise/WIDE_SLOPE_NOISE_result.md` pre-WSN-FIX run (superseded banner added).
- WSN-only helper; ANCHOR-ERR-VERIFY JOURNAL text (faint/mid/bright mmag ordering) was already
  physically correct and unchanged.

**Detector drift (path c):** DAO cutout centroid chain on
`Archive/Drafts/draft_000424/calibrated/lights/NoFilter_60_2/` (proc CSV -> BO_CVn_Light_NNN.fits).
Paths (a)(b) not available (`alignment_report.csv` absent). Distribution (n=148):
p50 span 8.6 px, p90 13.6 px; aligned-frame p90 0.21 px.

**P1 old vs new (corrected labels):**

| Tertile | | SD_obs | median SE | excess var | noise-dom |
|---------|---|-------:|----------:|-----------:|-----------|
| bright | old label "faint" | 0.055 | 0.018 | 0.00269 | no |
| bright | new | 0.055 | 0.018 | 0.00269 | no |
| faint | old label "bright" | 0.136 | 0.100 | 0.00850 | no |
| faint | new | 0.136 | 0.100 | 0.00850 | no |

SE ordering now physical (SE grows toward faint). Mislabeled old table showed impossible
faint SE 0.0001 (actually bright stars).

**P2 (corrected SEs, no se floor, chi2_red=2.20):**

| Group | in-sample share | q (FDR) | CV R2 |
|-------|----------------:|--------:|------:|
| spatial | 0.039 | 0.0003 | nan |
| drift_aligned | 0.023 | 0.0006 | nan |
| fwhm | 0.018 | 0.0065 | nan |
| colour | 0.013 | 0.0065 | 0.061 |
| drift_detector | 0.008 | 0.090 | nan |
| mag | 0.005 | 0.090 | nan |

No group >=50%. Bootstrap share CIs upper bounds <=20% (spatial). In-sample FDR-significant
groups collapse under CV (colour CV R2 ~6%; others ill-conditioned / nan) -- noise absorption,
not a dominant physical source.

**P3 H1 (detector drift):** eps=1% flat error attainable b_X p90 = 0.114 mag/airmass (testable);
eps=0.3% = 0.034 (below median SE floor). H1 moves from untestable (0.21 px aligned) to testable
at 1% flat gradient.

**P4:** RMS fitted spatial+detector+fwhm component = 35.5 mmag vs sigma_r 5.5 mmag --
**FAILED** (ratio 6.45). Consistent with overfit absorption (Part 3.2); fitted terms do not
integrate to PZQ red-noise scale.

**SUPERSEDED (WSN-2):** P4 fitted-RMS check replaced by excess-integration sigma_slope_pt;
see CURSOR_RESULT_wsn2.md. Faint tertile sigma_slope_pt 5.17 mmag aligns with sigma_r/rig constant.

**P5 re-verdict:** **EXCESS_UNATTRIBUTED** (honest). Real excess in all tertiles; no pre-registered
dominant source >=50% SS with FDR. P4 failure + CV collapse indicate multivariate shares are not
trustworthy magnitude estimates. H1 flat-drift at detector p90 is testable in principle (eps=1%)
but explains <1% SS in P2 -- not the dominant term. Action: bin2 flats / flat quality remains a
hypothesis, not a confirmed driver from this decomposition.

## Errors (if any)
None blocking. Full enrichment ~15 min; `--from-cache --refresh-detector` ~7 min for detector only.
Some DAO cutout frames emit NoDetectionsWarning (seed hold-last-good).

## Files changed
- docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md (APPROVED + SE/P2/H1 amendments)
- wide_slope_noise_core.py
- scripts/wide_slope_noise_run.py
- tests/test_wide_slope_noise_core.py (+8 tests)
- CURSOR_RESULT_wide_slope_noise.md (SUPERSEDED banner)
- CURSOR_RESULT_wsn_fix.md (this file)
- docs/VYVAR_STATE.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_JOURNAL.md
- tmp/wide_slope_noise/ (artifacts, gitignored)

pytest: 846 passed (+8 vs 838 baseline). ruff: clean on touched .py files.

NOT PUSHED -- Milan review first.
