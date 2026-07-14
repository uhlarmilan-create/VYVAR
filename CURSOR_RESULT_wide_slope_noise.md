CURSOR RESULT -- 2026-07-14

**STATUS: SUPERSEDED (2026-07-14 WSN-FIX).** The EXCESS_UNATTRIBUTED verdict below rested on
inverted brightness tertile labels (bright/faint swapped) and WLS residual SEs ~100x too
small for faint stars. Do not cite P1/P5 numbers from this file. See CURSOR_RESULT_wsn_fix.md.

What I did
Saved spec to docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md (DRAFT for Milan approval). Implemented
report-only analysis: wide_slope_noise_core.py, scripts/wide_slope_noise_run.py, tests.
Ran on draft_424 wide_CLEAR archival data; updated STATE/ROADMAP/JOURNAL.

## Output / findings

**Cell:** wide_CLEAR, draft_424, n=148, SD(b_X)=0.0948 mag/airmass (matches K2-COHORT).

**P3 (before P2):** Median slope SE floor 0.00145 mag/airmass. H1 attainable b_X at eps=1%
flat error: 0.00114 (untestable -- drift span p90 only 0.21 px). H2 p90 attainable ~0.044
(testable on full enrichment run).

**P1 (H0):** Excess variance significant in all brightness tertiles (CIs exclude zero) --
slope scatter is NOT fully explained by per-epoch photometric noise.

| Tertile | SD_obs | median SE | excess var |
|---------|-------:|----------:|-----------:|
| bright | 0.136 | 0.0058 | 0.0184 |
| mid | 0.072 | 0.0015 | 0.0052 |
| faint | 0.055 | 0.0001 | 0.0030 |

**P2:** Multivariate weighted decomposition (Type-II partial SS, se floor = SD/5): no group
>=50% share; chi2_red=21.6. Colour H4 check PASS (slope spread 0.0037 <= 0.031).

**P2b univariate (exploratory):** Largest linear R^2 ~0.05 (drift_span_px, r2_norm) -- far
below dominant-source threshold.

**P4:** Median cohort sigma_r ~5.5 mmag band; fitted spatial+drift+fwhm RMS component ~29 mmag
(order-of-magnitude; not a calibrated per-point noise estimate).

**P5 verdict:** **EXCESS_UNATTRIBUTED** -- real excess, no pre-registered dominant source named.

## Errors (if any)
None. Full enrichment run ~15 min (148 stars x LOO rebuild); use --from-cache for decomposition-only.

## Files changed
- docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md (new)
- wide_slope_noise_core.py (new)
- scripts/wide_slope_noise_run.py (new)
- tests/test_wide_slope_noise_core.py (new)
- docs/VYVAR_STATE.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_JOURNAL.md
- tmp/wide_slope_noise/ (artifacts, gitignored)

pytest: +6 tests (838 total suite). ruff: clean on touched .py files.

NOT PUSHED -- Milan spec approval + review.
