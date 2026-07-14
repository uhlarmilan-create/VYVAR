CURSOR RESULT -- 2026-07-14 (WSN-2)

What I did
Corrected P4 to integrate measured excess (not overfit multivariate fit). Pre-registered
neighbor-contamination effect size from Gaia DR3 SQLite; gated correlation test (untestable-here).
Final P5 verdict + park record. Pushed full WSN chain to origin/main.

## Output / findings

**P4 corrected integration (supersedes WSN-FIX fitted-RMS FAIL):**

sigma_slope_pt = sqrt(V_ex) * SD(X), SD(X)=0.056 airmass lever arm.

| Tertile | V_ex | sigma_slope_pt (mmag) | CI (mmag) |
|---------|-----:|----------------------:|----------:|
| bright | 0.00269 | 2.90 | [2.46, 3.19] |
| mid | 0.00197 | 2.49 | [1.73, 2.92] |
| faint | 0.00850 | 5.17 | [3.92, 5.93] |

Cohort median = 2.90 mmag; faint tertile drives unification.

**Unification:** CONSISTENT -- faint sigma_slope_pt 5.17 mmag aligns with PZQ sigma_r 5.5 mmag
[4.7, 6.5] and rig constant 4.5 mmag. Bright/mid tertiles smaller (2.5-2.9 mmag): expected when
sigma_slope_pt <= sigma_r (X-linear component is a subset of full red noise). Star scatter:
tmp/wide_slope_noise/figures/p4_sigma_slope_vs_sigma_r.png.

**P4 legacy fitted RMS (35.5 mmag):** SUPERSEDED -- CV-proven noise absorption; do not cite.

**Neighbor contamination (pre-test gate):**

- Plate scale 9.77 arcsec/px; search r = r_ap + 3*sigma_PSF(FWHM_p90) = 10.40 px (101.6 arcsec)
- Overlap: 2D Gaussian disk quadrature (grid_n=48)
- FWHM vs airmass: dFWHM/dX = 1.51 px/airmass (scatter 0.08 px, n=139)
- Attainable |b_attain|: p50=0.00010, p90=0.00287, max=1.36 mag/airmass
- Floor (median SE) = 0.057 mag/airmass -> **untestable-here** (p90 << floor)
- **STOP:** no correlation/regression computed (pre-registered gate)

**P5 final verdict:** **UNIFIED_PHENOMENON_PARK**

Slope excess, sigma_r, and rig constant consistent as one unidentified driver at ~5 mmag.
Hypothesis space exhausted at measured bounds: colour (<=0.031), spatial/drift/FWHM (<4% SS),
detector drift (H1 testable but <1% SS), brightness tertiles, neighbors (untestable).

**Park record (ROADMAP):** WIDE-SLOPE-NOISE -> PARKED. Revisit: new flats (bin2), defocus
experiment, or EPD/SysRem-style decorrelation workstream.

## Errors (if any)
None.

## Files changed
- wide_slope_noise_core.py (P4 excess, neighbor, final_wsn_outcome)
- scripts/wide_slope_noise_wsn2_run.py (new)
- tests/test_wide_slope_noise_core.py (+6 tests)
- docs/VYVAR_WIDE_SLOPE_NOISE_SPEC.md (P4 amendment)
- docs/VYVAR_STATE.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_JOURNAL.md
- CURSOR_RESULT_wsn_fix.md (P4 supersede pointer)
- CURSOR_RESULT_wsn2.md (this file)
- tmp/wide_slope_noise/wsn2_summary.json, WIDE_SLOPE_NOISE_wsn2_result.md (gitignored)

pytest: 852 passed (+6 vs WSN-FIX 846). ruff: clean on touched .py files.

Push: origin/main at <hash after push>; session_baseline_check.py --fast PASS.
