CURSOR RESULT - 2026-07-30 AUDIT REMEDIATION STAGE 0

What I did
Measured post-P-10 sigma_clipped_stats vs sigma_pp (0.1), re-verified P-10 with independent
pre-fix test (0.2), prepared AUDIT-T3 bundle for push (0.3).

## 0.1 Estimator necessity after P-10 - DECISION REQUIRED

Measurement: `tmp/audit_stage0_measure.py` on 10 P-10-preprocessed frames (draft_435 calibrated,
frames 001/017/034/050/067/083/100/116/133/150).

| frame | sigma_clipped_stats | sigma_pp | ratio std/sigma_pp |
|-------|---------------------|----------|---------------------|
| 001 | 56.23 | 51.94 | **1.083** |
| 017 | 46.69 | 44.69 | 1.045 |
| 034 | 45.65 | 43.91 | 1.040 |
| 050 | 44.89 | 43.29 | 1.037 |
| 067 | 44.36 | 42.86 | 1.035 |
| 083 | 43.54 | 42.19 | 1.032 |
| 100 | 42.51 | 41.21 | 1.032 |
| 116 | 41.40 | 40.27 | 1.028 |
| 133 | 40.70 | 39.64 | 1.027 |
| 150 | 41.06 | 39.75 | 1.033 |

Summary: median ratio **1.034**; spread **5.4%**; **within_5pct_of_1: false** (frame 001 at +8.3%).

**Interpretation.** P-10 removed the dominant doubled-gradient inflation; residual std/sigma_pp
offset is **~3% on dark-sky frames**, **~8% on twilight frame 001**. Estimator swap is no longer
mandatory for gradient immunity but still separates twilight sky structure from pixel noise.

**DECISION REQUIRED (Milan):**
- Drop sigma_pp, keep sigma_clipped_stats + threshold 3.8 recalibration? (simpler diff; ~3% typical)
- Keep sigma_pp? (handles frame 001 residual; already implemented)

## 0.2 P-10 independent verification (R1/R2)

**Literature/first principles:** fit sky as 2D polynomial to background pixels; subtract full surface
including mean term; large-scale residual p2p should be << input gradient p2p; median preserved.

**Pre-fix bug (origin/main):** `z_s = bg_median - work` (`pipeline.py` on `06ed950`).

**Post-fix:** `z_s = work - bg_median`.

**Test:** `dev/tests/test_preprocess_sky_surface.py::test_order1_prefix_bug_doubles_gradient_p2p_ratio_near_two`
- Reimplements pre-fix sign in test helper (independent of production code).
- **Fails before fix / passes after:** pre-fix ratio **1.99** (1.85-2.15 gate); post-fix ratio **<0.15**.
- Pedestal preserved (atol 6 ADU).

## 0.3 Bundle push

Single commit: P-10 + sigma_pp + `masterstar_dao_threshold_sigma=3.8` + I-12 PM logging + tests.
Not pushed alone (Tranche 3 rule).

## Files changed (Stage 0 commit)
`config.json`, `src_py/config.py`, `dev/validation/params_registry.json`, `src_py/pipeline.py`,
`src_py/vyvar_platesolver.py`, `dev/tests/test_preprocess_sky_surface.py`,
`dev/tests/test_dao_sigma_pp_estimator.py`, `dev/tests/test_pm_correction_logging.py`,
export test fixtures, `docs/VYVAR_DECISIONS.md` (P-10 section).
