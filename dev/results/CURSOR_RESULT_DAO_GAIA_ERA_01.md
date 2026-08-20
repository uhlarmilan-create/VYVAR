CURSOR RESULT - 2026-08-19 (DAO-GAIA-ERA-01, A-fix 4)

What I did
Implemented diagnostic-mode calibration measurement (A-fix 4): wide-open
DAO<->Gaia crossmatch and SNR-only seed truth, tail-corrected core p95,
full diagnostic distributions in certificate. Re-emitted draft 516
certificate. **Validation gate PASS.** Part C unblocked.

## A-fix 4 ù Diagnostic measurement mode (DONE)

DECISIONS principle (final): sample WITHOUT selection on the gated
quantity, measured in **diagnostic mode** with wide-open tolerances.

| Population | Method | Diagnostic radius / bound |
|------------|--------|---------------------------|
| detection_identity | Fresh pass1 DAO + greedy assign at R_diag; pass2 **accepted** at centroid cap 3.0 px | R_diag = max(10, 2ùFWHM) = **10.39 px** |
| seed_centroid | Unowned G?15: pass2 accepted + forced-seed SNR?min (centroid gate not applied) | centroid measure cap **3.0 px** |

Tail handling: `random_match_scale = sqrt(1/(??))`; core = separations
? **4.0 px** astrometric envelope; p95 on core. Method recorded in
certificate `derived.diagnostic.*.tail_method`.

## 516 diagnostic populations (measured)

| Population | n_raw | p95_raw | n_core | **p95_core** | ? derived |
|------------|-------|---------|--------|--------------|-----------|
| detection_identity | 2831 | 1.39 | 2824 | **1.36** | match **2.5 px** |
| seed_centroid | 374 | 2.45 | 374 | **2.45** | centroid **2.5 px** |

Hand-validated reference: match **3.0 px**, centroid **2.0 px** (? 4.5/4.0).

Honest diagnostic on live 516 yields **2.5/2.5 px** (identity p95 below
GAIA-00 reference 1.78 ù current pass1 DAO stack, not gate survivors).

## Validation gate ù **PASS**

| Check | Result |
|-------|--------|
| Status | **PASS** |
| max_regression_pp | **0.00063** |
| G2 empty-sky | **PASS** (all frames) |
| Empty-sky audit (derived 2.5/2.5) | pass2 0.14%, seed 0.27% ù **PASS** |

## Certificate

- `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/dao_gaia_calibration.json` ù **PASS**
- Includes `derived.diagnostic` block (n, p50, p95, n_raw, tail_estimate, tail_method, measurement_mode)
- Context: `dev/results/context/session_20260819_era01/era01_backfill_calibration_afix4.json`

## Part C - **STOP** (executed 2026-08-19)

Full rebuild ran (~3209 s). Certificate **PASS** on production MS path
(2.5/2.5, sigma 4.5/4.0). **L1 PASS; L2-L6 DEVIATE.** Overall STOP per
spec (no recut). Live 516 restored to 477dc8cf snapshot.

Detail: `dev/results/CURSOR_RESULT_DAO_GAIA_ERA_01_PART_C_STOP.md`
Raw: `dev/results/context/session_20260819_era01_part_c/part_c_rebuild_l1_l6.json`

## Part 0 ù SHA guard (unchanged GREEN)

| Check | Result |
|-------|--------|
| `compute_photometry_sha(draft_000516)` | **477dc8cf** n=97 PASS |

## Tests

`dev/tests/test_dao_gaia_calibration.py` ù **6/6 PASS** (includes slow
validation gate integration on live 516).

## Files changed

- `src_py/dao_gaia_calibration.py` ù A-fix 4 diagnostic mode + certificate audit fields
- `dev/tests/test_dao_gaia_calibration.py` ù updated for diagnostic path

Push not authorized.
