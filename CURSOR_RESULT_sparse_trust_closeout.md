CURSOR RESULT -- SPARSE-TRUST-CLOSEOUT -- 2026-07-14

What I did
Pushed the full SPARSE-TRUST local chain (Milan-authorized). Answered SS Cam stability-RED
branch question, full S3 tuple table, and photon-derivation audit. No band logic bug; no S3
re-run required.

## Part 0 -- Push

- Pushed: `7ed7459..43ea830` to `origin/main` (12 commits).
- HEAD: `43ea830b3604cf78e307024586d3a9c9122f84d1`
- `session_baseline_check.py --fast`: **PASS** (814 passed, 15 skipped on pushed HEAD).
- Post-closeout commits from this task: **local only** (not pushed).

## Part 1 -- SS Cam stability-RED branch check

### 1. Numeric x2_pair_mag2 (SS Cam, S3 artifacts)
`x2_pair_mag2 = 0.00029601` mag^2 (from `s3_verdict_tuples.csv` / sidecar).

### 2. trust_band() code path
Spec RED via stability: `p < 0.001 AND x2_pair > X2_RED` where `X2_RED = (0.02 mag)^2 = 4e-4`.

SS Cam inputs: `p_stab = 0.0`, `x2_pair = 0.00029601`.
- Condition 1 (p < 0.001): **MET**
- Condition 2 (x2 > 4e-4): **NOT MET** (0.00029601 < 0.0004)

Code path: `sparse_trust_core.py` lines 353-355 evaluate the stability-RED branch; branch
is entered but returns false on x2, so execution continues to YELLOW via `R_hi_exceeds_T_green`
(lines 366-368).

Constructed RED-via-stability test: `tests/test_sparse_trust_core.py`
`test_trust_band_green_red_yellow` (p=0.0, x2=0.0005 -> RED `comp_pair_unstable`).
SS Cam regression case in same test (p=0.0, x2=0.00029601 -> YELLOW, no `comp_pair_unstable`).

**No logic bug.** No fix commit; S3 numbers stand.

### 3. SS Cam band per spec (numbers filled in)
- R = 2.008, [R_lo, R_hi] = [1.224, 3.886] -> R_hi > T_green=1.5 -> not GREEN
- R_lo < T_red=4.0 -> not RED via R
- p=0.0 < 0.001 but x2_pair=0.00029601 <= X2_RED=0.0004 -> not RED via stability
- **Band: YELLOW** (flag: `R_hi_exceeds_T_green`)
- production_lc_err chi2 = 21.38 (baseline row unchanged)

SS Cam band **OPEN** for Milan final confirmation on top of this evidence.

## Part 2 -- Full S3 tuple table (r_60_4, external K)

| target | K_id | k_source | k_colour_offset | k_caveat | n | N | R [lo, hi] | R_det | p_stab | x2_pair_mag2 | flags | band |
|--------|------|----------|-----------------|----------|---|----|------------|-------|--------|--------------|-------|------|
| 1111749368289526912 | 1112110935816253440 | comp_pool_external | 0.104095 | 0 | 2 | 25 | 2.055 [1.253, 3.977] | 2.027 | 0.0 | 0.00029601 | R_hi_exceeds_T_green | YELLOW |
| 1112113066119992064 (SS Cam) | 1112110935816253440 | comp_pool_external | 0.104095 | 0 | 2 | 25 | 2.008 [1.224, 3.886] | 1.991 | 0.0 | 0.00029601 | R_hi_exceeds_T_green | YELLOW |
| 1112127291051695744 | 1111749157833870208 | comp_pool_external | 0.163477 | 0 | 2 | 25 | 2.944 [1.795, 5.698] | 2.914 | 0.999886 | 0.0 | R_hi_exceeds_T_green | YELLOW |
| 1112130898824233216 | 1112110935816253440 | comp_pool_external | 0.104095 | 0 | 2 | 25 | 2.006 [1.223, 3.881] | 1.987 | 0.0 | 0.00029601 | R_hi_exceeds_T_green | YELLOW |
| 1112121175018240768 | 1111749157833870208 | comp_pool_external | 0.163477 | 0 | 2 | 25 | 2.929 [1.786, 5.669] | 2.903 | 0.999886 | 0.0 | R_hi_exceeds_T_green | YELLOW |
| 1111931646701447424 | 1111749157833870208 | comp_pool_external | 0.163477 | 0 | 2 | 25 | 2.944 [1.795, 5.698] | 2.914 | 0.999886 | 0.0 | R_hi_exceeds_T_green | YELLOW |

SS Cam production_lc_err chi2 = 21.38. Source: `tmp/sparse_trust_validation/s3_verdict_tuples.csv`.

## Part 3 -- Photon derivation audit

### 1. Formula implemented
When proc row lacks `err`, `build_comp_photon_mag_from_frames()` calls production
`_photometric_error_with_bkg_mode()` (empirical path):

    variance = F / g + sigma_bkg_ap^2
    err_rel  = sqrt(variance) / F

then `photon_mag = MAG_ERR_SCALE * err_rel`. This includes the source photon term F/g, not
bkg-only. Matches `photometry_core.py` lines 1053-1056 (same function used at export
lines 2139-2147). **No formula fix required.**

### 2. Unit test
`tests/test_sparse_external_k.py::test_build_comp_photon_matches_production_err` -- derived
relative err agrees with production to <= 1e-9 rel; row-with-`err` column uses stored err.

### 3. Why err missing from r_60_4 proc CSVs
Proc CSVs on disk never store per-star `err`: `PROC_STORE_COLS` in `proc_frame_store.py`
projects columns to disk and **does not include `err`**. Production computes `err` in-memory
during aperture export (`photometry_core.py` ~2139-2148) before LC assembly; the 426 regen
is not defective -- this is the established store-projection design (same on draft_424).
Flag: PROC_STORE_COLS-class finding; sidecar derivation is the correct consumer path.

## Errors (if any)
None.

## Files changed
- tests/test_sparse_trust_core.py (stability-RED + SS Cam branch tests)
- tests/test_sparse_external_k.py (photon derivation agreement test)
- CURSOR_RESULT_sparse_trust_closeout.md
- docs/VYVAR_STATE.md (push snapshot)

## pytest
812 passed, 11 skipped (`-m "not slow"`); +2 new tests (trust_band branch + photon audit).
session_baseline_check --fast on pushed HEAD: 814 passed, 15 skipped.
ruff clean on touched files.
