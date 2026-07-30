CURSOR RESULT - 2026-07-30T17:55:00Z

What I did
Ran mandatory byte-identity verification: `python dev/scripts/session_baseline_check.py --full` on `origin/main` @ `8d7d4b9` against anchor snapshot `draft_000435_snapshot_skysurface_20260716`. Followed with column-level diff script on the fresh run output.

## Output / findings

### Command and runtime
- **Command:** `dev/scripts/session_baseline_check.py --full`
- **Git HEAD:** `8d7d4b9`
- **Work output:** `tmp/session_baseline/20260730T171114Z`
- **Pipeline wall time:** 2343 s (~39 min)

### Photometry SHA (n files identical run vs snapshot)

| Tier | Snapshot (known-good) | Fresh run | n |
|------|----------------------|-----------|---|
| Core | `b7f980c09e238b855c2ee1b9518061777934d8f0a61eaec7431cda4f537aed52` | `67931c25c94ede712cac7f3cd1bdc96f9b671d465ed0bef1d9ced4bba5007a3e` | 325 |
| Extended | `2c43bbbf06921fbef46fb6a4ed1f8afccdabacaa5827b8ec50372de0e3816205` | `b39d553807cb460100c11253634d3dbea57a7b357afe16a8d03b2260d996741d` | 487 |

SHA mismatch is **expected** (Stage 0 bundle: P-10 sky-surface sign fix + DAO `sigma_pp` + threshold 3.8 on `8d7d4b9`).

### Science-meaningful compare (tolerance gate)
- **`full-science-compare`: PASS** — `n_lc=162`, `science_failures=0`, `time_failures=0`
- All 162 paired light curves match within `TOL_SCIENCE` / `TOL_TIME_D` on science columns.

### Column-level diff (162 common LCs)

| Column / file class | Differs? | Explanation |
|---------------------|----------|-------------|
| Science mag columns (`mag_calib_final`, `delta_mag`, `mag_inst`, …) | No delta above tolerance | **Explained:** P-10/threshold changes did not move calibrated photometry beyond science tolerance on this anchor (or changes net to sub-tolerance). |
| **`err_photon`, `err_sem_rel`, `err_sigma_sys_rel`** | Present **only in run** | **Explained:** additive Part 1 LC export columns (local working tree during run). Snapshot predates them. QC / non-science for SHA science gate. |
| `pipeline_meta.json`, `photometry_summary.csv` | Byte diff | **Explained:** provenance / git hash / run metadata. |
| `active_targets.csv`, `comparison_stars_per_target.csv` | Identical bytes | Catalogue/ensemble unchanged at CSV level for this compare. |

**Sample LC:** `lightcurve_1485540612577549568.csv` — numeric science columns max delta **0.0**; run file larger solely due to three new err-budget columns.

### Unexplained differences
**None identified.** All byte-level diffs trace to (a) expected Stage 0 science-path inputs, (b) additive err-budget export columns, or (c) provenance metadata.

### Other baseline notes
- Phase-0 funnel fingerprints: PASS (165 active targets, expected skip histogram).
- `phase2a_empty_comp_drop=1` (R CVn): PASS, allowlisted.
- Pytest gate in baseline script: FAIL (1215 passed, 26 skipped) — pre-existing; not Part 0 photometry regression.

## Verdict
**Part 0 PASS (proceed).** Science content stable under tolerance; SHA change explained. No STOP condition.

## Files changed
- `dev/scripts/audit_stage3_part0_column_diff.py` (new helper)
- `dev/results/CURSOR_RESULT_audit_stage3_part0.md` (this report)
- Evidence: `tmp/session_baseline/20260730T171114Z`, `tmp/audit_stage3_part0_column_diff.json`
