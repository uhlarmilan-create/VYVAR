CURSOR RESULT ù 2026-07-09 (SIGMA-A2)

What I did
Part 0: committed and pushed SIGMA-BUDGET-A diagnostics (e2c9466). Part A2: rig parameter fixes,
sigma_floor variant, saturation margin check, archive rerun, docs update, second push.

## Output / findings

### Origin hashes

| Step | Hash | Message |
|------|------|---------|
| Part 0 push | `e2c9466` | Add sigma-budget Phase A and sparse-comp diagnostic harness (sandbox) |
| Part 5 push | `0b901aa` | SIGMA-A2 rig fixes, sigma_floor variant, rerun |

### Rig parameters (corrected, draft_424 NoFilter_60_2)

| Parameter | A-run | A2-run | Source |
|-----------|-------|--------|--------|
| D | 0.072 m | **0.200 m** | TELESCOPE.DIAMETER after `fix_telescope_diameter.py --apply` (72?200 mm) |
| alt | 0.0 m | **275.0 m** | pipeline_meta alt=0 ignored; LOCATION Jirny |
| exposure | 60 s | 60 s | setup name |

DB maintenance: `scripts/fix_telescope_diameter.py` (dry-run + `--apply` auditable).

### Joint fit (f_resid, sigma_floor) ù draft_424 calibrator ensemble

| Parameter | Value | Bootstrap CI (16ù84%) |
|-----------|-------|----------------------|
| f_resid | 0.74 | [0.0, 1.0] |
| sigma_floor | 10.5 mmag (0.0105 mag) | [9.5, 11.0] mmag |
| median chi2/dof | 1.000 | ù |
| per-star IQR (floor variant) | 0.158 | ù |

f_resid not pinned to edge. sigma_floor magnitude: 10.5 mmag (bookkeeping vs ~1ù5 mmag PRNU/flat-residual scale: number only).

### Chi2/dof per calibrator ù all four variants (draft_424, corrected rig)

| G | howell_only | howell_scint_full | howell_scint_fresid (f=0.74) | howell_scint_fresid_floor |
|---|-------------|-------------------|------------------------------|---------------------------|
| 9.29 | 12.26 | 9.07 | 10.28 | 1.23 |
| 10.47 | 2.13 | 2.02 | 2.07 | 0.90 |
| 11.15 | 1.92 | 1.86 | 1.89 | 1.08 |
| 11.85 | 1.16 | 1.16 | 1.16 | 0.99 |
| 12.35 | 1.15 | 1.14 | 1.15 | 1.01 |
| 12.68 | 0.83 | 0.83 | 0.83 | 0.78 |
| 12.90 | 1.04 | 1.03 | 1.03 | 0.97 |
| 13.23 | 0.55 | 0.55 | 0.55 | 0.52 |

Per-variant ensemble IQR: howell_only=0.982, howell_scint_full=0.916, howell_scint_fresid=0.945, howell_scint_fresid_floor=0.158.

Plot: `tmp/sigma_budget/chi2_vs_g_draft000424_NoFilter_60_2.png`
JSON: `tmp/sigma_budget/calibrator_chi2_summary.json`

### G9.3 calibrator saturation margin (1497674651102612992)

| Stat | Value |
|------|-------|
| N frames | 139 |
| fill p50 | 0.410 |
| fill p95 | 0.489 |
| fill max | 0.526 |
| peak_max_adu max | 29277 |
| saturate_limit_adu_85pct median | 55705 |
| likely_saturated_frames (fill?1) | 0 |

**g93_saturation_flagged:** false (threshold 0.85). No with/without-G9.3 exclusion table (calibrator retained).

### SS Cam / V0611 check-star chi2 (chi2-only rerun, corrected rig where D-dependent)

Updated rows in `tmp/sigma_budget/sparse_comp_diag.json` (4 cases: SS Cam g/i/r, V0611 g). Decomposition numbers unchanged from A-run.

### draft_425 trust reasons (K2 validation draft)

No GREEN rows in any setup `photometry_summary.csv`:

| Setup | trust values | n_rows |
|-------|--------------|--------|
| V_20_2 | YELLOW=96, RED=18 | 114 |
| B_20_2 | YELLOW=354, RED=17 | 371 |
| R_20_2 | YELLOW=352, RED=20 | 372 |

JSON: `tmp/sigma_budget/draft_425_trust.json`

### Deviations

1. min_frames still relaxed 200?120 (archive max 139 comp_n_frames).
2. f_resid bootstrap CI spans full [0,1] grid (wide; 8 calibrators).
3. sigma_floor grid upper bound 20 mmag (0.02 mag); fit landed at 10.5 mmag.
4. DB fix applied to local Milan dev DB only (not in git); script is committed.
5. G9.3 not excluded ù saturation margin well below 0.85 pipeline fraction.

## Errors (if any)

None.

## Files changed

- `sigma_budget.py` ù alt<=0 guard, `combine_sigma_mag_quadrature`, floor variant constant
- `scripts/chi2_sigma_gate.py` ù joint fit + bootstrap CIs, saturation margin, 4th variant
- `scripts/select_constant_calibrators.py` ù A2 rerun outputs, draft_425 trust probe
- `scripts/sparse_comp_diag.py` ù chi2_ci/scatter_mag_ci naming, `--chi2-only`
- `scripts/fix_telescope_diameter.py` (new)
- `tests/test_sigma_a2.py` (new)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`

**Gates:** pytest 669 passed, 15 skipped; ruff clean; no production-path changes.
