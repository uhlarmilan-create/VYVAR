CURSOR RESULT — 2026-07-09

What I did
Committed diagnostic sigma-budget + sparse-comp modules (`sigma_budget.py`, `scripts/chi2_sigma_gate.py`,
`scripts/select_constant_calibrators.py`, `scripts/sparse_comp_diag.py`, `tests/test_sigma_budget.py`).
Ran archive diagnostics on drafts 424/425/426; updated ROADMAP/STATE/JOURNAL. No production wiring.

## Output / findings

### Rig parameters used

| Draft | Setup | D (m) | alt (m) | exp (s) | Source |
|-------|-------|-------|---------|---------|--------|
| 424 | NoFilter_60_2 | 0.072 | 0.0 | 60 | TELESCOPE.DIAMETER (Carl-Zeiss 200mm); pipeline_meta alt; setup name |
| 425 | B/R/V_20_2 | 0.300 | 355.5 | 20 | TELESCOPE.DIAMETER (DDT 300/1200); pipeline_meta; setup name |
| 426 | g/i/r_60_4 etc. | 0.300 | 275 (fallback) | 60/70/90 | TELESCOPE + LOCATION fallbacks per `resolve_rig_scintillation_params` |

### Part 3 — Calibrators (draft_424 NoFilter_60_2)

Anchor target (GREEN): `1496278752372040832`. LOO path: `check_star_kmag.compute_check_ensemble_mag_calib`.
8 calibrators (G coverage 9.29–13.23):

| catalog_id | G | N frames | comp_rms |
|------------|---|----------|----------|
| 1497674651102612992 | 9.287 | 139 | 0.00595 |
| 1497894828305963776 | 10.472 | 139 | 0.00649 |
| 1485927228353411840 | 11.153 | 139 | 0.01415 |
| 1499195550562121728 | 11.846 | 139 | 0.02567 |
| 1485545044983800576 | 12.352 | 139 | 0.02555 |
| 1497564459421863680 | 12.676 | 139 | 0.03489 |
| 1498779694648503040 | 12.902 | 139 | 0.05624 |
| 1499178645570862336 | 13.233 | 139 | 0.04996 |

**draft_425** (B_20_2, R_20_2, V_20_2): zero calibrators — no GREEN anchor in `photometry_summary.csv`.

### Chi2/dof per sigma variant (draft_424, reduced chi2 vs weighted mean)

| G | howell_only | howell_scint_full | howell_scint_fresid (f=1.0) |
|---|-------------|-------------------|----------------------------|
| 9.29 | 12.26 [10.72, 13.62] | 4.97 [4.35, 5.52] | 4.97 |
| 10.47 | 2.13 [1.86, 2.33] | 1.75 [1.52, 1.91] | 1.75 |
| 11.15 | 1.92 [1.63, 2.16] | 1.70 [1.45, 1.92] | 1.70 |
| 11.85 | 1.16 [1.05, 1.26] | 1.13 [1.02, 1.23] | 1.13 |
| 12.35 | 1.15 [1.00, 1.28] | 1.13 [0.98, 1.25] | 1.13 |
| 12.68 | 0.83 [0.73, 0.91] | 0.82 [0.72, 0.91] | 0.82 |
| 12.90 | 1.04 [0.91, 1.14] | 1.03 [0.90, 1.13] | 1.03 |
| 13.23 | 0.55 [0.48, 0.61] | 0.55 [0.47, 0.60] | 0.55 |

Bootstrap CI in brackets. Baseline ~5.0 h, N=139 frames per calibrator.

**f_resid ensemble fit (draft_424 NoFilter_60_2):** f_resid=1.0, median chi2/dof=1.129, IQR spread=0.733.

**Plot:** `tmp/sigma_budget/chi2_vs_g_draft000424_NoFilter_60_2.png`

**JSON:** `tmp/sigma_budget/calibrator_chi2_summary.json`

### Part 4 — SS Cam sparse-comp decomposition (draft_426)

V0611 (`1112127291051695744`) on disk: **yes** (g_60_4). z_90_4 SS Cam: **not available** (no LC/proc).

| Setup | inter_star_rms | temporal_rms | headline_rms | ratio temp/headline | check scatter | N | cancel factor | comp_rms median |
|-------|----------------|--------------|--------------|---------------------|---------------|---|---------------|-----------------|
| g_60_4 SS Cam | 0.292 | 0.011 | 0.465 | 0.023 | 0.0369 | 23 | 0.079 | 0.0147 |
| r_60_4 SS Cam | 0.221 | 0.012 | 0.438 | 0.028 | 0.0233 | 25 | 0.053 | 0.0092 |
| i_70_4 SS Cam | 0.153 | 0.008 | 0.413 | 0.019 | 0.0113 | 25 | 0.027 | 0.0123 |
| g_60_4 V0611 | 0.935 | 0.016 | 1.002 | 0.016 | 0.0231 | 23 | 0.023 | 0.0113 |

Check-star chi2/dof (sigma variants) — SS Cam r_60_4 check `1112117498526276864`: howell_only=0.81,
howell_scint_full=0.80 (N=25, baseline 2.18 h). g_60_4 check: howell_only=0.16 (N=23).

### Healthy-field locus comparison

| Case | inter_star_rms | temporal_rms | headline_rms | check scatter | cancel factor |
|------|----------------|--------------|--------------|---------------|---------------|
| draft_424 NoFilter_60_2 (GREEN anchor) | 0.289 | 0.021 | 0.290 | 0.0123 (N=139) | 0.042 |
| draft_425 V_20_2 | — | — | — | unavailable (no LC on disk for target 458344517406391040) | — |

**JSON:** `tmp/sigma_budget/sparse_comp_diag.json`

### Deviations from spec

1. **min_frames=200:** archive max comp_n_frames on draft_424 is 139; script relaxed to 120 (`frame_gate_relaxed=true`). draft_425 relaxed to 10; still zero calibrators (no GREEN anchor).
2. **GREEN trust filter:** calibrators drawn from comp pool on GREEN anchor target only; draft_425 has no GREEN rows in `photometry_summary.csv`.
3. **TELESCOPE.DIAMETER draft_424:** DB value 72 mm ? 0.072 m (name "Carl-Zeiss 200mm"); spec rig D=0.2 m not applied (DB took precedence).
4. **Altitude draft_424:** pipeline_meta `alt_m=0.0` used; LOCATION fallback not triggered (0 is finite).
5. **f_resid CI:** grid search reports point estimate only; bootstrap CI on f_resid not implemented.
6. **Spec sandbox path:** modules committed at repo root / `scripts/` per task (not `tmp/phase12/`).
7. **Check-scatter bootstrap CI:** `scatter_ci_lo/hi` in JSON are chi2/dof bootstrap quantiles (not scatter-mag CI) — naming mismatch in `sparse_comp_diag.py`.

## Errors (if any)

- First `session_baseline_check.py` run: OVERALL FAIL on `git-staged` (transient; re-run recommended after clean tree).
- `session_baseline_check.py` has no `--fast` flag (default mode is fast).

## Files changed

- `sigma_budget.py` (new)
- `scripts/chi2_sigma_gate.py` (new)
- `scripts/select_constant_calibrators.py` (new)
- `scripts/sparse_comp_diag.py` (new)
- `tests/test_sigma_budget.py` (new)
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_sigma_sparse_diag.md` (this file)

**Gates:** pytest 666 passed, 15 skipped; ruff clean on new files; no production-path changes.

**Not committed** (per task: separate step / no push).
