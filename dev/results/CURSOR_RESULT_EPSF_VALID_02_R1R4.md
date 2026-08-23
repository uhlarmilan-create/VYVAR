CURSOR RESULT - EPSF-VALID-02 Phase 2 review fixups (R1-R4)

Date: 2026-08-22. HEAD: `93b3194`. Parent: CURSOR_TASK_EPSF_VALID_02_P2.md.
Scope: review findings only. No production model swap. No Part B/D. **STOP** — architect sign-off decides model-swap.

---

## R1 — 134 vs 135 frames (BLOCKING: explain, do not fix)

### Summary

| Path | Count | Enumerator |
|------|------:|------------|
| P1 / anchor / accept backup / aperture guard | **134** | `proc_BO_CVn_Light_*.csv` only (science lights after QC) |
| ACCEPT rerun / `export_per_frame_catalogs` / CSS denominator | **135** | `_iter_fits_recursive(detrended_aligned/lights/NoFilter_60_2)` — **all** `*.fits` |

**Extra frame:** `MASTERSTAR.fits` (stacked reference; `frame_index=134` in `epsf_photometry_job_summary.json`).

This is a **two-path defect**: anchor accounting and ACCEPT/export frame accounting use different enumerators. Not fixed in this task (report only).

### Extra frame detail

| Question | Answer |
|----------|--------|
| Which file? | `Archive/Drafts/draft_000516/detrended_aligned/lights/NoFilter_60_2/MASTERSTAR.fits` ? sidecar `proc_MASTERSTAR.csv` |
| Why did rerun enumerate it? | `export_per_frame_catalogs` sets `files = sorted(_iter_fits_recursive(root))` with no science-light filter (`pipeline.py` ~10874). MASTERSTAR lives in the same aligned-lights directory as science exposures. |
| Proc CSV outside anchor set? | **Yes.** Pre-accept backup (`dev/results/context/session_20260822_epsf_valid_02_accept/proc_backup_pre_accept/`) has **134** `proc_BO_CVn_Light_*.csv` files and **no** `proc_MASTERSTAR.csv`. ACCEPT rerun **created** `proc_MASTERSTAR.csv` as the 135th catalog. |
| QC drop set (134/16) involved? | **No.** The +1 is not a recovered QC frame. Draft 516 aligned science lights = **134** `BO_CVn_Light_*.fits` (150 raw ? 16 QC-dropped = 134 survivor count used by P1/anchor). `MASTERSTAR.fits` is a separate stacked reference co-located under `lights/`; it is outside the 150-science / 16-drop accounting. On disk: 134 Light FITS + 1 MASTERSTAR = 135 FITS total. |

### Path A vs Path B (code)

- **Path A (134):** P1a iterates `proc_BO_CVn_Light_*` only (`dev/results/context/session_20260822_epsf_valid_02_p1/p1a_target_*.csv`, 134 rows). Accept backup and aperture guard glob the same pattern.
- **Path B (135):** `export_per_frame_catalogs` ? `_iter_fits_recursive(root)` includes every FITS under aligned lights, including MASTERSTAR. Science-set filter (`build_epsf_science_set`) applies to **catalog IDs for PSF photometry**, not to the frame list.

### CSS and aperture guard

- CSS target coverage **135/135** counts all proc sidecars with a CSS row, including `proc_MASTERSTAR` (`accept_summary.json`: `n_frames_psf_ok=135`, `total_frames=135`). Science-light-only CSS on Light_* remains **134/134** for full-night stars.
- Aperture guard scoped to **134** Light backup files. `pre_aperture_column_hashes.json` was **`{}`** ? reported `aperture_hash_mismatches=0` was **vacuous** (empty pre-hash). Independent post-hoc column compare (dao_flux/mag/mag_inst/flux) shows widespread diffs vs backup — guard did not substantively run; follow-up belongs outside R1 report scope.

### On-disk verification (2026-08-22)

```
aligned/lights/NoFilter_60_2:  134 BO_CVn_Light_*.fits + 1 MASTERSTAR.fits = 135 FITS
proc sidecars:               134 proc_BO_CVn_Light_*.csv + 1 proc_MASTERSTAR.csv = 135 CSVs
```

Evidence: `dev/results/context/session_20260822_epsf_valid_02_accept/export_result.json`, `epsf_job_summary.json`, `css_target_coverage.csv`.

---

## R2 — `--fast` gate (BLOCKING)

Full `--fast` re-run at HEAD **`93b3194`**:

| Check | Result |
|-------|--------|
| pytest | **1497 passed**, 32 skipped |
| OVERALL | **PASS** |

Evidence: `dev/results/context/session_20260822_epsf_valid_02_r1r4/fast_baseline_stdout.txt`

Prior intermittent failure `test_v3d_fine_scale.py::test_v3d_run_structure` **did not reproduce** in the full sweep. No order-dependent polluter identified; **no fix commit** required.

---

## R3 — Funnel vs pre-registered Part C composition

### a) Funnel tables (516 and 517) with non-var and non-sat rows

Explicit rows added for pre-registered gates:

**Draft 516** (`r3_funnel_516.csv`):

| gate | n |
|------|--:|
| n_csv_input | 3610 |
| n_after_non_variable (VSX/Gaia-var excluded) | 3325 |
| n_after_non_saturated (likely_sat + is_saturated) | 2432 |
| n_after_photometry_ok | 2432 |
| n_after_not_noisy | 2264 |
| n_after_usable | 2264 |
| n_after_zone_linear | 2264 |
| n_after_clean_source_state | 2200 |
| n_after_science_scope | 68 |
| n_after_edge_safe_cutout | 68 |
| n_after_interim_top_n | 68 |
| n_before_isolation | 68 |
| n_after_isolation | **67** |
| n_stars_used (extract_stars) | **67** |

**Draft 517** (`r3_funnel_517.csv`):

| gate | n |
|------|--:|
| n_csv_input | 3606 |
| n_after_non_variable (VSX/Gaia-var excluded) | 3321 |
| n_after_non_saturated (likely_sat + is_saturated) | 2434 |
| … (downstream gates unchanged from F4) … |
| n_after_isolation | **66** |
| n_stars_used (extract_stars) | **66** |

Artifacts: `dev/results/context/session_20260822_epsf_valid_02_r1r4/r3_funnel_516.csv`, `r3_funnel_517.csv`.  
Sandbox: `dev/sandbox/epsf_valid_02_r3_build_census.py`.

### b) Build-star census (67 / 66)

Full lists with role, variability, and saturation flags:

- **516:** `dev/results/context/session_20260822_epsf_valid_02_r1r4/r3_build_stars_516.csv` — **67** stars
- **517:** `dev/results/context/session_20260822_epsf_valid_02_r1r4/r3_build_stars_517.csv` — **66** stars

Columns: `catalog_id`, `role`, `catalog_known_variable`, `vsx_known_variable`, `gaia_dr3_variable_catalog`, `likely_saturated`, `is_saturated`, `zone`, `source_state`, `mag`.

**516 composition:** all 67 = `per_target_comp`. Zero `target`, `check`, `blended`, or `other`. Zero VSX/Gaia-var flags; zero saturation flags.

**517 composition:** all 66 = `per_target_comp`. Same variability/saturation profile (all false).

### c) Pre-registration deviations

| Check | Status |
|-------|--------|
| Active targets (variables) in build set? | **No.** Build stars are per-target comps only; no science-set target IDs in the 67/66 extract_stars lists. |
| Non-variable gate skipped? | **No.** Applied upstream (`3325/3610` for 516; `3321/3606` for 517). |
| Non-saturated gate skipped? | **No.** Applied (`2432/3325` for 516; `2434/3321` for 517). |
| Science-scope gate | Applied (`68` ? isolation ? `67` for 516). Targets in the 333-ID science set are **not** in the build pool because they fail earlier gates (variable / photometry / isolation), not because gates were bypassed. |

**No silent deviation from pre-registered Part C gates.** Named expectation: ePSF build uses stable comps, not active variable targets — consistent with funnel and census.

---

## R4 — Split-half quick delta (informal preview)

Sandbox: `dev/sandbox/epsf_valid_02_r4_split_half.py`  
Models:

- **Production:** `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/masterstar_epsf.fits` (1475-star build)
- **Gated sandbox:** `dev/results/context/session_20260822_epsf_valid_02_f4/sandbox_516_gated_build/masterstar_epsf.fits` (67-star build)

Method: first **10** `BO_CVn_Light_*.fits` science frames; stars with existing `psf_fit_ok=True` in proc CSVs, capped 30/frame; science-set IDs; PSF re-fit only (no ledger, no production writes).

### Aggregate metrics

| Model | n_rows_ok | psf_dao_ratio median | chi2 median |
|-------|----------:|---------------------:|------------:|
| Production (1475) | 300 | 1.327 | 21.7 |
| Gated sandbox (67) | 300 | 0.997 | 1.48 |

**Per-star delta (gated ? prod PSF mag, mmag):**

- Median across stars: **?314.8 mmag** (gated fainter / higher mag on matched stars)
- RMS (median of per-star RMS): **86.7 mmag**
- Matched star-frame pairs (both ok): **300**

Artifacts:

- `dev/results/context/session_20260822_epsf_valid_02_r1r4/r4_split_half_summary.json`
- `r4_prod_psf_sample.csv`, `r4_gated_psf_sample.csv`, `r4_per_star_delta.csv`

**Caveat:** informal preview only; large systematic offset expected when comparing models built from different star pools. Not gate evidence for model swap.

---

## Gate status

| Gate | HEAD | Result | Evidence |
|------|------|--------|----------|
| `--fast` | `93b3194` | **OVERALL PASS** | `dev/results/context/session_20260822_epsf_valid_02_r1r4/fast_baseline_stdout.txt` |
| `--full` recut | `93b3194` | **OVERALL PASS** (from ACCEPT session) | `dev/results/context/session_20260822_epsf_valid_02_accept/full_baseline_stdout.txt` |
| core photometry SHA | — | `9902d918e9f48e0f…` n=121 | full baseline |
| extended photometry SHA | — | `472bc9e4446f13a8…` n=179 | full baseline |

---

## Files changed

| File | Action |
|------|--------|
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_R1R4.md` | created (this deliverable) |
| `dev/sandbox/epsf_valid_02_r3_build_census.py` | sandbox (R3; untracked) |
| `dev/sandbox/epsf_valid_02_r4_split_half.py` | sandbox (R4; untracked) |
| `dev/results/context/session_20260822_epsf_valid_02_r1r4/*` | R1R4 artifacts |

No production code changes. No commits. **STOP** — awaiting architect sign-off on model-swap question.
