CURSOR RESULT - ANCHOR-516-03

Date: 2026-08-18
Task: fix headless empirical ERR, name remaining deltas, then proceed toward rebuild/re-cut only if A-B are green.
Push: NOT authorized.

What I did
Started Part A and B. Verified the known leftover tracked diffs are documentation/result-stamp leftovers and not science-path edits. Implemented the headless empirical ERR cache/fail-loud fix and the stale `comp_qa` sidecar cleanup as separable code changes with focused tests.

## Output / findings

### Verified leftover tracked diffs left untouched

Verified by `git diff`:

- `dev/results/CURSOR_RESULT_PUSH_AUTH_20260817.md`
- `dev/results/CURSOR_RESULT_WIDE_ERR_04.md`
- `dev/results/WIDE_ERR_04_summary.json`

Finding: all three diffs are 2026-08-17-era push/result amendments or summary cleanup; no science-path code or draft data changed. They are intentionally left untouched and must be excluded from all commits in this task.

### Fast baseline

- `session_baseline_check.py --fast` @ `4a65675`: OVERALL PASS
- pytest: `1453 passed, 28 skipped`
- Follow-up commits after first `--fast` misses: `790291a`, `08cf443`, `4a65675`

### Part D -- `--full` measurement (do not recut 435 from this run)

`session_baseline_check.py --full` @ `4a65675`: OVERALL FAIL after 2540 s pipeline.

pytest still PASS (1453 / 28 skipped). The pipeline itself completed:
`tmp/session_baseline/20260818T102916Z`.

This is **not** an ERR-only SHA bump on the 435 golden:

| gate | observed | expected / snapshot |
| --- | --- | --- |
| `skip_photometry_true` | 137 | 2 |
| skip reasons | `vsx_type_out_of_scope` 82, `zone_noise` 53, `zone_flag` 2 | `{"": 162, no_comps: 1, zone_flag: 2}` |
| SHA `n` (core) | 57 (28 LCs) | snapshot 333 files |
| run core SHA | `70349514...` | snapshot `3d26f469...` |
| snapshot vs hardcoded expected | `3d26f469...` | `5bccd85a...` (already stale before this task) |

`--full` photometers **live `draft_000435`**, not the frozen snapshot inputs. Current `config.json` has `vsx_out_of_scope_types=["ROT"]`, which masks 82 targets. The 53 `zone_noise` skips match the expected noisy-zone census and are current TARGET-DEPTH / PFS-SEMANTICS policy; the 435 golden still treated those as photometry-ok.

Adopting this run as the new 435 / P1 golden would collapse the anchor from 333 SHA files to 28 LCs. That is not a 516 recut. The 516 product SHA **`be6191e0`** (48 LCs, MAG identical to `de6f7c8`, empirical ERR) remains the candidate canonical product.

P1 golden (`VL-P1-GOLD`, `draft_000435_p1mini`) is the same 435 lineage and was not recut.

### Separable commit A - Part A fail-loud empirical ERR fix

Commit: `96aa0d6`

Files changed:

- `src_py/photometry_core.py`
  - Added named Phase-2A proc-column requirements map.
  - Derived headless `_needed_cols_2a` from that requirements union instead of a hand-maintained list.
  - Added empirical ERR input guard `INV-ERR-MODE-01`: when Phase 2A runs with `err_background_mode=empirical` and `sigma_bkg_ap` is missing/NaN, raise instead of silently falling through to Howell.
  - Headless cache now explicitly carries empirical/provenance inputs including `sigma_bkg_ap`, `err_bkg_source`, and `sky_annulus_r_out_px`.
- `dev/tests/test_err_background_empirical.py`
  - Added coverage for the Phase-2A cache column contract.
  - Added fail-loud test for missing empirical `sigma_bkg_ap`.
  - Added projected-cache vs full-row equivalence test for empirical ERR under Phase 2A.

### Separable commit B - stale `comp_qa` cleanup

Commit: `1aa744c`

Files changed:

- `src_py/comp_qa_core.py`
  - `write_comp_qa_artifacts()` now removes stale `lightcurves/comp_qa_*.json` sidecars for targets not present in the current QA result set.
- `dev/tests/test_comp_qa_pool_guard.py`
  - Added fire-proof that stale `comp_qa_*.json` sidecars are removed while current ones are kept and written.

### Targeted verification run

- `python -m pytest dev/tests/test_err_background_empirical.py dev/tests/test_comp_qa_pool_guard.py dev/tests/test_proc_frame_store.py -q`
- Result: `31 passed`

### Separable commit C - sat-limit one-authority fix

Commit: `615ddda`

Measurement before fix on current tip (`d5ef039`, fresh 516 files):

- Writer of `55704.75`: `pipeline._annotate_masterstars_flux_zones()` via
  `saturate_limit_fraction=0.85`, producing `saturate_limit_adu_85pct`.
- Live science-path gates consuming the 0.85 authority:
  - `pipeline.select_comparison_stars_spatial_grid()` via
    `photometry_ok` / `likely_saturated` from fresh `masterstars_full_match.csv`.
  - `comp_selection_per_target._accumulate_per_frame_comp_metrics()` via
    per-frame `saturate_limit_adu_85pct` when counting over-threshold frames for
    Phase 1 comp rejection.
- Additional non-current or diagnostic consumers observed:
  - `comp_pool_rms.py` (pool RMS diagnostic path)
  - `psf_photometry.py` (PSF gated path)
  - UI/report readers (`ui_photometry_quality.py`, `sat_diag.py` sidecar/reporting)

Direct measurement on fresh 516 output:

- `0.80` threshold (`52428.0`) would mark **24** stars over the peak test.
- `0.85` threshold (`55704.75`) marked **23** stars.
- The one star in the `(52428.0, 55704.75]` band was
  `1497853802778923392` with `peak_max_adu=54581.773438`, and it was
  **admitted into fresh `comparison_stars.csv`** under the 0.85 path.

Conclusion: the 0.85 value was a live science gate, not metadata-only, so it
was corrected to the single INV-SAT-LIMIT authority.

Files changed:

- `src_py/config.py`
  - Default `saturate_limit_fraction` changed `0.85 -> 0.80`.
- `src_py/pipeline.py`
  - `_annotate_masterstars_flux_zones()` default now uses
    `SAT_LIMIT_NO_KNEE_FRAC` instead of `0.85`.
- `src_py/sat_diag.py`
  - SAT-DIAG default saturation/linearity threshold constants aligned to `0.80`.
  - `likely_saturated_threshold_adu()` now uses the shared constant.
- `src_py/comp_selection_per_target.py`
  - Phase 1 admission fallback literals aligned to `0.80`.
- `src_py/psf_photometry.py`
  - Saturated-core skip threshold aligned to `0.80`.
- `dev/tests/test_batch_e_recut.py`
  - Updated default-threshold proof to `0.70 / 0.80`.
- `dev/tests/test_sat_diag.py`
  - Updated default unresolved-threshold expectations to `52428.0`.
- `dev/tests/test_masterstar_zone_classifier.py`
  - Added fire-proof that a resolved 16-bit clip still uses `52428.0`.

Targeted verification run:

- `python -m pytest dev/tests/test_masterstar_zone_classifier.py dev/tests/test_pfs_semantics_01.py dev/tests/test_batch_e_recut.py dev/tests/test_sat_diag.py -q`
- Result: `28 passed`

### B4 ct_n_comp mechanism (measurement)

Evidence:

- `draft_000515/platesolve/NoFilter_60_2/comparison_stars.csv`
  - rows: `2374`
  - mtime: `2026-08-16T13:19:37`
  - `saturate_limit_adu_85pct`: missing on all rows
  - 17 IDs present that are saturated in the newer 515 `masterstars_full_match.csv`
- `draft_000515/platesolve/NoFilter_60_2/masterstars_full_match.csv`
  - rows: `3621`
  - mtime: `2026-08-17T09:16:35`
  - same 17 IDs are `zone=saturated`, `is_saturated=True`,
    `likely_saturated=True`, `is_usable=False`, `saturate_limit_adu_85pct=52428.0`
- `draft_000515/platesolve/NoFilter_60_2/photometry/pipeline_meta.json`
  - Phase 1 stage stamp: `2026-08-17T12:22:53+00:00`
  - Therefore the current 515 `comparison_stars.csv` predates the later 515
    Phase 1 rerun instead of being regenerated by it.
- `draft_000516/platesolve/NoFilter_60_2/comparison_stars.csv`
  - rows: `2357`
  - mtime: `2026-08-17T22:26:44`
  - fresh file carries `saturate_limit_adu_85pct=55704.75` on all rows
  - contains none of those 17 stale IDs
- `draft_000516/platesolve/NoFilter_60_2/photometry/pipeline_meta.json`
  - Phase 1 stage stamp: `2026-08-17T21:51:00+00:00`
  - The fresh 516 pool file was written after that run and reflects the current
    masterstar saturation classification.

Interpretation:

- The `ct_n_comp` delta `2363 vs 2346` tracks a **stale 515 `comparison_stars.csv`**
  pool, not a fresh rerun of the same Phase 1 state.
- The stale 515 pool was written by an earlier `2026-08-16` run and survived the
  later `2026-08-17` Phase 1 / Phase 2A activity recorded in `pipeline_meta.json`.
- It predates the later 515 `masterstars_full_match.csv` saturation
  reclassification and preserved 17 now-saturated candidates as usable.
- A fresh Phase 1 output **can** regenerate `comparison_stars.csv` (516 proves
  that), but the 515 provenance shows it did **not** do so on the later rerun in
  that draft tree. This is a history-dependence defect of the same class as stale
  `comp_qa`.

Answer to the task question:

- On **this input**, `ct_n_comp` is count-metadata-only: prior 516-02 measurement
  already showed `ct_c1` and MAG unchanged while only `ct_n_comp` drifted.
- In **general**, it can move `ct_c1`, because `fit_color_term_c1()` consumes the
  `comparison_stars.csv` pool directly; changing pool membership changes the fit
  inputs even when this specific night happened to leave `ct_c1` numerically
  unchanged.

### Part C - 516 rebuild on fixed tip `615ddda`

Harness: `tmp/anchor_516_03_phase012a_pfs_on.py` (PFS ON, `export_err_mode=calibrated`).
Runtime: Phase 0 0.4 s, Phase 1 4007 s, Phase 2A 571 s, total 4841 s. 48 LCs, no error.

| item | 515 (`de6f7c8`) | 516 rebuilt |
| --- | --- | --- |
| core SHA | `de6f7c8155d141376cf6df895144873f470555c5bb2de426ddad5b46cd981301` | `be6191e0efcb8016637445c8d5d5b186db693f9a2dacea4646a74c061f502f41` |
| extended SHA n | 193 (96 leftover `comp_qa`) | 145 (48 `comp_qa`) |
| MAG 48/48 | identical | max abs 0 |
| ERR | Howell-by-omission | empirical |
| BO median `err` | 8.945 mmag | 8.532 mmag |
| `ct_n_comp` | 2363 | 2346 |
| `ct_c1` | -0.373 | -0.373 |
| `saturate_limit_adu_85pct` in per-target CSV | 52428.0 | 55704.75 |
| `comp_qa_*.json` | 96 | 48 |

ERR carrying term is still `err_photon` only (`err_sem_rel` / `err_scint_rel` / `err_sigma_sys_rel` max abs 0). Max |ERR| 6.589 mag on faint `1498321301379345408` (same target as 516-01 UI empirical). Proc CSVs stamp `err_bkg_source=empirical`. Snapshot `err_background_mode=empirical`, `per_frame_saturation_enabled=true`, `saturate_limit_fraction=0.8`.

Sat-limit product caveat: Phase 0+1 consumed the existing `masterstars_full_match.csv` (still `55704.75` / band star `1497853802778923392` `zone=linear`). That star is in `comparison_stars.csv` (2357 rows) but in **zero** per-target ensembles, so MAG stayed identical. Landing `52428` in the 516 catalog requires re-annotating MASTERSTAR zones, then another Phase 1.

### Part C exports (BO CVn vs on-disk 515 `BO_CVn_20260423.txt`)

- AAVSO MAG: 134/134 identical.
- AAVSO MAGERR at 3-decimal export precision: 82/134 rows change; max delta 0.006 mag; median delta -0.001 mag.
- Median MAGERR remains `0.009` on both files (3-decimal rounding).
- 01B check MAD from `check_kmag_*.csv` (same formula `1.4826 * MAD(kmag) * 1000`): BO 7.1506 mmag, FW 8.2010 mmag, identical to 515 sidecars (MAG-path identity).

## Errors (if any)

- First rebuild attempt failed on a harness import (`tests.photometry_sha`); no science outputs written. Retry succeeded after adding `dev/` to `sys.path`.
- 516 product files still carry `saturate_limit_adu_85pct=55704.75` because MASTERSTAR catalog was not rebuilt.

## Files changed

- `src_py/photometry_core.py`
- `src_py/comp_qa_core.py`
- `src_py/config.py`
- `src_py/pipeline.py`
- `src_py/sat_diag.py`
- `src_py/comp_selection_per_target.py`
- `src_py/psf_photometry.py`
- `dev/tests/test_err_background_empirical.py`
- `dev/tests/test_comp_qa_pool_guard.py`
- `dev/tests/test_batch_e_recut.py`
- `dev/tests/test_masterstar_zone_classifier.py`
- `dev/tests/test_sat_diag.py`
- `dev/results/CURSOR_RESULT_ANCHOR_516_03.md`
- gitignored: draft 516 photometry rebuild; `tmp/anchor_516_03_*.py`

Committed so far:

- `96aa0d6` - Part A fail-loud empirical ERR cache fix
- `1aa744c` - stale `comp_qa` sidecar cleanup
- `615ddda` - sat-limit one-authority fix
- `790291a` - mixedframe fixture `sigma_bkg_ap` so `--fast` empirical default does not raise
- `08cf443` - remaining `read_flux_from_csv` fixtures given `sigma_bkg_ap`
- `4a65675` - docs stamp `saturate_limit_fraction` default 0.80

New 516 product SHA to supersede `de6f7c8`: **`be6191e0`**.

Docs impact: pending Part D re-cut. No living docs updated yet.
Recurrence: new tests `test_phase2a_empirical_requires_sigma_bkg_ap_input`, `test_phase2a_projected_cache_matches_full_row_empirical`, `test_write_comp_qa_artifacts_removes_stale_sidecars`, `test_resolved_equipment_limit_uses_inv_sat_limit_authority`.
