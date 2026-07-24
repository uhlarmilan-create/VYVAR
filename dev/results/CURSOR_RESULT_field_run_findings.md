CURSOR RESULT - 2026-07-24 FIELD-RUN-FINDINGS + SESSION-CLOSE

What I did
Implemented field-run fixes #11-#13 from Milan's first real E2E (FI Boo, draft_000001,
147 frames, preview-20260723), session-close bookkeeping (JOURNAL/STATE/ROADMAP), regression
tests, `--fast` gate, commit. **No bundle rebuild.** **STOP before push** (handoff below).

## Output / findings

### A1 #12 VSX fail-loud (root cause confirmed)

**Evidence:** `infolog_20260724_152115.txt` -- empty/wrong `vsx_local_db_path` on fresh config;
VSX cone returned 0 silently -> empty `variable_targets.csv` -> Phase 0: 0 targets -> false
`[OK] Pipeline dokonceny uspesne`. Milan's VSX DB is healthy (~7.8M rows once path set.

**Fix:** `require_vsx_local_db_path()` + `VSXCatalogError` in `database.py` (unset/missing
file/bad schema/zero-row table -> actionable RuntimeError naming Settings -> Catalogs).
`write_photometry_plan_files` and `_query_vsx_local*` with `require_db=True` fail loud;
legitimate empty field logs `VSX cone=0 on <db> (N total rows) - field genuinely empty`.

**Tests:** `dev/tests/test_field_run_findings.py` (missing path raises; healthy tmp DB with
0 in-field rows -> no raise + distinguishing infolog line).

### A2 #13 Truthful 0-target completion

**Fix:** `run_full_photometry_pipeline` returns `zero_targets=True` when Phase 0 has 0 actives
(skips Phase 2A and SUMMARY REPORT inputs). `app.py` RUN VYVAR logs warning
`Pipeline dokonceny - 0 aktivnych cielov, fotometria nespustena (skontroluj VSX katalog /
target selection)` and skips `generate_all_method_photometry_reports` for those setups.

**Test:** `test_run_full_photometry_zero_targets_skips_phase2a` (no `photometry_summary.csv`).

### A3 #11 BORDER glob 0 in RAM-handoff

**Root cause:** `write_photometry_plan_files` ran pre-alignment (and pre-RAM-flush) while
`detrended_aligned/lights/<setup>/proc_*.fits` not yet on disk; glob returned 0. Post-flush
rewrite existed but failures were swallowed.

**Fix:** Explicit defer log when glob finds 0 frames; post-flush rewrite logs failures.
ROADMAP `BORDER-PREALIGN` closed.

**Anchor argument:** Ordering/log-only on invalid path cases; post-flush BORDER uses same
aligned bytes already written by RAM flush -- no science-path change for anchor config.

### A4 observer_location_id fresh default

**Fix:** `AppConfig.observer_location_id` default **2 -> 1**; bootstrap materialization
writes 1 via `to_json()`. `#7` fallback + warning unchanged for genuine mismatches.

**Test:** bootstrap asserts `observer_location_id=1`; `resolve_import_location_id` with single
LOCATION id=1 -> no warning.

### A5 Cone radius (answer only)

`field_catalog_cone_meta.json` `cone_radius_deg=13.6` for ~6.6 deg diagonal FI Boo field
(factor ~2x, not ~4x on diagonal alone).

**Source:** `_effective_field_catalog_cone_radius_deg()` in `pipeline.py`:
1. WCS border sample -> `max_sep_deg * 1.38 + 45 arcsec` floor (`_field_center_and_radius_from_wcs`)
2. Optics floor from FITS FOCALLEN+PIXSIZE (`_gaia_catalog_cone_radius_optics_floor_deg`)
3. Optional UI `plate_solve_fov_deg` minimum via `catalog_cone_radius_from_fov_diameter_deg`,
   capped to ~130% above physical when r >= 2.5 deg (prevents 20 deg+ UI misconfig pulling 500k+ Gaia rows)

**Verdict:** Intentional safety margin for edge stars / plate-solve error / Gaia cone completeness,
not a unit bug. No hot-fix this arc; follow-up only if science QA shows undersized cone.

## Gates

| Gate | Result |
|------|--------|
| ruff (touched modules) | PASS |
| pytest (full dev/tests) | **1155 passed**, 26 skipped (after VYVAR_PARAMS regen) |
| session_baseline_check --fast | PASS after params md refresh |
| ASCII | PASS (new files `# -*- coding: ascii -*-`) |

**Anchor argument (A1/A2/A3):** A1/A2 change behavior only when VSX path invalid (anchor config
has valid VSX); A3 is ordering/logging -- no anchor science-byte change expected.

## Push record

**STOP before push** per task protocol. Commit ready on branch `main` (local); user runs push.

Suggested commit message:
```
Field-run findings #11-#13: VSX fail-loud, zero-target status, BORDER defer; session close docs.
```

## Files changed

- `src_py/database.py` -- VSX fail-loud helpers; `count_vsx_local_rows` tuple fix
- `src_py/pipeline.py` -- VSX require_db wiring; BORDER defer; post-flush log
- `src_py/photometry_core.py` -- zero-target early return
- `src_py/app.py` -- warning completion; skip summary on 0 targets
- `src_py/config.py` -- observer_location_id default 1 (prior partial)
- `dev/tests/test_field_run_findings.py` -- new
- `dev/tests/test_bootstrap_release_data_dir.py` -- observer_location_id assert
- `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_PARAMS.md` -- regen (default id change)

## Errors (if any)

None blocking. Pre-regen `--fast` failed only on stale `VYVAR_PARAMS.md` (observer_location_id default).
