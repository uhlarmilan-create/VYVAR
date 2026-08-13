CURSOR RESULT - 2026-08-13 22:45 UTC+2

What I did
Executed eight approved Wave 6 deletions (one commit + `--fast` each). Skipped W6-DEL-04 per Milan exclusion. Updated register. No push.

## 4.1 Deletion sequence

| commit | item | `--fast` |
|--------|------|----------|
| `181811b` | W6-DEL-01 qc_processing_run_exists + delete_qc_processing_run_by_hash | PASS 1323/27 skip |
| `b82e976` | W6-DEL-02 set_obs_draft_masterstar_path (+ ASCII-clean part_abc.md, CITATIONS.bib blocked `--fast` on prior attempt) | PASS |
| `1a7320e` | W6-DEL-03 fetch_draft_scanning_ids | PASS |
| `599e58d` | W6-DEL-05 importer legacy helpers (4 functions) | PASS |
| `e98b354` | W6-DEL-06 export helper stubs (3 functions) | PASS |
| `09f2b79` | W6-DEL-07 photometry_report PDF helpers (2 methods) | PASS |
| `ff6fba6` | W6-DEL-08 ProcFrameStore.frame_columns | PASS |
| `c6dc90d` | W6-DEL-09 param_resolver resolve_saturation/exptime | PASS |
| `76b18b4` | register Wave 6 outcomes (docs only) | not re-run |

**HEAD:** `76b18b4` (10 commits ahead of `origin/main` at `0e9786a`).

## 4.2 Skipped

| ID | reason |
|----|--------|
| W6-DEL-04 | Milan exclusion: `update_master_source_safety` + `count_final_data_for_*` are unwired library-delete guards, not dead. Moved to W6-PROP-05. |

No other items skipped; all eight approved deletes had zero callers on pre-delete re-grep.

## 4.3 Draft 510 verification (read-only)

From `trust_1498613634033133184.json` and on-disk CSVs (no photometry re-run):

| metric | expected | observed |
|--------|----------|----------|
| check-star scatter | 0.008629 | 0.008629278 (`check_scatter`) |
| trust | GREEN | GREEN |
| LC points | 134 | 134 (`lightcurve_1498613634033133184.csv`) |
| comparison stars | 5 | 5 (`n_good_comp`, comp_qa n_clean=5) |

**HOLD.**

## 4.4 PROPOSE reports

### W6-PROP-01 `detect_outliers`
- **Option A:** Honor `outlier_sigma` for non-variable targets only; keep `skip_sigma_clip=True` for VSX-known variables. Consequence: real outliers flagged on constants; variables still protected by feature_mask.
- **Option B:** Remove sigma-clip branch entirely; document mask-first-only API. Consequence: smaller surface; callers must use feature_mask for eclipses.
- **Option C:** Split into `detect_outliers_variable` vs `detect_outliers_constant`. Consequence: explicit contract; two call sites in `apply_reporting_postprocess`.

### W6-PROP-02 `preprocess_calibrated_to_processed` rename
- Add `preprocess_calibrated_lights_in_place()` as canonical name (matches INV-CAL-02 in-place semantics).
- Keep `preprocess_calibrated_to_processed` as deprecated shim calling the new name; emit `DeprecationWarning` once per process.
- Update `night_run.py`, `app.py`, docs. One release later: remove shim.
- Effort: ~4 call sites + CONFIG guide sentence; no science-path change.

### W6-PROP-03 `VY_QCBG` semantics (stale-write instance #1)
- **Option A (INV-CAL-02 compliant):** Stamp `VY_QCBG_PRE` at post-cal QC enrich; after sky subtract write `VY_QCBG` from preprocess annulus median in same FITS write. Report/PDF reads `_POST` if present else `_PRE`.
- **Option B:** Keep single keyword; change comment to "pre-skysf QC sky median"; document in MAP/GATES that preprocess invalidates pixel interpretation. No code move; provenance remains technically stale.
- Recommend Option A for referee defensibility.

### W6-PROP-04 reachability doc correction
- `unwired_ui` count: 1 (`ui_finalization` only). `ui_photometry_results`, `ui_suspected_lightcurves`, `ui_select_stars` deleted 2026-06.
- Update `docs/VYVAR_AUDIT_2026_REACHABILITY.md` counts: unwired_ui 1, not_statically_reachable unchanged.

### W6-PROP-05 library delete guards (ex W6-DEL-04)
- **Live path:** `ui_calibration_library.py` `_make_delete_confirm_dialog` -> `_delete_paths` unlinks file + DB row. No dependency check.
- **Wire plan:** Before OK in dialog, call `db.count_final_data_for_equipment_id(equipment_id)` (or telescope) for the master row's equipment; if count > 0, block delete with message listing N finalized drafts referencing that equipment via manifest. Optionally call `update_master_source_safety` when comp QA flags a master star unsafe.
- **Effort:** ~30 lines in `_make_delete_confirm_dialog`; need equipment_id on overview row dict (likely already in table). Test: mock DB count > 0 blocks delete.
- **Why not now:** post draft-435 recovery; wiring is product safety, not housekeeping.

## 3.1 Non-ASCII friction
Three failures today (`5dd2a4d`, `684a0ee` follow-up, DEL-02 step on tracked `CURSOR_RESULT_audit_part_abc.md`). Root cause: em-dashes/middle dots written into tracked markdown without `ascii_migrate`.
**Recommendation:** Run `python dev/tools/ascii_migrate.py --check --root docs` (or target file) before any audit-doc commit; optional: add `dev/scripts/check_ascii_docs.ps1` wrapper called from session_baseline_check when staged docs change. Low-cost: document in CLAUDE.md session-init for audit writes.

## 3.2 Untracked working documents (25)

| file | class | propose |
|------|-------|---------|
| `CURSOR_TASK.md` | ephemeral task stub | delete or gitignore |
| `dev/results/CURSOR_RESULT_audit_2026_waves1_5.md` | closed audit deliverable | **commit** with register |
| `dev/results/CURSOR_RESULT_audit_wave6_prepush.md` | closed | **commit** |
| `dev/results/CURSOR_RESULT_audit_wave6_execute.md` | this report | **commit** after approval |
| `dev/results/CURSOR_RESULT_placed_aperture.md` | closed SAT-DIAG arc | **commit group A** |
| `dev/results/CURSOR_RESULT_prepush_5dd2a4d.md`, `prepush_7ec4b09.md` | closed push logs | **commit group A** |
| `dev/results/MEMO_ensemble_zp_clip_literature.md` | closed ZP-clip memo | **commit group A** |
| `dev/results/CURSOR_RESULT_cal_diag_investigation.md` | closed CAL bundle | **commit group B** |
| `dev/results/CURSOR_RESULT_inv_cal_02_design.md` | closed INV-CAL-02 | **commit group B** |
| `dev/results/CURSOR_RESULT_phase3_obs_tables_drop.md` | closed | **commit group B** |
| `dev/results/CURSOR_RESULT_audit_retire_obs_tables.md` | closed | **commit group B** |
| `dev/results/CURSOR_RESULT_draft435_restore_plan.md` | closed recovery | **commit group C** (archive evidence) |
| `dev/results/CURSOR_RESULT_draft501_*` (5 files) | closed draft501 investigations | **commit group C** |
| `dev/results/CURSOR_RESULT_draft502_*`, `505_*`, `510_*` | closed comp/LC arcs | **commit group C** |
| `dev/results/CURSOR_RESULT_draft_manifest_phase22-28` (4 files) | closed manifest phases | **commit group C** |
| `dev/results/CURSOR_RESULT_dark_binning_physics.md`, `white_cores_v3.md` | closed physics memos | **commit group C** |
| `dev/tests/_tmp_batch_e_lc/` | scratch | gitignore / delete |
| `dev/tools/wide_err_e*.py`, `wide_err_a2b.py` | WIDE-ERR open work | keep untracked until WIDE-ERR closes |
| `src_py/tmp/` | scratch | gitignore |

**Proposed commit groups:** A = audit+SAT-DIAG+ZP memo; B = CAL-DIAG/INV-CAL; C = draft recovery/manifest history.

## A-1 before Wave 7

**Should not be closed as "principled-open"** - the audit measurements show three FWHM estimators on draft 510: MASTERSTAR Gaussian 3.30 px (drives SNR/aperture), per-frame moment 5.31 px (QC only), xval aligned-frame 2.96 px. The disagreement is **definitional** (stack PSF vs moment vs harness circular aperture), not unmeasurable.

**Closing A-1 would require:**
1. Aperture growth curves on the **same stars, same 134 frames** for at least two estimators (MASTERSTAR FWHM vs per-frame moment vs photutils fit).
2. Pick which FWHM drives `precompute_and_save_snr_aperture_table` (currently MASTERSTAR).
3. Decide COG/Stetson correction: wire `cog_aperture_correction_enabled` or document EE~54% at current r as accepted loss.
4. Re-cut anchor once (draft 510 or successor) if aperture radius changes.

Wave 7 referee statement can **document A-1 as measured-but-unresolved** without closing it; full closure needs the growth-curve experiment Milan described, not more code audit.
