CURSOR RESULT - 2026-08-13 21:45 UTC+2

What I did
Pushed C fixes (four commits + sat_diag provenance fix + ASCII audit docs). Repaired stale `sat_diag.json` provenance. Ran provenance sweep. Built Wave 6 disposition list for Milan approval (no deletions executed).

## 1. `--fast` (raw) and origin/main

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   0e9786a
git-staged                   PASS   none
git-untracked-known          WARN   24 known untracked
git-untracked                WARN   CURSOR_TASK.md; ...
git-origin-main              WARN   differs from origin/main (110716f); consider git pull
config-paths                 PASS   all present
pytest                       PASS   1323 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+96 other)
------------------------------------------------------------------------
OVERALL: PASS
```

**origin/main HEAD:** `0e9786a801c58392bfd337dceb9eeba6c9e0fc32`

**Pushed commits (6):** `043820f`, `f45af56`, `0e2e248`, `684a0ee`, `a49b0e4`, `0e9786a`

Draft 510 BO CVn on disk unchanged: 5 comps, GREEN trust, lc_rms 0.145389 (134 LC points per prior anchor).

## 2. sat_diag.json fix and provenance sweep

**Fix (C-SATDIAG-PROV):** Removed early `write_sat_diag_json` at align start. Per-frame catalog now propagates `raw_peaks_used` in row dict; after all frames, `commit_sat_diag_provenance()` writes `Archive/.../sat_diag.json` with correct flag. Test: `dev/tests/test_sat_diag.py::test_commit_sat_diag_provenance_sets_raw_peaks_flag`.

**Sweep:** See register section "Provenance stale-write sweep". Three known instances; sat_diag FIXED. Remaining OPEN: `VY_QCBG` (pre-skysf QC stamp), `preprocess_calibrated_to_processed` naming, `resolve_obs_file_to_processed_fits` naming.

## 3. Wave 6 disposition list (approval required before any delete)

### KEEP (do not delete)

| ID | Item | Why |
|----|------|-----|
| W6-KEEP-01 | `ui_finalization.render_finalization()` | Product functionality; untested not dead. Wire: add Streamlit tab in `app.py` calling `render_finalization(pipeline, draft_id)`. Banner already wired. |
| W6-KEEP-02 | Config-gated paths (`frame_align_residual_gate_enabled`, `sysrem_enabled`, `cog_aperture_correction_enabled`, blind-solve tiers, OSC subtree on mono) | Untested on anchor rig, not dead |
| W6-KEEP-03 | All gates / invariants / verifiers | INV-GATE-REMOVAL decision required per item |
| W6-KEEP-04 | `not_statically_reachable` modules (15) | Lazy import or config path; not proven dead |

### PROPOSE (no delete without alternative)

| ID | Item | Alternative |
|----|------|-------------|
| W6-PROP-01 | `detect_outliers` partial API (I-DETECT-OUT) | Wire `skip_sigma_clip`/`outlier_sigma` honestly for variables, or narrow signature and document mask-first-only path |
| W6-PROP-02 | `preprocess_calibrated_to_processed` alias | Rename to `preprocess_calibrated_lights_in_place`; keep shim one release |
| W6-PROP-03 | `VY_QCBG` header semantics | Stamp at preprocess as `VY_QCBG_PRE` or document as pre-skysf QC only |
| W6-PROP-04 | Reachability doc `unwired_ui` count | Update: `ui_photometry_results` / `ui_suspected_lightcurves` already deleted 2026-06 |

### DELETE candidates (one commit each if approved)

Evidence: grep shows **zero callers outside defining module** (2026-08-13 re-verify). Last reachable: pre-2026 or never wired.

| ID | Target | Was for | Breaks if needed |
|----|--------|---------|------------------|
| W6-DEL-01 | `database.delete_qc_processing_run_by_hash`, `qc_processing_run_exists` | QC run dedup housekeeping | Would need re-add for hash-based QC cache purge |
| W6-DEL-02 | `database.set_obs_draft_masterstar_path` | Legacy MASTERSTAR path column | Superseded by `set_obs_draft_masterstar_fits_path` / manifest |
| W6-DEL-03 | `database.fetch_draft_scanning_ids` | Pre-manifest scanning id list | Manifest `files[]` is live authority |
| W6-DEL-04 | `database.update_master_source_safety`, `count_final_data_for_*` | Equipment safety counters | Would need re-add for library UI delete guards |
| W6-DEL-05 | `importer._is_empty_or_missing`, `_first_fits_in_dir`, `_resolve_session_lights`, `_copy_fits_folder` | Legacy import helpers | Re-copy from git if old import path revived |
| W6-DEL-06 | `export_reports._observer_location_configured`, `_test_is_eclipsing`, `_comp_quality_map_for_export` | Abandoned export helpers | Re-implement from export spec |
| W6-DEL-07 | `photometry_report._lunar_risk_fill_color`, `_katalogy_cell_for_pdf` | PDF styling helpers | Cosmetic PDF only |
| W6-DEL-08 | `proc_frame_store.frame_columns` | Column introspection | Dev/debug only |
| W6-DEL-09 | `param_resolver.resolve_saturation`, `resolve_exptime` | Shadowed resolver API | SAT-DIAG / header paths live elsewhere |

**Not proposed for delete:** `ui_finalization`, OSC modules, `dao_reconcile.py`, any test-only or CLI entry modules.

## 4. Wave 7 readiness

**Draft referee statement is writable now** for: conventional differential photometry (xval PASS), CAL-DIAG/INV-CAL-02 positioning, deliberate CR absence, comp-tier design, SAT-DIAG placed-aperture contract (now provenance-correct in JSON).

**Must stay explicit caveats until closed:** A-1 aperture/COG, WIDE-ERR wide rig, D1-2 linearity, U-PED-01 header pedestal vs CAL-DIAG, VY_QCBG naming, Wave 6 deletions not executed, INV-ANCHOR-00 `--full` gap.

## Files changed

- src_py/sat_diag.py, src_py/pipeline.py, dev/tests/test_sat_diag.py (`a49b0e4`)
- docs/VYVAR_AUDIT_2026_*.md (`0e9786a`, `684a0ee` and prior C-fix commits)
- dev/results/CURSOR_RESULT_audit_wave6_prepush.md (this file)
