CURSOR RESULT - 2026-08-04 BATCH-E-PARAMS-REGISTRY

What I did
Registered six batch E AppConfig fields in params_registry.json, regenerated
VYVAR_PARAMS.md (277 entries), updated dashboard owner assert, synced FLOW doc
threshold 3.8, fixed BLE001 in closure Step 1 tool, and ASCII-migrated 34 tracked
files after contextual FFFD hand-repair. No science-path changes.

## Pytest hygiene subset (-k "params or ascii or docs_sync or ble001 or dashboard")

| Stage | passed | failed |
|-------|-------:|-------:|
| Baseline (ab0f669) | 26 | 6 |
| After STEP 1-2 (registry + PARAMS) | 28 | 4 |
| After STEP 3 (dashboard assert) | 29 | 3 |
| After STEP 4 (FLOW sync) | 30 | 2 |
| After STEP 5 (BLE001) | 31 | 1 |
| After STEP 6 (ASCII) | 32 | 0 |

Full suite after STEP 6: **1235 passed, 26 skipped** (was 1229 passed / 6 failed
in hygiene subset; full baseline not re-run at start).

## VYVAR_PARAMS.md diff summary

Header count: **270 -> 277** entries.

Expected axes in diff:
  (a) `preprocess_sky_surface_force_reapply` row (batch D, never regenerated),
  (b) `masterstar_dao_threshold_sigma` default **2.1 -> 3.8**,
  (c) six batch E rows: `admission_sat_peak_frac`, `dao_centroid_max_shift_fwhm`,
      `dao_detection_n_equiv`, `enable_lacosmic`, `lacosmic_sigclip`, `lacosmic_objlim`.

Owner summary after regen: db_static 9, config_runtime **249**, fits_dynamic 6,
internal 13.

Note: `gen_params_md.py` dropped a hand-appended "Parameter budget notes" tail;
removed in commit faa1782 so `test_generated_params_md_is_fresh` passes (generator
output only).

## STEP 6a hand-repaired occurrences (21 measured after_digit + 22 rule hits)

Measured U+FFFD "after digit / non-dash" set in 10 priority files (tmp/fffd_scan.txt)
before migrate; repaired by `dev/tools/batch_e_ascii_hand_repair.py` then
`ascii_migrate.py` for em-dash/range rest.

| file:line | before -> after |
|-----------|-----------------|
| docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md:49 | ?12.8 to ?13.4 deg -> -12.8 to -13.4 deg |
| docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md:50 | ~**1.7<FFFD>** -> ~**1.7x** |
| docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md:50 | (~1.39<FFFD>) -> (~1.39x) |
| docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md:51 | (~1.15<FFFD>) -> (~1.15x) |
| docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md:65 | 1.25<FFFD>log -> 1.25*log |
| dev/results/CURSOR_RESULT_audit_t4.md:17 | 3.8<FFFD>?_pixel -> 3.8*sigma_pixel |
| dev/results/CURSOR_RESULT_dao_sigma_stability.md:18 | 8<FFFD>FWHM -> 8x FWHM |
| dev/results/CURSOR_RESULT_dao_sigma_stability.md:41 | 3.8<FFFD>?_pp / 3.8<FFFD>bkg2d -> 3.8*sigma_pp / 3.8*sigma_bkg2d |
| dev/results/CURSOR_RESULT_dao_sigma_stability.md:57-58 | 3.8<FFFD>?_pp / bkg2d -> 3.8*sigma_* |
| dev/results/CURSOR_RESULT_dao_sigma_stability.md:82-87 | 3.8<FFFD>?_pp / bkg2d columns -> 3.8*sigma_* |
| dev/results/CURSOR_RESULT_dao_sigma_stability.md:90 | 3.8<FFFD>?_pp -> 3.8*sigma_pp (2 sites) |
| dev/results/CURSOR_RESULT_audit_t1.md:18 | =06ed950<FFFD> -> =06ed950... |
| dev/results/CURSOR_RESULT_audit_t1.md:22 | b7f980c0<FFFD> -> b7f980c0... |
| dev/results/CURSOR_RESULT_audit_t2.md:17 | 2<FFFD> amplif -> 2x amplif |
| dev/results/CURSOR_RESULT_sync_dev_to_github.md:155 | b7f980c0<FFFD> n= -> b7f980c0... n= |
| dev/results/CURSOR_RESULT_sync_dev_to_github.md:301 | b7f980c0<FFFD> n= -> b7f980c0... n= |

Additional rule-based fixes in same pass (not in FFFD scan): `?_pp` -> `sigma_pp`,
`rel_err?1.36` -> `rel_err~1.36`, `cal ? pre` -> `cal -> pre`, `42.5 ? 30.6`
-> `42.5 -> 30.6`, table `| <FFFD> |` -> `| - |`, SHA ellipsis patterns.

Remaining FFFD total **215** (pre-repair); **43** hand-repaired (21 measured after_digit
plus rule-based hits); **~172** em-dash/range FFFD folded to `-` via ascii_migrate CHAR_MAP
(mechanical).

Production touch: `src_py/vyvar_platesolver.py:150` em dash in GAIA-1 log string
folded to ASCII hyphen; string compiles and reads correctly.

## STEP 5C malformed noqa sites (REPORT ONLY)

`ruff check . --select BLE001,E722 --no-cache` emits 15 Invalid `# noqa` warnings
(EXCEPT-BULK 2026-07-08 truncated snippets):

| file | line |
|------|-----|
| src_py/inspect_drafts.py | 94, 112 |
| src_py/ui_masterstar_qa.py | 162, 402 |
| src_py/tess_verify.py | 643, 1398 |
| src_py/ui_quality_dashboard.py | 972, 1084 |
| src_py/utils.py | 568 |
| src_py/app.py | 1700 |
| src_py/calibration.py | 374 |
| src_py/variability_detector.py | 73 |
| src_py/ui_aperture_photometry.py | 1149, 1453 |
| src_py/photometry_report.py | 3234 |

Not fixed in this batch (touches 10 production files; Milan decision).

## FLOW PDF

Pages before: **36**. Pages after: **36**. Overflow: no overflow instrumentation exists
for the FLOW doc (`build_flow_doc.py`); not measured. (`dev/scripts/verify_pdf_overflow.py`
targets the SUMMARY MEASURE REPORT per draft, not the FLOW PDF.)

## Commit hashes (separate, not squashed)

| Group | hash | message |
|-------|------|---------|
| 1 | 8094af8 | Register batch E config fields and regenerate VYVAR_PARAMS.md (277 entries). |
| 1b | faa1782 | docs: align VYVAR_PARAMS.md with generator output for freshness test. |
| 2 | 416b274 | Update params dashboard owner distribution assert to 277 entries (batch D + E). |
| 3 | 6085ecd | Sync FLOW doc masterstar_dao_threshold_sigma 2.1 -> 3.8 and regenerate PDF. |
| 4 | f9af4f6 | Fix BLE001 in closure Step 1 tool; skip ble001 test when ruff absent. |
| 5 | 165c239 | ASCII policy: hand-repair non-dash FFFD then ascii_migrate 34 tracked files. |

## OUT OF SCOPE (report only)

- config.json regen (249 vs to_json() keys) -- Milan decides separately.
- config.py:2029 ValueError fallback asymmetry (3.8 default vs 1.8 fallback).
- lacosmic_sigclip / lacosmic_objlim have no clamp in config.py.
- test_g7_f003c_report_cfg_snapshot.py order dependence (STATE backlog).

## Errors

None.

## Files changed

dev/validation/params_registry.json, docs/VYVAR_PARAMS.md,
dev/tests/test_ui_params_dashboard.py, dev/tools/docs_pdf/flow_doc_facts.py,
dev/tools/docs_pdf/build_flow_doc.py, docs/VYVAR_FLOW_CZ.pdf,
dev/tools/closure_step1_aperture_fwhm_ground_truth.py,
dev/tests/test_ble001_regression.py, dev/tools/batch_e_ascii_hand_repair.py,
34 ASCII-migrated tracked text files, docs/VYVAR_ROADMAP.md,
dev/results/CURSOR_RESULT_batch_e_params_hygiene.md

---

## Follow-up 2026-08-04

### aperture_snr_sizing finding restored

Moved sharpened budget note from deleted `VYVAR_PARAMS.md` tail (commit `faa1782`) to
`docs/VYVAR_LIMITATIONS.md` new section "aperture_snr_sizing -- partially wired".
Status: partially wired -- live on `pipeline.py:187-188` aperture-bounds path; ignored
by `compute_snr_optimal_aperture_table` (hardcoded 0.8/2.5 x FWHM). Step 1b V7 FLOW
note retained.

### CONFIG guide split clause

`docs/VYVAR_CONFIG_GUIDE_EN.md:267` and `docs/VYVAR_CONFIG_GUIDE_CZ.md:269` now name
the consumer split (aperture-bounds path vs SNR-optimal sweep).

### Result file corrections

- FFFD arithmetic: **215 total** pre-repair; **43 hand-repaired**; **~172 mechanical**
  via ascii_migrate.
- FLOW PDF: no overflow instrumentation for `build_flow_doc.py`; not measured
  (`verify_pdf_overflow.py` is for SUMMARY MEASURE REPORT only).

### Full-suite baseline (ab0f669 vs HEAD) -- RETRACTED comparison

The ab0f669 run was taken in a **linked worktree**, which carries tracked files only.
`Archive/`, the exoplanet DB, and `GAIA_DR3/` are gitignored and absent there. The two
runs measured **different on-disk data states** and are **NOT comparable** as passed-count
totals. **Retract** any claim that "the correct baseline passed count is 1223".

| commit | passed | failed | skipped | note |
|--------|-------:|-------:|--------:|------|
| ab0f669 (worktree) | 1223 | 7 | 31 | invalid baseline -- missing gitignored data |
| HEAD (main tree) | 1235 | 0 | 26 | standing number on full data |

**Method defect:** a full-suite baseline for this repo must be taken on a tree that carries
the gitignored data (**stash on main**), never in a fresh worktree or clone without
`Archive/` / DB paths.

**Failures fixed by this batch: 6**, not 7. The seventh,
`test_exoplanet_promotion_restore.py::test_config_resolves_relative_exoplanet_path_against_data_root`
(`dev/tests/test_exoplanet_promotion_restore.py:106-110`), asserts `p.is_file()` on the
gitignored exoplanet DB. It fails in any worktree and passes on the main tree. Nothing in
this batch touched it.

The **31 -> 26 skip delta** is the same artifact: tests gated on gitignored on-disk drafts
(`test_alg_functions.py:418`, `test_photometry_completeness.py:15` and `:25`,
`test_fix_draft_equipment.py:102`, `test_f428_msstamp_coord.py:128`,
`test_pre_cal_proc_csv_naming_e2e.py:45` and `:49`). Not a code effect.

Standing full-suite on main tree: **1235 passed / 0 failed / 26 skipped**.

### batch_e_ascii_hand_repair.py fate

**Keep (a)** -- retained in repo as BATCH-E archaeology. One-shot migration artifact;
repair table is authoritative in this result file. No routine maintenance role. Docstring
left unchanged in this docs-only commit; a future code-touch may add the archaeology
banner if desired.

### Follow-up commit

Docs only: LIMITATIONS, CONFIG guides EN/CZ, this result append.
