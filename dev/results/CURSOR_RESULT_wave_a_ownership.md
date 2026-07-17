CURSOR RESULT - PARAM-OWNERSHIP-WAVE-A - 2026-07-17

What I did
Wave A of the parameters redesign (safe wave): closed out REPO-REORG Phase D,
fixed the config-write render side-effect by mechanism, added an ownership axis to
the registry + dashboard, and extended the SUMMARY MEASURE REPORT into an honest
full-config report (snapshot appendix + resolved-facts block) with a metadata-only
provenance extension. Full pytest green after every step; --fast PASS. Push is
GATED on Milan's explicit word (STEP 5).

## Commit stack (local, ahead of origin/main 62410c8)

```
5e311d4 feat(report): honest full-config report (snapshot appendix + resolved facts)  [STEP 4]
d4c7953 feat(ui): regroup Parameters dashboard by ownership                            [STEP 3]
c4b6885 feat(params): add ownership axis to the parameter registry                     [STEP 2]
083c8e0 fix(config): CONFIG-WRITE-GUARD - config.json persists only from explicit UI   [STEP 1]
10ce982 docs(layout): stamp REPO-REORG src_py/ + dev/ layout and Phase D closeout      [STEP 0]
860ebf7 chore(config): ratify observer block Jirny (id=2)                              [STEP 0]
cdd2277 docs(ledger): record REPO-REORG anchor gate PASS                               [STEP 0]
1345722 docs(audit): PARAM-SOURCE-AUDIT parameter provenance map                       [prior]
8f4d7b4 chore(layout): move VYVAR modules into src_py/ with entry shims                [REPO-REORG B]
c611353 chore(layout): move dev-side dirs and Cursor results into dev/                 [REPO-REORG A]
```

## STEP 0 - REPO-REORG Phase D closeout

- cdd2277 `dev/validation/VYVAR_VALIDATION_LEDGER.json` (5 +/- 5): anchor PASS record.
- 860ebf7 `config.json` (5 +/- 5): observer block Jirny ratified. Message states the
  ACCURATE root cause (Milan chose accurate wording over the task's premise): the
  write was NOT the headless run - it was the UI location-picker auto-save-on-render;
  fixed in STEP 1.
- 10ce982 docs stamp (301 insertions): CLAUDE.md repository-layout section;
  VYVAR_PROCESS.md ritual paths + result-file location rule (dev/results/);
  VYVAR_STATE.md + VYVAR_JOURNAL.md REPO-REORG arc + Phase C gate verdict;
  CURSOR_RESULT_repo_reorg.md Phase D confirmation block (Milan UI smoke:
  root-shim launch, Settings + Parameters tab, modified-counter=10, + draft_000436
  anchor run).

## STEP 1 - CONFIG-WRITE-GUARD (root cause, not symptom)

083c8e0 (123 +/- 13). Root-cause chain (investigated; differs from the task premise -
confirmed with Milan, accurate wording chosen):

- There is NO headless -> config.json write path. All 7 `save_config_json` callers
  are Streamlit UI. `run_full_photometry_pipeline`, `run_night_pipeline`, and
  `session_baseline_check --full` never call it; `AppConfig.__post_init__` hydrates
  the observer block in memory only (`src_py/config.py:1231-1244`, no write).
- The observed write (observer_location_id 1->2, Dablice->Jirny) came from the UI
  location picker in `render_live_view`:
  `app.py: render_live_view (1835)` -> location selectbox defaults to the DB
  IS_DEFAULT location (`app.py:1961-1973`, Jirny id=2) -> mismatch check at the old
  `app.py:2037-2047` -> `save_config_json(...)`. This fired on plain PAGE RENDER
  (e.g. launching the app for the smoke test) whenever config.json held a different
  location - an auto-save-on-render side effect, not an explicit user save.

Fix (mechanism, Milan approved "both"):
- `config.save_config_json` now raises `ConfigPersistError` unless called inside the
  new `config.ui_config_persist()` context. All 7 UI save handlers opt in explicitly;
  pipeline/headless code has no way to persist config.json.
- The location picker persists ONLY on a genuine user change of the selectbox
  (session-tracked baseline `vyvar_varstrem_location_persisted_id`); a plain render
  with IS_DEFAULT != config no longer writes config.json.

Regression guard `dev/tests/test_config_write_guard.py`: (a) unit tests - save raises
outside the context, succeeds inside, flag resets on exit; (b) contract scan asserting
pipeline.py / photometry_core.py / night_run.py / simulate_night_run.py never call
save_config_json nor open ui_config_persist.

## STEP 2 - ownership axis in the registry

c4b6885. Added `"owner"` to every one of the 304 params_registry.json entries, seeded
EXACTLY from the audit CSV proposed_owner column. Distribution (matches spec):
db_static 9 / config_runtime 277 / fits_dynamic 7 / internal 11.
- `params_registry.py`: OWNERS enum + "owner" in ENTRY_KEYS (full-schema guard now
  enforces owner presence).
- `test_params_registry.py`: owner guard (full coverage + valid enum; names offenders).
- `gen_params_md.py`: owner column + summary line; `docs/VYVAR_PARAMS.md` regenerated;
  freshness test green.

## STEP 3 - Parameters dashboard regrouped by ownership

d4c7953. `src_py/ui_params_dashboard.py` now groups by owner:
- CONFIG (config_runtime): editable, tiered exactly as before (basic/advanced/expert);
  Save writes ONLY config_runtime auto keys.
- OBSERVATORY FACTS (db_static): read-only cards, DB value + "managed in Settings ->
  Observatory / Database Explorer".
- RESOLVED AT RUNTIME (fits_dynamic): read-only cards; last-run value from provenance
  when available, else config fallback flagged "fallback only".
- internal: not rendered (subsumes widget=hidden).
- Modified-counter now counts owner=config_runtime ONLY (dead db_static fallbacks like
  observer_* no longer inflate it), relabeled accordingly.
Pure helpers group_keys_by_owner / editable_config_keys / count_modified(owners=) added;
dashboard smoke test extended (partition, editable-set disjointness, counter excludes
db_static).

## STEP 4 - honest full-config report

5e311d4. SUMMARY MEASURE REPORT Configuration section is now three parts:
1. Summary (existing) - deviations from default (owner-agnostic, from config_snapshot).
2. FULL CONFIG SNAPSHOT (new appendix) - complete as-run config_snapshot, every key,
   3-column layout grouped by phase.
3. RESOLVED FACTS BLOCK (new) - site (id/name/lat/lon/alt + source), gain (+source),
   read noise (+source), saturation, plate scale, frame dims, binning, filter, exposure.

Provenance writer extended (metadata only; NO numeric change; anchor comparator
ignores pipeline_meta - byte SHA fileset excludes pipeline_meta.json). New top-level
`pipeline_meta.resolved_facts`, captured at the Phase 2A resolver site
(`_build_phase2a_resolved_facts`), fields added:

- resolved_facts.site { location_id, name, lat, lon, alt_m, source, ok }
- resolved_facts.gain { value, source, key }
- resolved_facts.read_noise { value, source, key }
- resolved_facts.saturation_adu
- resolved_facts.plate_scale_arcsec_per_px
- resolved_facts.frame_width_px
- resolved_facts.frame_height_px
- resolved_facts.binning
- resolved_facts.filter
- resolved_facts.exptime_s

Fallback: legacy drafts (no snapshot / no resolved_facts) render what exists with a
visible warning and never crash. Builders (full_config_snapshot_model,
resolved_facts_model) are pure and unit-tested with synthetic provenance and the
missing-fields case; `_build_phase2a_resolved_facts` has its own writer unit tests.

Report verification (report-only, NO photometry rerun) - draft_000435 / NoFilter_60_2:
- PDF pages: 191.
- Part 1 (Summary): 10 deviations, source "run snapshot".
- Part 2 (Full snapshot): fallback=False, 292 keys, all 13 phases present.
- Part 3 (Resolved facts): fallback=True (draft predates the writer -> legacy path
  exercised): recovered Site id=2 Jirny lat=50.1122 lon=14.6983 alt=275 (source draft),
  gain 3.17, read noise 15.2, plate scale from dynamic_params; saturation/frame/binning/
  filter/exptime shown as "-" with the warning. A FRESH run populates resolved_facts
  (verified by the writer unit tests).
Output: tmp/wave_a_report_435.pdf.

## STEP 5 - gate + push (GATED)

Full pytest: 922 passed, 19 skipped (green after every step; STEP 1 910 -> STEP 2 911
-> STEP 3 914 -> STEP 4 922).

`python dev/scripts/session_baseline_check.py --fast`:

```
git-branch     PASS main        git-head       PASS 5e311d4
git-staged     PASS none        config-paths   PASS all present
pytest         PASS 922 passed, 19 skipped
ledger         PASS v1 14 items
OVERALL: PASS
```
(WARNs are pre-existing/expected: 4 known untracked scratch, git-origin-main differs -
unpushed, ledger-todo VL-ANCHOR-424 / VL-ANCHOR-DQ-430.)

Push: AWAITING Milan's explicit "push". On approval I will push ALL local commits
(REPO-REORG c611353 + 8f4d7b4, audit 1345722, wave A cdd2277..5e311d4) to origin/main.
`git log --oneline origin/main..HEAD` = the 10-commit stack listed above.

## Errors (if any)
None blocking. Note: `dev/tests/test_g7_f003c_report_cfg_snapshot.py` (2 tests) is
order-dependent in ISOLATION (its `_factory` monkeypatch self-recurses through
`config.AppConfig` when the qc-metrics path constructs AppConfig); it passes in the
full suite. Confirmed pre-existing (reproduces on committed STEP 3 code without any
wave-A edits) - not introduced here.

## Files changed
Per-step commits above. No numeric science change: STEP 1/2/3 touch config-persistence,
registry metadata, and UI only; STEP 4 adds provenance METADATA only (comparator
ignores pipeline_meta.json).
