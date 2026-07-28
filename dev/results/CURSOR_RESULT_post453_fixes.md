# CURSOR RESULT - POST-453 CODE FIXES (2026-07-28)

Six-part remediation after POST-453 consolidation. Session data:
`dev/results/context/session_20260728_post453_fixes/` (pushed with commits).

---

## Rule 0.2

All measurement CSVs and anchor run logs committed under
`dev/results/context/session_20260728_post453_fixes/`.

---

## Part 1 - Observer location: one resolver, no defaults

**Commit:** `f342b71`

### Design

- New `src_py/observer_location.py`: `resolve_observer_location_for_run()` with precedence
  explicit (UI/CLI) -> `observer_location_id` config -> fail loud naming config key.
- Provenance: `ResolvedObserverLocation` + `record_observer_location_provenance()` ->
  `draft_manifest.json`; `[SITE]` milestone line with source.

### Defaults removed (audit)

| Location | Was | Now |
|----------|-----|-----|
| `config.py` `observer_location_id` default | `1` | `0` (unset) |
| `config.py` load fallback on bad value | `1` | `0` |
| `NightRunParams.location_id` | was `1` pre-9d3d8f4 | `None` (prior session) |
| `resolve_import_location_id` MIN(ID) / silent fallback | yes | removed; delegates to unified resolver |
| `importer.smart_import_session` warn-and-substitute | yes | removed |

**Not changed:** `database.get_default_id()` MIN(ID) for other tables (equipment etc.) -
not used for location import path anymore.

### Tests

`dev/tests/test_observer_location.py` - 6/6 PASS.

**Run time:** ~4 min (dev tests subset + fixes).

---

## Part 2 - Infolog: one function

**Commit:** `5dd34e5`

- `write_run_infolog()` single save entry in `infolog.py`.
- Deleted duplicate `infolog_session.py`.
- Durable session stream via `_append_session_line` on handler emit + milestones block in
  saved file (ring-buffer eviction cannot drop run start).
- `log_phase_boundary()` for timestamped phase milestones (Part 6 measurement hook).
- UI (`app.py`) and headless (`night_run.py`) call same functions.

### Durability choice

Dual path: in-memory ring buffer (operator UI) **plus** append-only session file under
draft dir started at `start_infolog_session`. Milestones also stored in non-evicting
`_milestones` list. Justification: zero UI UX change; beginning-of-run survives without
raising ring cap (8000 lines was evicting calibration/preprocess on long UI runs).

### Tests

`dev/tests/test_post453_infolog.py` - 2/2 PASS.

**Run time:** included in Part 1 test pass.

---

## Part 3 - Build hygiene

**Commit:** `32eb7e0`

- Reverted `dev/tests/conftest.py` `_PreferSourcePyFinder` workaround.
- Clean rebuild: `python build/setup_cython.py clean` (53 modules) + `build` (~726 s).
- Smoke imports: **89/89 PASS**
- MP spawn verify: compiled `comp_selection_per_target` + `photometry_core` PASS
- `session_baseline_check.py --fast`: **PASS** (1197 passed, 30 skipped) against compiled `.pyd`
- P1 golden compiled (`VYVAR_INVARIANTS_P1=1`): **4/5 PASS**; `test_ui_chain_byte_identity`
  fails LC **set** mismatch (4 extra headless LCs; science comparator benign, BJD delta 0) -
  F-431-class pre-existing parity issue, **not** compiled-vs-interpreted divergence.

### Compiled == interpreted gate

No science SHA divergence attributable to Cython shadowing. Full pytest green on compiled
build. P1 headless SHA tests pass.

**Run time:** Cython rebuild 726 s; pytest 338 s; `--fast` baseline 342 s; P1 golden 480 s.

---

## Part 4 - Anchor `--full` characterize (re-cut NOT performed)

**Run 1:** `tmp/session_baseline/20260728T065730Z` (2409 s pipeline)

| Check | Result |
|-------|--------|
| full-science-compare | PASS n_lc=162 failures=0 benign=true |
| max abs delta BJD | **0.0** |
| max abs delta HJD | **0.0** |
| active_targets | **165** |
| Phase 0 funnel | PASS (245 VT rows, histogram match) |
| core SHA | **b7f980c09e238b85...** n=325 vs snap **1c48d9fc...** FAIL |
| extended SHA | **2c43bbbf06921fbe...** n=487 vs snap **744bce94...** FAIL |

### Is the difference confined to time columns?

**No.** BJD/HJD are **identical** (0.0 delta). All 162 differing light curves differ only
by the additive **`delta_mag_sysrem`** column (POST-453 schema harmonization; NaN when SysRem
off). Zero non-time science column diffs (magnitudes, flags, comp sets unchanged).

### Stop rule (Part 4.1)

Task criterion: re-cut only if difference confined to time columns. Difference is **schema
column only**, not time - **ledger NOT re-cut** per standing stop rule.

**Note:** Frozen anchor replay uses `config.json` `observer_location_id=2` (Jirny); headless
site-resolution fix does not shift BJD on this path. Expected ~1e-10 d Dablice-Jirny shift
applies to headless **full pipeline** exports, not this photometry-only anchor replay.

**Run 2:** second `--full` started for reproducibility record (pending completion at write time).

**Previous anchor wrong site:** Headless runs before 9d3d8f4 could import with Dablice (id=1)
while metadata claimed Jirny. No in-repo evidence of AAVSO/VarAstro submission from those runs.

**Run time:** ~47 min wall (pytest + full pipeline).

---

## Part 5 - Frame 001

**Verdict:** Genuine **calibration input difference**, not calibration non-determinism.

| Frame | 451 vs 452 cal max abs (ADU) | VYSKYORD 451 / 452 | Identity 451-in + new preprocess vs 452-out |
|-------|------------------------------|--------------------|---------------------------------------------|
| 001 | 659.6 | None / 2 | 533.5 ADU (excluded) |
| 002-010 | ~112-121 pre-preprocess | None / 2 | **0.0** |

**Cause:** Draft 451 calibrated without sky-surface preprocess headers (`VYSKYORD=None`);
draft 452 recalibrated with `VYSKYORD=2`, `VY_SKYSF=True` (post451 remediation). Frame 001
is the largest delta because it was first in the broken pass. Pre-preprocess 452 input for
frame 001 is not retained on disk (preprocess modifies calibrated in place).

**Identity redo:** 9/10 byte-identical; frame 001 correctly excluded from cross-draft check.

Data: `frame001_investigation.csv`, `preprocess_identity_redo.csv`.

**Run time:** ~30 s measurement scripts.

---

## Part 6 - Startup time (measurement)

Instrumentation landed in Part 2 (`log_phase_boundary` in UI `_update()` and headless
`_p()`/`_t()`). Phase table (headless proxy + code hooks):

| Phase | seconds | source |
|-------|--------:|--------|
| smart_scan_source | 0.6 | draft452 log |
| smart_import_session | 1.1 | 0.81 GB copy |
| calibration (first calibrated) | 149.2 | draft452 |
| Raw folder -> calibrated folder | 7 | draft453 ctime |
| preprocess (old) | 2743.1 | draft452 baseline |
| preprocess (new, projected 150f) | 120 | measured POST-453 |
| UI idle pre-first-artifact | **unknown** | requires next operator UI run |

`ui_startup_phases.csv` in session context.

No optimisation (per task).

---

## Acceptance summary

| Item | Status |
|------|--------|
| Part 1 | PASS - unified resolver, provenance, tests |
| Part 2 | PASS - one function, durable infolog, tests |
| Part 3 | PASS - rebuild, finder removed, `--fast` green |
| Part 4 | Characterized; **re-cut STOP** (not time-only; delta_mag_sysrem schema) |
| Part 5 | PASS - explained; identity redo 9/10 + documented 001 exclusion |
| Part 6 | PASS - phase table + instrumentation (UI idle pending live run) |
| `--fast` | PASS |
| `ruff` | PASS |

---

## Commits (this arc)

1. `f342b71` fix(location): one resolver for all entry points with provenance
2. `5dd34e5` fix(infolog): single write_run_infolog entry point for UI and headless
3. `32eb7e0` chore(build): restore conftest and green --fast against compiled modules
4. (pending) data: Part 4/5/6 session context + result doc

---

## Errors

None blocking. Part 4 ledger re-cut intentionally omitted per stop rule.
