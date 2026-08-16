# CURSOR RESULT - RUN-HARDEN-01

Date: 2026-08-16
Baseline tip at issue: da9cce4
Push: NO

## Verdict

Part A: headless draft 515 Phase 0+1+2A launched on SHA **da9cce4** (clean
re-run; no mid-Phase-1 comps on disk). Parts B-C hardenings committed while it
runs (imported code frozen at launch). Part D: RUN-WORKER-01 recorded open.

---

## Part A - Draft 515 headless

### A1 Streamlit

PID 27300 was **already stopped** at task start (no python streamlit process).
No UI RUN started for 515.

### A2 Inventory (reuse vs redo)

| Artifact | Status |
|----------|--------|
| Raw / calibrated lights | Present (150 cal FITS) - **reuse** |
| detrended_aligned + proc_*.csv | Present (135 aligned FITS, 134 proc CSV) - **reuse** |
| MASTERSTAR.fits + masterstars_full_match.csv | Present - **reuse** |
| per_frame_catalog_index / variable_targets | Present - **reuse** |
| photometry/active_targets.csv | Present (Phase 0) - overwritten by re-run |
| comparison_stars_per_target.csv | **Absent** - Phase 1 never persisted (abort at T16) |
| lightcurves | **Absent** |

**Choice:** clean headless **Phase 0+1+2A** via `run_full_photometry_pipeline` on
existing calibrated/aligned products. Mid-Phase-1 resume is impossible without
persisted comps for targets 1-15. Does not re-calibrate or re-platesolve.

### A3 Launch

| | |
|--|--|
| Command | `python dev/tools/draft_515_headless_phase012a.py` |
| Log | `tmp/draft_515_headless_phase012a.log` (also tee) |
| GIT_SHA at launch | **da9cce4a5edd1392b8ba842d3c8488589b9d0ac9** |
| START_UTC | **2026-08-16T13:51:45Z** |
| Early progress | Phase 0: 218 active; Phase 1: 97 targets; first status at ~53.9 s |

### A4 Status at report time

| | |
|--|--|
| State | **RUNNING** (not complete) |
| Last Phase 1 status | **24/97** RX CVn at **1192.9 s** (~19.9 min) |
| Prior milestones | 1/97 @ 53.9 s; 8/97 @ 351.6 s; 16/97 @ 730.2 s (past UI abort target) |
| Stack sample (earlier) | active in `_accumulate_per_frame_comp_metrics` (not idle) |
| Expected remaining | ~73 targets x ~45 s plus Phase 2A (order-of-magnitude) |
| D515-ACCEPT-01 | deferred to next task |

Log path remains `tmp/draft_515_headless_phase012a.log` (gitignored tmp).

---

## Part B - RUN exit visibility

- New `src_py/run_lifecycle.py`: `run_callable_with_exit_log`,
  `format_run_exit_line`, `is_vyvar_run_active`.
- UI entry in `app.py` wraps `_run_vyvar_full_pipeline` via
  `run_callable_with_exit_log(..., log_event)`.
- Lines: `[RUN] finished OK` / `[RUN] aborted: ...` /
  `[RUN] interrupted by script rerun`.

### Fire proof

`dev/tests/test_run_lifecycle.py::test_run_callable_logs_interrupted_on_streamlit_rerun`
raises real `RerunException` through the wrapper; asserts exact log line
`[RUN] interrupted by script rerun`. Also OK / aborted paths. **5 passed.**

---

## Part C - Gate auto-rerun + progress

- `ui_variability.py`: auto-crossmatch and auto-TESS no-op when
  `is_vyvar_run_active(footer)`.
- `ui_aperture_photometry.py`: Block B (auto crossmatch+TESS) same gate.
- Phase 1 progress: every target (removed `n//12` stride) in
  `photometry_core.py`.
- Unit tests for `is_vyvar_run_active` true/false (factorable, no Streamlit
  session required).

---

## Part D - DECISIONS (text)

See `docs/VYVAR_DECISIONS.md` entry **RUN-WORKER-01** (incident, B+C shipped,
subprocess/status-file sketch, awaiting Milan). ROADMAP lists it MED.

---

## Spec defects

1. Spec assumed Milan would stop Streamlit; process was already gone - noted,
   not a blocker.
2. Spec allowed mid-Phase-1 resume; disk had **no** `comparison_stars_per_target.csv`,
   so clean Phase 0+1+2A was the only honest path (named in A2).
3. Headless harness still imports modules that touch Streamlit session proxies
   (warnings in log); does not abort the run - cosmetic, not a hang.
4. Progress-every-target change does **not** affect the already-launched A run
   (frozen import at da9cce4); only future runs. Status lines at 1 and 16 match
   the old `n//12` stride on the live run.
5. "~44 s/target" understates first-target per-frame metric cost; T16 at 730 s
   implies ~45 s mean thereafter once the pool path is warm.

---

## Commits / --fast

| SHA | Message |
|-----|---------|
| af4a7ef | RUN-HARDEN-01 B: log RUN finished/aborted/interrupted with fire proof. |
| 7fbd88f | RUN-HARDEN-01 C: gate auto-rerun; Phase 1 progress every target. |
| acd22c4 | docs: RUN-WORKER-01 open architecture decision. |
| 849adcb | tools: headless Phase 0+1+2A harness for draft 515. |
| 9ff7de7 | fix BLE001 blind except in run_lifecycle + harness |
| dfed148 | results: RUN-HARDEN-01 report |

| | |
|--|--|
| `--fast` | **OVERALL PASS** @ **dfed148** |
| pytest | **1423 passed**, 27 skipped |