CURSOR RESULT - 2026-09-07 FRAME-QC-PARITY-02C

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 7c37c5e.
Class: diagnostic log only (D-NSTARS-DIAG-01). No drop, no status
change, no header change, no product change.
Live 516/517 read-only.

## Refute check

Quantity is qc_metrics `n_stars_detected` (robust helper
pipeline_calibrate.py:2884 / row write :3433), not Layer A capped
n_star. Status is assigned in `_qc_one_calibrated_file` before the
night csv write.

Architect suggested hook: completion of
`qc_enrich_calibrated_lights_in_place` (pipeline_preprocess.py:410)
or immediately after the prefilter status write that both paths share
(cited app.py:817-853 / night_run.py:241-277).

Refute of that pair as the *single* shared point:
- Those app.py / night_run.py ranges are stale. Current prefilter
  call is `build_prefilter_rejected_map` at app.py:425 and
  night_run.py:612 (duplicated call sites, same helper).
- That map is an *input* to enrich. Final `status` and
  `qc_metrics.csv` are written inside
  `_qc_enrich_calibrated_in_place` (pipeline_calibrate.py:3668-3673).
- UI (app.py:440) and night_run (night_run.py:625) both call
  `qc_enrich_calibrated_lights_in_place`, which only forwards to the
  inner function (pipeline_preprocess.py:455). OSC also calls the
  inner function directly (pipeline_calibrate.py:3235).
- Hooking only the preprocess wrapper would miss OSC.
- Cycle trap: pipeline_preprocess already imports pipeline_calibrate;
  planner/emitter stay in pipeline_calibrate.py.

Chosen shared point: after the csv write in
`_qc_enrich_calibrated_in_place` (emit at pipeline_calibrate.py:3675).
`--full` photometry-only path never runs preprocess; untouched.
No ePSF-graph name edited; `--full-epsf` skipped.

## What I did

- `N_STARS_DIAG_K = 5.0` in pipeline_constants.py (D-CONSTANTS-LEAF-01);
  facade re-export in pipeline.py.
- `plan_n_stars_diagnostic` / `emit_n_stars_diagnostic`: ok-set median
  +/- 5.0 * 1.4826 * MAD of `n_stars_detected`. If ok MAD==0, fall
  back to all-rows MAD; if still 0, one summary line and no per-frame
  warnings.
- Unit tests on a synthetic DataFrame only (no DAO, no fixtures).
  Full-suite first fail was caplog vs infolog
  (`pipeline` logger propagate=False); tests now attach a local
  handler.

## 516/517 replay (read-only qc_metrics.csv)

Planner on live drafts; identical tables.

ok n=134, median=98.0, MAD=1.0, sigma_MAD=1.4826,
lo=90.587, hi=105.413.

4 warned frames (matches n_stars_k_bounds.csv ok k=5.0):

| frame | n | side |
|-------|---|------|
| Light_010 | 90 | low |
| Light_012 | 107 | high |
| Light_029 | 263 | high |
| Light_037 | 106 | high |

n_low=1, n_high=3. Frames kept. Live Archive not written.

## Gates

G1 `--fast --clean` at 13a7863: OVERALL PASS (1625 passed, 32
skipped; clean-tree PASS). One intermediate G1 fail was sqlite
threading flake (`test_open_sqlite_connection_allows_cross_thread_use`);
re-run PASS. Earlier G1 fails were the caplog/infolog test isolation.

G2 `--full` aperture-only at 13a7863: OVERALL PASS. era04_aperture
`d55fcc9d` n=53 / ext `cc8b532e` n=157 byte-identical
(full-photometry-sha-core-aperture / -ext-aperture PASS).
full-pipeline 1358s. No `--full-epsf`.

G4 live 516 after G2: PASS csv `bfa24039` / fits `13e77cf8` /
epsf `172f9540`. Not written.

## Docs

D-NSTARS-DIAG-01 at top of docs/VYVAR_DECISIONS.md (verbatim).
FRAME-QC-PARITY moved to CLOSED this arc. CHECK-EPOCH-034 OPEN LOW
inserted after CAL-* / before COMP-POOL-R.

## Files / commits

9e66b6a FRAME-QC-PARITY-02C: n_stars diagnostic warning k=5.0
(D-NSTARS-DIAG-01).
b5c7aee / 13a7863 test isolation (pipeline logger / infolog).
docs commit: DECISIONS + ROADMAP + this report.

src_py/pipeline_constants.py, src_py/pipeline.py,
src_py/pipeline_calibrate.py, dev/tests/test_n_stars_diagnostic.py,
dev/tests/test_facade_inventory.py, docs/VYVAR_DECISIONS.md,
docs/VYVAR_ROADMAP.md, this file.

Push only `git push origin consolidate-01:consolidate-01`.
