CURSOR RESULT - 2026-07-21 -- CYTHON-SPIKE (profiling + compile feasibility)

What I did
Profiled anchor `--full` (py-spy flamegraph, OVERALL PASS) and P1 golden mini
(speedscope, 608 s). Added `build/setup_cython.py`, `build/README.md`,
`build/cython_build_win.bat`, `dev/scripts/summarize_speedscope.py`, and
gitignore rules for Cython artifacts. MSVC Build Tools 18 verified (2026-07-21
continuation): partial compile links OK; import proof + MP spawn OK; P1 byte-
identity FAIL on comp_selection_per_target compiled alone; photometry_core
still STOP at Cython translate.

Docs impact: ROADMAP (CYTHON-RELEASE arc, spike DONE); STATE one-liner.
Recurrence: n/a (first spike; compile blockers are environment + forward-ref +
comp_selection byte-identity under plain Cython).

## Part 1 -- profiling

### Full anchor run (mandatory flamegraph)

Command:
  py-spy record -o tmp/cython_spike/profile_full.svg --format flamegraph --
    python dev/scripts/session_baseline_check.py --full

Outcome: OVERALL PASS (2778 s pipeline; anchor SHAs 03d8fb64... / bbfcc92e...).
Flamegraph: tmp/cython_spike/profile_full.svg (278560 samples; not committed).

### Aggregate hotspots (P1 golden mini speedscope)

Command:
  VYVAR_INVARIANTS_P1=1 py-spy record --format speedscope --subprocesses --
    python -m pytest dev/tests/test_invariants_p1_golden.py -q

Outcome: 5 passed in 608 s. Summarized with dev/scripts/summarize_speedscope.py.
Note: table is from P1 mini (science path); full anchor shares the same
photometry_core / comp_selection hot path but adds longer phase0/1 on 139 frames.

#### Top VYVAR functions (src_py filter, P1 mini)

| function | module | self% | total% | class |
|----------|--------|-------|--------|-------|
| _accumulate_per_frame_comp_metrics | comp_selection_per_target.py | 52.0 | 10.7 | PY-LOOP |
| _apply_aperture_isolation_safe | photometry_core.py | 7.9 | 0.3 | PY-LOOP |
| _compute_comp_contamination_map | comp_selection_per_target.py | 7.5 | 0.4 | PY-LOOP |
| _normalize_gaia_id | photometry_core.py | 6.6 | 0.6 | PY-LOOP |
| save_target_field_map_png | photometry_core.py | 4.3 | 7.4 | IO (matplotlib) |
| _detrend_and_compute_comp_rms_map | comp_selection_per_target.py | 3.0 | 0.2 | PY-LOOP |
| run_full_photometry_pipeline | photometry_core.py | -- | 25.7 | MIX (orchestrator) |
| run_phase0_and_phase1 | photometry_core.py | -- | 13.4 | MIX |
| select_comparison_stars_per_target | photometry_core.py | -- | 13.2 | PY-LOOP |
| run_phase2a | photometry_core.py | -- | 12.3 | MIX |
| _phase2a_process_one_target | photometry_core.py | -- | 9.0 | PY-LOOP |

#### Global top-self (all stacks, P1 mini -- context)

| function | module | self% | class |
|----------|--------|-------|-------|
| _encode_tile | PIL/ImageFile.py | 10.2 | IO |
| _resample | matplotlib/image.py | 6.7 | IO |
| read | pandas/c_parser_wrapper.py | 6.3 | IO |
| _chop | pandas/ops.py | 4.0 | NUMPY-BOUND |

Classification summary:
- PY-LOOP (Cython typed candidate): comp_selection_per_target per-frame metrics,
  photometry_core normalization / phase2a per-target loops.
- NUMPY-BOUND (plain compile only): pandas/numpy/photutils inner loops; typed
  Cython unlikely to help without algorithm change.
- IO: FITS/pandas reads, matplotlib PNG (save_target_field_map_png); caching out
  of spike scope.
- sigma_floor_core, vyvar_alignment_frame, except_fix_counters: NOT in top-30
  VYVAR self time on P1 mini (deprioritize for typed work).

Spike module list (profile-driven): photometry_core, comp_selection_per_target,
photometry_phase2a.

## Part 2 -- compile spike

### Prereq: MSVC Build Tools -- PASS (2026-07-21 continuation)

VS 2026 Build Tools 18 + Desktop development with C++ installed.
VsDevCmd + cl.exe 14.51.36231 verified. hello-world Cython: RC 0.

### Build matrix (updated)

| module | Cython translate | link (.pyd) | P1 byte-identity |
|--------|------------------|---------------|------------------|
| photometry_core | **STOP** | n/a | n/a |
| comp_selection_per_target | OK | **PASS** | **FAIL** (alone) |
| photometry_phase2a | OK | PASS | not isolated (file lock) |

Partial build command:
  build\cython_build_win.bat  (CYTHON_MODULES=buildable; moves .pyd -> src_py/)

## Part 3 -- correctness proof

| check | status | detail |
|-------|--------|--------|
| Import proof (.pyd shadows .py) | **PASS** | comp_selection_per_target.cp312-win_amd64.pyd; photometry_phase2a.cp312-win_amd64.pyd under src_py/ |
| P1 golden compiled byte-identity | **FAIL** | comp_selection alone: core SHA 4ecbae9f... != VL-P1-GOLD 074ae881...; both modules: phase2a_empty_comp_drop=167/169 |
| Full pytest compiled | **MOSTLY PASS** | 1056 passed, 2 failed (P1 headless SHA + physics); 17 skipped |
| Multiprocessing worker __file__ | **PASS** | spawn worker: ...comp_selection_per_target.cp312-win_amd64.pyd |

Interpreted tree restored (.pyd removed): headless SHA **PASS** in 491 s;
P1 mini photometry regenerated at gold SHA.

## Part 4 -- measurements

| metric | result |
|--------|--------|
| P1 headless wall (interpreted, 1 run) | 491 s (post-restore) |
| P1 headless wall (compiled, broken) | 14 s (167 target drops -- not a valid speed win) |
| Speed median 3x3 | **not measured** (compiled path non-equivalent) |
| Protection (comp_selection .pyd) | 654336 bytes; source recoverable: **NO**; bytecode: **NO**; docstrings: **stripped** (__doc__ None); visible: Cython symbol names, build path to .c |

## Part 5 -- verdict

**GO/NO-GO: NO-GO for plain Cython compile as-is.**

Blockers:
1. photometry_core Cython translate STOP (_get_lc_psf_strict forward ref).
2. comp_selection_per_target plain compile **breaks science** (P1 SHA drift;
   isolated test confirms this module alone; 167/169 empty-comp drops when
   paired with photometry_phase2a compiled). Root-cause investigation required
   before any release compile of this module (likely Cython semantic delta in
   PY-LOOP closures/lambdas -- not acceptable for byte-identity release).

MSVC + build scaffolding: **GO** (ready for Linux sandbox link test).

Protection goal (closed-source): **GO** for .pyd format (no source in binary).

Speed: inconclusive until byte-identical compile achieved.

### Full-build sketch (post-spike)

| tier | modules | rationale |
|------|---------|-----------|
| Typed Cython (Phase 2) | comp_selection_per_target._accumulate_per_frame_comp_metrics, photometry_core phase2a per-target loops | PY-LOOP hotspots |
| Plain compile (closed-source) | comp_selection_per_target, photometry_phase2a, sigma_floor_core, pipeline frame glue, alignment helpers | coverage + modest gain |
| photometry_core | plain compile AFTER forward-ref fix | mandatory release target |
| Stay interpreted | app.py, ui_*, streamlit glue, dev/, config loader | UI layer, human-editable |

Bundling risks (open):
- Mixed .pyd + .py tree must ship together; import order shadows .py correctly.
- photometry_core compile blocker implies release bundle may ship comp_selection
  compiled before photometry_core until fix lands.
- MP pool workers must inherit compiled modules on sys.path (verify __file__ in workers).
- String literals (config keys, log tags, column names) remain visible in .pyd;
  source/bytecode not recoverable from plain compile (verify on first .pyd).

## Errors (if any)

- MSVC missing: Microsoft Visual C++ 14.0 or greater is required.
- winget Build Tools install exit 1602.
- photometry_core Cython: undeclared name `_get_lc_psf_strict` at line 7419.

## Files changed

- build/setup_cython.py (new)
- build/README.md (new)
- dev/scripts/summarize_speedscope.py (new)
- dev/results/CURSOR_RESULT_cython_spike.md (new)
- .gitignore (Cython artifact rules; build/ scripts now trackable)
- docs/VYVAR_ROADMAP.md (CYTHON-RELEASE arc)
- docs/VYVAR_STATE.md (one-liner)

Not committed: tmp/cython_spike/*, any *.pyd/*.so/*.c under src_py/ or build/_cython_out/
