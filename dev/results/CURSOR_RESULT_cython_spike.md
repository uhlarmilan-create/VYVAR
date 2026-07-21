CURSOR RESULT - 2026-07-21 -- CYTHON-SPIKE (profiling + compile feasibility)

What I did
Profiled anchor `--full` (py-spy flamegraph, OVERALL PASS) and P1 golden mini
(speedscope, 608 s). Added `build/setup_cython.py`, `build/README.md`,
`dev/scripts/summarize_speedscope.py`, and gitignore rules for Cython artifacts.
MSVC Build Tools absent on this Windows box; photometry_core fails Cython
translation without source edits. Parts 3-4 (import proof, P1 compiled, speed,
protection) BLOCKED pending MSVC + photometry_core fix on a build host.

Docs impact: ROADMAP (CYTHON-RELEASE arc, spike DONE); STATE one-liner.
Recurrence: n/a (first spike; compile blockers are environment + one forward-ref).

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

### Prereq: MSVC Build Tools -- FAIL (Windows)

Hello-world cythonize:
  error: Microsoft Visual C++ 14.0 or greater is required.

winget install Microsoft.VisualStudio.2022.BuildTools ... exit 1602 (installer
cancelled / likely needs elevation).

**Install on Windows dev box:**
  winget install Microsoft.VisualStudio.2022.BuildTools
  Select workload: "Desktop development with C++" (MSVC v143+, Windows SDK).
  Verify: python tmp/cython_spike/hello_cython_test.py -> RC 0, .pyd present.

Cython version recorded: **3.2.8**. py-spy: **0.4.2**.

### Build matrix

| module | Cython translate | link (.pyd/.so) | notes |
|--------|------------------|-----------------|-------|
| photometry_core | **STOP** | n/a | CompileError line 7419: undeclared `_get_lc_psf_strict` (forward ref to symbol defined only via lazy import in method_lc_output; needs source edit -- out of spike scope) |
| comp_selection_per_target | OK | BLOCKED (no MSVC) | Pure-Python mode, no source edits |
| photometry_phase2a | OK | BLOCKED (no MSVC) | Pure-Python mode, no source edits |

Scaffolding committed: build/setup_cython.py, build/README.md.
Docstring stripping: Cython.Compiler.Options.docstrings = False (not a
compiler_directive in 3.x).

## Part 3 -- correctness proof -- BLOCKED

| check | status | detail |
|-------|--------|--------|
| Import proof (.pyd shadows .py) | BLOCKED | No extension modules built |
| P1 golden compiled byte-identity | BLOCKED | photometry_core does not translate; no .pyd |
| Full pytest compiled | BLOCKED | same |
| Multiprocessing worker __file__ | BLOCKED | same |

Interpreted tree sanity: session_baseline_check --fast PASS after spike work
(1051 passed, 24 skipped).

## Part 4 -- measurements -- BLOCKED

| metric | status |
|--------|--------|
| P1 wall time interpreted vs compiled (3x median) | BLOCKED |
| --full wall interpreted vs compiled | BLOCKED |
| strings / protection on .pyd | BLOCKED |

Honest expectation (from profile): plain compile of numpy/pandas-bound paths
yields modest gains; comp_selection_per_target PY-LOOP is the best typed target;
IO/matplotlib slice is not fixed by Cython compile alone.

## Part 5 -- verdict

**GO/NO-GO: CONDITIONAL NO-GO for Windows compile on this machine; GO to proceed
with the CYTHON-RELEASE arc after two unblockers.**

Unblockers (ordered):
1. Install MSVC Build Tools on Windows (or use Linux sandbox for first linked build).
2. Fix photometry_core `_get_lc_psf_strict` forward reference (define or import
   at module level) -- science-path change, full gates required; not spike scope.

After unblockers, Linux sandbox should run from same build/setup_cython.py:
  python build/setup_cython.py build_ext --inplace
  import proof -> VYVAR_INVARIANTS_P1=1 pytest -> full pytest -> MP worker check.

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
