CURSOR RESULT - 2026-07-21 -- CYTHON ROOT-CAUSE (annotation_typing)

What I did
Confirmed Cython 3 science drift root cause: annotation_typing=True (default)
turns PEP-484 annotations into C declarations. Pin annotation_typing=False;
all three spike modules compile and P1 golden 7/7 byte-identical on Windows.
Latent-names fixes landed in prior commit (12887bc). Tree left interpreted.

Docs impact: DECISIONS (CYTHON-ANNOTATION-TYPING); STATE one-liner.
Recurrence: dev/tests/test_cython_annotation_typing.py pins directive;
ruff F821 from LATENT-NAMES arc.

## Hypothesis verdict: CONFIRMED

Cython 3 pure-Python compile with default annotation_typing=True changed
comp_selection_per_target science (P1 core SHA 4ecbae9f... vs gold 074ae881...;
167/169 phase2a_empty_comp_drop with both modules compiled, spike 2026-07-21).

With compiler_directives annotation_typing=False in build/setup_cython.py:
- comp_selection_per_target ALONE: P1 headless SHA PASS (422 s)
- ALL THREE modules: P1 golden 7/7 PASS (846 s total suite)

Mechanism: mypy-style annotations (int, float, optional containers) become C
type coercion boundaries; silent failures in comp-selection PY-LOOP paths.

No bisect ladder required beyond the single directive pin.

## Cross-platform build matrix (Windows measured; Linux per Claude sandbox)

| module | translate | link Windows | link Linux (Claude) | P1 compiled |
|--------|-----------|--------------|---------------------|-------------|
| photometry_core | OK | .pyd 3.6 MB | .so ~7.4 MB | PASS |
| comp_selection_per_target | OK | .pyd 654 KB | .so (Claude) | PASS |
| photometry_phase2a | OK | .pyd 20 KB | .so (Claude) | PASS |

Toolchain: Cython 3.2.8; MSVC 14.51.36231 (VS 2026 Build Tools 18);
gcc + Cython 3.2.8 on Linux (Claude sandbox, same setup_cython.py).

## Correctness (compiled, annotation_typing=False)

| check | status | detail |
|-------|--------|--------|
| Import proof | PASS | all three .pyd under src_py/ shadow .py |
| MP spawn worker __file__ | PASS | comp + photometry_core .pyd in worker |
| P1 golden 7/7 | PASS | VL-P1-GOLD byte-identical (846 s) |
| Full pytest | 1065 PASS, 1 incompatibility | see below |
| Recurrence test | PASS | test_cython_annotation_typing.py |

Compiled-mode incompatibility (do not patch blind):
- test_gs11_dilution_aperture_g2_f003::test_no_fixed_3_in_dilution_resolver_source
  uses inspect.getsource() on photometry_core function; fails when .pyd loaded.
  Release bundle keeps .py sources beside .pyd for guards/docs; or exclude from
  compiled-mode pytest subset.

## Speed (P1 headless, 3-run median)

| mode | runs (s) | median (s) |
|------|----------|------------|
| interpreted | 435.9, 480.0, 528.7 | **480.0** |
| compiled (3 modules) | 493.2, 374.2, 404.4 | **404.4** |

Speedup: **15.8%** (480.0 -> 404.4 s). Modest gain as expected for plain
compile of numpy/pandas-bound paths; comp_selection PY-LOOP contributes.

## Protection (photometry_core.pyd)

- size: 3_596_288 bytes
- source recoverable: NO
- bytecode recoverable: NO
- docstrings: stripped (run_phase2a.__doc__ is None)
- visible: Cython symbol names, EXC log string literals, build path to .c

## GO/NO-GO for CYTHON-RELEASE arc

**GO** for plain compile closed-source bundle WITH mandatory pins:
- annotation_typing=False (DECISIONS CYTHON-ANNOTATION-TYPING)
- Options.docstrings=False
- Latent-name gate: ruff F821 + compile translate check

**Next:** expand MODULE_LIST for release; typed Cython only with explicit
cython.* in dedicated modules; bundling/installer arc.

## Prior commits (this task)

1. 29bee9f spike(cython): windows build scripts + NO-GO addendum
2. 12887bc fix(latent): restore _get_lc_psf_strict (+AC guard); ...

## Errors (if any)

None blocking. EXC-0030 root cause fixed (sqlite3.Row.get -> row[] indexing).

## Files changed (commit 3)

- build/setup_cython.py (annotation_typing=False)
- build/cython_build_all.bat
- dev/tests/test_cython_annotation_typing.py
- docs/VYVAR_DECISIONS.md (CYTHON-ANNOTATION-TYPING)
- docs/VYVAR_STATE.md
- dev/results/CURSOR_RESULT_cython_rootcause.md
