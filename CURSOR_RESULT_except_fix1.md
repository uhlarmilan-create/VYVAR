CURSOR RESULT ù 2026-07-08 (EXCEPT-FIX-1)

What I did
Stage A probe (sandbox, uncommitted); TOP-10 T1 terminal-failure fixes; unit tests;
draft_424 validation; census/journal close.

## Stage A ù firing probe

| Site | draft_424 | draft_425 | draft_427 | Class |
|------|----------:|----------:|----------:|-------|
| EXC-0132 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0166 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0043 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0044 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0136 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0045 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0198 | 0 | 0 | 0 | NEVER-FIRES |
| EXC-0449/0452/0455 | 0 | 0 | 0 | NEVER-FIRES (PSF off) |

Artifact: `tmp/except_fix1_probe_light.json`

## Stage B ù fixes (3 files + counters module)

- `except_fix_counters.py` ù run-level counters ? `pipeline_meta.except_fix_summary`
- `photometry_core.py` ù EXC-0132/0166/0136/0198 + chk_mag UnboundLocalError fix
- `comp_selection_per_target.py` ù EXC-0043/0044/0045
- `psf_photometry.py` ù EXC-0449/0452/0455

## Validation

- `tests/test_except_fix_top10.py` ù 5 passed (synthetic failure paths)
- Full pytest: **593 passed**; ruff green
- draft_424 Phase2A rerun: `except_fix_summary` **all zeros** (hot path unchanged)
- photometry_summary sha changed on full rerun (0 comp-star LCs regenerated); not an except-fix fire

## Commits

- `c7227ae` fix(photometry): photometry_core + except_fix_counters + tests
- `a66ba18` fix(comp): comp_selection_per_target
- `dcf6f67` fix(psf): psf_photometry
- `323f44d` docs: census/journal/CURSOR_RESULT

## Errors (if any)

None.

## Files changed

- `except_fix_counters.py` (new)
- `photometry_core.py`
- `comp_selection_per_target.py`
- `psf_photometry.py`
- `tests/test_except_fix_top10.py` (new)
- `docs/VYVAR_EXCEPT_CENSUS.md`
- `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_except_fix1.md`
