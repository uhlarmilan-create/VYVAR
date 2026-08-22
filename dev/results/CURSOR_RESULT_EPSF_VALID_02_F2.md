CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 F2)

What I did
Replaced per-frame PSF exception swallow with full accounting, job summary JSON,
INV-PSF-FRAME-01 enforcement (>20% zero-ok frames = FAIL LOUDLY), honest log wording.

## Output / findings

| Item | Result |
|------|--------|
| Per-frame record | frame_name/index, n_fit, n_ok, exception_class, full message, traceback tail |
| Job summary | `platesolve/epsf_photometry_job_summary.json` via `finalize_epsf_frame_job` |
| INV-PSF-FRAME-01 | Wired in `invariants_runtime.WIRED_INV_IDS` + `docs/VYVAR_INVARIANTS.md` |
| Threshold | 20% zero-ok frames -> `InvariantViolation`; below with any zero-ok -> WARN |
| MP workers | Workers return `psf_frame_record`; parent merges and finalizes |
| Tests | `dev/tests/test_epsf_frame_accounting.py` (message capture, >20% trip, <=20% WARN) |

## Research cite

photutils `EPSFBuilder._process_iteration(stars, epsf, iter_num)` -- status codes 1/2/3
documented in photutils PSF module (fit region / convergence / off-cutout).

## Docs impact

- `docs/VYVAR_INVARIANTS.md` -- INV-PSF-FRAME-01 row added

## Gate status

`--fast` OVERALL: FAIL on unrelated `test_v3d_fine_scale.py::test_v3d_run_structure` (1488 passed, 32 skipped); ePSF tests green.

## Errors

None in F2 module tests.

## Files changed

- `src_py/epsf_frame_accounting.py` (new)
- `src_py/pipeline.py` (`_fill_psf_catalog_columns`, `export_per_frame_catalogs` finalize)
- `src_py/invariants_runtime.py`
- `src_py/app.py` (job summary log line)
- `dev/tests/test_epsf_frame_accounting.py` (new)
