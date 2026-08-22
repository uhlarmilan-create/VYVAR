CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 F3)

What I did
When `psf_photometry_enabled=True`, measurement IDs now come from the science set (333 on 516)
instead of `_epsf_lc_catalog_ids` (2505). Empty science set fails loud; composition in job meta.

## Output / findings

| Item | Result |
|------|--------|
| `_epsf_fit_catalog_ids` | Uses `build_epsf_science_set()` when PSF enabled |
| Empty guard | `ValueError` at export start and in ID resolver -- no fallback to LC pool |
| Job meta | `science_set` block in `epsf_photometry_job_summary.json` |
| Census test | `test_science_set_census_matches_p1_decisions`: n=333, 2172 pool-only excluded |

## Docs impact

None (behavior matches pre-registered P1-C / DECISIONS science set).

## Gate status

Same as F2 `--fast` note; F3-specific tests PASS.

## Errors

None.

## Files changed

- `src_py/pipeline.py` (`_epsf_fit_catalog_ids`, export preflight)
- `src_py/epsf_science_set.py` (shared with F1)
- `dev/tests/test_epsf_science_set.py`
