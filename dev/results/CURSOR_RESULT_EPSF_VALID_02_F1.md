CURSOR RESULT - 2026-08-22 (EPSF-VALID-02 F1)

What I did
Implemented F1 UI scope: shared science-set builder module, dashboard filter to 333 science stars,
ProgressColumn percent format fix, build-meta expander.

## Output / findings

| Item | Result |
|------|--------|
| ProgressColumn format | Explicit `format="%.1f%%"` on 0-100 stored `pct_psf_ok` (choice: keep 0-100 scale, not 0-1 fraction) |
| Science set builder | `src_py/epsf_science_set.py` -- single definition for UI + F3 |
| Dashboard table/selectbox | Filtered to science set via `build_epsf_science_set()` |
| Build meta expander | Reads `masterstar_epsf_meta.json` (n_stars_used, funnel, iteration curve when present) |
| Regression test | `dev/tests/test_epsf_dashboard_pct.py` inspects ProgressColumn config |

## Docs impact

None beyond F1 scope (invariant doc deferred to F2).

## Gate status

F1 is non-science UI only; no `--fast` required after F1 alone.

## Errors

None.

## Files changed

- `src_py/epsf_science_set.py` (new)
- `src_py/ui_epsf_dashboard.py`
- `dev/tests/test_epsf_science_set.py` (new)
- `dev/tests/test_epsf_dashboard_pct.py` (new)
