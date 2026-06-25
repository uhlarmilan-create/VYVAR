CURSOR RESULT — 2026-06-25 (F-BJD-1 Stage D)

What I did
Added `time_base` provenance column to per-target LC (`BJD_TDB` vs `JD_FALLBACK`) via
`_recompute_bjd_hjd_with_status`; kept 2-tuple wrapper for existing callers. Updated docs;
closed 2026-06-25 citation/error-model audit.

## Output / findings

### Implementation
- `_recompute_bjd_hjd_with_status` — cause-reported status on three fallback paths
- `_recompute_bjd_hjd_per_target` — thin 2-tuple wrapper (sandbox/tests unchanged)
- Production Phase 2A writes constant `time_base` column via `save_lightcurve_csv`
- `compare_photometry_science_meaningful`: `time_base` in QC exclusion set
- `tests/test_time_base_flag.py` (6 tests)

### Verification
- `pytest tests/` green (535 passed)
- Numeric columns unchanged; additive `time_base` only

## Errors (if any)
None.

## Files changed
- `photometry_core.py`, `method_lc_output.py`, `tests/photometry_sha.py`
- `tests/test_time_base_flag.py`
- `docs/VYVAR_MATH_PHYS_AUDIT.md`, `VYVAR_STATE.md`, `VYVAR_DECISIONS.md`, `VYVAR_JOURNAL.md`, `VYVAR_ROADMAP.md`, `VYVAR_AUDIT_LEDGER.md`
