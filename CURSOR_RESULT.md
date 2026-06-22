CURSOR RESULT — 2026-06-19

What I did
Implemented G1-F003 two-part fix: gated ref-grid pixel-fallback on `VY_ALGN`; flagged failed-alignment epochs in LC/summary/trust. Tests + ledger. Two commits, not pushed.

## Output / findings

**Part 1 — Correctness (`pipeline.py`)**
- `_fits_header_vy_algn_aligned(hdr)` reads FITS `VY_ALGN` (default True for legacy frames).
- Bad-WCS and no-WCS pixel NN paths skip when `VY_ALGN=False`; fall through to sky-only or nondetection instead of wrong-star ref-grid matches.
- Aligned frames with bad/missing WCS: pixel-fallback unchanged.

**Part 2 — Visibility (`photometry_core.py`, `trust_flag_core.py`)**
- `alignment_report.csv`: reads `aligned` + `reason` into `frame_time_lookup`.
- `read_flux_from_csv`: per-row `alignment_failed`; LC CSV column `alignment_failed`.
- Summary: `n_alignment_failed`, `alignment_failed_frac`.
- Trust: soft warning when ≥5% of LC epochs or ≥2 epochs failed alignment (existing YELLOW pattern).

**Do-no-harm**
- Fully-aligned path: `mag_inst` byte-identical in LC export; only new `alignment_failed` column (all False). Verified in `test_save_lightcurve_alignment_failed_column_do_no_harm`.

**Tests** — `tests/test_g1_f003_alignment_pixel_fallback.py`: 7 passed.

## Errors (if any)
None.

## Files changed
- `pipeline.py` — VY_ALGN gate on pixel-fallback
- `photometry_core.py` — alignment_failed propagation + LC/summary
- `trust_flag_core.py` — trust soft signal
- `tests/test_g1_f003_alignment_pixel_fallback.py` — new
- `docs/VYVAR_FULL_AUDIT_LEDGER.md` — G1-F003 FIXED
- Commits: (part 1 hash) + (part 2 hash + ledger) — not pushed
