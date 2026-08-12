CURSOR RESULT - 2026-08-12 saturated star core punch

What I did
Audited calibrate / preprocess / align for any step that replaces, masks, zeros,
or interpolates saturated / peak science pixels. Confirmed the only historical
punch was L.A.Cosmic (already removed). No further code deletion required.
Saturation flagging left intact (metadata only).

## Output / findings

**Pixel-alteration site (historical, already removed):**
- `pipeline.py::_remove_cosmics_lacosmic` (astroscrappy) -- deleted in `0ab686f`
- Evidence still on disk: drafts 505/506 with `VY_COSM=True` have sharp-star
  cores replaced by ~sky (e.g. frame 012 ~47460 ADU on 508 vs ~1847 on 506)

**Flagging kept (no pixel rewrite):**
- `pipeline.py::_star_saturation_flags` / `_vectorized_star_saturation_columns`
  / `_saturated_core_plateau*` -> `likely_saturated`, plateau/peak flags
- Zone `saturated` / `is_saturated` catalog columns
- `photometry_core.enhance_catalog_dataframe_aperture_bpm` BPM/`likely_nonlinear`
  flags; dark BPM JSON sidecar write in `importer.py`

**Draft 508 detrend_aligned verify:**
- Brightest cores solid (center/max = 1.0 at ~68566 ADU)
- Bright-ring hole scan: 0 hits
- `--fast` OVERALL PASS (1292 passed, 27 skipped)

**This arc code change:** none (fix already in `0ab686f`).
Fresh BO CVn re-run still needed for drafts whose FITS still carry `VY_COSM=True`.

Detail: `dev/results/CURSOR_RESULT_saturated_core_punch.md`

## Errors (if any)
None.

## Files changed
- `dev/results/CURSOR_RESULT_saturated_core_punch.md` (new)
- `CURSOR_RESULT.md` (this file)
