CURSOR RESULT - 2026-08-12

What I did
Made per-frame QA/header FWHM a median over many star-like detections so
cosmics/hot pixels cannot pull it to ~1-2 px (draft 508). No data clipping.

## Output / findings
- Cause: segmentation SourceCatalog FWHM after CR removal measured hot pixels.
- Fix: ``_robust_frame_fwhm_median`` (DAO + isolation + extended + median).
- Frame 62: 1.45 -> 5.14 px on recompute; night std 0.46 -> 0.08.
- Detail: ``dev/results/CURSOR_RESULT_fwhm_qa_robust.md``
- Fresh draft re-run needed for DB/headers/apertures.

## Errors (if any)
None.

## Files changed
src_py/pipeline.py, dev/tests/test_robust_frame_fwhm.py,
dev/results/CURSOR_RESULT_fwhm_qa_robust.md, CURSOR_RESULT.md
