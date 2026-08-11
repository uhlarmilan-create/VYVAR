CURSOR RESULT - 2026-08-11

What I did
Implemented RAM-handoff border filter: aligned frames in memory now feed the annulus-aware
safe bbox when disk glob is empty; post-alignment photometry-plan rewrite runs without
requiring draft_id+database_path; require_safe_bbox=True hard-errors instead of silent skip.

## Output / findings
- `common_field_intersection_bbox_px_from_arrays()` in photometry_core.py
- `write_photometry_plan_files(..., aligned_ram_frames=, require_safe_bbox=)`
- Post-align rewrite in `astrometry_align_and_build_masterstar` gated only on aligned frames
  existing (disk or RAM), not on DB presence
- Root cause confirmed: rewrite at pipeline.py:14657 required draft_id AND database_path;
  Milan's run logged pre-alignment defer and never got border-safe comparison_stars.csv
- Tests: `dev/tests/test_border_ram_handoff.py` (3 tests)

## Errors (if any)
None.

## Files changed
- src_py/pipeline.py
- src_py/photometry_core.py
- dev/tests/test_border_ram_handoff.py
