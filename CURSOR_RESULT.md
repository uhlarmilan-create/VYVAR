TASK_ID: unknown

CURSOR RESULT - 2026-08-12 remove L.A.Cosmic

What I did
Removed L.A.Cosmic and all destructive per-frame CR cleaning from QC preprocess.
Kept master dark/flat stacking and non-destructive stats clips. Registry 282 -> 279.
--fast OVERALL PASS. Commit + push.

## Output / findings
- Root cause path removed: `_qc_enrich_one_frame` no longer calls astroscrappy
- Config keys removed: enable_lacosmic, lacosmic_sigclip, lacosmic_objlim
- VY_COSM / VY_COSMNPX no longer written; legacy keys stripped on QC rewrite
- astroscrappy dropped from requirements.txt
- Other destructive light-pixel clip: none found (temporal_sigma_clip is dead unused API)
- Master stack (importer mean/median) KEPT
- Registry: 279 (config_runtime 251)

## Errors (if any)
None.

## Files changed
See commit message / CURSOR_RESULT_remove_lacosmic.md
