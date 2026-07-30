# session_20260730_preprocess_profile

Preprocess timing profile captured during release-build gate work (compiled Cython rebuild +
anchor `--full` on 2026-07-29). HEAD at run time: **226d269** (session-close commit; release gates
ran from this tree before **2e0909a** sync commits).

**Source context:** BO CVn wide rig, 10-frame bench (not a full draft night-run id). Produced while
measuring post-sky-surface-restore preprocess performance on the compiled build path.

**Schema note:** This file differs from `session_20260727_post453/preprocess_profile.csv`. The
post453 profile used a per-step breakdown (source masking, polynomial fit, surface eval, QC
metrics). This profile collapses masking+fit+eval into one step and replaces the metrics block
with `n_frames`, `total_s`, `per_frame_s`, and `max_abs_diff` (508.97 ADU on the compared set).

The post453 session folder was left untouched (immutable session evidence).
