CURSOR RESULT -- 2026-07-13T13:05:00Z

What I did
Part A root-cause archaeology (producer/collector/timeline). Part B collection-layer fix
(is_masterstar_proc_name + tests). Part C exposure audit. Part D r_60_4 check_kmag diagnosis.
Part E draft_426 regen on fixed code + fresh Newton baseline redo + docs amend.

## A -- Root cause

**Producer:** ``export_per_frame_catalogs`` (``pipeline.py:9791``) walks all FITS under
``detrended_aligned/lights/{setup}`` via ``_iter_fits_recursive``, including ``MASTERSTAR.fits``
copied there by ``_ensure_aligned_masterstar_copy`` (``pipeline.py:13529-13540``). Each frame
emits a sidecar at ``proc_csv_path_for_aligned_fits`` (``pipeline.py:10320``,
``proc_frame_store.py:100-110``) -> ``Archive/Drafts/draft_000426/detrended_aligned/lights/
{i_70_4}/proc_MASTERSTAR.csv``.

**Collector:** Phase-2A epoch list at ``photometry_core.py:7021-7024`` from
``proc_frame_store.keys()`` (PERF-5 build at ``photometry_core.py:13263-13269``) or flat
``glob("proc_*.csv")``. ``ProcFrameStore.build`` uses ``list_proc_csvs(..., recursive=True)``
(``proc_frame_store.py:181-182``). No MASTERSTAR exclusion before this task.

**Timeline:** Stale June LC mtime **2026-06-26**; ``proc_MASTERSTAR.csv`` mtime **2026-07-10**
(artifact created after June photometry). June run had **25** epochs because the proc file did
not exist at LC cut (answer **a**). Fresh 2026-07-13 regen picked up the later artifact ->
**26** epochs. PERF-5 (``ad6e788`` cluster) made the collector path explicit via store keys;
flat glob would also include MASTERSTAR once the file exists. Layout: ``MASTERSTAR.fits`` in
``detrended_aligned/lights`` for pre-cal draft_426 (confirmed on disk).

**Verdict:** Producer = aligned export loop + MASTERSTAR copy into lights tree; collector =
unfiltered ``proc_*.csv`` epoch set; regression trigger = **re-photometry after post-June
proc_MASTERSTAR artifact**, not a June-vs-July code path difference for the same on-disk file set.

## B -- Fix

- ``is_masterstar_proc_name()`` in ``proc_frame_store.py`` (case-insensitive ``proc_MASTERSTAR``).
- Excluded in ``list_proc_csvs`` and ``ProcFrameStore.build`` (canonical layer).
- Belt-and-braces filter + ``[MASTERSTAR-EPOCH]`` warning at ``photometry_core.py:7025``.
- **Choice:** keep ``proc_MASTERSTAR.csv`` on disk (legitimate SNR / ``noise_floor_adu`` artifact);
  exclude only from epoch collection.
- Tests: ``tests/test_proc_frame_store.py`` (+4 cases, case-variant name).

## C -- Exposure audit

Script: ``scripts/masterstar_epoch_exposure_audit.py`` -> ``tmp/masterstar_epoch_audit.json``.

| Draft | Setup | masterstar proc | phantom epoch in LCs (count/total) |
|-------|-------|-----------------|-------------------------------------|
| 424 | NoFilter_60_2 | no | 0 / 178 |
| 425 | B/R/V setups | no | 0 / all |
| 426 | g/i/r/z | yes (under lights) | **0 / 6** after fix regen (was 6/6 pre-fix) |
| 427 | (sweep) | no phantom LCs | 0 |

**Anchor 424:** CLEAN -- no masterstar proc, no phantom epochs. **No extra re-cut for
MASTERSTAR**; planned bundled re-anchor (unit fix + PROD-SIGMA-FLOOR) unchanged.

**Why 424/425 clean:** no ``proc_MASTERSTAR.csv`` under their ``detrended_aligned/lights``
trees (layout / pipeline history differs from pre-cal 426).

**AAVSO / VarAstro flag:** Stale draft_426 exports (evidence tree) did **not** carry phantom
epoch (25 epochs). Contaminated 26-epoch LCs existed only on local pre-fix regen (never pushed).
No export action taken.

## D -- r_60_4 check_kmag (follow-up, not auto-fixed)

**Symptom:** Fresh regen **0** ``check_kmag_*.csv``; stale had **6**.

**Cause:** Sparse comparison-star pool on regen -- ``comp_quality_*.json`` shows **2** good comps
vs **8** in stale June run for the same target. ``compute_check_ensemble_mag_calib`` returns
None when ``len(other_quality) < n_comp_min`` (``check_star_kmag.py:389``). ``select_check_star``
still finds a candidate; sidecar write fails on ensemble math, not on ``check_star_min_epochs``
(``trust_flag_core.py:73`` / config default 5).

**Not** caused by phantom MASTERSTAR epoch or frame-count change (25 vs 25 after fix).

**Recommendation:** Investigate r-band Phase-1 comp selection regression (why 2 vs 8 good comps
on regen). Re-run r photometry after comp-pool fix; do not patch check_kmag gates without that.

## E -- Redone baseline + science compare

Regen (fixed code): ``scripts/draft_426_regen.py --skip-preserve`` g/i/r/z.
Phase2a frame counts: **25** (ProcFrameStore 25 CSV). Shared epochs with stale: **25 vs 25**
(i_70_4), no ``only_fresh`` / ``only_stale`` proc rows.

Fresh Newton: ``tmp/sigma_newton_fresh/sigma_newton_fresh_summary.json`` (git **2beccfa** +
uncommitted fix, regen stamped **2beccfa**).

| Setup | V0611 err (mag) | V0611 chi2 | n |
|-------|-----------------|------------|---|
| g_60_4 | 0.0098 | 3.69 | 23 |
| i_70_4 | 0.0184 | **2.01** | 25 |
| r_60_4 | 0.0135 | **PENDING (COMP-POOL-R)** | 25 |

*Suspect: no check_kmag sidecar (LC mag fallback), same class as pre-fix flag.

**Supersedes:** ``a6b19df`` / ``CURSOR_RESULT_426_regen.md`` baseline (g=3.20, i=2.17@26 epochs).

Science compare (shared epochs): epoch alignment fixed (no extra MASTERSTAR row). Residual
milli-mag drift vs stale on mag_calib columns remains (expected post-June code path).

## Errors

- r_60_4 check_kmag still missing (flagged follow-up).
- i/r carrier_matches_normalize false in regen verify (err-path tolerance; frame sets now match stale).
- z_90_4 sparse (3 LCs); no V0611.

## Files changed

- proc_frame_store.py (is_masterstar_proc_name, list_proc_csvs, ProcFrameStore.build)
- photometry_core.py (phase2a belt-and-braces filter)
- tests/test_proc_frame_store.py (+4 tests)
- scripts/masterstar_epoch_exposure_audit.py (new)
- docs/VYVAR_STATE.md, docs/VYVAR_ROADMAP.md, docs/VYVAR_JOURNAL.md
- CURSOR_RESULT_masterstar_epoch.md (this file)
- Archive/Drafts/draft_000426/... (regen outputs, not in git)
- tmp/sigma_newton_fresh/, tmp/draft_426_regen/, tmp/masterstar_epoch_audit.json

## pytest

**779 passed**, 15 skipped (was 775 + 4 new tests). ``ruff check`` clean on touched files.

## Ready for push

Local chain **4d5c65b..2beccfa** (426-REGEN scripts + docs + PROVENANCE-GUARD) plus **uncommitted**
MASTERSTAR-EPOCH-FIX (this task). HOLD per Milan: one clean push after review of this result.
Do not push until authorized.
