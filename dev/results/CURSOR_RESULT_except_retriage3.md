CURSOR RESULT - 2026-07-08 (EXCEPT-RETRIAGE-3 + EXCEPT-FIX-3)

What I did
Executed all five parts of the tranche-3 task (platesolver / alignment / importer / database +
astrometry_optimizer 3b): refreshed the census scanner, added the tranche-3/3b evidence tables,
implemented the TOP-10 + 1-companion fix batch behind `except_fix_counters`, validated with the
full test suite, and synced state docs. Four separate commits per the FIX-1/2 convention.

## Commits
- Part 0 (scanner refresh): `e2444b5` - census: EXCEPT-RETRIAGE-3 Part 0 scanner refresh (stable-ID line-number update)
- Part B (FIX-3 code + tests): `c47e9b8` - EXCEPT-FIX-3: surface tranche-3 terminal failures (importer/platesolver/alignment/astrometry)
- Part A (census evidence): `a5b8bdf` - census: EXCEPT-RETRIAGE-3 Part A tranche-3 evidence (astrometry/import/database)
- Part D (state docs): `561bcb4` - docs: EXCEPT-RETRIAGE-3 + FIX-3 state sync (STATE/ROADMAP/JOURNAL)

Base commit: `9f3da34` (origin/main, SESSION-CLOSE-0708).
(Commit order is 0 ? B ? A ? D; A landed after B because the evidence tables were finalized with
the FIX-3-LANDED markers in place. No functional impact.)

## Pytest
- Baseline at 9f3da34: **604 passed**.
- After FIX-3: **615 passed, 15 skipped** (+11 new tests in `tests/test_except_fix3.py`). No test weakening.
- Ruff `--select BLE001,E722`: clean on all touched files.

## Part 0 - scanner refresh
- Root-caused a doubled site count (1230 vs 625): a leftover `.worktrees/except_fix1_*` git worktree
  held duplicate `.py` files. Fix: added `.worktrees` to `EXCLUDE_DIRS` in `sandbox/_except_census_scan.py`.
- Added a stable-ID line-refresh mode: with an existing census, the scanner preserves every EXC-####
  ID + curated prose/disposition and updates ONLY the `file:line` field, matching surviving
  (non-`FIXED`) rows to new sites by within-file line order.
- **102 line numbers refreshed; all EXC IDs preserved** (no remapping needed).
- Deferred `pipeline.py`: 160 surviving old rows vs 159 scanned sites (one non-`FIXED` pipeline site
  no longer detected beyond the 10 FIX-2 rows). Out of tranche-3 scope; its lines were left untouched
  to avoid misaligning tranche-2 IDs, and a warning is emitted. Recorded in the census scanner-refresh note.

## Part A - evidence tables
Added "Tranche 3 - astrometry/import/database (EVIDENCE, 2026-07-08)" to `docs/VYVAR_EXCEPT_CENSUS.md`:
84 core sites (database 20, importer 17, alignment 9, blind 2, platesolver 36) + tranche 3b
`astrometry_optimizer.py` 14 (marked as Milan-approved scope extension) = 98. Includes tier/disposition
summaries (T1 2 / T2 25 / T3 35 / T4 36; fix-now 11, narrow+log-ERROR 4, narrow+log 19,
narrow+comment 45, delete-dead 19), the grounded `log_event` dead-code fact, the platesolver
"mostly HEALTHY" insight, all per-file tables with FIX-3-LANDED markers, the FIX-3 #1
fail-closed-import-abort open question, and a tranche status table. Bulk dispositions are recorded
only (deferred to the bulk pass, per Part C.4).

## Part B - per-fix confirmation
New counters added to `except_fix_counters.ExceptFixCounters` (dataclass fields + `snapshot()`), all 10:
`importer_filter_read_fail`, `dark_bpm_sidecar_write_fail`, `calib_scope_conflict_check_fail`,
`calib_library_register_fail`, `importer_capture_date_fallback`, `importer_imagetyp_read_fail`,
`importer_obs_group_meta_skip`, `wcs_header_key_copy_fail`, `align_unique_sample_fail`,
`platesolve_match_rate_meta_fail`.

- **#1 EXC-0095 `_read_filter`** (surfacing): counter + `logging.error`, still returns `"NoFilter"`.
  Code comment + census pointer + open question recorded. ?
- **#2 EXC-0100 BPM sidecar** (surfacing): counter + `logging.error` incl. `out_path`; master creation still succeeds. ?
- **#3 EXC-0090 scope-conflict** [BEHAVIOR CHANGE, fail-open?fail-closed]: exception ? counter + `logging.error` + return `True` ("assume conflict"). Unit test: raising DB method ? returns True + counter. ?
- **#4 EXC-0092 library register** (surfacing): counter + `logging.error` (path+kind), returns `False`. ?
- **#5 EXC-0089 capture date** [BEHAVIOR CHANGE, better fallback]: counter + `logging.error`, falls back to file `st_mtime` date, then `now()` only if mtime unavailable. Unit test: unreadable file ? mtime date + counter. ?
- **#6 EXC-0094 IMAGETYP** (surfacing): counter + `logging.error`, returns `"unknown"`. ?
- **#7 EXC-0102 obs-group meta** (surfacing): counter + `logging.error` (frame path), keeps `continue`. ?
- **#8 EXC-0625 + EXC-0010 WCS copy** [BEHAVIOR CHANGE, abort on core-key failure]: new shared
  `wcs_header_io.copy_wcs_header_keys(dst, src, *, context)`. Copies with the skip-list, collects
  failed keys, classifies core celestial keys (CRVAL/CRPIX/CD/PC/CDELT/CTYPE/CUNIT/PV/LONPOLE/
  LATPOLE + SIP A_/B_/AP_/BP_ families). Core-key failure ? counter + `logging.error` + abort with
  `dst` untouched (staged via a probe header, so nothing is flushed); non-core failure ?
  `logging.warning`, proceed. EXC-0625 sibling recovery pre-validates on a scratch header and
  returns unrecovered on core failure; EXC-0010 SIP refit pre-validates and skips the refit on core
  failure. Unit tests: (a) passthrough, (b) core-failure abort + counter, (c) non-core warn-only. ?
- **#9 EXC-0586 alignment unique-sample** [BEHAVIOR CHANGE, don't reject on helper error]: helper
  promoted to module-level `_alignment_n_unique_spread_sample`, returns `-1` ("check unavailable")
  on exception + counter + `_alignment_emit_log` warning + `logging.error`. Callers treat only
  `0 <= n <= 3` as constant; `n < 0` ? accept. Unit tests: valid counts, error ? `-1` (not 0) +
  counter, sentinel does not mark frame constant. ?
- **#10 EXC-0605 match-rate meta** (surfacing): counter + `logging.error`, sets
  `sip_meta["match_rate_final"]=nan` and `["match_rate_scope"]="error"` sentinels. ?

## Part C - gates
1. Full pytest **615 passed, 15 skipped** (baseline 604 + 11 new). ?
2. New tests present for #3, #5, #8 (a/b/c), #9 + counter smokes for #1/#4/#6. ?
3. Happy-path invariance: all edits are inside except handlers / post-exception sentinels; the WCS
   helper is byte-identical to the old per-key loop on success; the alignment refactor is
   behavior-preserving (nested wrapper delegates to the module fn) and unit-tested; existing
   byte-identity gate tests remain green. draft_424 headless anchor was NOT re-run - the touched
   paths do not fire on a healthy draft (deviation noted below).
4. Bulk dispositions recorded in Part A only; not implemented. ?

## Files changed
- `sandbox/_except_census_scan.py` (Part 0)
- `docs/VYVAR_EXCEPT_CENSUS.md` (Part 0 + Part A)
- `except_fix_counters.py`, `importer.py`, `wcs_header_io.py` (new), `vyvar_platesolver.py`,
  `vyvar_alignment_frame.py`, `astrometry_optimizer.py`, `tests/test_except_fix3.py` (Part B)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md` (Part D)

## Deviations from spec
- **Commit order** is 0 ? B ? A ? D rather than 0 ? A ? B ? D (evidence tables were finalized with
  FIX-3-LANDED markers after the code landed). No content difference.
- **`pipeline.py` line refresh deferred** (count mismatch, out of tranche-3 scope) - see Part 0.
- **draft_424 headless regression not executed** (Part C.3 was "recommended"); happy-path invariance
  argued structurally + covered by existing byte-identity tests. No site was found to have drifted
  into different code than the spec described, so no per-site STOP was required.
- No secrets or out-of-scope working-tree changes were committed (pre-existing deleted CURSOR_RESULT_*
  files and untracked scripts left as-is).

## Errors
None outstanding.
