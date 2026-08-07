CURSOR RESULT - 2026-08-07 (session-close)

What I did
Amended DAO detection reference wording (confusion-blend untestable; unmeasurable-fraction
caveat), updated STATE/JOURNAL/ROADMAP, synced to GitHub. No science path changes.

## Git inventory (pre-commit)

Working tree clean for tracked files at start. **No uncommitted `config.json` or
`src_py/ui_calibration.py` diffs** at session close (nothing to exclude from commit).

Untracked (left untracked): `CURSOR_TASK.md`, draft_501 diag results, wide_err tools,
`_tmp_batch_e_lc/`.

## Amendments (section 1)

### 1a Confusion-blend

`docs/VYVAR_DAO_DETECTION.md` section 4.5 and LIMITATIONS: hypothesis **not testable** with
Gaia censored at G=17.5; control self-match artefact documented; corrected neighbour table;
closure verdict unchanged (undecidable).

### 1b Unmeasurable fraction

`docs/VYVAR_DAO_DETECTION.md` section 3.5 and LIMITATIONS: wide-rig fraction dominated by
`zp_residual_rms` floor; draft_501 reflects faint rows.

## State documents (section 2)

- `docs/VYVAR_JOURNAL.md` -- Czech entry 2026-08-05..07 arc
- `docs/VYVAR_STATE.md` -- HEAD, closed/open lists
- `docs/VYVAR_ROADMAP.md` -- D1 DONE, D1b/D2 open; Post-DAO carry-forward table (7 items);
  A-6 and DAO-THRESHOLD-PARAMS closed (unchanged)

## Sync (section 3)

See commits below. `git pull --rebase origin main` clean; push success.

## Open carry-forward (explicit)

1. P1-RECUT (ledger stale since a9d7eb0; interim A/B standard)
2. Task A regression remediation (test reimplements flow)
3. D1b (defaults await review)
4. D2 (storage choice)
5. F-B01/F-B02 (calpath provenance)
6. QHY294MM RN double-count
7. BPM sidecar path unresolved

## Errors (if any)

None.

## Files changed

- docs/VYVAR_DAO_DETECTION.md
- docs/VYVAR_LIMITATIONS.md
- docs/VYVAR_STATE.md
- docs/VYVAR_JOURNAL.md
- docs/VYVAR_ROADMAP.md
- dev/results/CURSOR_RESULT_session_close_20260807.md
