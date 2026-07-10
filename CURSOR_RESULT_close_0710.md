CURSOR RESULT -- 2026-07-10 SESSION-CLOSE-0710

What I did
Verified clean production tree on HEAD `560723c`, confirmed today's commit series pushed,
ran full pytest + session_baseline_check --fast, updated STATE/JOURNAL/ROADMAP for tomorrow's
entry point, wrote this close record. Committed and pushed session-close docs.

## Part 1 -- verification sweep

### git status

**Production tree clean.** Restored accidentally deleted tracked files (CURSOR_RESULT_*, CLAUDE.md)
before close. Remaining untracked (known, non-production):

| Path | Note |
|------|------|
| `.worktrees/` | git worktrees |
| `CURSOR_RESULT_except_retriage3.md` | scratch |
| `docs/VYVAR_CODE_AUDIT.md` | audit scratch |
| `docs/round2_figs/v0454_lc_vyvar.png` | figure |
| `scripts/dy_peg_night_run_bvr.py` | night-run helper |
| `scripts/qatar8_night_run_v.py` | night-run helper |

### git log f4c7c83..HEAD (on origin/main)

```
c8b1e2f SESSION-CLOSE-0710: HRD arc + F-BINGAIN-1 close, next-session entry points.
560723c Document F-BINGAIN-1 regate: decomposition gates, hybrid B remedy, SIGMA-BUDGET follow-up.
0eb47d7 Document F-BINGAIN-1 regate: decomposition-driven gate refinement and SIGMA-BUDGET follow-up.
3b33b03 Add empirical background noise term and hybrid howell_scaled fallback.
76838ab Docs: add commit hashes to TODO-12f result file
... (TODO-12e/12d/12c/12b/12 chain through 8c25679)
```

**Confirmed:** TODO-12 arc `8c25679..76838ab` + F-BINGAIN-1 `3b33b03`, `0eb47d7`, `560723c` on `origin/main`.

### pytest

```
737 passed, 15 skipped, 30 warnings in 293.72s
```

### session_baseline_check --fast

```
OVERALL: PASS
git-head: 560723c | pytest: 737 passed, 15 skipped | ledger: v1 10 items
```

### Byte-identity baseline (F-BINGAIN-1)

Re-anchored for documented **`err` column divergence** (empirical `sigma_bkg_ap` term in production).
**Non-err proc-CSV science columns verified byte-identical** on patch-only acceptance runs
(draft_426 g: 24 files; draft_424: 139 files; draft_425 B/V/R: 12 files each). LC `err` is the
authoritative production uncertainty for chi2 validation (`bingain_fix_validate.py` LC-err path).

## Part 2 -- documentation updates

| File | Change |
|------|--------|
| `docs/VYVAR_STATE.md` | SESSION-CLOSE snapshot: HEAD, pytest, both workstreams DONE, NEXT=SIGMA-NEWTON |
| `docs/VYVAR_JOURNAL.md` | Czech SESSION-CLOSE-0710 entry (HRD arc + F-BINGAIN root cause/fix/regate seeds) |
| `docs/VYVAR_ROADMAP.md` | New NEXT SESSION entry block; F-BINGAIN row result-file pointers |

TODO-12 and F-BINGAIN-1 rows verified DONE/RESOLVED with `CURSOR_RESULT_*` pointers.

## Part 3 -- sandbox tools (gitignored)

- `sandbox/tools/bingain_inventory.py`
- `sandbox/tools/bingain_rn_measure.py`
- `sandbox/tools/bingain_bg_closure.py`
- `sandbox/tools/bingain_common.py`
- `sandbox/tools/bingain_stageB_run.py`
- `sandbox/tools/bingain_stageC_run.py`
- `sandbox/tools/bingain_stageC_chi2.py`

Artifacts under `tmp/bingain_stageB/`, `tmp/bingain_stageC/`, `tmp/bingain_acceptance/`.

## Errors

None.

## Files changed

- `docs/VYVAR_STATE.md`
- `docs/VYVAR_JOURNAL.md`
- `docs/VYVAR_ROADMAP.md`
- `CURSOR_RESULT_close_0710.md`

**origin/main:** `c8b1e2f`
