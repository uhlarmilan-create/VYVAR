CURSOR RESULT - 2026-08-21 (PUSH + DOCS-SYNC-517 Part A + FRAME-QC-PARITY-01 Part B)

What I did
Verified local series and Part A/B deliverables; filled two doc gaps (518 XVAL
addendum, PRECAL XVAL note); refreshed FRAME-QC measurements and line refs at
tip 66a0813. **Part 0 (push) not executed** -- origin/main still 8dea595.

## Part 0 - PUSH status

| Check | Result |
|-------|--------|
| `git rev-parse HEAD` | **66a0813** |
| `git rev-parse origin/main` | **8dea595** (after fetch) |
| Tracked tree | Clean (untracked dev/results scratch only) |
| `--full` running | No |

**STOP:** Milan must run `git push origin main`. Expected post-push origin:
**66a0813**.

**Local series** (`git log --oneline origin/main..HEAD`, 11 commits):

```
66a0813 D10-1-CLOSE
e0db635 D10-1b
489d9d3 D10-1
5d015fc ERR-518-02b
5f46469 INV-ERR-SIGMA-ACCT-01 wired
ea0b4ad ERR-518-02 docs
30d631d ERR-518-02 commit 2
70a6022 ERR-518-02 commit 1
34f9cf1 ERR-518-01
3175cd2 FRAME-QC-PARITY-01 phase 1
49e6795 DOCS-SYNC-517
```

Note vs task brief expected list: includes **49e6795**, **3175cd2**, **34f9cf1**
(Part A, Part B, ERR-518-01) ahead of the ERR-518-02 block -- content matches
task spec; order differs because Part A/B landed before ERR-518-01 in this
session history.

After push, tell Cursor "pushed, origin at 66a0813".

## Part A - DOCS-SYNC-517 (complete)

| Item | Status | Evidence |
|------|--------|----------|
| A1 DRAFT-517-REVIEW | **Done** commit 49e6795 | `dev/results/CURSOR_RESULT_DRAFT_517_REVIEW.md` + context |
| A2 ROADMAP sync | **Done** + amended this session | FRAME-QC-PARITY-01, MS-POOL-POLICY rescope, MS-QA-DISPLAY, CV-CVN-SKIP, COMP-HISTORY-DB, EMPTY-DAO-01 closed, PRECAL XVAL note added |
| A3 DECISIONS 2026-08-20 | **Done** + 518 XVAL addendum | Product model section; EMPTY-DAO-01 closed |

## Part B - FRAME-QC-PARITY-01 phase 1 (complete)

| Item | Status |
|------|--------|
| Mechanism named | **Done** commit 3175cd2 |
| Path x gate table | In result file |
| 16-frame table | In result file (FWHM prefilter, not HFR) |
| Phase 2 options | In result file, no recommendation |
| Re-measurement 2026-08-21 | 516 vs 517 identical (0 status/header diffs, 134 proc each) |

**Key finding (unchanged):** architectural split is **full pipeline vs
photometry-only `--full` replay**, not live 516 vs 517 QC divergence.

Line refs updated to **66a0813** (`pipeline.py:16361`, `16016`, `16246`).

## Docs impact (DOCS-SYNC)

| File | Change |
|------|--------|
| `docs/VYVAR_DECISIONS.md` | 518 XVAL TEST-only addendum |
| `docs/VYVAR_ROADMAP.md` | Local tip 66a0813; PRECAL XVAL evidence sentence |
| `dev/results/CURSOR_RESULT_FRAME_QC_PARITY_01.md` | Line refs at 66a0813 |
| `dev/results/context/session_20260821_frame_qc_parity/measurements.json` | Re-run refresh |

## Errors

None.

## Files changed (this amendment)

- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`
- `dev/results/CURSOR_RESULT_FRAME_QC_PARITY_01.md`
- `dev/results/context/session_20260821_frame_qc_parity/measurements.json`
- `dev/results/CURSOR_RESULT_PUSH_DOCS517_QCPARITY.md` (this file)

STOP -- await Milan push of full series, then Milan review of Part A+B before
next push of any amendment commit.
