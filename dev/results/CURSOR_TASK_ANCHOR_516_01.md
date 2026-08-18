# CURSOR TASK - ANCHOR-516-01 (architect copy)

Date issued: 2026-08-18
Status: **STOP at Part B** (see CURSOR_RESULT_ANCHOR_516_01.md)

Premise check: draft_000516 (created 2026-08-17 22:45, after session
close) is claimed identical to draft_000515 (product SHA de6f7c8,
48 LCs). This task verifies that claim by measurement, re-cuts all
frozen references onto 516, and produces an explicit delete list.
Comparable things: 516 export products vs 515 export products, same
pipeline tip. Not comparable: anything from da9cce4/8f107cf eras.

## Part A - Provenance of 516 (read, do not assume)
1. Report: git HEAD the 516 run executed on, run timestamps,
   pipeline_meta.json contents, calibration_mode, PFS setting,
   config fingerprint. Confirm it ran on d5ef039 (or a0d326c content).
   If it ran on an older tip: STOP, report, do not re-cut anything.

## Part B - Identity measurement 516 vs 515
2. Product SHA of 516 LC set. Expected: de6f7c8 if truly identical.
3. If SHA differs: per-file diff (which LCs, mag or err or header),
   report and STOP for architect review. Do not proceed to Part C.
4. Report LC count (expected 48), gating breakdown (expected
   45 zone_noise + 3 below_target_depth + 1 per_frame_saturation),
   ERR_MODEL line (expected gain=g_pt=0.6371).

## Part C - Re-cut frozen references onto 516
5. Cut new P1 golden from 516 (or a 516-mini if runtime demands;
   mini must include cal_diag.json - do NOT reproduce the 435_p1mini
   gap that caused INV-CAL-01 errors). Register new SHAs.
6. Re-cut --full anchor on current tip with 516 as reference draft.
7. Fire proofs: run session_baseline_check.py --fast and --full;
   both must be OVERALL PASS with zero stale-golden failures
   (test_headless_chain_sha, test_p1_snapshot_sha_matches_registered,
   test_p1_census_fingerprint_in_meta must pass, not xfail).
8. Runtime per part (Rule 0.3).

## Part D - Retirement list
9. Grep repo (tests, tools, validation ledger, docs) for hard path
   references to draft_000435*, 436, 437, 509, 513, 514, 515.
   Report every live reference. Anything referenced only by closed
   JOURNAL/DECISIONS prose is fine to delete; anything referenced by
   a test or gate must be repointed to 516 in this task.
10. Output: explicit DELETE-OK list for Milan. Milan deletes manually;
    Cursor does not delete draft directories.

## Docs impact
STATE (new anchor stamp), ROADMAP (close FULL-ANCHOR-RECUT, P1-RECUT,
A-1-435-RECUT as superseded-by-516), JOURNAL entry, validation ledger
(new golden SHAs). Note in DECISIONS: 515 retired, provenance of
de6f7c8 carried by 516 identity measurement.

Commit separable: Part C reference re-cut is its own commit; docs its
own. Push only on Milan's authorization.
