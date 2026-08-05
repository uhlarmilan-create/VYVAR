CURSOR RESULT - 2026-08-05 (INV-MS-01 removal)

What I did
Removed runtime invariant INV-MS-01 and restructured masterstars CSV write so annotate/bp_rp
failures cannot skip _vyvar_df_to_csv (A2 fix). Retained dao_only_fraction_from_masterstars as
informational census only. Updated tests and docs. Verified historical script greppers.

## Premise check (measured)

| item | value |
|------|-------|
| Anchor fixture | dev/results/context/session_20260727/draft_452_masterstars_full_match.csv |
| Anchor n / DAO_ONLY | 2951 / 109 |
| Anchor dao_only_fraction | 0.0369 (pytest.approx 0.0369 abs=0.002) |
| draft_501 (Newton pre-cal) | 0.417 (from prior diagnosis) |
| WARN / FAIL thresholds removed | 0.10 / 0.25 |

Conclusion unchanged: thresholds seeded on one rig/calibration mode are not portable; removal
not re-tuning.

## Output / findings

Acceptance criteria:
1. grep INV-MS-01 src_py/ -- **empty**
2. grep check_dao_only_fraction src_py/ dev/tests/ -- only test_dao_only_fraction_is_informational_only negative assertions
3. dao_only_fraction_from_masterstars callers preserved (pipeline census, audit scripts, tests)
4. Tests: **1236 passed, 26 skipped, 0 failed** (was 1235 passed baseline; +2 tests, -1 test)
5. Census line text: "MASTERSTAR DAO_ONLY census: N/M (fraction=X.XXX) -- informational, not a gate"
6. A2 regression: test_masterstars_csv_write_survives_bp_rp_failure PASS
7. P1 golden E2E: not re-run (--full gate non-authoritative per STATE.md ANCHOR-RESTORE-1)
8. df_final binding: assigned unconditionally at pipeline.py:12459 via _annotate_masterstars_flux_zones
   immediately before annotate try; earlier paths set df_final at 12382/12388/12407. CSV write at
   :12491 is outside annotate try and always reached if function reaches annotate block.

Historical script greppers (read-only):
- dev/scripts/draft_ui_equivalence_check.py _grep pattern absent -> returns None (no raise)
- dev/scripts/draft454_analysis.py _grep_line pattern absent -> returns None (no raise)

## Errors

None.

## Files changed

Runtime + tests (commit 1):
- src_py/pipeline.py
- src_py/invariants_runtime.py
- dev/tests/test_invariants_p2.py
- dev/tests/test_post451_part_b.py
- dev/tests/test_post453_infolog.py

Docs (commit 2):
- docs/VYVAR_INVARIANTS.md
- docs/VYVAR_LIMITATIONS.md
- docs/VYVAR_DECISIONS.md
- CHANGELOG.md
- dev/results/CURSOR_RESULT_inv_ms01_removal.md
