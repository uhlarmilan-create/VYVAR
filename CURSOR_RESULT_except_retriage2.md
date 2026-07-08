CURSOR RESULT - 2026-07-08 13:57

What I did
Applied EXCEPT-RETRIAGE-2 evidence triage to 170 pipeline.py census rows (EXC-0275-EXC-0444).
Inserted Tranche 2 summary + TOP-10 block into docs/VYVAR_EXCEPT_CENSUS.md.

## Output / findings

| Tier | Count |
|------|------:|
| T1-SCIENCE | 83 |
| T2-INTEGRITY | 41 |
| T3-UI | 20 |
| T4-LEGIT | 26 |
| ? | 0 |

| Disposition | Count |
|-------------|------:|
| narrow+log-ERROR | 80 |
| fix-now | 26 |
| narrow+comment(T4) | 26 |
| leave+comment | 20 |
| delete-dead | 18 |

**CAL-DIAG:** 14 | **silent_drop:** 10 | **fix-now:** 26

## Errors (if any)
None

## Files changed
- docs/VYVAR_EXCEPT_CENSUS.md
- docs/VYVAR_JOURNAL.md
- CURSOR_RESULT_except_retriage2.md
- sandbox/_except_retriage2_apply.py (local, gitignored)

**Commit:** `d183261` — pushed to `main`.
