CURSOR RESULT — 2026-07-08 (EXCEPT-BATCH-S0)

What I did
Stage 0: HRD-PLOT-TUPLE fix + PDF placeholder; Stage 1: production silent-failure census;
docs close + push.

## Stage 0 — HRD root cause

**Hypothesis:** missing `row_factory` ? **REFUTED** (already `sqlite3.Row` at line 73).

**Actual mechanism:** `sqlite3.Row` iteration yields **values**, not keys. `{k: row[k] for k in row}`
at `hrd_analysis.py:113` used Gaia `source_id` integers as indices ? `IndexError: tuple index out of range`.

**Fix:** `{key: row[key] for key in row.keys()}`.

**PDF:** `_report_hrd_unavailable_page`; narrowed `_hrd_build_errors`; `logging.exception` at ERROR;
missing inputs also emit placeholder (no silent skip).

**Validation:**
- draft_424 PDF: page 15 = Field astrophysics + HRD highlights table; 6088 stars / 2474 reliable
- `overflow_violations`: 0
- pytest: **588 passed**; ruff green

## Stage 1 — Census

`docs/VYVAR_EXCEPT_CENSUS.md` — **625** sites, stable IDs EXC-0001…

| Tier | Count |
|------|------:|
| T1-SCIENCE | 354 |
| T2-INTEGRITY | 82 |
| T3-UI | 76 |
| T4-LEGIT | 3 |
| ? | 110 |

F-EXCEPT-TIER1 reconcile: pipeline pass/continue ~95 + photometry_core ~66 ? 160.

## Commit
(pending)
