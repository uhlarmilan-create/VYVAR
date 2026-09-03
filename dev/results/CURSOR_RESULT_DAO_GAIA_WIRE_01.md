CURSOR RESULT - 2026-08-19 13:57 UTC+2 (DAO-GAIA-WIRE-01)

What I did
Wired DAO-GAIA-STAGE-01 iter4 config into production (`src_py/`), rebuilt
MASTERSTAR catalog + census on live draft 516, delivered Milan UI
artifacts. **HARD STOP** - no push, no Phase 0/1/2A, no pool regen.

## Premise check

Compared: **iter4 sandbox harness** (star-masked sky ? pass1, born-owned
pass2, forced seed) vs **production** (was rms_conv pass1, global pass2
merge, pass2 ?=5). Same frames conceptually; production census uses
`field_catalog_cone.csv` on-chip rows (n=4131) vs iter4 full Gaia DB
query eligible set (~6168 G?16). Numbers are not byte-comparable; state
*semantics* are.

## Part A - Production wiring

| Item | Change |
|------|--------|
| Pass1 threshold | `masterstar_dao_threshold_sigma` x **star-masked sky ?** (NOT rms_conv); rms_conv retained diagnostic-only |
| Pass2 | ?=**4.0**, tol=**2 px**, seeds G?**15**, **born-owned** (`dao_pass2_born_owned_rows`, `merge_dao_pass1_pass2_born_owned`) |
| Pass1 dedup | **0.75 px** spatial before pass2 |
| Catalog match | Born-owned pass2 pre-lock by `vy_seed_catalog_id` (no greedy reassignment) |
| FORCED_SEED | unchanged path via `enrich_masterstar_gaia_complete`; SNR?4, centroid?2 px |
| Lock assign | Pass2 catalog_id wins over pass1 proximity (`lock_existing_and_leftover_assign`) |
| Config defaults | `config.json` + `config.py`: pass1=4.5, pass2=4.0, depth=15.0 |
| Overlay/UI | `masterstar_qa_plot.py`: green hollow=P1/P2, cyan fill=FORCED_SEED, gold slash=ambiguous; caption in `ui_masterstar_qa.py` |
| PARAMS | Updated keys in `docs/VYVAR_PARAMS.md` |

Key files: `src_py/masterstar_gaia_accounting.py`, `src_py/pipeline.py`,
`src_py/config.py`, `config.json`.

Tests: `dev/tests/test_masterstar_gaia_01.py` - **5/6 PASS**
(INV-DET-FALSEFILL-01, INV-SEED-FALSEFILL-01, INV-MS-CENSUS-01 OK at
?=4.0/4.0). `test_part_a_report_exists` fails pre-existing report schema
(E1 key absent).

## Part B - MS build draft 516 (47 s detect + 34 s census re-enrich)

Command: `python tmp/dao_gaia_wire_01_ms_build.py`

Config logged: `sky_sigma=39.85`, `threshold=179.3` (=4.5x39.85),
`pass2_sigma=4.0`.

### Census (production cone, after re-enrich)

Path: `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/gaia_source_state_census.csv`

| State | n |
|-------|--:|
| DETECTED_P1 | 2157 |
| DETECTED_P2 | **367** |
| FORCED_SEED | 123 |
| BLENDED | 221 |
| SEED_REJECTED | 672 |
| TOO_FAINT | 511 |
| EDGE | 80 |
| **Total** | **4131** |

INV-MS-CENSUS-01: **PASS** (4131 rows = on-chip cone count).

### vs iter4 sandbox (MASTERSTAR, different denominator)

| Metric | iter4 sandbox | production wire01 |
|--------|---------------|-------------------|
| Eligible set | ~6168 G?16 (Gaia DB query) | 4131 (cone on-chip) |
| Pass1 detections | 2636 | 2245 (vy_dao_pass=1 in CSV) |
| Pass2 detections | 217 | **367** (vy_dao_pass=2) |
| FORCED_SEED | 163 | 123 |
| ambiguous_owner | 8 | **1** (pass2 peak flag) |
| red X G?14 TOO_FAINT | 41 | **0** (cone census; different faint band accounting) |

**Deviation notes:**
- Cone catalog (4131) ? iter4 Gaia query set ? completeness denominators differ.
- `astrometry_optimizer` still runs after DAO; `vy_dao_pass` preserved in CSV; pipeline inline enrich needed provenance re-merge (fixed in code; wire script re-enriches census post-build for correct P1/P2 split).
- `ambiguous_owner`: fixed NaN?True bug in census propagation (was 257 false flags).
- INV-DAG-01 FAIL logged (pre-existing stage seq stamp); not introduced by this wire.

### Deliverables (Milan UI)

Context: `dev/results/context/session_20260819_wire01/`
- `gaia_source_state_census.csv`
- `masterstars_full_match.csv`
- `wire01_build_summary.json`
- `ambiguous_owner_list.csv` (if any)
- `red_x_list.csv` (empty - 0 red X at G?14 TOO_FAINT in cone census)

Live draft paths updated (catalog layer only):
- `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/masterstars_full_match.csv`
- `Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/gaia_source_state_census.csv`

**Photometry products:** `generate_masterstar_and_catalog` also refreshed
comparison_stars / photometry_plan sidecars - verify SHA **477dc8cf**
before any GO; revert source remains
`draft_000516_snapshot_cleanrebuild_20260818` (read-only).

## HARD STOP

No push. No Phase 1/2A. Milan reviews VYVAR UI MASTERSTAR QA tab:
- green hollow = DAO P1/P2
- cyan filled = FORCED_SEED
- gold slash = ambiguous_owner
- census table in QA caption

Verdict options per task: **GO new era** or **NO-GO revert** from snapshot.

## Files changed

- `src_py/masterstar_gaia_accounting.py`
- `src_py/pipeline.py`
- `src_py/config.py`
- `config.json`
- `src_py/masterstar_qa_plot.py`
- `src_py/ui_masterstar_qa.py`
- `docs/VYVAR_PARAMS.md`
- `tmp/dao_gaia_wire_01_ms_build.py`
- `dev/results/context/session_20260819_wire01/*`
- `dev/results/CURSOR_RESULT_DAO_GAIA_WIRE_01.md`

Push not authorized. Wall **~81 s** (build + census).
