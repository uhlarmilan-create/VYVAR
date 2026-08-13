CURSOR RESULT - 2026-08-11 13:45 UTC+2

What I did
Completed Phase 3 final table drops: OBS_FILES then OBS_DRAFT. Repointed all readers/writers
to draft_manifest.json; rebuilt FINAL_DATA integrity via manifest scan (equipment/telescope
pairs from rig + QC run draft_ids). Removed CREATE/migrations/Explorer picker entries.
DROP TABLE IF EXISTS on open for both. Gates: --fast PASS, local P1 A/B byte-identical both drops.

## Output / findings

**Final HEAD:** `1c6ca81` (after tmp artifact cleanup commit if applied)

**Drop commits:**
| Table | Commit | Message |
|-------|--------|---------|
| OBS_FILES | `d4ced2a` | Phase 3: drop OBS_FILES table (manifest files[] sole store) |
| OBS_DRAFT | `1c6ca81` | Phase 3: drop OBS_DRAFT table (manifest sole draft store) |

**Prior Phase 3 (already on main):**
| Table | Commit |
|-------|--------|
| SCANNING | `84ff9a1` |
| OBSERVATION | `3eea08c` |

**Gates (both drops):**
- `--fast`: OVERALL PASS (1292 passed, 27 skipped)
- P1 A/B core SHA: `24820ee282e5c03020e16757201bad624050d0a4bc78e3b137584f23debe517b` (n=325) -- byte-identical pre/post each drop
- manifest-db-parity: PASS draft_id=435

**grep (src_py):** No CREATE/SELECT/INSERT/UPDATE/DELETE/JOIN on SCANNING/OBSERVATION/OBS_FILES/OBS_DRAFT; only DROP TABLE IF EXISTS on open (+ benign comment/doc mentions).

**FINAL_DATA:** Manifest-sourced via `iter_manifest_final_data_pairs()`; SQL view dropped.

**Explorer:** Picker lists TELESCOPES/EQUIPMENTS/LOCATION only; draft manifests in dedicated section.

**Done state:** All four staging tables gone from src_py SQL; manifest is sole draft/file store; anchor draft_000435 runnable.

## Errors (if any)
None blocking. Accidental inclusion of dev/tests/_tmp_batch_e_lc/ in OBS_DRAFT commit -- removed in follow-up commit.

## Files changed
- src_py/database.py, draft_provenance.py, ui_database_explorer.py, pipeline.py (+ comment cleanup)
- dev/tests (manifest-only tests), dev/scripts/session_baseline_check.py, fix_draft_equipment.py
- Commits `d4ced2a`, `1c6ca81` (+ cleanup if pushed)
