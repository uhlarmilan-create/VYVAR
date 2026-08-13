CURSOR RESULT - 2026-08-11 Phase 2.3 rig/location read consolidation

What I did
Added manifest-first `get_draft_equipment_id`, `get_draft_telescope_id`, and
`get_draft_location_id` on VyvarDatabase; routed all direct OBS_DRAFT rig/location
FK reads through them; simplified `_draft_location` / `_draft_id_location`; added
tests; fixed null-island mock tests for the new accessor path.

## Output / findings

### Accessors (database.py)
- `_draft_rig_resolve_row()` - single raw DB fetch for rig FKs + path hints
- `_resolve_draft_rig_id()` - manifest-first via `resolve_rig_id_manifest_first`
- `get_draft_equipment_id()`, `get_draft_telescope_id()`, `get_draft_location_id()`

### Direct reads replaced
| site | change |
|------|--------|
| crowding_index.py `_gain_rn_for_draft` | `get_draft_equipment_id` |
| photometry_core.py `_resolve_phase2a_equipment_id` | accessor |
| photometry_core.py Phase 2A site log | `get_draft_location_id` |
| pipeline.py post-cal QC linkage | `get_draft_equipment_id` |
| psf_runner.py `_draft_equipment_id` | accessor |
| database.py `get_combined_metadata` | equipment + telescope accessors |
| database.py `fetch_obs_draft_telescope_equipment` | uses accessors |
| param_resolver.py `_draft_location` / `_draft_id_location` | accessor only |

### Grep (rig FK bypass reads)
Only `database.py` `_draft_rig_resolve_row` (central accessor DB source) plus
writer/parity raw paths in `draft_provenance.py` (`SELECT *`, `ARCHIVE_PATH`).

### Shadow report
```
drafts=55 equal=440 absent=0 mismatch=0 fallback=0
```

### Tests
- `dev/tests/test_manifest_rig_accessors.py` (3 new)
- `dev/tests/test_obsloc_null_island.py` updated for `get_draft_location_id` mocks
- manifest flip/shadow suite: **12/12 PASS**

### Gates
| gate | result |
|------|--------|
| `--fast` | **OVERALL PASS** (1288 passed, 27 skipped) |
| P1 A/B (2.3 vs prior 2.2) | **byte-identical** `24820ee2...` n=325 |

P1 with Phase 2.3 code: core SHA `24820ee282e5c03020e16757201bad624050d0a4bc78e3b137584f23debe517b`
n=325 -- matches Phase 2.2 P1 from prior session. Manifest mirrors DB on all 55 drafts,
so consolidated accessor routing is science-neutral.

### Transition note (unchanged from 2.2)
Raw external OBS_DRAFT edits bypass manifest refresh; recovery =
`dev/tools/backfill_draft_manifests.py`. Gone at Phase 3.

## Errors (if any)
None blocking.

## Files changed
- src_py/database.py, draft_provenance.py (2.2+2.3 cumulative)
- src_py/crowding_index.py, photometry_core.py, pipeline.py, psf_runner.py, param_resolver.py
- dev/tests/test_manifest_rig_accessors.py (new)
- dev/tests/test_obsloc_null_island.py

Not committed (awaiting Milan approval).
