CURSOR RESULT - 2026-08-10 Phase 2.2 manifest-first rig-id reads

What I did
Flipped four rig-id accessors to manifest-first (DB fallback), closed the Database
Explorer OBS_DRAFT edit trap via post-commit manifest refresh, fixed numpy int64 FK
failure in editor saves, and validated with unit tests + P1 science-path A/B.

## Output / findings

### 1. Manifest-first rig-id reads (Phase 2.2)
Accessors now return manifest rig FK ids when present; DB fallback + MANIFEST_FALLBACK
counter + infolog when manifest absent or field missing:
- `database.py` `fetch_obs_draft_by_id` (equipment/telescope/location/scanning)
- `database.py` `fetch_obs_draft_telescope_equipment` (equipment/telescope + JOIN)
- `param_resolver.py` `_draft_location` (location_id / site coords)
- `param_resolver.py` `_draft_id_location` (location_id)

Helpers in `draft_provenance.py`: `resolve_rig_id_manifest_first()`, `apply_manifest_rig_to_draft_row()`,
`MANIFEST_FALLBACK` counter. Mismatch logging retained via `observe_manifest_rig_ids()`.
`manifest_db_parity_errors()` uses raw DB row fetch (not flipped accessor).

### 2. Edit trap closed
`apply_main_table_editor_save`: OBS_DRAFT added to `_EDITABLE_EDITOR_TABLES`; tracks
insert/update/soft-delete draft ids; after commit calls `_try_refresh_draft_manifest()` +
`clear_manifest_shadow_load_cache()`. EQUIPMENTS/TELESCOPE/LOCATION edits do NOT trigger
manifest refresh (resolver reads those tables live by id).

Fix for pandas/numpy editor path: `_coerce_sql_param` coerces `np.generic` via `.item()`
(FK failed under BEGIN with `np.int64` params).

### 3. Shadow report (all 55 drafts)
```
drafts=55 equal=440 absent=0 mismatch=0 fallback=0
```

### 4. Tests
`dev/tests/test_manifest_rig_flip.py` (4 tests):
- manifest present -> manifest rig id returned
- manifest absent -> DB fallback + fallback counter
- OBS_DRAFT editor save refreshes manifest (draft-438 scenario)
- telescope/equipment JOIN uses manifest telescope id

`dev/tests/test_manifest_shadow_rig.py` updated for 2.2 behavior.
**6/6 manifest tests PASS.**

### 5. Gates
| gate | result |
|------|--------|
| `--fast` | **OVERALL PASS** (1285 passed, 27 skipped) |
| P1 A/B (science-path) | **byte-identical** |
| `--full` snapshot vs run | see note below |

**P1 A/B (mandatory science-path check, committed config.json):**

| code | core SHA | core n |
|------|----------|--------|
| Phase 2.1 baseline (55cb365) | `24820ee282e5c03020e16757201bad624050d0a4bc78e3b137584f23debe517b` | 325 |
| Phase 2.2 (this task) | `24820ee282e5c03020e16757201bad624050d0a4bc78e3b137584f23debe517b` | 325 |

Identical. Manifest-first reads produce the same science outputs as 2.1 shadow-observe
(all 55 drafts equal=220 in 2.1; equal=440 field checks in 2.2 report).

**`--full` note:** Run completed but OVERALL FAIL on stale gate seeds (known
ANCHOR-GATE-SEED / ROADMAP item): `EXPECTED_PHOTOMETRY_SHA_CORE` in
`session_baseline_check.py` still points at draft_500 batch E (`5bccd85a...` n=497)
while snapshot on disk is draft_435 (`3d26f469...` n=333). Run photometry SHA with
restored committed config: `b9c9489a...` n=325 (P1-mini scope). This mismatch is
pre-existing and unrelated to the rig-id flip; P1 A/B is the authoritative check for
this science-path change.

### 6. Transition note (Phase 2 coexistence)
While OBS_DRAFT and manifest coexist, a raw external edit to OBS_DRAFT (sqlite CLI /
DB Browser, bypassing the app) will NOT refresh the manifest; the flipped read would
then use stale manifest rig ids. Recovery: re-run `dev/tools/backfill_draft_manifests.py`.
This limitation disappears at Phase 3 when OBS_DRAFT is dropped and manifest is the
only store.

## Errors (if any)
None blocking. `--full` OVERALL FAIL due to pre-existing anchor gate seed drift (documented above).

## Files changed
- `src_py/draft_provenance.py` - manifest-first resolution, counters, parity fix
- `src_py/database.py` - flipped accessors, OBS_DRAFT editor save + manifest refresh, numpy coerce
- `src_py/param_resolver.py` - flipped location accessors
- `dev/tests/test_manifest_rig_flip.py` - new (4 tests)
- `dev/tests/test_manifest_shadow_rig.py` - updated for 2.2
- `dev/tools/report_manifest_shadow_rig.py` - reports fallback counter

Not committed (awaiting Milan approval).
