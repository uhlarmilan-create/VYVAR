CURSOR RESULT - 2026-08-10 (read-only audit)

What I did
Validated Claude's migration audit against repo HEAD 14b198f (read-only). Counted SQL
read/write sites, draft-related modules, database API surface, manifest state, and linchpin
(equipment_id) call paths. No code or schema changes.

## Output / findings

### Verdict: audit direction is correct; sequencing and guards stand

Retiring SCANNING / OBSERVATION / OBS_DRAFT / OBS_FILES is a multi-session migration, not a
DROP. The manifest must carry rig ids and per-frame reject/QC state before any table removal.
PHOTOMETRY_LIGHT_CURVE Phase-0 drop is independently safe.

### Validated counts (src_py, literal SQL)

| Table | Reads | Write sites | Files with reads |
|-------|------:|------------:|------------------|
| SCANNING | 6 | 2 | database, photometry_core, ui_database_explorer, ui_finalization |
| OBSERVATION | 7 | 3 | database, ui_database_explorer |
| OBS_DRAFT | 31 | 16 | 11 files (database, pipeline, param_resolver, ui_*, photometry_core, ...) |
| OBS_FILES | 32 | 17 | 9 files (database, pipeline, photometry_core, ui_*, calibration, ...) |
| PHOTOMETRY_LIGHT_CURVE | 0 | 0 | schema-only in database.py |

Minor count deltas vs brief: OBS_DRAFT reads 31 (not 30), OBS_FILES reads 32 (not 29).
SCANNING/OBSERVATION/PHOTOMETRY_LIGHT_CURVE match exactly.

### Module footprint

- **35 modules** in src_py reference draft_id / OBS_DRAFT / OBS_FILES / fetch_obs_draft (matches brief).
- **52 database.py methods** touch draft/obs/scanning (create_draft, fetch_draft_light_rows_for_quality, insert_draft_files, finalize_draft_to_observation, ...).
- **67 DB API call sites** across src_py (pipeline.py 40; ui_finalization 5; app/importer 3 each).
- Broad identifier touchpoints (includes comments/docstrings): pipeline.py ~466, database.py ~527.
  Brief's 266/217 likely used a narrower definition; SQL-site counts above are the actionable gate for Phase 3.

### Linchpin confirmed

Rig FKs enter at draft creation from explicit user scan-source selection:

```4066:4095:src_py/database.py
    def create_draft(self, data: dict[str, Any]) -> int:
        ...
        id_equipments = int(data.get("id_equipments", 1))
        id_telescope = int(data.get("id_telescope", 1))
        id_location = int(data.get("id_location", 1))
        id_scanning = int(data.get("id_scanning", 1))
        ...
        INSERT INTO OBS_DRAFT (ID_EQUIPMENTS, ID_TELESCOPE, ID_LOCATION, ID_SCANNING, ...)
```

Runtime equipment_id resolution today reads OBS_DRAFT only (no manifest fallback):

- `pipeline.py:1732` -- observation hash (ID_EQUIPMENTS + ID_TELESCOPE)
- `pipeline.py:3400-3401` -- plate-scale FOV hint
- `pipeline.py:15306-15310` -- post-cal QC config
- `param_resolver.py:646-692` -- site via OBS_DRAFT JOIN LOCATION

After migration, manifest must supply equipment_id / telescope_id / location_id before any DROP,
or DB-first gain/RN/saturation/focal/site resolution silently degrades.

### draft_manifest.json today

Confirmed minimal payload in `draft_provenance.py:71-89`:

- `draft_id`, `calibration_mode`, `updated_utc`, optional `extra` (e.g. `observer_location`)
- Dual-write for calibration_mode already exists (`record_draft_calibration_provenance`, line 153)
- **No rig ids, paths, status, or files[] yet**
- No `draft_manifest.json` found under `Archive/` in this workspace (manifests may exist only on
  imported drafts at runtime; anchor draft_000435 currently DB-backed on disk layout)

### Table mapping -- spot checks

**SCANNING:** Header-redundant confirmed. Reads are JOIN/BINNING lookups + UI preview; resolver
already has `resolve_exptime`, `resolve_binning`, gain-index mapping in param_resolver.

**OBSERVATION:** Light duplicate of draft FKs + center/JD; 7 reads all database.py or explorer UI.
Lifecycle effectively INGESTED/FINALIZED.

**OBS_DRAFT:** Heavy hub. Beyond SQL, 52 database API methods and ui_components status-panel
persistence (`update_obs_draft_status_panel_values`) depend on it.

**OBS_FILES:** Heaviest. Schema evolution at database.py:2648-2710 confirms QC + inspection +
reject + grouping + calib columns. DB-first frame selection at pipeline.py:2363-2404
(`FWHM IS NOT NULL` gate) coexists with disk glob elsewhere -- migration must unify on sidecar/manifest.

**Non-derivable state (must migrate):** IS_REJECTED, REJECTED_AUTO, QC_PASSED, inspection metrics
after human/pipeline decisions; brief is correct.

### Migration sequencing -- endorsed with one addition

| Phase | Action | Gate |
|-------|--------|------|
| 0 | DROP PHOTOMETRY_LIGHT_CURVE + 2 indexes | grep 0 readers (PASS today) |
| 1 | Widen manifest; dual-write rig ids, paths, status, files[] | diff manifest vs DB |
| 2 | Rewire readers module-by-module | --fast per module; --full on science-path |
| 3 | DROP tables when grep = 0 per table | anchor draft_000435 runnable |

**Suggested Phase 2 order (matches brief):** SCANNING -> OBSERVATION -> OBS_FILES sidecar ->
OBS_DRAFT manifest last.

**Addition:** Rewire `database.py` API layer (52 methods) in Phase 2 alongside call sites -- many
modules call `fetch_draft_light_rows_for_quality` etc., not raw SQL. Phase 3 grep should include
method names / table string literals / fetch_* wrappers.

### Anchor / baseline guards

- draft_000435 is disk-resident; backfill manifest for it before dropping OBS_DRAFT.
- Any change to frame selection (OBS_FILES reject path) or resolver equipment_id sourcing is
  science-path -> `--full` anchor A/B required.
- Independent safe items (ACTIVE YES/NO, EQUIPMENTS.FOCAL drop, PHOTOMETRY_LIGHT_CURVE drop) remain
  separable from this arc.

### Risks unchanged

1. Linchpin first (equipment_id in manifest).
2. Frame reject decisions must not be lost.
3. Do not bundle with config/DB-facts cleanup.
4. Byte-identity gate on science-path rewires.

## Errors (if any)

None (read-only).

## Files changed

None (audit only). This report: `dev/results/CURSOR_RESULT_audit_retire_obs_tables.md`.
