CURSOR RESULT - 2026-07-18 (DB-SEED-SPLIT)

What I did
Split the author's observatory seed out of the product DB init path. A new user's
fresh database is now EMPTY of Location / Telescope / Equipment / Scanning rows;
the author rows live as a harness-only fixture in `dev/tools/reference_seed.py`
and are wired only where Step 1 found real consumers. Installer FINISH text,
INSTALL.md, and the CZ install-guide PDF (v1.2 narrative) updated to match.
Stack stays HELD for the Lenovo test + PUSH protocol.

## STEP 1 - consumer trace

Question: does the `--full` gate create a fresh DB, or use the production one?

Answer: **uses the production DB** (`VyvarDatabase(cfg.database_path)` in
`dev/scripts/session_baseline_check.py` ~L360). The author's populated
`vyvar.sqlite3` is never re-initialised. Draft FKs
(`ID_LOCATION` / `ID_EQUIPMENTS` / ...) on production drafts therefore do **not**
depend on `initialize_database()` seed - those rows already exist.

Who actually needed the seed on a *fresh* DB:

| Consumer | Path | Needs seed? | Wiring |
|----------|------|-------------|--------|
| `--full` anchor harness | `dev/scripts/session_baseline_check.py` | Soft: INSERT OR IGNORE on production is a no-op; needed if gate ever opens an empty DB so draft optics FKs resolve | `seed_reference_observatory(db)` before `run_full_photometry_pipeline` |
| pytest that `create_draft` with FK ids 1/1/1/1 | `dev/tests/test_pre_calibrated_run.py` (2 tests) | Yes on fresh tmp DB | call seed before `create_draft` |
| pytest that UPDATEs equipment id=4 + draft FK telescope id=1 | `dev/tests/test_fix_draft_equipment.py` | Yes | call seed in `_seed_equipment_db` |
| Installer SMOKE / new users | `install_vyvar.ps1` / `.sh` | Must NOT seed | schema-only; assert empty when file is brand-new |
| Production drafts (Archive) | author's live DB | No - DB never re-init | n/a |

Nothing besides tests/harness needs the seed on a fresh DB. Product path stays empty.

Exact pre-split seed (moved verbatim - task text mentioning Jirny/Zdanice/C3-26000
was approximate; the committed seed matches HEAD's old `initialize_database`):

- EQUIPMENTS: QHY294MM(1), C5A-150M(4)
- TELESCOPE: Carl-Zeiss(1), AZ800(6)
- LOCATION: Dablice(1)
- SCANNING: Clear 120s bin11 ?10  degC (1)

## STEP 2 - split

1. `VyvarDatabase.initialize_database()` - no-op (schema from `_create_tables`;
   reference tables stay empty). Docstring documents the product decision.
2. `dev/tools/reference_seed.py::seed_reference_observatory(db)` - INSERT OR IGNORE
   of the exact author set (kept out of `src_py`).
3. Wired in: session_baseline_check `--full`, three pytest sites above.
   Installer does **not** call it.
4. Installer SMOKE: if the DB file did not exist before open, assert
   EQUIPMENTS/TELESCOPE/LOCATION counts == 0. FINISH text + INSTALL.md: fresh DB
   empty; user creates their own observatory records.

## STEP 3 - guards

- `dev/tests/test_reference_seed.py`: empty init; seed pins exact ids/names;
  second seed call is idempotent.
- `dev/tests/test_fresh_machine_startup.py`: asserts empty reference tables;
  `observer_location_id=2` hydrates to None on empty DB (graceful).

## Coordination - install guide v1.2

Refreshed `dev/tools/docs_pdf/build_install_guide.py` from Claude's delivery
(stripped a stray module-level `pass` that would have been a no-op syntax wart).
Regenerated `docs/VYVAR_INSTALL_GUIDE_CZ.pdf`: 5 pages; contains `PRAZDNA`,
`v1.2`, and the unresolved-location note.

## STEP 4 - gates

```
pytest:  973 passed, 19 skipped
--fast:  OVERALL PASS
--full:  OVERALL PASS (2158s photometry)
  full-science-compare         PASS   n_lc=166 failures=0
  full-snapshot-sha-core       PASS   3d26f4692ac81fc5... n=333
  full-photometry-sha-core     PASS   3d26f4692ac81fc5... n=333
  full-photometry-sha-extended PASS   6420f1daa53a0d5d... n=499
  full-counters-expected       PASS   allowlisted phase2a_empty_comp_drop=1
```

Byte-identical: the seed call on the production DB is INSERT OR IGNORE (no-op),
and the gate proves the split preserved the anchor context.

## Files changed
- src_py/database.py
- dev/tools/reference_seed.py (new)
- dev/scripts/session_baseline_check.py
- dev/tests/test_reference_seed.py (new)
- dev/tests/test_fresh_machine_startup.py
- dev/tests/test_pre_calibrated_run.py
- dev/tests/test_fix_draft_equipment.py
- install_vyvar.ps1 / install_vyvar.sh
- INSTALL.md
- dev/tools/docs_pdf/build_install_guide.py
- docs/VYVAR_INSTALL_GUIDE_CZ.pdf
- dev/validation/VYVAR_VALIDATION_LEDGER.json (if --full re-stamped)
- dev/results/CURSOR_RESULT_db_seed_split.md

## Push
HELD - Lenovo stranger test + Milan's PUSH protocol.
