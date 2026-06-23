CURSOR RESULT — 2026-06-23 (Tier-1 broad-except STEP 2 — high-value core only)

What I did
Implemented BATCH 1–3 (10 sites): narrowed broad `except Exception` handlers at 4 MEDIUM core locations + 4 photometry_core debug-only sites + infolog `log_event` (broad kept, DEBUG swallow log added). Deferred 38 SAFE UI sites — logged as TIER1-UI-DEBT in ledger. Added `tests/test_config_observer_location_hydrate.py` (2 cases). No commit/push (awaiting Milan sign-off).

## Output / findings

### BATCH 1 — MEDIUM core (fallback/control flow unchanged; logging added)

| Site | Narrowed types | Fallback unchanged |
|------|----------------|-------------------|
| `config.py:1061` | `sqlite3.Error`, `TypeError`, `ValueError` | On DB hydrate failure, json-loaded `observer_lat`/`lon`/`alt_m`/`location_name` are left as set before the try block (including 0.0 coords when json has zeros). No DB values applied. |
| `ui_quality_dashboard.py:434` | `ImportError`, `KeyError`, `TypeError`, `ValueError`, `AttributeError` | Success path (st.info / st.warning for missing RN / st.caption) unchanged. Failure now surfaces `st.warning` + `logging.WARNING` instead of silent pass. |
| `app.py:1568` | `sqlite3.Error`, `TypeError`, `ValueError`, `KeyError` | `_equip_sat` from `_equipment_saturate_adu_from_db` retained when combined-metadata lookup fails; only adds `log_event` WARNING. |
| `app.py:2409` | `sqlite3.Error`, `TypeError`, `ValueError`, `KeyError` | Import completion flow unchanged; failed SATURATE_ADU log now emits WARNING via `log_event` instead of silent pass. |

### BATCH 2 — photometry_core (narrow only; DEBUG level unchanged)

| Site | Narrowed types |
|------|----------------|
| `:7214` | `ImportError`, `KeyError`, `TypeError`, `ValueError`, `AttributeError` |
| `:7867` | `ImportError`, `KeyError`, `TypeError`, `ValueError`, `AttributeError`, `OSError` |
| `:9632` | `ImportError`, `AttributeError`, `ValueError`, `TypeError` |
| `:9882` | `ValueError`, `TypeError` |

### BATCH 3 — infolog.py:35

- Kept broad `except Exception` with `# noqa: BLE001` + comment (logging must not crash pipeline).
- Added `logging.debug("log_event append failed: …")` for swallowed errors.

### config.py:1060 test

`tests/test_config_observer_location_hydrate.py`:
- `test_observer_location_hydrate_db_failure_keeps_json_coords_and_warns` — patched `sqlite3.Error` → json lat/lon/alt/name preserved + WARNING logged.
- `test_observer_location_hydrate_db_failure_keeps_zero_coords` — 0.0/0.0 coords stay 0.0 on failure + WARNING.

### Validation

- `ruff check . --select BLE001,E722` → **All checks passed!** (0 unmarked)
- `pytest -q` → **447 passed, 15 skipped** (was 445/15; +2 new tests)

### Ledger

- `docs/VYVAR_FULL_AUDIT_LEDGER.md` — new **TIER1-UI-DEBT** entry (38 SAFE UI/plotly sites deferred LOW cosmetic).

### Deferred (not touched)

- 38 SAFE UI/plotly `pass` sites → TIER1-UI-DEBT
- 299 pipeline/photometry_core defensive passes → existing phased-audit item

## Errors (if any)

None.

## Files changed

- `config.py` — observer hydrate narrow + WARNING; `import sqlite3`
- `ui_quality_dashboard.py` — Gain/RN panel narrow + st.warning + logging.WARNING
- `app.py` — saturate_adu sites narrow + log_event WARNING; `import sqlite3`
- `infolog.py` — DEBUG swallow log; noqa comment
- `photometry_core.py` — 4 narrowed debug handlers
- `tests/test_config_observer_location_hydrate.py` — new
- `docs/VYVAR_FULL_AUDIT_LEDGER.md` — TIER1-UI-DEBT
- `CURSOR_RESULT.md` — this report

No commit hash (awaiting Milan sign-off → approved 2026-06-23 for commit/push).
