CURSOR RESULT — 2026-06-19 (Group 6 audit — config / orchestration / utils)

What I did
AUDIT-ONLY Group 6 checkpoint: 12 modules (110 functions), AST inventory + L1–L11 lens scans, `AppConfig.__post_init__` read, config↔UI parity cross-check vs `VYVAR_PARAMS.md` / `config.json`. Appended ledger section. No code fixes.

## Module set + counts

| Module | Funcs | Lines |
|--------|-------|-------|
| `config.py` | 19 (+ `__post_init__` load surface) | 2294 |
| `utils.py` | 34 | 736 |
| `night_run.py` | 19 | 1128 |
| `infolog.py` | 10 | 138 |
| `vyvar_ui_status.py` | 4 | 90 |
| `inspect_drafts.py` | 4 | 154 |
| `lunar_context.py` | 6 | 139 |
| `orchestrator/vyvar_orchestrator.py` | 6 | 206 |
| `scripts/_build_vyvar_params.py` | 4 | 578 |
| `simulate_night_run.py` | 2 | 164 |
| `run_crowding_index.py` | 1 | 86 |
| `run_smoothness_report.py` | 1 | 61 |
| **Total** | **110** | ~4634 |

Excluded (prior groups): `param_resolver`, `time_utils`, `draft_provenance`, `fits_suffixes`, `masterstar_context` (G3). `ui_*.py` / `app.py` scanned for parity only → Group 7.

## Findings summary

| Sev | Count | IDs |
|-----|-------|-----|
| HIGH | 4 | G6-F001 rig literals; G6-F002 validity-day defaults; G6-F003 `max_bv_diff` missing field; G6-F004 `phase01_use_bprp_primary` getattr-only |
| MED | 7 | G6-F005–F011 (silent config load/swallow, orphan json key, PARAMS drift bucket) |
| LOW | 1 | G6-F012 inspect_drafts silent except |
| CLEAN | 4 | G6-P001–P004 |

**DEAD reclassification:** 0 TRULY-DEAD / 110 functions (heuristic DEAD 0 — do not apply G3 88% over-count here).

**Coverage:** 22 FLAGGED · 49 NEEDS-TEST · 4 LIVE-DYNAMIC (nested closures).

## Config ↔ UI parity (headline)

- `AppConfig` fields **278** · `config.json` **252** · UI refs **95** · PARAMS keys **259**
- PARAMS summary: 82 exposed · 136 intentionally-hidden · **34 config-only (`no`)** · **2 UI-without-field**
- Verified: `phase01_comparison_max_bv_diff` **not** on `AppConfig` — `ui_select_stars.py` direct access → AttributeError risk
- Orphan json: `phase2a_variable_xy_fallback_mag_tol`

Artifacts: `tmp/audit_group6_parity.py`, `tmp/audit_group6_parity.json`

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_FULL_AUDIT_LEDGER.md` (Group 6 checkpoint)
- `CURSOR_RESULT.md`
- `tmp/audit_group6_parity.py`, `tmp/audit_group6_parity.json` (gitignored)

**No fix steps** — checkpoint for Claude review before Group 7.
