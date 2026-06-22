CURSOR RESULT — 2026-06-19 (Group 7 audit — UI shell — 7-group map complete)

What I did
AUDIT-ONLY final group: 14 UI modules (144 functions), AST + L1–L11 scans, Phase B verification of G6 parity items from UI source. Appended ledger + **7-group map complete** summary. No code fixes.

## Module set + counts

14 modules · **144 functions** · ~9.8k LOC (`app.py` + `ui_*.py` except Group 4: `ui_variability`, `ui_hrd`, `ui_masterstar_qa`).

| Module | Funcs |
|--------|-------|
| `app.py` | 29 |
| `ui_aperture_photometry.py` | 25 |
| `ui_quality_dashboard.py` | 14 |
| `ui_select_stars.py` | 13 |
| + 10 smaller `ui_*.py` | 52 |
| **Total** | **144** |

## Phase B — G6 parity resolution (UI side)

| Symbol | Pattern | Verdict |
|--------|---------|---------|
| `phase01_comparison_max_bv_diff` | Direct `cfg.` | **AttributeError** on Select Stars page (expander `st.code` + run) — **HIGH** G7-F001 |
| `phase01_use_bprp_primary` | `getattr(..., True)` | No crash; non-persist — **MED** G7-F003 (downgrades G6-F004 severity) |
| `max_bv_diff=` kwarg | `run_phase0_and_phase1(...)` | **Not in core signature** — stale API **HIGH** G7-F002 |

34 PARAMS config-only keys: mostly **intentionally-hidden** (blind cluster, neighbor-sub, session observer json).

## Findings summary

| Sev | Count | Headline |
|-----|-------|----------|
| HIGH | 2 | G7-F001 crash + G7-F002 stale `max_bv_diff` kwarg |
| MED | 5 | Silent except handlers; frame gates OFF-by-design; display column drift risk |
| LOW | 1 | Minor silent passes |
| CLEAN | 3 | Settings save path; no rig literals in shell; delegates to core |

**DEAD:** 0 TRULY-DEAD / 144 · **29 FLAGGED** · **52 NEEDS-TEST**

## 7-group map complete

**1728 functions** audited across 7 groups (see ledger table). Fix-pass queue: G1-F001/F002, G3-F002, G6/G7 config parity.

Artifacts: `tmp/audit_group7_inventory.py`, `tmp/audit_group7_results.json`

## Files changed

- `docs/VYVAR_FULL_AUDIT_LEDGER.md` (Group 7 checkpoint + map summary)
- `CURSOR_RESULT.md`

**No fix steps** — checkpoint for Claude review before HIGH fix-pass.
