CURSOR RESULT — 2026-06-22 (remove unwired Select Stars page — G7-F001/F002)

What I did
Confirmed `ui_select_stars.py` is unwired (no imports, no tests), deleted the file, verified phantom field removal and test suite, updated ledger G7-F001/F002 → RESOLVED. G7-F003 untouched.

## Step 1 — Isolation (current tree)

```
grep -rn "render_select_stars" --include=*.py .
→ ui_select_stars.py:28 (comment), ui_select_stars.py:358 (definition only)

grep -rn "ui_select_stars" --include=*.py . | grep -v ui_select_stars.py
→ (no matches)

grep -rln "ui_select_stars" tests/
→ (no matches)
```

**Verdict:** No live references — safe to delete.

## Step 2 — Delete

`git rm ui_select_stars.py` (622 lines removed)

## Step 3 — Post-delete verification

| Check | Result |
|-------|--------|
| `phase01_comparison_max_bv_diff` repo-wide | **0 matches** |
| Live `phase01_comparison_*` fields | Present in `config.py`, `ui_settings.py`, `photometry_core.py`, `ui_dao_stars.py` |
| `vyvar_ui_status` helpers | Still imported in `ui_aperture_photometry.py:25`, used at `:787-791` |
| `python -c "import app"` | **OK** |
| Full pytest | **406 passed, 15 skipped** |

## Step 4 — Ledger

- **G7-F001** → **RESOLVED** (dead-code removal)
- **G7-F002** → **RESOLVED** (dead-code removal)
- **G7-F003** → **STILL OPEN** (`phase01_use_bprp_primary` in `ui_aperture_photometry.py` only — defer to parity fix-pass)

## Errors (if any)

None.

## Files changed

- `ui_select_stars.py` (deleted)
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`

**Not pushed** — stop for Claude review.
