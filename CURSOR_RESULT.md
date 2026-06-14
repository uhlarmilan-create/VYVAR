CURSOR RESULT — 2026-06-14 (session consolidation)

What I did
Gate tail fix (`_brno_check` → draft_400 / skip); synced STATE/DECISIONS/PROCESS/ROADMAP/JOURNAL;
pre-commit verification; full commit + push.

## Output / findings

### Step 1 — Gate tail fix
`sandbox/anchor387_legacy_vs_scoped_gate.py`: `_brno_check` prefers `draft_000400`, falls back to
399, skips cleanly if absent. Added `--compare-only` for fast re-verify when `tmp/anchor387_*` exist.

### Step 2 — Docs synced
See `docs/VYVAR_STATE.md`, `VYVAR_DECISIONS.md`, `VYVAR_PROCESS.md`, `VYVAR_ROADMAP.md`,
`VYVAR_JOURNAL.md`.

### Step 3 — Verification

| Check | Result |
|-------|--------|
| `pytest tests/` | **284 passed, 12 skipped** |
| `ruff` BLE001/E722 (touched) | **clean** |
| Gate `--compare-only` | **0 science failures**, Brno **94.2%**, **exit 0** |

### Step 4 — Commit

**SHA:** `70c23d0` — pushed to `origin/main`.

### Open confirmation
Brno **not fully locked** until Milan's **draft_401 UI sign-off + overlay**.

## Files changed
Gate tail, docs, solver fix (prior session), CURSOR_RESULT.md
