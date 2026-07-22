CURSOR RESULT - 2026-07-22 (encoding remediation)

What I did

Corrected misclassification of `test_ascii_policy` failures in
`CURSOR_RESULT_skipproc_qc_fix.md`, repaired four locally introduced non-ASCII
files via `ascii_migrate.py`, fixed cp1252-corruption `?` artifacts in result
docs, added workspace UTF-8 encoding settings, re-ran gates to **OVERALL PASS**.

---

## Push inventory

### `git log origin/main..HEAD --oneline`

```
4f140ef docs(invariants): QC-01 skip-processed allowlist contract + DECISIONS entry
263c6e7 fix(skipproc): qc_metrics allowlist gates alignment; full-set QC + prefilter stamping
a5ddfa1 fix(ui): material calibration icon and structured params JSON widgets
cff5502 fix(invariants): INV-FLUX-02 checks median (matches normalize_flat_master)
9533953 docs(osc): M71 L/B/G/R extraction experiment results
627bd35 docs(osc): IMX533 M71 re-run appended to gap inventory
cad6002 docs(osc): IMX533 discovery gap inventory; park OSC-SUPPORT arc
```

### `git status --short` (before remediation commit)

```
 M dev/results/CURSOR_RESULT_skipproc_qc_fix.md
 M dev/results/CURSOR_RESULT_skipproc_qc_leak.md
 M dev/results/OSC_GAP_INVENTORY.md
 M docs/VYVAR_ROADMAP.md
?? .vscode/settings.json
?? dev/results/CURSOR_RESULT_skipproc_qc_remediation.md
?? dev/scripts/dy_peg_night_run_bvr.py
?? dev/scripts/qatar8_night_run_v.py
```

(config.json local edit `skip_processed_directory=true` was **reverted** to match
origin; it was causing `test_flow_doc_config_facts` FAIL and is unrelated to push.)

### Per-commit annotations (non-SKIPPROC / non-pre-step)

| Commit | Session / arc | One-line description |
|--------|---------------|----------------------|
| `cad6002` | OSC-SUPPORT (2026-07-21) | Initial IMX533 M71 discovery gap inventory markdown |
| `627bd35` | OSC-SUPPORT (2026-07-21) | Append IMX533 M71 re-run section to gap inventory |
| `9533953` | OSC-EXTRACT (2026-07-21) | M71 L/B/G/R superpixel extraction experiment results + ROADMAP note |

SKIPPROC-QC arc: `263c6e7`, `4f140ef`. Pre-step: `cff5502`, `a5ddfa1`.

---

## Repairs

| File | Introducing commit(s) | What was repaired |
|------|----------------------|-------------------|
| `docs/VYVAR_ROADMAP.md` | `9533953` | UTF-8 em dash U+2014 in "extraction experiment - L/B/R..." -> ASCII `-` |
| `dev/results/OSC_GAP_INVENTORY.md` | `cad6002`, `627bd35`, `9533953` | 17x cp1252 0x97 (Windows em dash) -> `-`; table em-dash cells -> `-` |
| `dev/results/CURSOR_RESULT_skipproc_qc_fix.md` | `4f140ef` | 5x cp1252 0x97 section headers (`C -`, `A -`) -> `-`; corrected "pre-existing" claims to locally introduced + commit table |
| `dev/results/CURSOR_RESULT_skipproc_qc_leak.md` | `4f140ef` | 89 non-ASCII chars (cp1252 em dashes + curly quotes) via `ascii_migrate.py`; residual `?` line-wrap artifacts -> `->` / `~` / `sigma` where context-clear |

Tool: `python dev/tools/ascii_migrate.py` (full transliteration table in
`dev/tools/ascii_migrate.py` CHAR_MAP). No content deleted; semantic text preserved.

---

## Full-tree sweep

```
ascii_migrate [CHECK] root=C:\ASTRO\python\VYVAR
migrated_or_would=0
stop=0
```

Only the four files above required repair; no additional tracked text offenders.

`test_ascii_policy.py`: **3/3 PASS** after repairs.

---

## Editor encoding finding

**User settings** (`%APPDATA%\Cursor\User\settings.json`):

```json
{
    "window.commandCenter": true,
    "window.autoDetectColorScheme": false,
    "claudeCode.preferredLocation": "panel"
}
```

No `files.encoding` or `files.autoGuessEncoding` was set (cp1252 fallback risk on
Windows when saving UTF-8 content as single-byte).

**Change applied (local workspace):** created `.vscode/settings.json`:

```json
{
  "files.encoding": "utf8",
  "files.autoGuessEncoding": false
}
```

Note: `.vscode/` is **gitignored** in this repo; the file is not in the remediation
commit. Milan should keep these settings locally (or add a documented team convention).

This is the second occurrence of the cp1252/U+FFFD corruption class; UTF-8 enforcement
should prevent reintroduction when editing through Cursor/VS Code.

---

## Skewed-flat correction commit id

INV-FLUX-02 skewed-flat index fix (`flat[:15, :]` so median=1.0) is in:

**`263c6e7`** (`fix(skipproc): qc_metrics allowlist gates alignment...`)

via `dev/tests/test_invariants_p2.py` bundled with QC-01 registry regex update.
Not dangling in the working tree.

(`cff5502` originally had `flat[:16, :]` which failed median=1.0; corrected before
skipproc commit landed.)

---

## Gates

### pytest

```
1068 passed, 24 skipped, 49 warnings
```

### ruff

Clean on touched source (no new Python changes in remediation commit).

### session_baseline_check.py --fast

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   4f140ef
git-staged                   PASS   none
git-untracked-known          WARN   2 known untracked
git-origin-main              WARN   differs from origin/main (302e81b); consider git pull
config-paths                 PASS   all present
pytest                       PASS   1068 passed, 24 skipped
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.1 (+89 other) - gated upgrade, see docs/DEPS_POLICY.md
------------------------------------------------------------------------
OVERALL: PASS
```

(post-remediation commit; HEAD will advance after `fix(encoding)` commit)

---

## Docs impact

None. Repairs restore ASCII form of existing prose; no `flow_doc_facts.py` counts
changed. `test_docs_sync_guard`: 4/4 PASS (with `config.json` reverted to origin).

---

## STOP before push

Milan authorizes push. Push inventory: **7 unpushed commits** (+ remediation commit
= 8) on `main` ahead of `origin/main` @ `302e81b`.

---

## Push (2026-07-22, authorized by Milan)

### Pre-push checks

| Check | Result |
|-------|--------|
| Stack `git log origin/main..HEAD --oneline` | 8 commits - exact match (see below) |
| `git status --short` | Clean; allowlisted untracked only (`dy_peg_night_run_bvr.py`, `qatar8_night_run_v.py`) |
| `git fetch origin`; `origin/main` | `302e81b` (no upstream movement) |
| `session_baseline_check.py --fast` | **OVERALL PASS** - 1068 passed, 24 skipped |

### Pushed commits (8, newest first)

```
63ed2b4 fix(encoding): repair locally introduced non-ASCII (ROADMAP, OSC inventory, result files)
4f140ef docs(invariants): QC-01 skip-processed allowlist contract + DECISIONS entry
263c6e7 fix(skipproc): qc_metrics allowlist gates alignment; full-set QC + prefilter stamping
a5ddfa1 fix(ui): material calibration icon and structured params JSON widgets
cff5502 fix(invariants): INV-FLUX-02 checks median (matches normalize_flat_master)
9533953 docs(osc): M71 L/B/G/R extraction experiment results
627bd35 docs(osc): IMX533 M71 re-run appended to gap inventory
cad6002 docs(osc): IMX533 discovery gap inventory; park OSC-SUPPORT arc
```

Base: `302e81b` -> stack tip: `63ed2b4` (`git push origin main` succeeded).

### Bookkeeping commits

- `bf03340` - `docs(results): record SKIPPROC-QC stack push (tip 63ed2b4)`
- `6d6f51f` - `docs(results): fix final tip hash in SKIPPROC-QC push record`
- `f2f44c6` - `docs(results): align SKIPPROC-QC push record with origin/main tip 6d6f51f`

### Final origin/main tip

Stack landed at `63ed2b4`; bookkeeping commits above followed. At push
completion (2026-07-22): `f2f44c6cd512ee764409246f4adb4b72f20a3b95` (`f2f44c6`).
For current tip after any later commits: `git rev-parse origin/main`.
