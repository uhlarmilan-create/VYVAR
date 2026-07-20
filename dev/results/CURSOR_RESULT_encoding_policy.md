# CURSOR RESULT - ENCODING-POLICY (ASCII migration + EOL)

CURSOR RESULT - 2026-07-20 (local; not pushed)

## Verdict

Two commits prepared (not pushed; tip = `git log -2 --oneline`):
1. chore(encoding): ASCII migration of 40 legacy text files +
   ascii-policy guard + PROCESS rule
2. chore(eol): normalize line endings (* text=auto) + editorconfig

## Docs impact

- PROCESS.md: ENCODING-POLICY block (replaces DOCS-FIX-ARC1 README UTF-8 exemption)
- ROADMAP: ENCODING-POLICY -> DONE
- STATE: one-liner
- FLOW / flow_doc_facts: none

## Commit 1 - migration + guard

### Tool

`dev/tools/ascii_migrate.py`
- Walk: tracked `.md .py .json .txt .cfg .toml .yml .yaml .ps1 .sh` +
  `.gitignore` `.gitattributes` `LICENSE`; skip `Archive/`
- Decode: UTF-8 else cp1252 (`errors=replace` so undefined 0x9D -> U+FFFD)
- Explicit CHAR_MAP (dashes, quotes->single, double-prime->" arcsec",
  ellipsis, arrows, Czech fold, common science/UI symbols). Unmapped ->
  STOP (no write). `.py` post-fold `compile()` gate refuses syntax-breaking
  folds.
- `--check` idempotent report mode

### Migration outcome

- STOP list: empty
- Rewritten: 302 files (41 legacy non-UTF-8/cp1252 + UTF-8 typographic /
  Czech / emoji carriers). Commit message says "40" for the legacy class;
  full walk rewrote 302 to satisfy the empty-allowlist guard.
- `dev/results/specs/VYVAR_SIMPLE_DIFFERENTIAL_SPEC.md`: 13x U+FFFD -> `-`
- Per-file non-ASCII char counts: `tmp/_ascii_migrate_out.txt` (scratch).
  Highlights: README_CZ.md (Czech folded), INSTALL.md, LICENSE,
  DEPS_SCOUT.md, VYVAR_CALIBRATION.md, test_fresh_machine_startup.py,
  many CURSOR_RESULT_*.md.

### Guard

`dev/tests/test_ascii_policy.py` - every tracked text byte < 0x80;
`ASCII_POLICY_ALLOWLIST` present and EMPTY.

### Gates (commit 1)

| Gate | Result |
|------|--------|
| ascii_migrate --check | exit 0 |
| test_ascii_policy | 3 passed |
| compileall src_py dev | clean |
| ruff check src_py + new files | All checks passed |
| --fast | OVERALL PASS (1027 passed, 24 skipped) |
| P1 golden | 7 passed (seed+golden; VYVAR_INVARIANTS_P1=1) |

## Commit 2 - EOL normalization

### Changes (ONLY these two paths in the commit)

- `.gitattributes`: `* text=auto` + `*.pdf binary` + `*.png binary`
- `.editorconfig` (new): root=true; [*] utf-8 + lf + final newline;
  [*.{ps1,bat}] crlf

### Renormalize note

Repo blobs were already LF after commit 1. `git add --renormalize .`
staged no content files beyond the two policy paths. Working-tree CRLF is
`core.autocrlf=true` checkout smudge; `* text=auto` makes that explicit.

### Pure-EOL machine proof

`git diff HEAD^ HEAD --ignore-cr-at-eol --name-only` for commit 2:

```
.editorconfig
.gitattributes
```

PASS (RESULT lives in commit 1 so it does not pollute this proof).

### Gates (commit 2)

| Gate | Result |
|------|--------|
| pure-EOL proof | PASS (attrs + editorconfig only) |
| --fast | OVERALL PASS (1034 passed, 17 skipped) |
| P1 golden | 7 passed |
| git status | clean except allowlisted untracked |

## Untracked left alone

- dev/scripts/dy_peg_night_run_bvr.py (allowlisted)
- dev/scripts/qatar8_night_run_v.py (allowlisted)
- dev/scripts/forensic_disc_ui_match2.py (no Milan box)
