# CURSOR RESULT - PUSH RETRY 2026-07-20

CURSOR RESULT - 2026-07-20 (local)

## Verdict (abort path, then retry-2)

First attempt ABORT on dirty tree / failed pure-EOL proof.
Retry-2 executed under Milan-approved R1/R2/R3/R4 resolutions.

---

## Step 1a - `.gitattributes` (diagnosis, unchanged)

- `git hash-object .gitattributes` == `git rev-parse HEAD:.gitattributes`
  (`4bbe705420fe5b6f8dace4bd2bd0c3885c42358d`) - byte-identical blob.
- Status `M` is `core.autocrlf=true` noise (CR-strip equal: True).
- HEAD policy unchanged: `*.pdf binary` only.
- R3: no renormalize; leave harmless-M noise alone for this push.

## Step 1b - Pure-EOL proof (diagnosis)

`git diff --ignore-cr-at-eol --stat` was NOT empty (16 files).

### Real content (committed in retry-2 / R1)

- `dev/validation/VYVAR_VALIDATION_LEDGER.json` - K2 --full auto-stamp:
  date 2026-07-20, commit 69432ee
- `dev/results/CURSOR_RESULT_per_frame_sat.md` - footer `69432ee` (not pushed)

### Encoding corruption (restored in retry-2 / R2)

HEAD had cp1252 `0x97`; working tree had UTF-8 U+FFFD. Restored from HEAD:

- INSTALL.md
- LICENSE
- README_CZ.md
- dev/results/CURSOR_RESULT_config_human_edit.md
- dev/results/CURSOR_RESULT_db_seed_split.md
- dev/results/CURSOR_RESULT_docs_fix_arc1.md
- dev/results/CURSOR_RESULT_docs_pdf.md
- dev/results/CURSOR_RESULT_install_arc.md
- dev/results/CURSOR_RESULT_push_2026-07-18.md
- dev/results/CURSOR_RESULT_readme_detail.md
- dev/results/CURSOR_RESULT_wave_b_reduction.md
- dev/results/DEPS_SCOUT.md
- dev/tests/test_fresh_machine_startup.py
- docs/VYVAR_CALIBRATION.md

Post-restore: `grep`/byte scan for `ef bf bd` over those 14 -> empty.

### Harmless status-M (R3, no action)

24 files including `.gitattributes`, `app.py`, scripts/tests, etc.

### Untracked (left as-is)

- `dev/scripts/dy_peg_night_run_bvr.py` - allowlisted
- `dev/scripts/qatar8_night_run_v.py` - allowlisted
- `dev/scripts/forensic_disc_ui_match2.py` - neither commit nor delete marked

## R4 - roadmap

Added under OPEN low-priority hygiene:

`ENCODING-POLICY | MED | ... one-time migration to UTF-8 + guard + attrs/editorconfig`

---

## RETRY-2 outcome (pre-push bookkeeping)

- Remediation case: R2 restore (14) + R1 bookkeeping commit (ledger,
  per_frame_sat footer, this RESULT, roadmap ENCODING-POLICY row).
- R1 commit message:
  `chore(bookkeeping): ledger auto-stamp from k2 --full + result footer stamps`
- Final RESULT tip / --fast / origin verify: appended after push below.

## Push / sanity (filled after push)

| Check | Result |
|-------|--------|
| `origin/main` pre-push | (pending) |
| `--fast` | (pending) |
| `git push` | (pending) |
| Pushed tip | (pending) |
