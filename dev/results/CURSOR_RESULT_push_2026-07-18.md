CURSOR RESULT - PUSH 2026-07-18

## PUSH

Pushed HEAD: `f24f4ab` (then this report commit on top).

Outgoing stack (`7e88c3b`..`f24f4ab`):

```
f24f4ab docs(pdf): ship regenerable CZ technical pipeline flow guide
12dc186 docs(result): DB-SEED-SPLIT consumer trace, wiring, --full byte-identical PASS
59694a9 docs(install): empty-DB first-run narrative (v1.2 CZ guide + installer FINISH)
1c0bd2a test(db): guard empty init + pin reference seed; wire fixtures that need FKs
a0743a2 fix(db): fresh database empty; author observatory seed is harness-only
b3fcef9 docs(result): DOCS-PDF arc result (guard diff, builder logs, links, deviation)
8f50c4c docs(links): cross-link the CZ PDF guides from the docs index
6c4b815 docs(pdf): ship regenerable CZ parameter handbook + install guide
e358b7e test(docs): allow *.pdf in docs/ when a committed builder regenerates it
aa0f6cf docs(result): INSTALL-ARC result + ledger anchor re-stamp
1aae2d6 feat(install): Windows/Linux installer, INSTALL.md, config path writer
cabd9e6 fix(config): treat blank archive/calibration/database paths as project-root default
```

Working tree: clean of intended changes (only known scratch untracked:
`dev/scripts/dy_peg_night_run_bvr.py`, `forensic_disc_ui_match2.py`,
`qatar8_night_run_v.py`). After the stack push: `main` == `origin/main` at
`f24f4ab`.

### STEP 2 gates used

- pytest: **973 passed, 19 skipped**, 31 warnings in 317.27s
- `--fast`: **OVERALL PASS** (pre-existing WARNs only: known untracked,
  ahead-of-origin before push, ledger TODOs, deps-outdated informational)
- SCIENCE-PATH RULE: stack touches `src_py/config.py` (`cabd9e6`) and
  `src_py/database.py` (`a0743a2`). `--full` was run after the last
  science-path commit (`a0743a2` / DB-SEED-SPLIT) and **PASS**ed
  byte-identical: core `3d26f469...`, extended `6420f1da...`, n_lc=166
  failures=0 (recorded in `CURSOR_RESULT_db_seed_split.md`). Subsequent
  commits (docs/PDF/FLOW) did not touch science path; no re-`--full`
  required.

Push: `7e88c3b..f24f4ab  main -> main` (clean fast-forward; origin/main
had not moved at fetch).
