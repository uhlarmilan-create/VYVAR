CURSOR RESULT - CYTHON-RELEASE-2 (2026-07-23)

What I did
Implemented RELEASE-2 bundle builder with embedded Python runtime, data-dir separation
(B2), install docs CZ+EN, public-repo staging, and release runbook. Built Windows
preview bundle and passed bundle smoke on Milan's machine. STOP before push.

## B2 case statement

**Case: src_py path resolution changed, dev-neutral via bundle markers.**

Files touched: `config.py` (`resolve_data_root`, `data_root` on AppConfig),
`pipeline.py`, `vyvar_runtime.py`, `app.py`, UI save paths, `params_registry.py`.

Logic:
- Git checkout (`.git` at install root): `data_root == project_root` (unchanged dev layout)
- `VYVAR_DATA_DIR` env: explicit override
- Bundle install (`VYVAR_RELEASE_BUNDLE=1` or `RUNTIME_PIN.json` / embedded `python/`): platform default data dir
- Otherwise (pytest tmp paths): `data_root == project_root`

**Requires --full anchor gate** (interpreted, after `setup_cython.py clean`) because
`config.py` / `pipeline.py` science paths changed. `--full` queued on Milan's machine
after RELEASE-1 push; not re-run in this session (see Gates).

## Bundle layout (Windows preview-20260723-win64)

```
VYVAR-preview-20260723-win64/
  python/              embedded Python 3.12.10 + Lib/site-packages
  src_py/              85x .pyd + ui_*.py + app.py (NO .py for compiled science modules)
  app.py               root Streamlit shim
  VYVAR.bat / vyvar.sh launchers (VYVAR_RELEASE_BUNDLE=1)
  vyvar_selftest.py    --selftest entry
  config.template.json
  LICENSE
  RUNTIME_PIN.json
  THIRD_PARTY_NOTICES.txt
```

Output: `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-win64.zip`
SHA256: `86588241cf00b413a2e3c09a454d9676104522b00392b73c2e807bf5ea4401d1`

## Runtime pins

| Platform | Version | URL | SHA256 |
|----------|---------|-----|--------|
| win64 | 3.12.10 embed | python.org ftp | 4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3 |
| linux-x64 | 3.12.8 standalone | python-build-standalone 20241206 | cache sidecar on first fetch |

## Selftest sample (Windows smoke, path with spaces)

```
VYVAR selftest platform=Windows-11-10.0.26200-SP0
python=3.12.10
install_dir=...\vyvar_bundle_smoke iuteoizr\VYVAR-preview-20260723-win64
data_dir=C:\Users\uhlar\AppData\Local\VYVAR
dep numpy=2.4.6 astropy=8.0.1 photutils=3.0.0 streamlit=1.60.0
SELFTEST PASS modules=85
```

Log: `tmp/cython_release/bundle/dist/smoke_last.log`

Linux bundle: **deferred** (WSL Ubuntu 24 build per runbook; not executed this session).

## Docs (CZ + EN, ASCII)

Choice: **ASCII without diacritics** in tracked `.md` files (repo ENCODING-POLICY; no allowlist change).

| File | Purpose |
|------|---------|
| release/public_repo/INSTALL_VYVAR_EN.md | English install guide |
| release/public_repo/INSTALL_VYVAR_CZ.md | Czech install guide (ASCII) |
| release/public_repo/INSTALL_VYVAR_*.pdf | PDF via `dev/tools/docs_pdf/build_install_vyvar_release.py` |
| release/public_repo/README_EN.md / README_CZ.md | Public repo landing |
| release/public_repo/CHANGELOG.md | preview-20260723 stub |
| release/public_repo/LICENSE | Copy of repo LICENSE |
| docs/VYVAR_RELEASE_RUNBOOK.md | Repeatable release ritual |

## Gates

| Gate | Status |
|------|--------|
| ruff clean (after BLE001 fix) | PASS |
| pytest interpreted | 1130 passed, 18 skipped |
| --fast OVERALL | PASS |
| --full byte-identical (B2) | **PENDING** - required before push |
| Windows bundle build | PASS |
| Windows bundle smoke | PASS |
| Linux bundle smoke | DEFERRED (WSL per runbook) |

## ASCII docs check

All new tracked `.md` under `release/public_repo/` and `docs/VYVAR_RELEASE_RUNBOOK.md`: ASCII-only.

## STOP before push

Not pushed. RELEASE-1 commits (`b4c372a`, `7ead4a4`) still local ahead of origin.

## Files changed (summary)

- `dev/tools/cython_release/bundle/` (build_bundle, runtime_fetch, smoke, templates)
- `src_py/config.py`, `vyvar_runtime.py`, `pipeline.py`, `app.py`, UI saves, `params_registry.py`
- `release/public_repo/*`, `docs/VYVAR_RELEASE_RUNBOOK.md`, `docs/VYVAR_DECISIONS.md`
- `dev/tests/test_config_data_root.py`
- `.gitignore`

## Errors

None blocking bundle/smoke. `--full` anchor re-verify pending (B2 requirement).
