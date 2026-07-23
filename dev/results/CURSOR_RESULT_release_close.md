CURSOR RESULT - RELEASE-CLOSE (2026-07-23)

What I did
Closed the CYTHON release arc: explained RELEASE-1 skip delta (24 vs 18),
verified interpreted `--full` B2 gate and ledger stamp, pushed the private stack,
populated `VYVAR-release`, attempted WSL Linux bundle (blocked), updated runbook
Step 6 wording. **STOP before preview release** (no Linux bundle; Milan not
confirmed in-session).

## Skip delta (24 -> 18) -- LEGITIMATE, proceed

The RELEASE-1 result file reported **1126 passed, 18 skipped** on interpreted
`--fast`. That count is **not** comparable to the pre-RELEASE-1 baseline of
**24 skipped** @ `8f85b2d` without understanding shell env and one new test.

| Observation | Skips | Passed | Cause |
|-------------|-------|--------|-------|
| Pre-RELEASE-1 baseline @ `8f85b2d` (clean env) | 24 | 1117 | OSC-3 gate record |
| HEAD @ `893d5d2` (clean env, re-run 2026-07-23) | 25 | 1123 | +1 new `test_cython_mp_spawn` skip on interpreted path |
| HEAD @ `893d5d2` (stale `VYVAR_INVARIANTS_P1=1` in shell) | 18 | 1130 | 7 P1 opt-in tests **ran** instead of skip; mp_spawn still skips |

**Net 24 -> 18 (RELEASE-1 leftover):** seven P1 tests stopped skipping because
`VYVAR_INVARIANTS_P1=1` was left set after P1 golden / compiled gates; one new
test started skipping (`test_cython_mp_spawn` without compiled `.pyd`). Net -6.

| Test | File | Why it stopped skipping (24 -> 18 session) |
|------|------|-----------------------------------------------|
| test_p1_snapshot_sha_matches_registered | test_invariants_p1_seed.py | `VYVAR_INVARIANTS_P1=1` set; opt-in P1 gate enabled |
| test_p1_census_fingerprint_in_meta | test_invariants_p1_seed.py | same |
| test_mini_present_or_buildable | test_invariants_p1_golden.py | same |
| test_headless_chain_sha | test_invariants_p1_golden.py | same |
| test_ui_chain_byte_identity | test_invariants_p1_golden.py | same |
| test_census_bands | test_invariants_p1_golden.py | same |
| test_physics_asserts | test_invariants_p1_golden.py | same |

**Counterbalancing new skip (not in 24 baseline):**

| Test | Why it adds +1 skip on clean interpreted path |
|------|--------------------------------------------------|
| test_mp_spawn_loads_compiled_photometry_core | RELEASE-1: skips when no compiled `.pyd` in `src_py` |

**Legitimate code-only delta (clean env):** 24 -> 25 (+1 mp_spawn only). No
suspicious skip regression. **Proceed.**

**Gate hygiene:** always `Remove-Item Env:VYVAR_INVARIANTS_P1` before `--fast` /
interpreted pytest counts.

## Interpreted --full (B2 gate) -- PASS

After `python build/setup_cython.py clean`, interpreted tree @ feature commit
`6a05390` (before bookkeeping stamp `893d5d2`):

| Check | Result |
|-------|--------|
| full-snapshot-sha-core | PASS `03d8fb6491bc3c221f89f87acf22b929cece74c60951cf19bda80699180fb989` n=333 |
| full-photometry-sha-extended | PASS `bbfcc92e7ac5c4c5edfe0f99353aca9d03a987f99407352217e82875ed342892` n=499 |
| full-science-compare | PASS failures=0 |
| OVERALL | PASS |
| Pipeline wall | ~2429 s |

Ledger `VL-ANCHOR-WCSINV` stamped in commit `893d5d2` (interpreted --full after
B2 `resolve_data_root` / bundle data-dir separation). No anchor drift.

## Private repo push -- DONE

| Step | Result |
|------|--------|
| `git fetch origin`; verified tip | `8f85b2d` |
| Stack pushed | `b4c372a`, `7ead4a4`, `3369832`, `6a05390`, `893d5d2` |
| Final `--fast` (clean env) | OVERALL PASS -- 1123 passed, 25 skipped |
| Post-push `origin/main` | `893d5d2c548ff3d9123e7652d9c15716f49fc92b` |

No fix commits beyond the five above were required for skip-delta or B2 gate.

## Windows bundle (PREVIEW-CUT rebuild)

Original RELEASE-2 zip SHA `86588241...` was lost when `dist/` was reset during
Linux WSL staging work. Tree @ `893d5d2` unchanged; win64 **rebuilt** after
restoring `.pyd` in `src_py` (same release stack, no source drift).

| Artifact | SHA256 |
|----------|--------|
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-win64.zip` | `5573d15f299f0c3abdeaafc0c869102ad24e4a1303745728d2f8d1349b4eadea` |

Win64 smoke PASS: path with spaces, selftest **85 modules**, no `.py` for compiled science.

## Linux bundle (WSL Ubuntu 24) -- DONE (PREVIEW-CUT)

| Step | Result |
|------|--------|
| `python3 -m pip install --user --break-system-packages cython setuptools` | OK |
| `python3 build/setup_cython.py build` | **85** `.so` in `src_py` (MODULE_LIST n=85); pinned flags PASS (`annotation_typing=False`, `Options.docstrings=False`) |
| Build wall | ~1960 s |
| Bundle + repack | `VYVAR-preview-20260723-linux-x64.tar.gz` |
| Linux smoke / selftest | PASS -- 85 modules, glibc **2.39** (python-build-standalone 3.12.8) |

| Artifact | SHA256 |
|----------|--------|
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-linux-x64.tar.gz` | `02d96e2a1264ba117eaee425195a09bee7825f7dcbea8e24b8db6a9d384d22b1` |

**WSL notes:** Linux runtime extract to `/mnt/c` hits drvfs terminfo symlink loops;
fixed in `runtime_fetch.py` (extract via `/tmp`, copy without symlinks). `vyvar.sh`
updated for nested `python/python/bin` layout + LF line endings.

## Preview release -- STOP (RELEASE-CLOSE)

(See **Preview cut (PREVIEW-CUT)** section below for Milan-authorized cut outcome.)

## VYVAR-release public repo -- DONE (docs only)

| Step | Result |
|------|--------|
| Clone | `C:\ASTRO\python\VYVAR-release` |
| Staged from | `release/public_repo/` (8 files: README EN+CZ, INSTALL EN+CZ + PDFs, LICENSE, CHANGELOG) |
| Commit | `75b1d68` on `main` |
| Push | `origin/main` @ `75b1d68` |

**Internal-reference grep (before push):** patterns
`uhlarmilan-create/VYVAR[^-]`, `C:\ASTRO`, `C:/ASTRO`, `MiUh-PC`, `ghp_`,
`github_pat_`, `token`, `VYVAR_INVARIANTS` -- **no matches** in staged files.
Only public release URL references (`VYVAR-release/releases`) in README.

**Repo metadata:** `gh` CLI not on PATH on this Windows host; description + topics
(astronomy, photometry, variable-stars, aavso) **not set** via CLI. Milan can run:

```bat
gh repo edit uhlarmilan-create/VYVAR-release --description "VYVAR variable-star photometry pipeline - preview release bundles" --add-topic astronomy --add-topic photometry --add-topic variable-stars --add-topic aavso
```

## Preview cut (PREVIEW-CUT 2026-07-23) -- Milan authorized

| Item | Value |
|------|-------|
| Private tag | `preview-20260723` @ pushed tip (see push record below) |
| `--fast` before push | OVERALL PASS -- 1119 passed, 29 skipped (`.so` + `.pyd` in tree) |
| SHA256SUMS | `tmp/cython_release/bundle/dist/SHA256SUMS` (both assets) |
| GitHub Release | see release URL below |
| Round-trip verify | see below |

### Push record (PREVIEW-CUT)

| Step | Result |
|------|--------|
| Pre-push `origin/main` | `893d5d2` |
| Commits pushed | `1c05bc7` fix(cython), `b859d3f` docs(release) |
| Post-push `origin/main` | `b859d3f` |
| Tag | `preview-20260723` pushed to private repo |

### GitHub Release (VYVAR-release)

**BLOCKED in Cursor shell:** `gh auth status` reports not logged in (no
`%APPDATA%\GitHub CLI\hosts.yml`). Milan completed auth in interactive session;
Cursor subprocess does not inherit it. Run locally:

```bat
"C:\Program Files\GitHub CLI\gh.exe" auth login
"C:\Program Files\GitHub CLI\gh.exe" repo edit uhlarmilan-create/VYVAR-release --description "VYVAR variable-star photometry pipeline - preview release bundles" --add-topic astronomy --add-topic photometry --add-topic variable-stars --add-topic aavso
"C:\Program Files\GitHub CLI\gh.exe" release create preview-20260723 --repo uhlarmilan-create/VYVAR-release --prerelease --title "VYVAR preview 2026-07-23" --notes-file tmp\release_close\release_notes_preview-20260723.md tmp\cython_release\bundle\dist\VYVAR-preview-20260723-win64.zip tmp\cython_release\bundle\dist\VYVAR-preview-20260723-linux-x64.tar.gz tmp\cython_release\bundle\dist\SHA256SUMS
```

Release URL: **pending Milan gh release create** (assets + SHA256SUMS ready in
`tmp/cython_release/bundle/dist/`).

### Round-trip integrity

**STOP (2026-07-23 PREVIEW-CUT-FINAL):** GitHub release not visible to anonymous API
or Cursor `gh` (no auth in subprocess). Cannot download release assets for round-trip.

| Check | Result |
|-------|--------|
| `gh release view` (Cursor subprocess) | FAIL -- not logged in |
| `GET .../releases/tags/preview-20260723` | 404 Not Found |
| `GET .../releases` | `[]` (empty) |
| Asset URL HEAD (win64 zip) | 404 Not Found |

**Local sanity (not a substitute for round-trip):** local `tmp/cython_release/bundle/dist/`
files match expected SHA256 and local `SHA256SUMS`:

| Asset | Expected / local SHA256 | Match |
|-------|-------------------------|-------|
| VYVAR-preview-20260723-win64.zip | `5573d15f299f0c3abdeaafc0c869102ad24e4a1303745728d2f8d1349b4eadea` | PASS |
| VYVAR-preview-20260723-linux-x64.tar.gz | `02d96e2a1264ba117eaee425195a09bee7825f7dcbea8e24b8db6a9d384d22b1` | PASS |

**Action:** Milan verify release URL in browser; if public, re-run round-trip from
release download URLs. Do not re-upload until mismatch root cause is known.

---

CURSOR RESULT - BUNDLE-ISOLATION-FIX (2026-07-23)

What I did
Fixed bundled launcher environment isolation (Milan Linux field bug: host numpy
1.26.4 / astropy 7.2.0 / photutils 1.13.0 shadowed bundle -> ImagePSF missing).
Launchers use `python -I`; selftest loads `RUNTIME_PIN.json` `dep_versions` and
FAILs early on version or origin mismatch. Smoke adds contamination regression
(isolated launcher PASS with poison PYTHONPATH; direct inject FAIL with clear
message). Cosmetic: generic ui placeholders; public README.md + Releases link.
Rebuilt win64 + linux-x64; new SHA256SUMS. **STOP before gh re-upload** (Milan).

## Launcher isolation

| File | Change |
|------|--------|
| `dev/tools/cython_release/bundle/templates/vyvar.sh` | `unset PYTHONPATH PYTHONHOME PYTHONUSERBASE`; `python -I` for selftest + streamlit |
| `dev/tools/cython_release/bundle/templates/VYVAR.bat` | clear host env vars; `python -I` for selftest + streamlit |

`-I` verified: streamlit and selftest import from bundled site-packages only.

## Selftest pin verification

| File | Change |
|------|--------|
| `dev/tools/cython_release/bundle/build_bundle.py` | `_pinned_dep_versions()` + `_write_runtime_pin()` writes numpy/astropy/photutils/streamlit/pandas/scipy |
| `dev/tools/cython_release/bundle/templates/vyvar_selftest.py` | fail-fast contamination messages (version + `__file__` under bundle tree) |
| `dev/tools/cython_release/bundle/bundle_layout.py` | `VYVAR_BUNDLE_DIST` override (WSL native `/tmp/vyvar_bundle_dist` per runbook) |

## Smoke contamination regression

| File | Change |
|------|--------|
| `dev/tools/cython_release/bundle/smoke_bundle.py` | poison fake numpy via PYTHONPATH (isolated) + sys.path inject (non-isolated) |

Win64 + linux-x64 smoke PASS (including contamination case).

## Cosmetic / public repo

| File | Change |
|------|--------|
| `src_py/ui_photometry_quality.py` | generic `/path/to/...` placeholders (no personal paths) |
| `release/public_repo/README.md` | default README; [Cesky](README_CZ.md) + Releases link at top |
| `release/public_repo/README_EN.md` | removed (renamed) |
| `release/public_repo/CHANGELOG.md` | preview refreshed: launcher isolation fix |

`VYVAR-release` @ `42aef8c` (docs only; pushed earlier this arc).

## Rebuilt preview artifacts (isolation fix)

| Asset | SHA256 | Smoke |
|-------|--------|-------|
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-win64.zip` | `1b1188ac49fb4f5f1e6343b78a6ac2c4337c901de4d96b0dea372acf96d80dd0` | PASS |
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-linux-x64.tar.gz` | `d80239c7e6c136c15c523337cef4ebd12adb2bd6597bcdb1d3de541f1d69b7b6` | PASS (WSL) |

Combined `tmp/cython_release/bundle/dist/SHA256SUMS` updated (both lines above).

**Previous SHAs (pre-isolation, superseded):**
win64 `5573d15f...`; linux `02d96e2a...`.

Linux build note: use `VYVAR_BUNDLE_DIST=/tmp/vyvar_bundle_dist` on WSL (avoid NTFS
terminfo case collision on `/mnt/c`).

## Session baseline

`session_baseline_check.py --fast` with `VYVAR_INVARIANTS_P1=1`: pytest FAIL only
on unstaged README rename (`README_EN.md` missing on disk); resolves after commit.

## Milan re-upload (STOP -- run in authenticated terminal)

Use **`gh release upload ... --clobber`** (same tag `preview-20260723`; replaces assets):

```bat
"C:\Program Files\GitHub CLI\gh.exe" release upload preview-20260723 --repo uhlarmilan-create/VYVAR-release --clobber "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\VYVAR-preview-20260723-win64.zip" "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\VYVAR-preview-20260723-linux-x64.tar.gz" "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\SHA256SUMS"
```

Do **not** delete/recreate the release unless upload fails.

## Files changed (private repo)

- `dev/tools/cython_release/bundle/` (templates, build_bundle, smoke_bundle, bundle_layout)
- `src_py/ui_photometry_quality.py`
- `release/public_repo/` (README.md, CHANGELOG.md; README_EN.md deleted)
- `CHANGELOG.md`, `docs/VYVAR_JOURNAL.md`
- `dev/results/CURSOR_RESULT_release_close.md` (this append)

## ASCII check

This append written with ASCII-only punctuation (no em dash, no Unicode arrows).

---

CURSOR RESULT - BUNDLE-DATA-FILES (2026-07-23)

What I did
Fixed field bug #2: Settings tab crashed on missing `dev/validation/params_registry.json`.
Runtime dependency sweep, ship required install-root data files, manifest guard in
`build_bundle.py`, selftest + smoke loader checks. Graceful UI fallback if registry
missing. Rebuilt win64 + linux-x64 preview bundles. **STOP before gh re-upload**.

## Runtime dependency sweep

| File (install-relative) | Consumer | Trigger | Disposition |
|-------------------------|----------|---------|-------------|
| `dev/validation/params_registry.json` | `params_registry.load_registry()` | Settings -> Parameters; config save; PDF config appendix | **(a) Shipped** |
| `CITATIONS.bib` | `citations.load_citations_bib()` | Phase 2A export; PDF methods/citations | **(a) Shipped** |
| `img/VYVAR_logo.png` | `photometry_report` cover | PDF after VAR-STREM / Aperture Photometry | **(a) Shipped** |
| `config.template.json` | `vyvar_runtime.ensure_release_data_dir()` | First bundled app launch | **(a) Shipped** (template) |
| `dev/scripts/diagnose_psf_elongation_362.py` | `psf_photometry._epsf_augment_*` | ePSF broad-pool augment | **(b) Degrades** (warn + skip; not shipped) |
| `docs/*`, validation ledger | (none) | n/a | Not read at runtime |
| `.git/` / git subprocess | provenance helpers | pipeline_meta / HRD caption | **(b) Degrades** to null/nogit |

## Manifest guard

- `bundle_layout.REQUIRED_RUNTIME_FILES` + `RUNTIME_FILE_SOURCES`
- `build_bundle._copy_runtime_data_files()` + `_assert_runtime_files()` (build fails if missing)
- `RUNTIME_PIN.json` gains `required_files` list
- `vyvar_selftest.py`: presence check + `load_registry()` / `load_citations_bib()` calls
- `smoke_bundle.py`: `_runtime_loaders_smoke()` from bundled interpreter

## Code hardening

- `params_registry.load_registry()` / `load_registry_meta()`: explicit `FileNotFoundError`
- `ui_params_dashboard.render_params_dashboard()`: warning + return (no crash)

## Rebuilt preview artifacts

| Asset | SHA256 | Smoke |
|-------|--------|-------|
| `VYVAR-preview-20260723-win64.zip` | `129ec09ecd976c7ed5ad4a15b7b88c2a217f12a5c1d7ddde4bb2153cadfb1ae6` | PASS |
| `VYVAR-preview-20260723-linux-x64.tar.gz` | `3afe17b2c37f80142b903a7dffbfacedaa5a1ed9f904ea36b5b34070a5d99e43` | PASS (WSL) |

Supersedes isolation-only SHAs: win64 `1b1188ac...`; linux `d80239c7...`.

## Milan re-upload (STOP)

```bat
"C:\Program Files\GitHub CLI\gh.exe" release upload preview-20260723 --repo uhlarmilan-create/VYVAR-release --clobber "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\VYVAR-preview-20260723-win64.zip" "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\VYVAR-preview-20260723-linux-x64.tar.gz" "C:\ASTRO\python\VYVAR\tmp\cython_release\bundle\dist\SHA256SUMS"
```

## Files changed

- `dev/tools/cython_release/bundle/` (bundle_layout, build_bundle, smoke_bundle, vyvar_selftest template)
- `src_py/params_registry.py`, `src_py/ui_params_dashboard.py`
- `CHANGELOG.md`, `release/public_repo/CHANGELOG.md`, `docs/VYVAR_JOURNAL.md`
- `dev/results/CURSOR_RESULT_release_close.md` (this append)

## ASCII check

This append written with ASCII-only punctuation (no em dash, no Unicode arrows).

### Module count 84 vs 85

RELEASE-1 @ `b4c372a` compiled **84** modules (`module_list()` excluded
`vyvar_runtime.py`, which did not exist yet). RELEASE-2 commit **`3369832`**
added `src_py/vyvar_runtime.py` (bundled first-launch bootstrap for
`resolve_data_root()` / `%LOCALAPPDATA%\\VYVAR`); MODULE_LIST became **85** for
PREVIEW-CUT builds and selftest.

### Release verify (GitHub)

Release URL: **not confirmed** (anonymous API 404; see Round-trip STOP above).

Expected when visible: pre-release flag true; exactly 3 assets (win64 zip,
linux-x64 tar.gz, SHA256SUMS).

## Runbook update -- committed

`docs/VYVAR_RELEASE_RUNBOOK.md` Step 6: gh CLI workflow (Milan authorizes per release).

## ASCII check

This file written with ASCII-only punctuation (no em dash, no Unicode arrows).
