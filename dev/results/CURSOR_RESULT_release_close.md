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

## Preview cut - final verification (PREVIEW-CUT-FINAL 2026-07-23)

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
