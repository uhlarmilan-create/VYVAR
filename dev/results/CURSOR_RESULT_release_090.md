CURSOR RESULT - RELEASE BUILD preview-VYVAR.0.9.0 (2026-07-29)

What I did
Cleared stale `.pyd.stale` artefacts, full Cython rebuild (90 modules), verified all release gates
on the **compiled** build, built Windows + Linux bundles, smoke-tested both, generated SHA256SUMS
and release notes. **Nothing published** (no tag push, no `gh release`).

---

## Part 0 - Preconditions

| Check | Result |
|-------|--------|
| Commit | `226d269f8648419ee834dbf58b140599453f5f3a` |
| Tag `preview-VYVAR.0.9.0` | absent locally and on `origin` |
| Tag `preview-20260723` | present locally + on `origin` (`fe574c0`) |
| `.pyd.stale` / disabled compiled | **3 removed** at start (`infolog`, `photometry_core`, `pipeline`); **0** after rebuild |
| `git status --porcelain` | dirty (untracked result docs, local sqlite wal); release work on clean commit base |

**Run time:** premise check ~15 s.

---

## Part 1 - Clean Cython rebuild (Windows)

```
python build/setup_cython.py clean   # 0.45 s
python build/setup_cython.py build   # 813.2 s
python dev/tools/cython_release/smoke_imports.py   # 53.8 s
python dev/tools/cython_release/verify_mp.py       # (included above)
```

| Item | Value |
|------|------:|
| Modules built | **90** |
| `.pyd` in `src_py/` after build | 90 |
| `.pyd.stale` / disabled anywhere | **0** |
| Import smoke | **90/90 PASS** (all `.pyd`) |
| MP spawn verify | **PASS** (`photometry_core`, `comp_selection_per_target` compiled in worker) |

Post-bundle post-clean removed `.pyd` from `src_py/` (85 paths; 7 locked with WinError 5).

---

## Part 2 - Release gates (COMPILED build)

`VYVAR_INVARIANTS_P1` cleared before plain pytest (runbook hygiene).

| Gate | Result | Run time |
|------|--------|----------|
| `ruff` | **PASS** | 0.9 s |
| `--fast` | **PASS** (1198 passed, 30 skipped) | **487.9 s** |
| P1 golden (`VYVAR_INVARIANTS_P1=1`) | **7/7 PASS** (5 golden + 2 seed; both UI + headless chains executed) | **978.6 s** |
| anchor `--full` | **PASS** | harness **3276.7 s**; pipeline **2889 s** |

**Anchor (compiled) - must match interpreted baseline:**

| Check | Value |
|-------|-------|
| `full-photometry-sha-core` | **b7f980c09e238b85... n=325** |
| `full-photometry-sha-extended` | **2c43bbbf06921fbe... n=487** |
| `full-plan-regen` | 875 rows |
| `full-phase0-funnel` | active **165** |
| `full-catalog-provenance` | **PASS** |

**Compiled == interpreted (CYTHON-RELEASE-1 identity):** anchor byte-SHA reproduced exactly on
compiled build; 0 pytest failures; conditional skip delta only (1198/30 compiled vs 1206/22
interpreted @ session close) -- same pattern as RELEASE-1, not a science divergence.

---

## Part 3 - Windows bundle

```
python dev/tools/cython_release/bundle/build_bundle.py --platform win64 --tag preview-VYVAR.0.9.0
python dev/tools/cython_release/bundle/smoke_bundle.py --artifact tmp/cython_release/bundle/dist/VYVAR-preview-VYVAR.0.9.0-win64.zip
```

| Item | Value |
|------|------:|
| Path | `tmp/cython_release/bundle/dist/VYVAR-preview-VYVAR.0.9.0-win64.zip` |
| Size | **339,837,962 B** (~324.1 MB) |
| SHA256 | `ace78b6ae6d415a624a9eac2b97b8d2f0185c9422b2c1c063cceec6e8278b30b` |
| Build wall | **147.8 s** |
| Smoke | **PASS** (108.2 s) |

**`--selftest` (excerpt):** python=3.12.10; deps numpy 2.4.6, astropy 8.0.1, photutils 3.0.0,
streamlit 1.60.0; **SELFTEST PASS modules=90**.

---

## Part 4 - Linux bundle (WSL Ubuntu 24)

| Item | Value |
|-------|-------|
| Windows commit | `226d269f8648419ee834dbf58b140599453f5f3a` |
| WSL commit | `226d269f8648419ee834dbf58b140599453f5f3a` |
| **Match** | **yes** |
| glibc (WSL) | **2.39** (pin >= 2.39 satisfied) |

**Build notes:** first WSL bundle attempt failed (`.so` left at repo root -- fixed
`build_release._relocate_pyd_artifacts` for Linux). Second attempt failed on `/mnt/c/` staging
(terminfo case collision on Windows-mounted FS). **Successful build** used
`VYVAR_BUNDLE_DIST=/tmp/vyvar_bundle_dist_linux` (native ext4).

| Item | Value |
|------|------:|
| Path | `/tmp/vyvar_bundle_dist_linux/VYVAR-preview-VYVAR.0.9.0-linux-x64.tar.gz` |
| Size | **514,282,323 B** (~490.6 MB) |
| SHA256 | `848b6e39dd2e8d3d33b0aeff77553c5b2b1c8deac1ae3fbfb461e2ea6592c2a6` |
| Build wall | **406 s** |
| Smoke | **PASS** (38.2 s) |

**`--selftest` (excerpt):** python=3.12.8; platform glibc2.39; **SELFTEST PASS modules=90**.

---

## Part 5 - SHA256SUMS, release notes, CHANGELOG

- `release/public_repo/SHA256SUMS` (both artefacts)
- `release/public_repo/RELEASE_NOTES_preview-VYVAR.0.9.0.md` (reprocessing warning first)
- `release/public_repo/CHANGELOG.md` updated; `preview-20260723` marked withdrawn
- `docs/VYVAR_RELEASE_RUNBOOK.md` R4 line corrected to `preview-VYVAR.<semver>`

Binary artefacts **not** committed (gitignored under `tmp/`; copy from paths above for upload).

---

## Part 6 - Handover for Milan (NOT executed)

**Bundle naming flag:** `bundle_name()` -> `VYVAR-preview-VYVAR.0.9.0-<platform>` (VYVAR twice).

```bat
git tag preview-VYVAR.0.9.0 226d269f8648419ee834dbf58b140599453f5f3a
git push origin preview-VYVAR.0.9.0
```

(In `VYVAR-release` public repo, after copying `release/public_repo/*` and uploading assets:)

```bat
gh release create preview-VYVAR.0.9.0 ^
  --repo uhlar/VYVAR-release ^
  --title "preview-VYVAR.0.9.0" ^
  --notes-file release/public_repo/RELEASE_NOTES_preview-VYVAR.0.9.0.md ^
  --prerelease ^
  tmp\cython_release\bundle\dist\VYVAR-preview-VYVAR.0.9.0-win64.zip ^
  \\wsl$\Ubuntu\tmp\vyvar_bundle_dist_linux\VYVAR-preview-VYVAR.0.9.0-linux-x64.tar.gz ^
  release\public_repo\SHA256SUMS
```

**Verify uploads:**

| File | SHA256 |
|------|--------|
| `VYVAR-preview-VYVAR.0.9.0-win64.zip` | `ace78b6ae6d415a624a9eac2b97b8d2f0185c9422b2c1c063cceec6e8278b30b` |
| `VYVAR-preview-VYVAR.0.9.0-linux-x64.tar.gz` | `848b6e39dd2e8d3d33b0aeff77553c5b2b1c8deac1ae3fbfb461e2ea6592c2a6` |

Adjust `gh release create` paths to where Milan stages files; run in **interactive** terminal (`gh auth`).

---

## Part 7 - Retiring preview-20260723 (record only)

**Reason:** missing order-2 sky-surface subtraction -> inflated detection catalogue (~40% DAO_ONLY).
Recorded in CHANGELOG + release notes. **Recommend:** delete GitHub pre-release + assets; **keep**
git tag `preview-20260723` on private repo.

**Local copies still on disk:**

| Path | Size |
|------|-----:|
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-win64.zip` | ~324 MB |
| `tmp/cython_release/bundle/dist/VYVAR-preview-20260723-linux-x64.tar.gz` | ~489 MB |
| `tmp/cython_release/bundle/dist_win/VYVAR-preview-20260723-win64.zip` | ~324 MB |

Agent did **not** delete anything.

---

## Errors / fixes during build

1. Linux `.so` relocation: extended `_relocate_pyd_artifacts` in `build_release.py`.
2. WSL bundle on `/mnt/c/`: terminfo `File exists` (case-insensitive mount) -- use native
   `VYVAR_BUNDLE_DIST=/tmp/vyvar_bundle_dist_linux`.

---

## Files changed

- `dev/tools/cython_release/build_release.py` (Linux `.so` relocate)
- `docs/VYVAR_RELEASE_RUNBOOK.md`
- `release/public_repo/CHANGELOG.md`
- `release/public_repo/RELEASE_NOTES_preview-VYVAR.0.9.0.md`
- `release/public_repo/SHA256SUMS`
- `dev/results/CURSOR_RESULT_release_090.md`
