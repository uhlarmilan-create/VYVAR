# VYVAR release runbook (REPEATABLE)

Repeatable ritual for cutting a preview release bundle. **STOP before push** unless
Milan explicitly approves.

## Prerequisites

- CYTHON-RELEASE-1 tooling green (`dev/tools/cython_release/`)
- Windows: Visual Studio Build Tools (MSVC) for compile
- Linux bundle: WSL Ubuntu 24+ with `python3-dev`, `gcc`, repo cloned in WSL

## Step 1 - Clean tree + interpreted `--fast`

**Gate hygiene:** clear stale P1 env before pytest counts:

```bat
Remove-Item Env:VYVAR_INVARIANTS_P1 -ErrorAction SilentlyContinue
```

```bat
cd VYVAR
git status   REM must be clean except release work
python dev\scripts\session_baseline_check.py --fast
```

Expect **OVERALL PASS**.

## Step 2 - RELEASE-1 compile + gates

```bat
python build\setup_cython.py build
python dev\tools\cython_release\smoke_imports.py
python dev\tools\cython_release\verify_mp.py
set PYTHONPATH=src_py
pytest dev\tests -q
set VYVAR_INVARIANTS_P1=1
pytest dev\tests\test_invariants_p1_seed.py dev\tests\test_invariants_p1_golden.py -q
python dev\scripts\session_baseline_check.py --full
```

Compiled `--full` must match anchor SHAs (core `03d8fb64...` n=333, extended
`bbfcc92e...` n=499, science compare 0 failures).

If `src_py` path-resolution changed (B2), interpreted `--full` after `setup_cython.py clean`
must also PASS byte-identical.

## Step 3 - Windows bundle

```bat
python dev\tools\cython_release\bundle\build_bundle.py --platform win64 --tag preview-YYYYMMDD
python dev\tools\cython_release\bundle\smoke_bundle.py --artifact dist\release\VYVAR-preview-YYYYMMDD-win64.zip
```

`build_bundle.py` runs `setup_cython.py clean` automatically after a successful bundle
(unless `--no-post-clean`). The checkout should have no `.pyd`/`.so` under `src_py/` before
the next interpreted dev session.

## Step 4 - Linux bundle (WSL Ubuntu 24)

One-time WSL setup:

```bash
sudo apt update && sudo apt install -y python3-dev gcc build-essential
cd /mnt/c/ASTRO/python/VYVAR   # or clone inside WSL
python3 build/setup_cython.py build
export VYVAR_BUNDLE_DIST=/tmp/vyvar_bundle_dist   # native FS avoids terminfo case collisions on /mnt/c/
python3 dev/tools/cython_release/bundle/build_bundle.py --platform linux-x64 --tag preview-VYVAR.0.9.0
python3 dev/tools/cython_release/bundle/smoke_bundle.py --artifact /tmp/vyvar_bundle_dist/VYVAR-preview-VYVAR.0.9.0-linux-x64.tar.gz
```

**Future:** GitHub Actions matrix (win64 + linux-x64) - not implemented in RELEASE-2.

## Step 5 - Tag (private repo)

```bat
git tag preview-YYYYMMDD
```

Do not push until Milan approves.

## Step 6 - Public release (Cursor via gh CLI, Milan authorizes per release)

1. Copy `release/public_repo/*` to `VYVAR-release` repository (no binaries, no source).
2. Grep staged content for private repo URLs, machine paths, tokens; fix before push.
3. Commit and push `VYVAR-release` (repo-local git identity if needed).
4. Set repo description + topics via `gh repo edit`.
5. When **both** platform bundles exist and Milan confirms in-session:
   - Tag `preview-YYYYMMDD` in the **private** repo.
   - Create GitHub Release in `VYVAR-release` with **pre-release** flag via `gh release create`.
   - Upload `VYVAR-*-win64.zip`, `VYVAR-*-linux-x64.tar.gz`, `SHA256SUMS`.
6. Do **not** cut a single-platform preview without Milan's explicit say-so.

**gh auth:** `gh release create` runs in Milan's **interactive terminal**; Cursor
subprocess does not inherit `gh auth` (`%APPDATA%\GitHub CLI\hosts.yml`).

## Platform build hygiene

**`dist/` is NOT shared between Windows and WSL builds.** Use separate dist roots
(e.g. Windows native `tmp/cython_release/bundle/dist/` vs WSL `/tmp/vyvar_bundle_dist`
symlink) or fully clean `dist/` between platforms. Mixing WSL symlink staging with
Windows `build_bundle.py` causes **WinError 1920** on the staging tree.

## Step 7 - Bookkeeping

- Update `dev/validation/VYVAR_VALIDATION_LEDGER.json` (preview bundle stamp)
- Add `docs/VYVAR_JOURNAL.md` entry
- Write `dev/results/CURSOR_RESULT_cython_release2.md`

## Runtime pins

See `dev/tools/cython_release/bundle/runtime_pins.py`:

| Platform | Python | Source |
|----------|--------|--------|
| win64 | 3.12.10 embed | python.org |
| linux-x64 | 3.12.8 | python-build-standalone (glibc >= 2.39) |

## Preview versioning (R4)

Tags: `preview-VYVAR.<semver>` (e.g. `preview-VYVAR.0.9.0`) with GitHub pre-release flag.
Earlier previews used `preview-YYYYMMDD`; semantic `v1.0.0` later.
