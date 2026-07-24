# VYVAR release bundle - English install guide (RELEASE-2)

Preview bundles ship a **bundled Python 3.12 runtime** (R1). Your system Python is
not used. User data (config, database, Archive, catalogs) lives in a separate
**data directory** from the install folder.

## 1. Download and unpack

1. Download `VYVAR-<tag>-win64.zip` or `VYVAR-<tag>-linux-x64.tar.gz` from GitHub
   Releases on the public `VYVAR-release` repository.
2. Verify SHA256 against `SHA256SUMS` in the same release.
3. Unpack to a permanent location, e.g.:
   - Windows: `C:\Program Files\VYVAR\`
   - Linux: `~/apps/vyvar/`
4. Paths with spaces are supported.

## 2. First launch

**Windows:** double-click `VYVAR.bat` or run from cmd:

```
cd C:\Program Files\VYVAR\VYVAR-preview-20260723-win64
VYVAR.bat
```

**Linux:**

```
chmod +x vyvar.sh
./vyvar.sh
```

On first launch the data directory is created automatically:

| Platform | Default data directory |
|----------|------------------------|
| Windows  | `%LOCALAPPDATA%\VYVAR` |
| Linux    | `~/.local/share/vyvar` |

Override before launch: set environment variable `VYVAR_DATA_DIR` to any writable path.

The initializer creates `Archive/Drafts/`, `CalibrationLibrary/`, catalog folders,
`logs/`, an empty `vyvar.sqlite3`, `config.json` from template, and `NEXT_STEPS.txt`.

Catalog and observation data never live inside the install folder (B1 design).

## 3. Health check (selftest)

```
VYVAR.bat --selftest
./vyvar.sh --selftest
```

Prints Python version, platform, resolved data directory, per-item bootstrap
report (`bootstrap <path>: created|preexisting|FAILED:...`), key dependency versions,
runtime file checks, and imports every compiled science module. On first run against
a fresh data directory, bootstrap lines should read ``created`` and the directories,
``config.json``, and ``vyvar.sqlite3`` must exist on disk under the printed
``data_dir``. Exit code 0 means the install is healthy; any ``FAILED:`` line or
missing on-disk artifact yields a non-zero exit.

## 4. Building the catalogs (required - never pre-built)

Catalogs are **always built by the user** (R2). The release bundle ships builder
scripts under `scripts/catalogs/` (same bundled Python; no system Python).

### 4.1 Order (do not skip Gaia)

1. **Gaia DR3 SQLite** (largest; network download from ESA Gaia TAP)
2. **Gaia blind indexes** (local CPU; requires Gaia DB from step 1)
3. **VSX local DB** (VizieR download)
4. **Exoplanet local DB** (NASA Exoplanet Archive TAP)

### 4.2 Commands (bundled launcher)

Use `--` before script flags. Outputs default to your **data directory** (section 2).

**Linux:**

```
./vyvar.sh --tool build_gaia -- --help
./vyvar.sh --tool build_gaia -- --mag-limit 16.5
./vyvar.sh --tool build_blind_index --
./vyvar.sh --tool build_vsx --
./vyvar.sh --tool build_exoplanets --
```

**Windows:**

```
VYVAR.bat --tool build_gaia -- --help
VYVAR.bat --tool build_gaia -- --mag-limit 16.5
VYVAR.bat --tool build_blind_index --
VYVAR.bat --tool build_vsx --
VYVAR.bat --tool build_exoplanets --
```

See also `scripts/catalogs/README.md` in the install folder.

### 4.3 Where files land (data directory)

| Step | Default output path (under data dir) |
|------|-------------------------------------|
| Gaia DR3 | `GAIA_DR3/vyvar_gaia_dr3.db` |
| Blind indexes | `GAIA_DR3/gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl` |
| VSX | `VSX/vyvar_vsx_local.db` |
| Exoplanets | `exoplanets/vyvar_exoplanet_local.db` |

Override with script flags (`--out`, `--db`, etc.) or set paths in Settings /
`config.json` after building.

### 4.4 Time and disk (typical)

| Step | Network source | Download | Output size (typical) |
|------|----------------|----------|------------------------|
| Gaia G<=16.5 full sky | esa.gaia.eu TAP | hours to days | **~9-10 GB** SQLite |
| Blind indexes | (local Gaia DB) | none | ~100-500 MB PKL |
| VSX | VizieR B/vsx/vsx | minutes | ~10-50 MB |
| Exoplanets | exoplanetarchive.ipac.caltech.edu | minutes | ~1-5 MB |

Gaia build is **resumable** (`strip_progress` in the DB). Safe to restart after
interruption. Narrow `--dec-min`/`--dec-max` for testing before a full-sky run.

### 4.5 Verification

```
./vyvar.sh --selftest
sqlite3 ~/.local/share/vyvar/GAIA_DR3/vyvar_gaia_dr3.db "SELECT COUNT(*) FROM gaia_dr3;"
```

Row counts depend on mag limit and sky coverage. After building, confirm paths in
Settings (Paths section) or `config.json` point at the files above.

### 4.6 When to rebuild

| Event | Action |
|-------|--------|
| First install | Full sequence (4.1) |
| Gaia mag limit raised | Re-run Gaia + blind indexes |
| VSX mag limit raised | Re-run VSX only (incremental) |
| Exoplanet archive update | Re-run exoplanet builder (incremental) |
| New VYVAR install | Keep data directory; rebuild only if you want fresh catalogs |

## 5. Equipment setup (DB Explorer)

Open **Database Explorer** in the UI. Add:

1. **Location** (lat/lon/alt)
2. **Telescope**
3. **Equipment** (camera, filters, gain)

For **OSC cameras**, set **BAYERMASK** on the equipment record (required for correct
debayer/channel handling).

## 6. Upgrade

1. Close VYVAR.
2. Replace the **install directory** with the new bundle (or unpack over it).
3. Do **not** delete the data directory - config, DB, and catalogs are preserved.

## 7. Uninstall

1. Delete the install directory.
2. Optionally delete the data directory if you no longer need observations/catalogs.

## 8. Troubleshooting

| Issue | Action |
|-------|--------|
| App will not start | Run `--selftest`; check antivirus / Windows SmartScreen (unsigned binaries) |
| No stars / no comps | Catalog paths empty or wrong - build catalogs (section 4) |
| Linux import errors | Requires **glibc >= 2.39** (Ubuntu 24.04 baseline) |
| Wrong data location | Set `VYVAR_DATA_DIR` before launching |

## License

See `LICENSE` in the install directory (proprietary; same terms as the dev repository).
