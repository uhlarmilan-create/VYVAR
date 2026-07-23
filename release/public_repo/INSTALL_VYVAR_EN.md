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

The initializer creates `Archive/`, `CalibrationLibrary/`, catalog folders, an empty
`vyvar.sqlite3`, `config.json` from template, and `NEXT_STEPS.txt`.

## 3. Health check (selftest)

```
VYVAR.bat --selftest
./vyvar.sh --selftest
```

Prints Python version, platform, data directory, key dependency versions, and imports
every compiled science module. Exit code 0 means the install is healthy.

## 4. Build catalogs (required - never shipped)

Catalogs are **always built by the user** (R2). Use the scripts from a dev checkout
(or copy outputs into your data directory):

| Catalog | Script | Typical output under data dir |
|---------|--------|----------------------------------|
| Gaia DR3 SQLite | `GAIA_DR3/build_gaia_catalog.py` | `GAIA_DR3/vyvar_gaia_dr3.db` |
| Blind indexes | `GAIA_DR3/build_blind_index.py` | `GAIA_DR3/gaia_triangles_*.pkl` |
| VSX local | `VSX/vsx_make.py` | `VSX/vyvar_vsx_local_v2.db` |
| Exoplanets | `exoplanets/exoplanet_make.py` | `exoplanets/vyvar_exoplanet_local.db` |

Point paths in Settings or edit `config.json` in the **data directory** after building.

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
