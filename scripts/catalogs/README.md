# VYVAR catalog builders (release bundle)

These scripts build local SQLite/PKL catalogs into your **data directory**
(not the install folder). Default data directory:

| Platform | Path |
|----------|------|
| Windows  | `%LOCALAPPDATA%\VYVAR` |
| Linux    | `~/.local/share/vyvar` |

Override with `VYVAR_DATA_DIR` before running.

## Recommended order

1. **Gaia DR3 SQLite** (largest download; do this first)
2. **Gaia blind indexes** (requires Gaia DB)
3. **VSX local DB**
4. **Exoplanet local DB**

## Invocation (bundled install)

Linux/macOS:

```bash
./vyvar.sh --tool build_gaia -- --help
./vyvar.sh --tool build_gaia -- --mag-limit 16.5
./vyvar.sh --tool build_blind_index --
./vyvar.sh --tool build_vsx --
./vyvar.sh --tool build_exoplanets --
```

Windows:

```bat
VYVAR.bat --tool build_gaia -- --help
```

The `--` separates launcher flags from script flags.

Direct (same bundled Python):

```bash
./python/bin/python3 -I scripts/catalogs/build_gaia_catalog.py --help
```

## Outputs (under data dir)

| Script | Default output |
|--------|----------------|
| `build_gaia_catalog.py` | `GAIA_DR3/vyvar_gaia_dr3.db` |
| `build_blind_index.py` | `GAIA_DR3/gaia_triangles_fine.pkl`, `gaia_triangles_wide.pkl` |
| `vsx_make.py` | `VSX/vyvar_vsx_local.db` |
| `exoplanet_make.py` | `exoplanets/vyvar_exoplanet_local.db` |

## Network and disk (approximate)

| Step | Source | Download | Output size |
|------|--------|----------|-------------|
| Gaia DR3 (G<=16.5, full sky) | ESA Gaia TAP | hours-days | **~9-10 GB** SQLite typical |
| Blind indexes | local Gaia DB | none | ~100-500 MB PKL (tier/mag dependent) |
| VSX | VizieR B/vsx/vsx | minutes | ~10-50 MB |
| Exoplanets | NASA Exoplanet Archive TAP | minutes | ~1-5 MB |

Gaia build is resumable (`strip_progress` table). Re-run safely after interruption.

## Verification

```bash
./vyvar.sh --selftest
sqlite3 "$VYVAR_DATA_DIR/GAIA_DR3/vyvar_gaia_dr3.db" "SELECT COUNT(*) FROM gaia_dr3;"
```

Set paths in Settings or `config.json` (`gaia_db_path`, blind index paths, `vsx_local_db_path`,
`exoplanet_local_db_path`) if you use non-default locations.
