# VYVAR star catalogs

All catalogs live in one folder, `catalogs/` in the VYVAR data directory (the VYVAR folder of a
checkout; `%LOCALAPPDATA%\VYVAR` or `~/.local/share/vyvar` for a release install, override with
`VYVAR_DATA_DIR`). The builders and the maintenance script are in the same folder of the install.
VYVAR reads the paths set in Settings -> Paths and catalogs (`gaia_db_path`, `objects_db_path`,
`astrometry_index_dir`); the builders write to the same paths by default.

| File | Content | Built by | Size |
|---|---|---|---|
| `vyvar_gaia_dr3.db` | Gaia DR3 stars (table `gaia_dr3`), coverage, `catalog_meta` | `build_gaia_catalog.py` | ~38 GB at G <= 17.5, ~10 GB at G <= 16 |
| `vyvar_objects.db` | AAVSO VSX (`vsx_data`) + NASA Exoplanet Archive (`exoplanet_data`), `catalog_meta` | `objects_make.py` | ~0.9 GB |
| `vyvar_astro_index/` | blind plate-solve index (optional: only frames without a position in the header) | `build_astro_index.py` | ~7 GB |

Every catalog DB has a `catalog_meta` table (source, build date, limits, row count; content
fingerprint of the objects tables). Run provenance and the Settings cards read it.

## Build from Settings

Settings -> Paths and catalogs -> Catalog builder runs the scripts below as separate processes
(log and progress on the page; a build of hours keeps running when the page reloads; job files in
`<data>/jobs/catalog_*`):

- Gaia DR3: build a missing DB, resume an unfinished build, report, register (`catalog_meta`), drop
  unused indexes. Gaia DR3 does not change; Gaia DR4 (2 December 2026) will be a new file.
- VSX + exoplanets: update = complete new download into a staging table; the old table is replaced
  only when everything arrived (old table kept on any error or a download < 50 % of the old size);
  rows added / removed / changed are shown on the card.
- Blind index: build or complete it from the Gaia DB.

## Build (from the VYVAR folder)

```bash
python catalogs/build_gaia_catalog.py --mag-limit 17.5      # ESA Gaia TAP, hours to days; resumable
python catalogs/objects_make.py --vsx --exoplanets          # VizieR + NASA archive, minutes
python catalogs/build_astro_index.py                        # from the Gaia DB, optional
```

Release install: the same scripts through the launcher, e.g. `./vyvar.sh --tool build_gaia -- --mag-limit 17.5`
(tools: `build_gaia`, `build_objects`, `build_astro_index`, `gaia_maintain`; `VYVAR.bat` on Windows).

The Gaia build is resumable (`strip_progress`): run it again after an interruption. VYVAR fetches
fainter stars of one field online when a frame reaches deeper than the DB (`gaia_coverage`);
those rows are added to `vyvar_gaia_dr3.db` and to its `catalog_meta` row count.

## Gaia maintenance (`gaia_maintain.py`, standard library only; close VYVAR first)

```bash
python catalogs/gaia_maintain.py --report                   # indexes, free space, catalog_meta, coverage
python catalogs/gaia_maintain.py --drop-unused --register   # drop indexes VYVAR never uses; write catalog_meta
python catalogs/gaia_maintain.py --vacuum-into catalogs/vyvar_gaia_dr3_new.db   # compact copy, checked
```
