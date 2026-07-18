# Installing VYVAR

This guide mirrors the installer (`install_vyvar.ps1` on Windows,
`install_vyvar.sh` on Linux) phase by phase. The installer is interactive and
idempotent - it is safe to re-run (a healthy `.venv` is reused, catalog files are
verified rather than recopied, and `config.json` is rewritten from its current
contents).

For a narrated, step-by-step walk-through in Czech, see
`docs/VYVAR_INSTALL_GUIDE_CZ.pdf`. For what every configuration key means, see
`docs/VYVAR_CONFIG_GUIDE_EN.md` (English) or `docs/VYVAR_CONFIG_GUIDE_CZ.md`
(Czech), or the in-depth `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` (Czech). For a
project overview, see `README.md` / `README_CZ.md`.

## Prerequisites

- Windows 10/11 (primary, PowerShell installer) or Linux (bash twin, best-effort).
- Python 3.12 (the pinned science stack is validated on 3.12).
- git (to clone) OR a release ZIP of the repository.
- Disk budget:
  - ~2 GB for the code tree plus the Python virtual environment (`.venv`).
  - ~12.5 GB for the recommended catalog set (see the manifest below).
  - plus room for your own raw/processed observation data.
- 16 GB RAM recommended for full-night runs. No compiler or IRAF is needed - the
  entire production pipeline runs from wheels on Windows/Python 3.12.

## 1. Obtain the tree

Clone with git (use a personal access token for the private repository):

```
git clone https://github.com/uhlarmilan-create/VYVAR.git
cd VYVAR
```

Or download the release ZIP, extract it, and open a terminal in the extracted
folder. Everything below runs from the repository root.

## 2. Run the installer

Windows (PowerShell):

```
powershell -ExecutionPolicy Bypass -File .\install_vyvar.ps1
```

Linux (bash):

```
./install_vyvar.sh
```

The installer walks seven phases, each printing `[OK]` or `[FAIL]`:

1. **PYTHON** - detects Python 3.12 (`py -3.12`, then `python`). If it is missing
   the installer prints the download link and the `winget` command and stops.
2. **VENV** - creates `.venv`, upgrades `pip`, installs `requirements.txt`, and
   runs `pip check`. The science wheels (numpy, scipy, astropy, photutils) are
   large; this phase can take several minutes.
3. **CATALOGS** - the three options below.
4. **PATHS** - prompts for the archive root, calibration-library root, and
   database path (catalog paths are prefilled from phase 3) and writes them into
   `config.json`. The author's absolute `C:\ASTRO\...` paths are never kept: any
   path not chosen here is blanked so the project-root-relative default resolves.
   Location, telescope, and camera facts are NOT asked here - they belong to the
   app UI.
5. **VALIDATE** - runs `dev/scripts/validate_config.py`; it must exit clean.
6. **SMOKE** - imports the app through the `src_py` bootstrap (no server) and
   confirms the database self-initialises (the sqlite file and its reference
   tables are created).
7. **FINISH** - prints the exact first-run steps.

## 3. Catalogs

VYVAR needs a local Gaia catalog (and a few companions) to identify stars, select
comparison stars, and plate-solve. The installer offers three options.

### Option 1 - Copy from an existing VYVAR installation (recommended)

Point the installer at an existing VYVAR root (an external disk or another
machine). It copies the recommended "zaloha" set - the reproducible anchor
catalog (Gaia G<=16 subset), far smaller than the full 50 GB build - and verifies
each file's size after copying.

| File (destination)                     | Purpose                                   | Size     |
|----------------------------------------|-------------------------------------------|----------|
| `GAIA_DR3/vyvar_gaia_dr3.db`           | Local Gaia DR3 star catalog (G<=16 subset)| ~9.4 GB  |
| `GAIA_DR3/gaia_triangles_fine.pkl`     | Blind plate-solve index (narrow field)    | ~1.3 GB  |
| `GAIA_DR3/gaia_triangles_wide.pkl`     | Blind plate-solve index (wide field)      | ~0.7 GB  |
| `VSX/vyvar_vsx_local_v2.db`            | AAVSO VSX known-variable catalog          | ~0.87 GB |
| `exoplanets/vyvar_exoplanet_local.db`  | NASA exoplanet-host cross-match           | ~2 MB    |
| **Total**                              |                                           | **~12.1 GB** |

The installer reports the total size upfront and checks free space on the target
drive before copying. The set fits on a 16 GB USB stick. (The full 50 GB
`GAIA_DR3/vyvar_gaia_dr3.db` is only needed for fields fainter than the G<=16 cut.)

### Option 2 - Build from sources

Downloads and builds the catalogs from the ESA Gaia archive, AAVSO VSX, and the
NASA exoplanet archive. Expect large downloads and hours-to-days depending on your
connection. This is not the quick-setup path. The builders are:

```
python GAIA_DR3/build_gaia_catalog.py     # Gaia DR3 -> sqlite (astroquery TAP)
python GAIA_DR3/build_blind_index.py      # triangle indexes from the DB
python VSX/vsx_make.py                     # AAVSO VSX -> sqlite
python exoplanets/exoplanet_make.py        # NASA archive -> sqlite
```

### Option 3 - Skip for now (LIMITED MODE)

The installer completes but the app starts in LIMITED MODE: without the Gaia
catalog there is no star identification or comparison-star selection, so the
pipeline is not usable for science until catalogs arrive. Re-run the installer
(option 1) to add them later.

## 4. First run

Start the app from the repository root:

```
streamlit run app.py
```

The database ships self-initialised with the author's example reference rows
(equipment such as `QHY294MM`, telescopes such as `Carl-Zeiss`, and the location
`Dablice`). These are examples, not your setup. In the app:

1. Open **Settings** and create your own **Location**, **Telescope**, and
   **Equipment**, then select them. Do not submit observations under the seeded
   example rows.
2. Import your first night and run the pipeline.

## Troubleshooting

| Symptom                                   | Cause / fix                                                                                          |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `Python 3.12 not found`                   | Install it: `winget install -e --id Python.Python.3.12` (tick "Add to PATH"), or python.org. Re-run.|
| `pip install` fails with SSL / proxy      | Behind a corporate proxy: set `HTTPS_PROXY`/`HTTP_PROXY`, or `pip config set global.proxy ...`. Retry.|
| `not enough free space on ...`            | Choose a target data drive with room for ~12.5 GB of catalogs plus your data (installer prompts).    |
| Catalogs skipped / LIMITED MODE warning   | You chose option 3. Re-run the installer, option 1, and point it at a VYVAR install with the catalogs.|
| App starts but finds no stars / no comps  | `gaia_db_path` / `vsx_local_db_path` empty or wrong. Re-run installer option 1, or edit `config.json` and run `python dev/scripts/validate_config.py`. |
| `Port 8501 is already in use`             | Another Streamlit is running. Close it, or start on another port: `streamlit run app.py --server.port 8502`. |
| Antivirus locks / slows the sqlite files  | Add an exclusion for the VYVAR data folder (Gaia DB is large; real-time scanning can lock or slow it).|
| Config error after hand-editing           | Run `python dev/scripts/validate_config.py` - it reports the line/column and the closest valid key.  |
