# VYVAR

Automated differential photometry pipeline for variable stars: from raw FITS to
calibrated, plate-solved, measured light curves with AAVSO / VarAstro export and
PDF reports.

## Features

- End-to-end night processing: import, calibration, alignment, plate solve, photometry
- Gaia DR3 cross-match, comparison-star selection, quality gates
- OSC multi-channel extraction and Johnson-band export paths
- Streamlit UI plus headless pipeline (same science code paths)
- Compiled science core for release performance; UI stays interpreted for debuggability

## Screenshots

*(Placeholder - add UI screenshots before public release.)*

## Context

VYVAR targets advanced amateur and small-observatory workflows aligned with AAVSO
reporting and VarAstro-style analysis. Catalogs (Gaia DR3, VSX, exoplanet host DB)
are built locally by the user - not shipped with the installer.

## Install

See [INSTALL_VYVAR_EN.md](INSTALL_VYVAR_EN.md) (English) or
[INSTALL_VYVAR_CZ.md](INSTALL_VYVAR_CZ.md) (Czech, ASCII diacritics-free).

Download preview bundles from [GitHub Releases](https://github.com/uhlarmilan-create/VYVAR-release/releases)
(`VYVAR-preview-YYYYMMDD-win64.zip` / `linux-x64.tar.gz`).

## License

Proprietary - see LICENSE. Contact Milan Uhlar for permission.
