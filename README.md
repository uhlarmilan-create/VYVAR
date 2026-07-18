# VYVAR — Automated Differential Photometry Pipeline

*Read this in Czech: [README_CZ.md](README_CZ.md) — česká verze.*

VYVAR turns a night's worth of raw FITS frames into publication-ready variable-star
light curves. It is built for the amateur observer who wants observatory-grade
differential photometry without hand-tuning every step: point it at a folder of
science frames plus calibration frames, and it calibrates, plate-solves, builds a
comparison-star ensemble, measures every star, judges how much to trust each light
curve, and writes out the reports and AAVSO/VAR.ASTRO submission files.

The goal is *honest* photometry. VYVAR does not just produce a pretty curve — it
quantifies the noise, selects comparison stars the way the Broeg (2005) algorithm
intends, cross-checks candidates against Gaia DR3 and TESS, and attaches a trust
verdict so you know whether a detected variation is real or an artefact. Everything
runs from a single Streamlit app, and every configurable knob is documented in plain
language so you can drive the pipeline from the UI or by editing `config.json` directly.

![VYVAR user interface (screenshot placeholder — img/vyvar_ui.png)](img/vyvar_ui.png)

## What VYVAR does

- **Calibration** — bias/dark/flat reduction, sky-background modelling, and quality
  gates on each frame before anything is measured.
- **Detection & Gaia** — DAOPHOT-style source detection, blind plate-solving, and
  Gaia DR3 catalog matching with HRD classification for every measured star.
- **Ensemble photometry** — Broeg, Fernández & Neuhäuser (2005) variability-weighted
  differential photometry with SNR-optimal apertures (Howell 1989) and automatic
  comparison-star selection, floor policies, and graceful degradation on sparse fields.
- **Trust** — per-light-curve trust scoring, check-star witnesses, sigma-budget
  accounting, and TESS cross-validation so questionable detections are flagged, not hidden.
- **Reports & submission** — a PDF Summary Measure Report with per-star light curves,
  plus AAVSO and VAR.ASTRO exports carrying full citation headers.

## Project status (honest numbers)

- **963 tests** pass (19 skipped) on the current tree.
- **269** registered, documented configuration parameters (see `docs/VYVAR_PARAMS.md`).
- **Anchor discipline:** photometry output is guarded byte-for-byte against a frozen
  reference run, so refactors and config changes cannot silently move the science
  numbers (see `docs/VYVAR_VALIDATION.md`).
- Photometry has been cross-validated against AstroImageJ, Muniwin, IRAF apphot, and
  SExtractor; details and current agreement figures live in `docs/VYVAR_VALIDATION.md`.

## Installation

A full step-by-step installer and setup guide (**INSTALL.md**, EN + CZ) is landing in
the next release. Until then, the short version:

```bash
pip install -r requirements.txt
streamlit run app.py
```

You also need a local Gaia DR3 catalog. Build it from inside the repo (the builders
import `gaia_catalog_id.py` from `src_py/`); use `--out` to place the large database
wherever you like:

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <path-to-db>
python GAIA_DR3/build_blind_index.py --db <path-to-db> --tier both
```

Quick smoke build (small sky patch):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

Requirements: Python 3.12 (developed and tested on 3.12), Windows 10/11 or Linux,
8 GB RAM minimum (16 GB recommended). An NVIDIA GPU is optional and only accelerates
plate solving.

## Documentation

| Topic | Document |
|-------|----------|
| Configuration guide (English) | `docs/VYVAR_CONFIG_GUIDE_EN.md` |
| Configuration guide (Czech) | `docs/VYVAR_CONFIG_GUIDE_CZ.md` |
| All 269 parameters (reference) | `docs/VYVAR_PARAMS.md` |
| Pipeline manual (Czech) | `docs/VYVAR_PIPELINE_CZ.md` |
| Operating runbook | `docs/VYVAR_RUNBOOK.md` |
| Validation & anchor discipline | `docs/VYVAR_VALIDATION.md` |

Editing `config.json` by hand is fully supported: the file is grouped, commented, and
tolerant of `//` comments, and `python dev/scripts/validate_config.py` checks it before
a run. See the configuration guides for details.

## Algorithm references

See `CITATIONS.bib` for the full list. Key algorithms:

- **Differential photometry:** Broeg, Fernández & Neuhäuser (2005) AN 326:134
- **CCD photometric error:** Howell (1989) PASP 101:616
- **ZP ensemble sigma-clip / DAOPHOT detection:** Stetson (1987) PASP 99:191
- **Catalog:** Gaia DR3 — Gaia Collaboration (2023) A&A 674, A1

## For developers

Tests are configured through `pyproject.toml` (`testpaths = dev/tests`,
`pythonpath = [".", "src_py", "dev"]`), so just run:

```bash
python -m pytest            # full suite
python -m pytest -q         # quiet
```

Production modules live in `src_py/`; developer material (tests, scripts, tools,
validation, results) lives under `dev/`. The root `app.py` is a thin shim that puts
`src_py/` on `sys.path`. See `CLAUDE.md` for the repository layout and workflow.

## License

VYVAR is proprietary. Copyright © 2026 Milan Uhlár. All rights reserved. No use,
copying, modification, or distribution is permitted without prior written permission.
The software is provided without warranty of any kind. See [LICENSE](LICENSE).

## Citation

If VYVAR contributes to your work, please cite:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
