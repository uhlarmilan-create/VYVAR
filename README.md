# VYVAR — Automated Differential Photometry Pipeline

VYVAR is an automated differential photometry pipeline for amateur variable star observers.
It processes raw FITS images through calibration, astrometry, aperture photometry,
and variability detection, producing light curves and reports ready for AAVSO/VAR.ASTRO submission.

## Features

- Automated pipeline: calibration → plate solving → MASTERSTAR → aperture photometry → variability detection
- Broeg (2005) weighted differential photometry with optimal comp star selection
- Howell (1989) CCD photometric error model with per-star SNR-optimal apertures
- TESS cross-validation for variability candidates
- Gaia DR3 catalog matching and HRD classification
- PDF Summary Measure Report with per-star light curves
- AAVSO and VAR.ASTRO export with full citation headers

## Validation

VYVAR photometry has been cross-validated against four independent tools:

| Tool | Method | Agreement |
|------|--------|-----------|
| photutils 3.0 | Differential LC (67 stars, mag 8–13) | Δ < 0.001 mag |
| Muniwin 2.1.36 | Differential LC (same comp stars) | ±5–15% RMS |
| IRAF apphot | Single-frame flux (48 stars) | 2.2% scatter |
| SExtractor 2.28 | Single-frame flux | 6% offset (growth curve) |

## Installation

### Requirements

- Python 3.10+
- Windows 10/11 or Linux (Ubuntu 22.04+)
- 8 GB RAM minimum, 16 GB recommended
- NVIDIA GPU optional (plate solving acceleration)

### Dependencies

```bash
pip install -r requirements.txt
```

### Building the Gaia catalog (new user)

After cloning VYVAR, run the catalog builders **from inside the repo** (they import
`gaia_catalog_id.py` from the clone root). Use `--out` / `--fine-out` / `--wide-out` to
write the large database and index files anywhere you like.

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <path-to-db>
python GAIA_DR3/build_blind_index.py --db <path-to-db> --tier both
```

Quick smoke (small sky patch):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

### Running VYVAR

```bash
streamlit run app.py
```

## Algorithm References

See `CITATIONS.bib` for full references. Key algorithms:

- **Differential photometry:** Broeg, Fernandez & Neuhäuser (2005) AN 326:134
- **CCD photometric error:** Howell (1989) PASP 101:616
- **ZP ensemble sigma-clip:** Stetson (1987) PASP 99:191
- **Star detection (DAOPHOT):** Stetson (1987) PASP 99:191
- **Catalog:** Gaia DR3 — Gaia Collaboration (2023) A&A 674, A1

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -m slow   # include slow integration tests
```

Full suite (2026-07-14): **852 passed**, 15 skipped.

## License

See `LICENSE` for details.

## Citation

If you use VYVAR in your research, please cite:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star Observers.
> (paper in preparation)

## Log files

Pipeline logs are written to the project root during runs.
They are gitignored. For persistent logs use `--log-dir logs/` (future TODO-ORCHESTRATOR).
