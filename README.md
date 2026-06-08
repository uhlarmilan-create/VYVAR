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
python -m pytest tests/test_photometry_core.py -v
```

## License

See `LICENSE` for details.

## Citation

If you use VYVAR in your research, please cite:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star Observers.
> (paper in preparation)

## Log files

Pipeline logs are written to the project root during runs.
They are gitignored. For persistent logs use `--log-dir logs/` (future TODO-ORCHESTRATOR).
