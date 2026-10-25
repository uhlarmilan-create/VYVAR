# VYVAR - Automated Differential Photometry Pipeline

*Turn a night of raw FITS frames into trustworthy, submission-ready variable-star light curves.*

*Read this in Czech: [README_CZ.md](README_CZ.md) - ceska verze.*

---

## What it is

VYVAR is an automated differential photometry pipeline for the amateur variable-star
observer who wants observatory-grade results without hand-tuning every step. Point it at a
folder of science frames plus calibration frames and it calibrates, plate-solves, builds a
comparison-star ensemble, measures every star, judges how much to trust each light curve,
and writes out the reports and AAVSO / VarAstro submission files. Everything runs from a
single Streamlit app, and every configurable knob is documented in plain language so you can
drive the pipeline from the UI or by editing `config.json` directly.

The goal is *honest* photometry. VYVAR does not just draw a pretty curve - it quantifies the
noise with a real CCD error model, selects comparison stars the way the Broeg (2005)
algorithm intends, cross-checks candidates against Gaia DR3, VSX and TESS, and attaches a
GREEN / YELLOW / RED trust verdict so you know whether a detected variation is real or an
artefact. Its aperture extraction has been cross-validated against four independent tools and its
photometry outputs are guarded byte-for-byte against a frozen reference so refactors cannot
silently move the science numbers.

**Validation status.** Aperture photometry is validated end-to-end and cross-checked against
AstroImageJ (4.86 mmag RMS, 134 epochs, BO CVn). ePSF photometry is available as **beta**
(default OFF): internally consistent with aperture on the reference night (era06);
independent external validation and multi-rig validation are pending.

---

## Pipeline in detail

Each stage below is a real, testable step in the production path.

### Calibration
Dark / dark-flat / flat reduction (CMOS, no bias; see dev/results/specs/VYVAR_CALIBRATION_SPEC.md) builds and applies master calibration frames from a temperature-
and age-matched calibration library. A **CAL-DIAG radiometry gate** (default ON) checks the
calibrated frames for sane radiometry and records its provenance before anything is measured.
A **sky-surface preprocess** models and flattens residual background gradients (moonlight,
light pollution) with a low-order surface fit prior to detection.

### Quality control
Every frame passes QC before it contributes: FWHM, background, and star-count checks reject
clouded, trailed, or defocused frames so a bad frame never poisons the ensemble.

### Alignment
Frames are registered to a common pixel grid (astroalign) with a measured, stable centroid
drift (median ~0.4 px across a night on the validation field), independently confirmed by the
cross-validation study.

### Plate solving (astrosolve + Gaia verification)
Every MASTERSTAR is solved by `astrosolve` (PLATESOLVE-2026-10), an astrometry.net-style solver
(Lang et al. 2010): hinted with the position from the FITS header, blind with a multi-scale index
of 4-star patterns built from the local Gaia DB (`astro_index.py`, folder `vyvar_astro_index/`)
when the header has no usable position. A solution is accepted by Bayesian log-odds against the
Gaia stars of its own footprint and refined as TAN+SIP with the SIP order chosen by BIC. A WCS
handed over from another MASTERSTAR (sibling filter, OSC channel) is confirmed the same way.

### MASTERSTAR
A deep, aligned master frame gives stable centroids, WCS, and FWHM for the whole night,
against which per-frame photometry is anchored.

### Gaia DR3 / VSX / exoplanet crossmatch
Every measured star is matched to a local Gaia DR3 catalog (SQLite, built full-sky by
`build_gaia_catalog.py`), tagged with known-variable flags from the AAVSO VSX (Watson, Henden
& Price 2006), and cross-checked against exoplanet host catalogs. Gaia BP-RP colours and
GSP-Phot stellar parameters (Andrae et al. 2023) feed colour terms and the HRD.

### Ensemble differential photometry
Comparison stars are chosen and weighted following **Broeg, Fernandez & Neuhauser (2005)** -
a variability-weighted artificial comparison star with iterative down-weighting of variable
references. Selection ranks stars by **BP-RP colour tiers** (Gaia colour transforms; Jordi et
al. 2010, Riello et al. 2021) then RMS, with an RMS floor (`1e-6`) that
drops isolated-bin artefacts. The path **adapts to field density**: graceful degradation on
sparse fields keeps an honest result instead of failing, and an order-independent comp-QA
locus (Sokolovsky indices) makes comparison-star flagging reproducible regardless of target
order. The canonical ensemble combination is flux-sum (AstroImageJ-validated; Collins,
Kielkopf & Stelzer 2017); Broeg inverse-variance weighting is available but parked until the
per-measurement sigma budget validates.

### Error model
Per-star uncertainties use the **Howell (1989)** CCD signal-to-noise equation with SNR-optimal
apertures, an empty-aperture background term (Labbe et al. 2003), **Honeycutt (1992)**
common-mode ensemble-residual removal, and a per-rig systematic floor (`sigma_sys`) so formal
errors do not under-report real scatter (Merline & Howell 1995). Airmass uses the Kasten &
Young (1989) optical air-mass formula. SysRem, Savitzky-Golay and Democratic Detrender
detrending, PyTICS comparison re-weighting and the "Method B" aperture correction were
evaluated and removed from the code after the 2026-10 physics audit.

### Camera gain and read noise
The gain (e-/ADU) and read noise of every frame come from one resolver, in this order: the FITS
header when it states e-/ADU (`EGAIN`, or `GAIN` with an e-/ADU comment, e.g. Moravian SIPS);
otherwise the camera's gain curve at the gain setting written in the header (QHY / ZWO through
NINA or ASCOM write the setting, not e-/ADU) - a point measured from raw flats wins over a manually
entered one, the camera's default point and the typical values of the shipped camera catalog
(`src_py/camera_catalog.json`). Binning is applied by the camera's convention (digital average:
gain x N^2; digital sum: gain x 1; charge binning: both x 1). When nothing is known the run stops
with a message instead of using a default. Cameras are added from the catalog in Database Explorer
> EQUIPMENTS; Calibration Library > Measure gain derives gain and read noise from raw flat and dark
pairs (photon transfer, Janesick 2007). The report shows where the gain came from and an
error-model check (check-star scatter / reported errors). Lights of one observation group must
share the gain setting.

### Trust model
Each light curve receives a **GREEN / YELLOW / RED** trust band from check-star witnesses,
comparison-star count (`comp_trust_min_comps`), and scatter diagnostics. Sparse fields fall
back to a check-star ensemble at n>=2 with Howell, Warnock & Mitchell (1988) variance
triangulation and confidence-interval trust bands, so a thin field degrades gracefully instead
of silently over-claiming precision.

### Variability detection + TESS cross-analysis
Candidates are flagged with robust variability indices (Sokolovsky et al. 2017; von Neumann
1941 ratio) and periods from Lomb-Scargle analysis (Lomb 1976; Scargle 1982; VanderPlas
2018). Every candidate is auto-cross-checked against TESS (via Lightkurve) with a blend check
and period-reliability classification.

### Reports
A PDF **Summary Measure Report** carries per-star light curves, an HR diagram with Gaia-based
classification, and a full **configuration-provenance page** - the exact parameter snapshot,
git head, and generation timestamp baked into every report so any figure is reproducible.

### AAVSO / VarAstro export
Submission files are written with full citation headers driven by `CITATIONS.bib` (the single
source of truth), citing only the methods that actually ran for that dataset.

---

## Validation

VYVAR's photometry has been cross-validated against four independent professional tools on the
draft_310 (BO CVn) field. Every row below is sourced from the project record
(`docs/VYVAR_JOURNAL.md`); tool versions, star counts, and agreement figures are as measured,
not aspirational.

| Tool | Method | Stars | Agreement |
|------|--------|-------|-----------|
| photutils 3.0 | Differential LC vs VYVAR `dao_flux`, mag 8-13 | 67 | Delta < 0.001 mag |
| Muniwin 2.1.36 (c-munipack) | Differential LC, same comparison stars | 3 | +-5-15% RMS |
| IRAF apphot (Community IRAF 2.17.1) | Single-frame flux on MASTERSTAR.fits | 48 | 2.2% scatter (after ZP) |
| SExtractor 2.28 | Single-frame flux | 273 | 6% offset (growth-curve / PSF wings) |

An independent end-to-end study on draft_000365 (V842 Her) built its own Gaia catalog,
detection, apertures, and background: a SExtractor-class mesh-background pipeline (SEP; Barbary
2016) reproduces VYVAR aperture extraction to **~0.2% per frame**, and three engines
(photutils + sep + VYVAR) reproduce the science light-curve RMS to ~1% with no systematic
offset.

**Anchor discipline.** Photometry outputs are held to a byte-identical **SHA-256 regression
baseline**, backed by a science-meaningful numeric comparator at the ~`1e-6` level, so a refactor
or a config change either reproduces the frozen science numbers exactly or is flagged.

---

## Reproducibility & engineering

- **963 tests** pass (19 skipped) on the current tree.
- **Anchor gates**: a fast/full session baseline check re-verifies pytest, config paths, and
  the frozen science anchor before new work is accepted.
- **Provenance in every report**: exact config snapshot + git head + timestamp on the report's
  configuration page.
- **Human-editable config**: `config.json` is grouped, commented, and tolerant of `//`
  comments; `python dev/scripts/validate_config.py` checks a hand-edited file before a run.
- **Documented parameter surface**: **56** registered parameters (config.json persists
  **49**; untuned knobs are fixed constants in the code); all documented in
  `docs/VYVAR_PARAMS.md` and the config guides.

---

## Screenshots

<!-- SCREENSHOT 1: Capture the main Streamlit dashboard after a full run - the RUN VYVAR
     view with the pipeline phases complete and the variability/trust dashboard visible.
     Save as img/readme_dashboard.png -->
![VYVAR Streamlit dashboard after a completed run](img/readme_dashboard.png)

<!-- SCREENSHOT 2: Capture a representative page of the PDF Summary Measure Report - ideally
     one showing the HR diagram or the configuration-provenance page. Save as img/readme_report.png -->
![VYVAR PDF Summary Measure Report page](img/readme_report.png)

<!-- SCREENSHOT 3: Capture a single clean variable-star light curve (e.g. an eclipsing binary)
     with comparison stars and error bars. Save as img/readme_lightcurve.png -->
![Example variable-star light curve produced by VYVAR](img/readme_lightcurve.png)

---

## Hardware it runs on

VYVAR is a desktop Python application, not a service. It runs on:

- **OS:** Windows 10/11 or Linux (developed on both).
- **Python:** 3.12 (developed and tested on 3.12).
- **RAM:** 8 GB minimum, 16 GB recommended for large nights.
- **GPU:** optional NVIDIA GPU - only accelerates plate solving; not required.
- **Data:** any FITS-producing telescope + monochrome/mono-CMOS or CCD setup, from a wide
  short-focus rig (~9.8 arcsec/px) to a long-focus Newton (~0.65 arcsec/px). VYVAR is scale-aware and
  picks the aperture-vs-PSF path accordingly.

You also need a local Gaia DR3 catalog (built once; see Installation).

---

## Installation

A full step-by-step installer and setup guide (**[INSTALL.md](INSTALL.md)**, EN + CZ) is available.
The short version for a git dev checkout:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Build the local Gaia DR3 catalog once (the builders in `catalogs/` import their modules from
`src_py/`; by default they write to the paths set in Settings, `catalogs/` in the data directory;
`--out` places the database anywhere). See `catalogs/README.md`.

```bash
python catalogs/build_gaia_catalog.py --mag-limit 16.5 --out <path-to-db>
python catalogs/build_astro_index.py --db <path-to-db>
```

Quick smoke build (small sky patch):

```bash
python catalogs/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
```

Run the tests (configured via `pyproject.toml`: `testpaths = dev/tests`,
`pythonpath = [".", "src_py", "dev"]`):

```bash
python -m pytest
```

---

## Documentation

| Topic | Document |
|-------|----------|
| Configuration guide (English) | `docs/VYVAR_CONFIG_GUIDE_EN.md` |
| Configuration guide (Czech) | `docs/VYVAR_CONFIG_GUIDE_CZ.md` |
| All 269 parameters (reference) | `docs/VYVAR_PARAMS.md` |
| Parameter handbook (Czech, PDF) | `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf` |
| Install & first-run guide (Czech, PDF) | `docs/VYVAR_INSTALL_GUIDE_CZ.pdf` |
| Technical pipeline description (Czech, PDF) | `docs/VYVAR_FLOW_CZ.pdf` |
| Pipeline manual (Czech) | `docs/VYVAR_PIPELINE_CZ.md` |
| Magnitude calibration data-flow (Czech) | `docs/VYVAR_CALIBRATION.md` |
| Operating runbook | `docs/VYVAR_RUNBOOK.md` |
| Validation harness & anchor discipline | `docs/VYVAR_VALIDATION.md` |
| Algorithm & software citations | `CITATIONS.bib` |

Editing `config.json` by hand is fully supported - see the configuration guides and the
`validate_config.py` checker.

---

## Project status & license

VYVAR **0.10.0a1** is an alpha for external testers: aperture path is the validated product;
ePSF is **beta** (see Validation status above). Current tree keeps byte-identical aperture
anchor discipline. The public paper is in preparation. See `docs/ALPHA_TESTING.md`.

VYVAR is **proprietary**. Copyright (c) 2026 Milan Uhlar. All rights reserved. No use, copying,
modification, or distribution is permitted without prior written permission. The software is
provided without warranty of any kind. See [LICENSE](LICENSE).

## Citation

If VYVAR contributes to your work, please cite:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
