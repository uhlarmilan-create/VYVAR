# VYVAR — Automated Differential Photometry Pipeline

*Turn a night of raw FITS frames into trustworthy, submission-ready variable-star light curves.*

*Read this in Czech: [README_CZ.md](README_CZ.md) — česká verze.*

---

## What it is

VYVAR is an automated differential photometry pipeline for the amateur variable-star
observer who wants observatory-grade results without hand-tuning every step. Point it at a
folder of science frames plus calibration frames and it calibrates, plate-solves, builds a
comparison-star ensemble, measures every star, judges how much to trust each light curve,
and writes out the reports and AAVSO / VarAstro submission files. Everything runs from a
single Streamlit app, and every configurable knob is documented in plain language so you can
drive the pipeline from the UI or by editing `config.json` directly.

The goal is *honest* photometry. VYVAR does not just draw a pretty curve — it quantifies the
noise with a real CCD error model, selects comparison stars the way the Broeg (2005)
algorithm intends, cross-checks candidates against Gaia DR3, VSX and TESS, and attaches a
GREEN / YELLOW / RED trust verdict so you know whether a detected variation is real or an
artefact. Its extraction has been cross-validated against four independent tools and its
photometry outputs are guarded byte-for-byte against a frozen reference so refactors cannot
silently move the science numbers.

---

## Pipeline in detail

Each stage below is a real, testable step in the production path.

### Calibration
Bias / dark / flat reduction builds and applies master calibration frames from a temperature-
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

### Plate solving (blind solver + Gaia verification)
A rebuilt **blind plate-solver** (`vyvar_blind_solver.py`) uses a density-matched Gaia DR3
triangle index (8-NN triangle hashes), DBSCAN vote clustering (haversine metric), and
cluster-level RANSAC WCS verification with geometric match-fraction scoring rather than a bare
vote count. A **two-tier index** (fine for long focal length, wide for short) lets a
scale-aware orchestrator pick the right tier. Solutions are verified against Gaia to
`verify_mag_limit = 14` (A/B-tested: as reliable as mag 16 at ~28% less runtime). WCS writes
are fail-closed — a write error blocks Phase 2A rather than shipping stale astrometry. Blind
astrometric calibration follows the hint-as-prior philosophy of Lang et al. (2010).

### MASTERSTAR
A deep, aligned master frame gives stable centroids, WCS, and FWHM for the whole night,
against which per-frame photometry is anchored.

### Gaia DR3 / VSX / exoplanet crossmatch
Every measured star is matched to a local Gaia DR3 catalog (SQLite, built full-sky by
`build_gaia_catalog.py`), tagged with known-variable flags from the AAVSO VSX (Watson, Henden
& Price 2006), and cross-checked against exoplanet host catalogs. Gaia BP-RP colours and
GSP-Phot stellar parameters (Andrae et al. 2023) feed colour terms and the HRD.

### Ensemble differential photometry
Comparison stars are chosen and weighted following **Broeg, Fernández & Neuhäuser (2005)** —
a variability-weighted artificial comparison star with iterative down-weighting of variable
references. Selection ranks stars by **BP-RP colour tiers** (Gaia colour transforms; Jordi et
al. 2010, Riello et al. 2021) then RMS, with a `comp_select_rms_floor` (default `1e-6`) that
drops isolated-bin artefacts. The path **adapts to field density**: graceful degradation on
sparse fields keeps an honest result instead of failing, and an order-independent comp-QA
locus (Sokolovsky indices) makes comparison-star flagging reproducible regardless of target
order. The canonical ensemble combination is flux-sum (AstroImageJ-validated; Collins,
Kielkopf & Stelzer 2017); Broeg inverse-variance weighting is available but parked until the
per-measurement sigma budget validates.

### Error model
Per-star uncertainties use the **Howell (1989)** CCD signal-to-noise equation with SNR-optimal
apertures, an empty-aperture background term (Labbé et al. 2003), **Honeycutt (1992)**
common-mode ensemble-residual removal, and a per-rig systematic floor (`sigma_sys`) so formal
errors do not under-report real scatter (Merline & Howell 1995). Airmass uses the Kasten &
Young (1989) optical air-mass formula. SysRem systematic detrending (Tamuz, Mazeh & Zucker
2005) has been evaluated and is available but is not enabled in the default production path.

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
classification, and a full **configuration-provenance page** — the exact parameter snapshot,
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
| photutils 3.0 | Differential LC vs VYVAR `dao_flux`, mag 8–13 | 67 | Δ < 0.001 mag |
| Muniwin 2.1.36 (c-munipack) | Differential LC, same comparison stars | 3 | ±5–15% RMS |
| IRAF apphot (Community IRAF 2.17.1) | Single-frame flux on MASTERSTAR.fits | 48 | 2.2% scatter (after ZP) |
| SExtractor 2.28 | Single-frame flux | 273 | 6% offset (growth-curve / PSF wings) |

An independent end-to-end study on draft_000365 (V842 Her) built its own Gaia catalog,
detection, apertures, and background: a SExtractor-class mesh-background pipeline (SEP; Barbary
2016) reproduces VYVAR aperture extraction to **~0.2% per frame**, and three engines
(photutils + sep + VYVAR) reproduce the science light-curve RMS to ~1% with no systematic
offset.

**Anchor discipline.** Photometry outputs are held to a byte-identical **SHA-256 regression
baseline** (frozen reference sets `770966c3…` / `edbd97e7…`), backed by a science-meaningful
numeric comparator at the ~`1e-6` level, so a refactor or a config change either reproduces the
frozen science numbers exactly or is flagged.

---

## Reproducibility & engineering

- **963 tests** pass (19 skipped) on the current tree.
- **Anchor gates**: a fast/full session baseline check re-verifies pytest, config paths, and
  the frozen science anchor before new work is accepted.
- **Provenance in every report**: exact config snapshot + git head + timestamp on the report's
  configuration page.
- **Human-editable config**: `config.json` is grouped, commented, and tolerant of `//`
  comments; `python dev/scripts/validate_config.py` checks a hand-edited file before a run.
- **Documented parameter surface**: **269** registered parameters (config.json persists
  **249**); all documented in `docs/VYVAR_PARAMS.md` and the config guides.

---

## Screenshots

<!-- SCREENSHOT 1: Capture the main Streamlit dashboard after a full run — the RUN VYVAR
     view with the pipeline phases complete and the variability/trust dashboard visible.
     Save as img/readme_dashboard.png -->
![VYVAR Streamlit dashboard after a completed run](img/readme_dashboard.png)

<!-- SCREENSHOT 2: Capture a representative page of the PDF Summary Measure Report — ideally
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
- **GPU:** optional NVIDIA GPU — only accelerates plate solving; not required.
- **Data:** any FITS-producing telescope + monochrome/mono-CMOS or CCD setup, from a wide
  short-focus rig (~9.8″/px) to a long-focus Newton (~0.65″/px). VYVAR is scale-aware and
  picks the aperture-vs-PSF path accordingly.

You also need a local Gaia DR3 catalog (built once; see Installation).

---

## Installation

A full step-by-step installer and setup guide (**INSTALL.md**, EN + CZ) is in preparation.
Until then, the short version:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Build the local Gaia DR3 catalog once (run from inside the repo — the builders import
`gaia_catalog_id.py` from `src_py/`; use `--out` to place the large database anywhere):

```bash
python GAIA_DR3/build_gaia_catalog.py --mag-limit 16.5 --out <path-to-db>
python GAIA_DR3/build_blind_index.py --db <path-to-db> --tier both
```

Quick smoke build (small sky patch):

```bash
python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db
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
| Pipeline manual (Czech) | `docs/VYVAR_PIPELINE_CZ.md` |
| Magnitude calibration data-flow (Czech) | `docs/VYVAR_CALIBRATION.md` |
| Operating runbook | `docs/VYVAR_RUNBOOK.md` |
| Validation harness & anchor discipline | `docs/VYVAR_VALIDATION.md` |
| Algorithm & software citations | `CITATIONS.bib` |

Editing `config.json` by hand is fully supported — see the configuration guides and the
`validate_config.py` checker.

---

## Project status & license

VYVAR is in active development and used for real variable-star submissions. Current tree:
963 tests green, 269 documented parameters, byte-identical photometry anchor discipline. The
public paper is in preparation.

VYVAR is **proprietary**. Copyright © 2026 Milan Uhlár. All rights reserved. No use, copying,
modification, or distribution is permitted without prior written permission. The software is
provided without warranty of any kind. See [LICENSE](LICENSE).

## Citation

If VYVAR contributes to your work, please cite:

> VYVAR: An Automated Differential Photometry Pipeline for Amateur Variable Star
> Observers. (paper in preparation)
