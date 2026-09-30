# VYVAR alpha testing guide (0.10.0a1)

Date: 2026-09-30. Audience: invited external alpha testers.

## What this alpha is

- **Aperture photometry:** validated end-to-end, cross-checked against AstroImageJ
  (4.86 mmag RMS, 134 epochs, BO CVn). This is the path to trust for science
  submissions.
- **ePSF photometry:** available, default **OFF**, marked **beta**. Internally
  consistent with aperture on the reference night (era06). Independent external
  PSF comparison and multi-rig validation are still pending - do not treat ePSF
  as submission-ready without Milan's OK.

## Supported setups (as far as tested)

Primary validation field and rig:

- Field: BO CVn night (draft 516 / era05-era06 snapshot chain).
- Optics class: wide / coarse plate-scale (~9.8 arcsec/px class; Carl-Zeiss 200 mm
  + QHY294MM in the reference run).
- Filters: unfiltered and filtered sessions exist in-house; alpha claims for
  aperture are tied to the BO CVn cross-check above.

Not yet claimed for alpha:

- Independent second observatory / second rig (debt **EPSF-CROSSRIG-01**).
- Independent external ePSF comparison + blend stress test (**EPSF-XVAL-EXT-01**).

## What to test

1. Install from the alpha tag / branch (`docs/INSTALL.md`). Prefer the catalog
   copy option (~12 GB "zaloha" set) so the app is not in LIMITED MODE.
2. First run: in Settings create **Location**, **Telescope**, and **Equipment**,
   then select them on the VAR-STREM page. RUN VYVAR stays disabled until a
   camera and telescope are selected (`src_py/app.py` run gate).
3. Process one of your own nights (calibrated or raw with library masters).
4. Inspect aperture light curves, Summary Measure Report PDF, and one AAVSO or
   VarAstro export.
5. Optional: turn on ePSF Auto Run (beta) or use the ePSF tab - report only; do
   not submit ePSF products as final photometry unless asked.

## How to report a problem

Attach all of the following when possible:

| Item | Why |
|------|-----|
| VYVAR version string (`0.10.0a1` / UI or export `#SOFTWARE`) | reproduce build |
| `git` hash + dirty flag (PDF provenance page, or `git rev-parse HEAD` + `git status -sb`) | exact tree |
| `config.json` (redact absolute paths if needed; keep keys) | settings |
| Draft id / draft folder name under `Archive/Drafts/` | data identity |
| Run log for that night (Streamlit / session log under the draft) | failure trail |
| Short description: OS, Python 3.12, telescope+camera, what you clicked, expected vs actual | triage |

Send the package to Milan Uhlar (alpha contact). Do not open a public issue with
proprietary data attached.

## Known issues (plain language)

These are tracked debts, not fixed in this alpha:

- **PIN-ISOLATION-01** - some pinned comparison stars sit closer than the usual
  3-FWHM isolation rule (~2.6 FWHM nearest neighbour).
- **SAT-COLNAME-01** - a saturation column is named `..._85pct` but holds 0.80.
- **CHI2-THRESH-LEGACY-01** - legacy `psf_chi2_threshold` still appears in
  config/UI; the live ePSF gate uses a data-derived chi2 locus instead.
- **EPSF-PERF** - ePSF runtime / forced-linear refit performance work deferred.
- **EPSF-XVAL-EXT-01** - independent external PSF comparison + blend test pending.
- **EPSF-CROSSRIG-01** - second night / second rig validation pending.

## Short CZ note / kratka CZ poznamka

Aperturni fotometrie je end-to-end validovana (AstroImageJ 4.86 mmag RMS,
134 epoch, BO CVn). ePSF je **beta** (vychozi VYPNUTO): interni konzistence s
aperturou na referencni noci (era06); nezavisla externi a multi-rig validace
jeste ceka. Pri chybe poslete verzi, git hash, `config.json`, draft id a log
behu. Znamy dluh: izolace pinu, misnamed sat sloupec, legacy chi2 prah, ePSF
vykon, externa ePSF validace, druha montaz.
