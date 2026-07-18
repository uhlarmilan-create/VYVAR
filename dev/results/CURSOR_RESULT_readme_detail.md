CURSOR RESULT — README-DETAIL — 2026-07-18

What I did
==========
Rewrote `README.md` (EN) and `README_CZ.md` (CZ) in lockstep as a detailed GitHub front
door with all 10 required sections, restoring and **verifying** the old README's
cross-validation table against the repo record. Every number and claim is sourced below
(HONESTY RULE). One commit on top of the held stack; pytest + `--fast` green; push HELD.

Baseline: local HEAD `d879ccf` (recon audit + arc 1, unpushed).

## Cross-validation table — verification

The old README's four-row table survives, corrected and completed from the canonical
"Finálna valida?ná tabu?ka VYVAR (draft_310, BO CVn pole)" in `docs/VYVAR_JOURNAL.md`
(lines 4991–4996) plus the individual validation records:

| Row | README value | Source (docs/VYVAR_JOURNAL.md) |
|-----|--------------|--------------------------------|
| photutils 3.0 | diff LC, 67 stars, mag 8–13, ? < 0.001 mag | 3738, 4963, 4993 |
| Muniwin 2.1.36 (c-munipack) | diff LC, same comps, 3 stars, ±5–15% RMS | 4967–4986, 4994 |
| IRAF apphot (Community IRAF 2.17.1) | single-frame flux, 48 stars, 2.2% scatter (after ZP) | 4413–4417, 4964, 4995 |
| SExtractor 2.28 | single-frame flux, 273 stars, 6% offset (growth curve) | 4401–4425, 4996 |

**Rows dropped: NONE.** All four rows are sourced. Corrections vs the old README:
- IRAF: added star count (48) and "after ZP" qualifier (JOURNAL 4416: ZP offset 24.977 ?
  25.0 after gain fix).
- SExtractor: added star count (273) and "growth curve / PSF wings" mechanism (JOURNAL
  4425/4431).
- Muniwin: added "(c-munipack)" and star count (3).
- Version nuance recorded: photutils cross-val was run on photutils **3.0.0** (JOURNAL
  4385, Lenovo/Kubuntu/Py3.12); the repo now pins `photutils>=2.3,<3` for the anchor. The
  table documents the version used at validation time, which is the honest historical record.

## Claim -> source table (all other numeric/technical claims)

| Claim in README | Source |
|-----------------|--------|
| 963 tests pass, 19 skipped | full `pytest -q` this session; `docs/VYVAR_STATE.md` test line |
| 269 registered params / 249 persisted | `params_registry` len=269; `docs/VYVAR_CONFIG_GUIDE_EN.md` L36–37; DECISIONS WAVE-B |
| sep reproduces VYVAR extraction ~0.2%/frame | STATE L628–629; DECISIONS L811/922; JOURNAL L2661/2696 |
| 3 engines reproduce science RMS to ~1%, no offset | JOURNAL L2660 (draft_000365 table L2652–2657) |
| byte-identical SHA anchors 770966c3 / edbd97e7 | `docs/VYVAR_PIPELINE_CZ.md` L166–170; `docs/VYVAR_VALIDATION.md` L146–148 |
| ~1e-6 comparator | `src_py/config.py:534` `comp_select_rms_floor=1e-6`; `src_py/pipeline.py:3536` `abs(pv-nv)<1e-6` |
| Broeg (2005) variability-weighted comp selection | `CITATIONS.bib` broeg2005; STATE L453 |
| Honeycutt (1992) common-mode removal | `CITATIONS.bib` honeycutt1992 |
| Howell (1989) CCD SNR equation + optimal aperture | `CITATIONS.bib` howell1989 |
| Kasten & Young (1989) air mass | `CITATIONS.bib` kastenyoung1989 |
| SysRem (Tamuz, Mazeh & Zucker 2005) evaluated, not default-on | `CITATIONS.bib` tamuz2005; STATE (sysrem_enabled default off) |
| Sparse trust: check-star ensemble n>=2, Howell/Warnock/Mitchell (1988) triangulation | CHANGELOG L29–31; `CITATIONS.bib` howellwarnockmitchell1988 |
| GREEN/YELLOW/RED trust bands; comp_trust_min_comps | STATE L59, L630 |
| flux-sum canonical (AIJ-validated); Broeg IVW parked | STATE L451–453; DECISIONS |
| Sokolovsky (2017) + von Neumann (1941) variability indices | `CITATIONS.bib` sokolovsky2017/vonneumann1941 |
| Lomb-Scargle (Lomb 1976 / Scargle 1982 / VanderPlas 2018) | `CITATIONS.bib` |
| TESS auto cross-check via Lightkurve, blend + period reliability | CHANGELOG; `CITATIONS.bib` lightkurve2018 |
| VSX known-variable flags (Watson, Henden & Price 2006) | `CITATIONS.bib` watson2006 |
| Gaia BP-RP colour tiers (Jordi 2010 / Riello 2021) | `CITATIONS.bib` jordi2010/riello2021 |
| Gaia GSP-Phot stellar params (Andrae 2023) for HRD | `CITATIONS.bib` andrae2023 |
| blind solver: 8-NN triangle index, DBSCAN votes, RANSAC WCS | `docs/VYVAR_PIPELINE_CZ.md` L145–157 |
| verify_mag_limit=14 as reliable as 16, ~28% less runtime | `docs/VYVAR_PIPELINE_CZ.md` L155–156 |
| stars_per_cell=95 fine-tier index density | `docs/VYVAR_PIPELINE_CZ.md` L145 |
| median centroid drift ~0.4 px over a night | JOURNAL L2669 (0.39 px, 127 frames) |
| CAL-DIAG radiometry gate default ON | `docs/VYVAR_CAL_DIAG_SPEC.md`; STATE |
| sky-surface preprocess (low-order surface fit) | `dev/tests/test_preprocess_sky_surface.py`; pipeline |
| WCS write fail-closed blocks Phase 2A | `docs/VYVAR_PIPELINE_CZ.md` L204–206 |
| provenance snapshot (config + git head + timestamp) in every report | STATE L57, L282 |
| human-editable commented config.json + validate_config.py | `docs/VYVAR_PROCESS.md` L167–172; `dev/scripts/validate_config.py` |
| Python 3.12 developed/tested (dropped unverified "3.10+") | `pyproject.toml` (ruff target py312, pytest pythonpath); env Python312 |
| plate scales ~9.8"/px wide, ~0.65"/px Newton | `docs/VYVAR_PIPELINE_CZ.md` L121–122; STATE L451 |

## Sections delivered (EN; CZ mirrors section-for-section, natural Czech, UTF-8)
1. Title + tagline + CZ/EN cross-link. 2. What it is (2 paras). 3. Pipeline in detail
(13 stage subsections, literature cited inline). 4. Validation (verified cross-val table +
anchor-discipline sentence). 5. Reproducibility & engineering. 6. Screenshots — three slots
`img/readme_dashboard.png`, `img/readme_report.png`, `img/readme_lightcurve.png` with HTML
capture instructions + meaningful alt text. 7. Hardware it runs on (generic; no private
site/rig details beyond the old README's published class). 8. Installation (pointer +
correct pytest/dev paths). 9. Docs index (extended). 10. Project status + proprietary license.

## Screenshot slots (Milan to capture)
- `img/readme_dashboard.png` — Streamlit RUN VYVAR view after a full run (phases done + trust
  dashboard).
- `img/readme_report.png` — a PDF Summary Measure Report page (HRD or config-provenance page).
- `img/readme_lightcurve.png` — one clean variable-star light curve with comps + error bars.
Until captured, GitHub shows broken-image alt text (placeholders, by design).

## README rendered-preview note
Standard GitHub-Flavored Markdown: H1 + italic tagline + italic CZ/EN link line, prose,
bullet lists, four tables (validation, docs index, hardware bullets), fenced `bash` blocks,
and three image lines with HTML comments (comments do not render). `README_CZ.md` mirrors it
with UTF-8 diacritics. No raw HTML beyond comments; renders cleanly.

## Gates
- Full `pytest -q`: 963 passed, 19 skipped.
- `dev/scripts/session_baseline_check.py --fast`: OVERALL PASS.
- docs-layout / params guards: green (README + README_CZ are at repo root, not docs/).

## Commit
`docs(readme): detailed EN+CZ front door (verified cross-val table)` — one commit on top of
the stack.

## HOLD
Not pushed. Full stack to push on Milan's word (oldest first):
`0e67786` (recon) -> `bfac696` -> `3e6ecfd` -> `e4074d6` -> `e03cd06` -> `385cadf` ->
`df79775` -> `d879ccf` -> <this commit>.
