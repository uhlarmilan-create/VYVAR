CURSOR RESULT - 2026-09-30 ALPHA-READY-01

What I did
Closed EPSF-CHI2-LOCUS-01 Phase 2 LOCK (era06), wrote honest ePSF-beta
capability text, measured/fixed alpha hygiene H1-H8, prepared merge/tag
commands for Milan. STOP for PUSH_AUTH (no main merge/tag from this agent).

## Step 1 - LOCK

era06 G3 refs SET: BO 15.372 / FW 5.360 mmag, n_full=134 each.
epsf01 anchor: c94cf4fedd60b381... n=53.
era05 epsf01 552ace75 SUPERSEDED.
Aperture unchanged: era05 87197716 / dd92e99d.
Ledger VL-ANCHOR-EPSF01 + DECISIONS D-EPSF-CHI2-LOCUS-01 + STATE + ROADMAP debt:
PIN-ISOLATION-01, SAT-COLNAME-01, CHI2-THRESH-LEGACY-01, EPSF-PERF-01,
EPSF-XVAL-EXT-01, EPSF-CROSSRIG-01.

Stamps (artifacts under
dev/results/context/session_20260930_alpha_ready_01/):
- --full-epsf OVERALL PASS (full_epsf.log; exit 0; ~17862 s)
  full-photometry-sha-core-aperture PASS 87197716...
  full-photometry-sha-ext-aperture PASS dd92e99d...
  full-photometry-sha-core-psf PASS c94cf4fe...
  full-g3-residual PASS dem BO=15.372 FW=5.360 n_full=134
- --fast --clean OVERALL PASS (fast_clean.log; exit 0; 1651 pytest)

## Step 2 - honest capability

README.md, docs/README_FULL.md, docs/README_CZ.md, UI ePSF tab caption,
params_registry + CONFIG guides EN/CZ: aperture validated (AIJ 4.86 mmag
RMS, 134 epochs, BO CVn); ePSF beta default OFF; era06 internal consistency;
external / multi-rig pending. No overclaim of ePSF as production-validated.

## Step 3 - hygiene table

| ID | Status | Notes |
|----|--------|-------|
| H1 clean install | FIXED | install_vyvar.ps1 -NonInteractive on fresh local clone failed smoke (required retired OBSERVATION/OBS_DRAFT). Smoke assertion FIXED in .ps1 and .sh. Retry INSTALL COMPLETE. Headless streamlit HTTP 200. night_run on bundled sample: MISSING-needs-Milan (NonInteractive skips catalogs; no small night FITS bundle). Linux container: MISSING-needs-Milan (sh smoke fixed statically). |
| H2 first-run | PRESENT | RUN disabled without camera+telescope: src_py/app.py:1791-1797. Empty DB messages + Settings guidance in FINISH / INSTALL.md. Location empty does not disable RUN (gap noted in INSTALL). |
| H3 provenance | PRESENT + gaps | PDF: git_hash+git_dirty_code (p71). PSF LC CSV: git_hash+git_dirty. AAVSO/VarAstro: version SOFTWARE line only (now 0.10.0a1 via vyvar_version). Aperture LC CSV: no version/git headers. Full PROV-FIX on all products: MISSING-needs-Milan (aperture byte lock). See h3_provenance_check.txt. |
| H4 version | FIXED | Single source: src_py/vyvar_version.py __version__=0.10.0a1; pyproject.toml [project].version; export_reports imports VYVAR_SOFTWARE_VERSION. |
| H5 CHANGELOG | FIXED | [0.10.0a1] user-facing entry since preview-VYVAR.0.9.0. |
| H6 ALPHA_TESTING | FIXED | docs/ALPHA_TESTING.md EN + short CZ; debt list plain language. |
| H7 catalogs | PRESENT | Repo GAIA_DR3 ~64 GB full local; recommended install set ~12.1 GB (INSTALL.md). VSX ~0.87 GB. Testers need catalogs (else LIMITED MODE). Attribution: CITATIONS.bib gaia2023 + VSX Watson/Henden/Price; INSTALL names ESA Gaia / AAVSO VSX. |
| H8 LICENSE | PRESENT | Proprietary unchanged. Alpha use/redistribution: none without prior written permission (LICENSE). JOSS needs OSI-approved licence - Milan decides. |

## Step 4 - STOP for Milan (PUSH_AUTH)

Proposed version / tag: 0.10.0a1 / VYVAR-0.10.0a1

Exact commands (DO NOT RUN from this agent):

  git checkout main
  git pull origin main
  git merge --ff-only consolidate-01
  # if ff-only fails, use merge commit as in the previous merge:
  # git merge --no-ff consolidate-01 -m "Merge consolidate-01: alpha 0.10.0a1 LOCK"
  git tag -a VYVAR-0.10.0a1 -m "VYVAR 0.10.0a1 alpha"
  git push origin main
  git push origin VYVAR-0.10.0a1

## Output / findings
- Aperture bytes did not move (87197716 / dd92e99d gates PASS).
- full-epsf-stage 15551 s; full-pipeline 1701 s.
- Artifacts: full_epsf.log, fast_clean.log, h1_summary.txt,
  h1_install_retry.log, h1_streamlit_*.log, h3_provenance_check.txt.

## Errors (if any)
- First --fast failed on test_aavso_software after version bump; tests
  updated to follow VYVAR_SOFTWARE_VERSION; re-run PASS.
- NonInteractive first install FAIL before smoke table fix (FIXED).

## Files changed
Commits: (1) LOCK 91194ea; (2) docs/version 4ad612a.
