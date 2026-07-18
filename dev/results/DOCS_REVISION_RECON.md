DOCS-REVISION-RECON -- full documentation audit before the rewrite (ANALYSIS ONLY)

Date: 2026-07-18
Audited HEAD: 23b8b2e ("docs(result): record --full PASS in ledger + WAVE-B result file"),
post-push, main == origin/main.
Scope: analysis only. The only change is this document. ASCII-only English.

Method: git log (last SUBSTANTIVE commit, not mtime) + targeted greps of the live code for
the top claims of each doc (pre-reorg paths, deleted WAVE-B parameters, 304-count, stale
anchor/test numbers, dead cross-links). Where a claim is contradicted the code location is
cited.

Baseline facts used to judge staleness (current as of HEAD 23b8b2e):
- Repo layout is POST-REORG (2026-07-17): production modules in src_py/ (flat imports),
  dev material under dev/ (dev/tests, dev/scripts, dev/tools, dev/validation, dev/results,
  dev/orchestrator). Root keeps config.json, pyproject.toml, requirements.txt, app.py shim,
  README.md, CLAUDE.md, CHANGELOG.md, CITATIONS.bib.
- Parameter surface is POST-WAVE-B + CONFIG-HUMAN-EDIT: 269 registered params, 249 persisted
  in config.json; config.json is now a grouped, commented JSONC-lite file.
- Current anchor: draft_435 sky-surface (core/photometry SHA 3d26f469, extended 6420f1da).
- Current test suite: 963 passed, 19 skipped.
- No LICENSE file exists in the repo (see STEP 2).

================================================================================
STEP 1 -- INVENTORY & STALENESS MATRIX (docs/ 31 + root 3 = 34 files)
================================================================================

Legend for disposition: OK | UPDATE(what) | REWRITE | MERGE-INTO:<doc> | ARCHIVE(->dev/results)
Audience: USER (an observer running VYVAR) | DEV (maintainer) | BOTH.
Translate?: proposed YES only for USER-audience docs (taxonomy is Milan's call, STEP 4).

--- ROOT FILES ---

### README.md  (3.0 KB; last substantive 2026-07-14 `0f1c941`)  Audience: BOTH  Translate: YES
STALENESS EVIDENCE:
- "python -m pytest tests/ -v" (lines 79-80) -- PRE-REORG path; tests now live in dev/tests
  and pytest is configured via pyproject testpaths=["dev/tests"]. Contradicted.
- "Full suite (2026-07-14): 852 passed, 15 skipped" (line 83) -- now 963 passed / 19 skipped.
- "## License / See LICENSE for details" (lines 85-87) -- there is NO LICENSE file (STEP 2).
  Contradicted.
- "Python 3.10+" (line 32) vs ruff target-version = py312 and dev on 3.12 (pyproject).
  Understated/loose; slots dataclasses + PEP 604 unions are used (3.10 OK but untested).
- Catalog build instructions (lines 43-58) say builders "import gaia_catalog_id.py from the
  clone root" -- FALSE post-reorg: gaia_catalog_id.py is in src_py/ and build_gaia_catalog.py
  cannot find it (see STEP 3.2, real breakage).
- Non-ASCII throughout (em-dash, arrows, Delta, Neuhauser) -- fine for GitHub, but the repo's
  hard rule is ASCII; README should be brought in line or explicitly exempted.
DISPOSITION: REWRITE (it is the GitHub front door; every install/run/test line is wrong or
outdated). Fold in the new docs index, install pointer, license note, current numbers.

### CLAUDE.md  (2.3 KB; 2026-07-17 `55fef5e`)  Audience: DEV  Translate: NO
STALENESS EVIDENCE:
- "plain-language guide to all 304 config.json parameters" (line 10) -- now 269 registered /
  249 persisted (WAVE-B + CONFIG-HUMAN-EDIT). Contradicted.
- "Root keeps only: ... CLAUDE.md, CHANGELOG.md, CITATIONS.bib, and the app.py shim" (line 27)
  omits README.md, which is tracked at root. Minor inconsistency.
- Layout section itself is accurate (post-reorg) and fresh.
DISPOSITION: UPDATE (fix the 304 count; add README to the root list).

### CHANGELOG.md  (5.2 KB; 2026-07-15 `e210d90`)  Audience: BOTH  Translate: NO (EN-only)
STALENESS EVIDENCE:
- [Unreleased] stops at 2026-07-16 (VALIDATE-429). MISSING the three most recent arcs:
  REPO-REORG (2026-07-17), WAVE-B-PARAM-REDUCTION and CONFIG-HUMAN-EDIT (2026-07-18).
- "tests/test_photometry_core.py" (line 38) -- pre-reorg path.
- Non-ASCII (em-dash, arrow, times sign).
DISPOSITION: UPDATE (append REORG/WAVE-B/CONFIG-HUMAN-EDIT; fix path; consider first tagged
release now that a GitHub README ships).

--- docs/ FRESH (updated in the current stack; spot-checks pass) ---

### VYVAR_STATE.md  (42.5 KB; 2026-07-18 `2cca6e8`)  Audience: DEV  Translate: NO
- Mostly fresh (header + WAVE-B + CONFIG-HUMAN-EDIT stamps present). One stale line:
  "Tests: 852 passed / 15 skipped ... 2026-07-14 close" (line 631) -- now 963/19.
DISPOSITION: UPDATE (test count line only).

### VYVAR_JOURNAL.md (323 KB; 2026-07-18 `2cca6e8`)  Audience: DEV  Translate: NO
- Append-only historical log; current entries fresh. Contains pre-reorg paths only inside old
  dated entries (e.g. line 4504 `pytest tests/test_photometry_core.py`) -- historically
  correct, not a live instruction. DISPOSITION: OK (do not rewrite history).

### VYVAR_PROCESS.md (15.2 KB; 2026-07-18 `2cca6e8`)  Audience: DEV  Translate: NO
- Uses correct dev/scripts path; CONFIG-HUMAN-EDIT note present. DISPOSITION: OK.

### VYVAR_DECISIONS.md (97.5 KB; 2026-07-18 `df93984`)  Audience: DEV  Translate: NO
- WAVE-B decision entry present and correct. Large but current. DISPOSITION: OK.

### VYVAR_CONFIG_GUIDE_EN.md (66.5 KB; 2026-07-18 `0b75a69`)  Audience: USER  Translate: has CZ
- Post-WAVE-B, "Editing without the UI" section present, per-key table is the source of the
  registry help. DISPOSITION: OK (this is the reference the USER_GUIDE will point at).

### VYVAR_CONFIG_GUIDE_CZ.md (67.4 KB; 2026-07-18 `0b75a69`)  Audience: USER (CZ)
- CZ ASCII mirror of EN, fresh. DISPOSITION: OK.

### VYVAR_PARAMS.md (33.2 KB; 2026-07-18 `1307b73`)  Audience: DEV/BOTH  Translate: NO
- GENERATED from params_registry.json + AppConfig; freshness is a tested property. Current.
  DISPOSITION: OK (generated; never hand-edit).

### VYVAR_ROADMAP.md (64.3 KB; 2026-07-16 `ded815b`)  Audience: DEV  Translate: NO
- Mostly current. STALE: line 497 "scripts/session_baseline_check.py ... --full re-verifies
  draft_424" -- path is pre-reorg (now dev/scripts) and anchor is draft_435.
DISPOSITION: UPDATE (path + anchor name).

--- docs/ STALE (pre-reorg paths / deleted params / dead links) ---

### VYVAR_RUNBOOK.md (3.8 KB; content 2026-07-17 was a chore move, body pre-reorg) Audience: BOTH Translate: maybe
STALENESS EVIDENCE:
- "python scripts/session_baseline_check.py" (lines 12-13) -- pre-reorg path (now dev/scripts).
- --full table (line 19) describes draft_424 anchor and SHA "92939fab / 76642318" -- current
  anchor is draft_435 (SHA core 3d26f469 / extended 6420f1da).
- Timings "--fast ~3 min / --full ~25 min" understate the current 963-test suite + ~40 min full
  run observed today (2311 s pipeline alone).
DISPOSITION: UPDATE (paths, anchor, SHAs, timings). Coverage is thin vs the real night
workflow -- STEP 2 proposes a separate USER quickstart; keep this as the DEV/operator gate
runbook.

### VYVAR_CLAUDE_OPERATING_PRINCIPLES.md (5.3 KB; 2026-07-08 `00dd0cd`) Audience: DEV Translate: NO
- "python scripts/session_baseline_check.py" (line 89) -- pre-reorg path.
DISPOSITION: UPDATE (path). Otherwise normative and current.

### VYVAR_VALIDATION.md (8.5 KB; 2026-06-09 `18b3b18`)  Audience: DEV  Translate: NO
- "tests/validation/" (line 3) and "python -m tests.validation.recover --all" (line 19) --
  pre-reorg; the harness is now under dev/tests/validation/. Contradicted.
DISPOSITION: UPDATE (module paths + run command).

### VYVAR_CODE_MAP.md (18.1 KB; 2026-06-08 `10b81fa`)  Audience: DEV  Translate: NO
STALENESS EVIDENCE:
- Dated 2026-06-08, PRE-REORG and PRE-WAVE-B. Cross-refs use old paths: "scripts/archive/...",
  "scripts/chiandh_*", "orchestrator/vyvar_orchestrator.py" (now under dev/).
- Claims to be a "living document ... update on module add/delete" but is 6 weeks stale; module
  set has changed since (e.g. WAVE-B hardcode moves, config writer additions).
- Written in mixed CZ/SK, not clean EN.
DISPOSITION: REWRITE (regenerate from current src_py docstrings+imports) -- high value for the
future maintainer (feeds SUCCESSION.md). Alternatively ARCHIVE if a generated map replaces it.

### docs/config_schema.md (19.8 KB; 2026-07-13 `0608739`)  Audience: DEV/BOTH  Translate: NO
STALENESS EVIDENCE (most contradicted doc in the set):
- Lists DELETED WAVE-B parameters as if live: comp_tier1..4_bprp_limit, comp_tier1..4_weight
  (lines 91-102), aperture_fwhm_factor_small/medium/large (lines 142-144). All merged/removed
  in WAVE-B (now aperture_snr_sizing, comp_color_tiers, phase01_tiers).
- Lists DB-dup keys removed from config.json: gain (line 22), read_noise (line 23); and the
  hardcoded sky_adu_fallback (line 27).
- Stale defaults: masterdark_validity_days=80 (code default 90), masterflat_validity_days=524
  (code default 200).
- Fully redundant with the GENERATED VYVAR_PARAMS.md and the hand-authored CONFIG_GUIDE.
DISPOSITION: ARCHIVE(->dev/results) or MERGE-INTO:VYVAR_PARAMS.md -- it duplicates two
better-maintained sources and is the worst-drifted file. Recommend ARCHIVE.

### VYVAR_CALIBRATION.md (12.4 KB; 2026-06-22 `07e6f69`)  Audience: BOTH (science, CZ ASCII)  Translate: has CZ
STALENESS EVIDENCE:
- DEAD cross-link: references "VYVAR_FULL_AUDIT_LEDGER.md" (line 9) which does not exist in the
  repo (only this doc references it).
- Anchored to commit be3e193 (pre-reorg / pre-WAVE-B); magnitude-flow formulas likely still
  valid but parameter names/gates should be re-checked against current config.
DISPOSITION: UPDATE (remove/replace the dead link; re-anchor to current HEAD; verify gates).

### VYVAR_PIPELINE_CZ.md (12.3 KB; 2026-06-09..07-08 `0913665`)  Audience: BOTH (article manual, CZ) Translate: EN counterpart
STALENESS EVIDENCE:
- Anchored to baseline commit 28fdafa + session 2026-06-09 (pre-reorg / pre-WAVE-B); parameter
  names and numbers predate WAVE-B (e.g. tier scalars).
DISPOSITION: UPDATE (re-anchor values to current code) -- this is the CZ manual "pro clanek";
its EN counterpart is effectively the paper, not a repo doc.

--- docs/ DESIGN / SPEC RECORDS (DEV, EN, mostly closed arcs) ---

The following are design/spec records tied to specific arcs. They are DEV-audience, EN-only,
and generally historical (normative content has since landed in code + DECISIONS/STATE). None
are USER-facing. Proposed disposition: keep the RECENT/active ones in docs/; ARCHIVE the closed
older ones to dev/results so docs/ stays a lean, current surface. Milan decides per item.

RECENT (2026-07, likely still referenced) -- propose OK / light UPDATE:
- K2_BAND_AWARE_SPEC.md (07-14), VYVAR_K2_DESIGN_SPEC.md (07-08)
- VYVAR_CAL_DIAG_SPEC.md (07-14)
- VYVAR_SPARSE_TRUST_SPEC.md (07-14), VYVAR_SIGMA_FLOOR_SPEC.md (07-14),
  VYVAR_SIGMA_BUDGET_SPEC.md (07-13)
- VYVAR_WIDE_SLOPE_NOISE_SPEC.md (07-14, parked verdict)

OLDER (2026-06, closed design records) -- propose ARCHIVE(->dev/results):
- VYVAR_COMP_DEGRADATION_SPEC.md (06-16), VYVAR_NEIGHBOR_SUB_DESIGN.md (06-22),
  VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md (06-11), VYVAR_SIMPLE_DIFFERENTIAL_SPEC.md (06-19),
  VYVAR_CANONICAL_COMBINATION_LOGIC.md (06-16), VYVAR_CHECKSTAR_SELECTION_SPEC.md (06-11),
  VYVAR_COMP_FLOOR_POLICY_SPEC.md (06-11), VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md (06-11)

NORMATIVE PROCESS (keep in docs/):
- VYVAR_DECISION_GROUNDING_RULE.md (06-16) -- short reasoning rule; OK (keep).

STALENESS NOTE for specs: spot-check did not find pre-reorg run commands inside these (they are
prose design docs), but any code-path references should be re-checked at rewrite time. They are
NOT the priority of this revision (user-facing + install docs are).

SUMMARY COUNTS
- Fresh/OK: ~10 (STATE, JOURNAL, PROCESS, DECISIONS, CONFIG_GUIDE_EN/CZ, PARAMS, plus recent
  specs).
- UPDATE: README(->REWRITE), CLAUDE, CHANGELOG, ROADMAP, RUNBOOK, CLAUDE_OPERATING_PRINCIPLES,
  VYVAR_VALIDATION, VYVAR_CALIBRATION, VYVAR_PIPELINE_CZ, STATE(minor).
- REWRITE: README.md, VYVAR_CODE_MAP.md.
- ARCHIVE/MERGE: config_schema.md (worst drift) + ~8 older June spec records.

================================================================================
STEP 2 -- GAP LIST (docs that do NOT exist yet) + outlines
================================================================================

Confirmed ABSENT (git ls-files): INSTALL*, USER_GUIDE*/QUICKSTART*, SUCCESSION*, LICENSE.
README.md exists but is stale (STEP 1) -> treat as REWRITE, outline below.

LICENSE CHECK: NO license file of any kind (LICENSE / LICENCE / COPYING) is tracked. README
"See LICENSE" is a dangling promise. Milan must choose a license before the repo is shared
(decision in STEP 4). Without a license the default is "all rights reserved" -- strangers
technically may not reuse it.

1) INSTALL manual (EN + CZ)  -- audience USER
   - Prerequisites: Windows 10/11 (primary) or Ubuntu 22.04+; Python 3.12 (match dev/ruff);
     ~15 GB free for the zaloha catalog set (or ~55 GB for the full Gaia build); 16 GB RAM.
   - Get the code: git clone; create venv; pip install -r requirements.txt (after the
     requirements fix in STEP 3.1).
   - Catalog bootstrap: two paths -- (a) COPY the prebuilt catalog set from a known-good
     machine (recommended for the Lenovo; exact file list + sizes in STEP 3.2), or (b) BUILD
     with GAIA_DR3/VSX/exoplanet scripts (hours-to-days; only after the STEP 3.2 fix).
   - Configure paths: point config.json (or the install script) at the copied catalog/DB/
     archive locations (currently hardcoded to C:\ASTRO -- STEP 3.3).
   - First-run check: streamlit run app.py -> Settings -> create Location/Telescope/Equipment
     rows -> run validate_config.py -> import one night of FITS.
   - Troubleshooting: ImportError (missing dep), catalog-not-found, empty reference tables,
     wrong absolute paths, sep/pyraf optional-skip messages.

2) USER_GUIDE / quickstart (EN + CZ)  -- audience USER
   - "I have a night of FITS" -> "AAVSO submitted", UI-first.
   - Steps: Session upload; equipment/site selection; calibration (masters from
     CalibrationLibrary); RUN VYVAR (calibration -> plate solve -> MASTERSTAR -> photometry ->
     trust); read the PDF Summary Report; interpret trust GREEN/YELLOW/RED; AAVSO / VarAstro
     export.
   - Reference pointers: config knobs -> VYVAR_CONFIG_GUIDE_EN/CZ; gate/anchor discipline ->
     RUNBOOK; concepts -> (rewritten) README.
   - One or two annotated screenshots (placeholder now).

3) README.md for GitHub (REWRITE)  -- audience BOTH
   - What VYVAR is (one paragraph) + capabilities bullets (keep, de-jargon).
   - Screenshot placeholder (UI + a sample PDF page).
   - Install pointer (-> INSTALL manual), quickstart pointer (-> USER_GUIDE).
   - Docs index table (STATE/PROCESS/CONFIG_GUIDE/RUNBOOK/PARAMS + specs).
   - Status/CI badge placeholder; test count as a single generated line (avoid hardcoding).
   - License note (real, once chosen); citation block (keep); validation table (keep, verify).

4) SUCCESSION.md (EN)  -- audience DEV (future maintainer letter)
   - Where things live (post-reorg map; point at rewritten CODE_MAP).
   - The never-do list (never hand-edit generated docs/config; never write config.json outside
     the UI persist context; never push without the gate; ASCII-only; byte-identical anchor
     rule).
   - How the AI-agent workflow runs: session ritual (read STATE/ROADMAP/JOURNAL/PROCESS/
     OPERATING_PRINCIPLES), the CURSOR_TASK/CURSOR_RESULT pattern, the orchestrator, the
     --fast/--full gate discipline and the anchor cut protocol.
   - Keys to the kingdom: catalog rebuild vs copy, the validation ledger, the anchor snapshots.

================================================================================
STEP 3 -- INSTALLATION FEASIBILITY RECON
================================================================================

### 3.1 Dependency reality
requirements.txt (15 lines, UNPINNED, no versions): streamlit, pandas, numpy, psutil, scipy,
astropy, photutils, pillow, plotly, lightkurve, astroquery, astroalign, reproject, pytest,
reportlab.

Imports scanned across src_py vs requirements:
- DIRECTLY imported but NOT explicitly listed (present only as TRANSITIVE deps today):
  * matplotlib  -- used for all PNG/PDF plots (photometry_core.py, photometry_report.py,
    hrd_analysis.py); several call sites are UNGUARDED (e.g. photometry_report.py:2217,
    hrd_analysis.py:1122). Currently satisfied transitively via lightkurve, but must be an
    explicit dependency.
  * scikit-image (skimage) -- used in the alignment path
    (vyvar_alignment_frame.py:621 phase_cross_correlation). Satisfied transitively via
    astroalign; should be explicit.
- OPTIONAL, correctly GUARDED (try/except; not needed for the core pipeline):
  * pyarrow (pipeline.py:5716 parquet fast-path), sep (xval_run.py:114 cross-val),
    pyraf (validate_lc_crossval.py:31 IRAF cross-val), cupy (pipeline.py GPU path).
- No version pins anywhere: reproducibility risk (a future numpy/astropy/photutils major
  could break byte-identity or import). Python version is not constrained in requirements
  (pyproject ruff target is py312; dev runs 3.12).
VERDICT: `pip install -r requirements.txt` will PROBABLY work today (matplotlib/scikit-image
arrive transitively) but is fragile. Fix: add matplotlib + scikit-image explicitly, add an
[optional] extras group for pyarrow/sep, pin at least the science-critical trio
(numpy/astropy/photutils) to the versions the anchor was cut with.

### 3.2 Catalog bootstrap
Builders PRESENT in the repo:
- GAIA_DR3/build_gaia_catalog.py  -- Gaia DR3 -> local SQLite via astroquery TAP (anonymous or
  GAIA_USER/GAIA_PASS). Source: ESA Gaia archive TAP. Build cost: HUGE (the full G<=16.5 DB is
  50.7 GB; a full download/build is hours-to-days on a home link).
  BROKEN POST-REORG: _find_vyvar_root (line 41-55) walks up from GAIA_DR3/ looking for
  gaia_catalog_id.py at the repo root, but that module moved to src_py/. It will raise
  "needs gaia_catalog_id.py from the VYVAR repo". README's build instructions inherit this bug.
  Fix is one line (search src_py/ too, or insert src_py on sys.path).
- GAIA_DR3/build_blind_index.py -- builds the triangle PKLs from the Gaia DB (SQL only; no repo
  module import). Works standalone given the DB.
- VSX/vsx_make.py, exoplanets/exoplanet_make.py -- standalone builders (no repo-module import).
  VSX source: AAVSO VSX; exoplanet source: NASA/exoplanet archive.

Existing catalog artifacts and sizes (on this machine):
  GAIA_DR3/vyvar_gaia_dr3.db ............ 50.7 GB   (full, G<=16.5)
  GAIA_DR3/gaia_triangles_fine.pkl ...... 2.1 GB
  GAIA_DR3/gaia_triangles_wide.pkl ...... 1.1 GB
  GAIA_DR3/zaloha/vyvar_gaia_dr3.db ..... 9.6 GB    (backup subset, G<=16; the reproducible
                                                     anchor catalog per RUNBOOK)
  GAIA_DR3/zaloha/gaia_triangles_fine.pkl 1.3 GB
  GAIA_DR3/zaloha/gaia_triangles_wide.pkl 0.66 GB
  VSX/vyvar_vsx_local_v2.db ............. 866 MB
  exoplanets/vyvar_exoplanet_local.db ... 2.2 MB

PRAGMATIC LENOVO ANSWER: COPY-FROM-EXISTING is far more viable than rebuilding. Copy the
"zaloha" set (the anchor catalog) rather than the 50 GB full DB:
  - GAIA_DR3/zaloha/vyvar_gaia_dr3.db        (9.6 GB)
  - GAIA_DR3/zaloha/gaia_triangles_fine.pkl  (1.3 GB)
  - GAIA_DR3/zaloha/gaia_triangles_wide.pkl  (0.66 GB)
  - VSX/vyvar_vsx_local_v2.db                (866 MB)
  - exoplanets/vyvar_exoplanet_local.db      (2.2 MB)
  Total ~= 12.5 GB (fits on a USB stick). Then point config.json at the zaloha DB + PKLs.
  (The full 50 GB DB is only needed for fields fainter than the zaloha G<=16 cut.)

### 3.3 First-run surface on a fresh machine
- Database: SELF-INITIALISES. database.py creates every table with CREATE TABLE IF NOT EXISTS
  (EQUIPMENTS/TELESCOPE/LOCATION/OBSERVATION/OBS_DRAFT/... at lines 893-968+). sqlite3.connect
  creates vyvar.sqlite3 on first use. So NO crash from a missing DB -- but the reference tables
  are EMPTY: the user MUST create a Location, Telescope and Equipment row (Settings) before a
  run resolves site/optics. Document this as first-run step 1.
- Archive / CalibrationLibrary: the loader defaults are project_root-relative
  (archive_root = project_root/Archive, calibration_library_root = project_root/
  CalibrationLibrary, database_path = project_root/vyvar.sqlite3), so the MAIN app self-locates
  when config.json omits them. They are created/used lazily.
- ABSOLUTE-PATH LANDMINES (the biggest fresh-machine risk):
  * config.json (tracked) PINS Milan's machine paths in the paths section:
      archive_root, calibration_library_root, database_path, gaia_db_path, vsx_local_db_path,
      blind_index_fine_path, blind_index_wide_path -> all "C:\\ASTRO\\python\\VYVAR\\...".
    On the Lenovo (any other clone path) these point at a nonexistent tree. The install script
    (or a first-run wizard) MUST rewrite or blank these so the relative defaults / copied
    catalog paths take over.
  * Hardcoded C:\ASTRO in auxiliary DEV runner scripts (NOT the main app path):
      src_py/psf_runner.py:22, src_py/run_smoothness_report.py:12,
      src_py/run_crowding_index.py:20, src_py/inspect_drafts.py:159, and a UI PLACEHOLDER
      string in src_py/ui_photometry_quality.py:79. These break only if a stranger runs those
      specific dev tools; the production app (app.py -> AppConfig) is relocatable.

### 3.4 Windows specifics
- Core stack has Windows/py3.12 wheels: numpy, scipy, astropy, photutils, pandas, streamlit,
  plotly, pillow, reportlab, astroquery, reproject, matplotlib, scikit-image, lightkurve.
  No compiler needed for these.
- astroalign: pure-Python but pulls scikit-image (+ historically sep). scikit-image ships
  wheels; fine.
- RISK deps (all OPTIONAL / guarded, so not required for a basic install):
  * sep -- original `sep` has had Windows/newer-Python build gaps; `sep-pjw` fork provides
    3.12 Windows wheels. Only needed for the xval harness.
  * pyraf -- IRAF Python; effectively unavailable on Windows. Only the LC cross-val uses it
    and it is guarded. Leave it OUT of the default install.
  * cupy -- needs CUDA; GPU-only path, guarded. Leave OUT of default install.
VERDICT: a fresh Windows/py3.12 machine can run the FULL production pipeline with only the
(fixed) requirements.txt; no compilers or IRAF needed.

================================================================================
STEP 4 -- PROPOSED REVISION PLAN FOR MILAN
================================================================================

Ordered work packages (effort S<=1h, M=half-day, L=1-2 days):

WP1 (S) requirements.txt fix + pin -- add matplotlib, scikit-image; optional-extras note for
   pyarrow/sep; pin numpy/astropy/photutils to anchor versions. Unblocks every install.
WP2 (S) Fix build_gaia_catalog.py root detection (src_py aware). Unblocks catalog build path.
WP3 (M) README.md REWRITE (front door): what/capabilities/screenshot placeholder/install+
   quickstart pointers/docs index/license note/current numbers. EN (+ CZ optional).
WP4 (L) INSTALL manual EN + CZ (STEP 2.1 outline) incl. the copy-vs-build catalog decision and
   the config.json path-rewrite step. Pair with a first-run checklist.
WP5 (M) install script (see arc below) that WP4 documents.
WP6 (L) USER_GUIDE/quickstart EN + CZ (STEP 2.2). Depends on WP3 wording.
WP7 (S) UPDATE stale DEV docs: RUNBOOK, ROADMAP, CLAUDE_OPERATING_PRINCIPLES, VYVAR_VALIDATION,
   CLAUDE.md, CHANGELOG, STATE test-count (pure path/number fixes).
WP8 (M) VYVAR_CODE_MAP.md REWRITE (regenerate from src_py) -> feeds WP9.
WP9 (M) SUCCESSION.md (STEP 2.4).
WP10 (S) ARCHIVE config_schema.md + the older June spec records to dev/results; fix the dead
   VYVAR_FULL_AUDIT_LEDGER link in VYVAR_CALIBRATION.md.
WP11 (S) Add a LICENSE file once Milan chooses (blocks nothing technical, blocks sharing).

Suggested order: WP1, WP2 (unblock install) -> WP7, WP10 (cheap correctness) -> WP3 -> WP4,
WP5 -> WP6 -> WP8, WP9 -> WP11 (whenever license is chosen).

TRANSLATION set (if taxonomy = "user docs bilingual, dev docs EN-only"):
  Bilingual (EN+CZ): README, INSTALL, USER_GUIDE, CONFIG_GUIDE (already both), PIPELINE
  (already CZ; EN = the paper). EN-only: all DEV docs (STATE/PROCESS/DECISIONS/ROADMAP/
  JOURNAL/PARAMS/RUNBOOK/CODE_MAP/SUCCESSION/specs/CLAUDE*).

INSTALL-SCRIPT ARC (outline for the follow-on task):
  Script steps (Windows PowerShell + a Linux bash twin):
   1. check Python 3.12 present; create/activate venv.
   2. pip install -r requirements.txt (fixed).
   3. prompt for catalog location; either verify a copied catalog set or offer to build
      (calls the fixed builders).
   4. write a machine-local config.json paths block (archive/calibration/db/gaia/vsx/blind)
      from the chosen locations -- do NOT ship Milan's absolute paths.
   5. run validate_config.py; run a tiny smoke (build_gaia_catalog small patch OR just import
      app) to confirm the environment.
   6. print next steps (Settings -> create Location/Telescope/Equipment; import a night).
  Manual chapters mirror the script: Prereqs / Get code / Dependencies / Catalog / Configure /
  First run / Troubleshooting.

TOP 5 RISKS for the Lenovo stranger test:
  1. config.json ships absolute C:\ASTRO paths -> app points at a nonexistent tree until the
     paths block is rewritten (STEP 3.3). HIGH.
  2. Catalog volume: ~12.5 GB (zaloha) to copy, or a multi-hour/50 GB build; plus the broken
     Gaia builder (WP2) if they try to build. HIGH.
  3. requirements.txt relies on transitive matplotlib/scikit-image and is unpinned -> an env
     that resolves differently can ImportError or drift (WP1). MEDIUM.
  4. Empty reference tables on first run -> confusing "no site/optics" failures until the user
     creates Location/Telescope/Equipment (needs the USER_GUIDE step). MEDIUM.
  5. No LICENSE + a README that promises one -> legal/clarity blocker for sharing (WP11). LOW
     technically, but blocks public release.

OPEN DECISIONS FOR MILAN (this recon does not decide them):
  A. Translation taxonomy: confirm "user docs bilingual (EN+CZ), dev docs EN-only" (the matrix
     "Translate?" column assumes this).
  B. LICENSE choice (MIT / BSD-3 / GPL-3 / proprietary) -- required before sharing.
  C. Distribution / visibility: public vs private repo, and the "compiled-library"
     (Cython/binary) idea -- both are Milan's separate calls; referenced here only because they
     shape the INSTALL manual and the SUCCESSION notes.
  D. config_schema.md fate: ARCHIVE (recommended) vs keep-and-regenerate.
  E. Spec-record archiving: move the closed June specs to dev/results, or keep them in docs/.

================================================================================
Deliverable: this file only. Full pytest untouched (no code changed). NOT pushed.
