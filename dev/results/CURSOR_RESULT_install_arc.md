CURSOR RESULT - INSTALL-ARC - 2026-07-18

What I did
Implemented the installer, the EN install manual, and the first-run hardening so
the (parallel) VYVAR_INSTALL_GUIDE_CZ.pdf flow is true on a fresh machine.
Baseline: origin/main 7e88c3b. HOLD push for the Lenovo stranger test.

================================================================================
STEP 1 - install_vyvar.ps1 (+ install_vyvar.sh twin)
================================================================================

install_vyvar.ps1 (repo root, Windows PowerShell). Interactive, idempotent,
each phase prints [OK]/[WARN]/[FAIL]. Phases:

1 PYTHON   - detect `py -3.12`, then `python --version` (require 3.12). If
             missing: print winget command + python.org link, exit 1.
2 VENV     - reuse healthy .venv or create it; `pip install --upgrade pip`;
             `pip install -r requirements.txt`; `pip check` (WARN, not FAIL).
             Uses the venv python directly (no activation-policy dependency).
3 CATALOGS - menu:
             [1] Copy from an existing VYVAR install (default): prompt source
                 root + target data root; resolve each manifest file (zaloha
                 preferred); report ~total size upfront; free-space check on the
                 target drive; copy; per-file size verification after copy;
                 already-present + size-matching files are skipped (idempotent).
             [2] Build from sources: prints the (fixed) builder commands with a
                 volume/time warning. Not the Lenovo path.
             [3] Skip: LIMITED MODE warning (no Gaia matching until catalogs
                 arrive); installer still completes.
4 PATHS    - prompt archive root / calibration-library root / database path
             (catalog paths prefilled from phase 3); call
             dev/scripts/apply_install_config.py to write config.json through the
             canonical comment-preserving writer. Author C:\ASTRO paths are
             never kept (blanked -> project-root defaults). Location/telescope/
             camera facts are NOT asked here (they belong to the app UI).
5 VALIDATE - `python dev/scripts/validate_config.py <config>`; must exit 0.
6 SMOKE    - import `app` through the src_py bootstrap (no server) + construct
             VyvarDatabase(cfg.database_path) and assert the reference tables
             self-initialise. Fails the phase on any traceback.
7 FINISH   - print verbatim next steps: `streamlit run app.py`, then Settings ->
             create Location/Telescope/Equipment; import a first night. Points to
             VYVAR_INSTALL_GUIDE_CZ.pdf, docs/VYVAR_CONFIG_GUIDE_*.md, INSTALL.md.

Parameters for CI/repeat: -NonInteractive, -CatalogSource <path>, -DataRoot
<path>. Syntax verified with the PowerShell AST parser (no parse errors).

install_vyvar.sh (repo root, Linux bash twin): same seven phases, best-effort.
Env knobs: NONINTERACTIVE=1, CATALOG_SOURCE=..., DATA_ROOT=.... Verified with
`bash -n` (syntax OK).

Helper: dev/scripts/apply_install_config.py - writes only the file/catalog path
keys, sanitises any author C:\ASTRO absolute path that is not explicitly chosen
(blanks it so the project-root default resolves), and writes via the canonical
render_config_jsonc (comment-preserving). Unit-tested (test_apply_install_config).

================================================================================
Catalog manifest as implemented (verified on the author machine 2026-07-18)
================================================================================

Recommended copy = the "zaloha" (G<=16 subset) anchor catalog, ~12.1 GB total:

  File (destination)                    Purpose                              Bytes
  GAIA_DR3/vyvar_gaia_dr3.db            Gaia DR3 catalog (G<=16 subset) 10,066,063,360
  GAIA_DR3/gaia_triangles_fine.pkl      Blind-solve index (narrow)       1,406,291,841
  GAIA_DR3/gaia_triangles_wide.pkl      Blind-solve index (wide)           706,509,280
  VSX/vyvar_vsx_local_v2.db             AAVSO VSX known variables          908,324,864
  exoplanets/vyvar_exoplanet_local.db   NASA exoplanet-host cross-match      2,334,720
  Total                                                                  ~12.09 GB

Source resolution tries GAIA_DR3/zaloha/<file> first, then GAIA_DR3/<file>, so a
source machine that only has the full 50 GB DB still works (it copies that).
Verification after copy is exact source-vs-destination byte-size equality.

================================================================================
STEP 3 - first-run hardening (fixes list)
================================================================================

Traced the real fresh-machine startup (temp project root, blanked paths, no DB).

FIX 1 (src_py/config.py, AppConfig.__post_init__): a blank ("" / whitespace)
  archive_root / calibration_library_root / database_path was treated as an
  explicit value, so Path("") resolved to "." . database_path="." would make
  sqlite3.connect fail on first run (crash-class). Now a blank value means "use
  the project-root default", identical to omitting the key. This is exactly what
  the installer and hand-editors produce when dropping the author's absolute
  paths. Minimal 3-key change; non-blank values are unchanged (anchor config has
  real absolute paths -> byte-identical, confirmed by --full below).

No other first-run crash found: VyvarDatabase self-initialises the sqlite file +
schema on construction; get_observer_location_by_id returns None for a missing id
(the shipped observer_location_id=2 does not exist in the seed - handled
gracefully); importing `app` runs no top-level Streamlit code (main() is guarded).

Guard tests added:
  dev/tests/test_fresh_machine_startup.py - 4 tests: AppConfig resolves local
    roots when paths blanked; blanked catalog keys stay empty (LIMITED MODE);
    DB self-initialises on a fresh file; location hydration is graceful for a
    missing id.
  dev/tests/test_apply_install_config.py - 3 tests: author paths blanked when not
    chosen; chosen value wins; end-to-end write validates and drops author paths.

================================================================================
Guide deviations (for Claude's PDF update BEFORE the Lenovo test)
================================================================================

D1. "Empty reference tables on first run" (recon 3.3) is INACCURATE. database.py
    initialize_database() SEEDS the author's own observatory on every fresh DB:
    EQUIPMENTS (QHY294MM id1, C5A-150M id4), TELESCOPE (Carl-Zeiss id1, AZ800
    id6), LOCATION (Dablice id1), SCANNING (id1). This seed is anchor-critical
    (draft_435 depends on these ids) and was deliberately NOT removed. Guide/first
    -run wording must say: the app ships with the author's EXAMPLE rows; the user
    creates and selects THEIR OWN Location/Telescope/Equipment and must not submit
    under the seeded "Dablice"/author rows. (Reflected in INSTALL.md and the ps1
    FINISH text.)

D2. Shipped config.json has observer_location_id=2, but the seed only creates
    LOCATION id 1. On a fresh DB id 2 does not exist; hydration returns None
    (handled, no crash). The user selecting their own Location in Settings fixes
    it. Installer does NOT touch this (location facts belong to the UI). Guide
    should mention selecting the Location in Settings resolves the "no site" state.

D3. Catalog copy: the recommended set is the ~12.1 GB "zaloha" subset, not the
    full 50 GB DB. The guide's size/USB wording should use ~12.5 GB.

D4. requirements.txt now installs cleanly with matplotlib/scikit-image explicit
    and numpy/astropy/photutils pinned (prior WP1). No transitive-only surprises.

================================================================================
STEP 4 - gates
================================================================================

Full pytest:  970 passed, 19 skipped (was 963; +7 install-arc guard tests).
--fast:       OVERALL PASS (WARNs: known untracked scratch, ledger-todo, and the
              by-design deps-outdated informational line).
--full:       OVERALL PASS - BYTE-IDENTICAL (SCIENCE-PATH RULE: src_py/config.py
              changed). vs anchor 435 at tree 7e88c3b: full-science-compare
              n_lc=166 failures=0; full-photometry-sha-core 3d26f4692ac81fc5...
              n=333; extended 6420f1daa53a0d5d... n=499; pipeline 2331s. Same core
              +extended SHAs as the pre-fix anchor -> the blank-path fix does not
              move any number. Ledger anchor items auto-stamped.

Files changed / added
  M src_py/config.py                              (blank-path -> default fix)
  A dev/scripts/apply_install_config.py           (config path writer)
  A dev/tests/test_fresh_machine_startup.py       (fresh-machine guard)
  A dev/tests/test_apply_install_config.py        (installer writer guard)
  A install_vyvar.ps1                             (Windows installer)
  A install_vyvar.sh                              (Linux twin)
  A INSTALL.md                                    (EN install manual, ASCII)
  A dev/results/CURSOR_RESULT_install_arc.md      (this file)

Push: HELD for the LENOVO STRANGER TEST. After Milan runs the PDF-only flow
(clone -> installer -> catalogs option 1 copy -> first launch -> create
Location/Telescope/Equipment -> import one small night), findings are appended
here, fixes iterate on this stack, then push via the PUSH protocol.

================================================================================
Lenovo-test findings (appended after Milan's run)
================================================================================
(pending)
