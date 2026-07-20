CURSOR RESULT - 2026-07-18 (DOCS-PDF)

What I did
Shipped the two CZ PDF guides into `docs/` with committed, repo-relative builders,
relaxed the docs-layout guard to allow `.pdf` (backed by a builder-existence check),
wired cross-links from the READMEs / VYVAR_PARAMS.md / INSTALL.md, and ran the gates.
Docs-only stack: no `--full` needed. Push HELD for Milan's PUSH protocol.

## STEP 1 - docs-layout guard relaxed (deliberately)

`dev/tests/test_docs_layout.py`:
- Renamed `test_docs_contains_only_markdown` -> `test_docs_contains_only_allowed_types`;
  allowed suffix set is now `{.md, .pdf}` (subdir ban and `CURSOR_*` ban unchanged).
- Added `test_docs_pdfs_have_a_committed_builder`: if any `*.pdf` lives in `docs/`,
  there must be a `dev/tools/docs_pdf/` dir with at least one `build_*.py` - so a
  committed binary can never be orphaned from its source.
- Module docstring updated to state the md+pdf rule.

`docs/VYVAR_PROCESS.md` - docs/ rule paragraph: added one sentence that docs/ is
Markdown-only except for `*.pdf` guides that a committed builder under
`dev/tools/docs_pdf/` regenerates.

## STEP 2 - builders + PDFs committed

Builders committed verbatim (Claude's repo-relative versions, run from repo root,
output into `docs/`):
- `dev/tools/docs_pdf/build_parameter_handbook.py`
- `dev/tools/docs_pdf/build_install_guide.py`

Builder run logs (from repo root):

```
$ python dev/tools/docs_pdf/build_parameter_handbook.py
built: params=250, detailed=83, deep boxes=13

$ python dev/tools/docs_pdf/build_install_guide.py
ok
```

Output verification (via pypdf):

```
handbook pages: 41            (spec: 41)                 PASS
handbook entries: 250         (spec: 250)                PASS  (249 config.json keys + saturate_limit_fraction code-default)
install pages: 5              (spec: 5)                  PASS
install contains "3.3 Stavba katalogu": True             PASS
install contains "3B. Linux": True                       PASS
```

Committed the REGENERATED PDFs:
- `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf`
- `docs/VYVAR_INSTALL_GUIDE_CZ.pdf`

### Regeneration diff vs Claude's delivered binaries
Only the two builder `.py` files were delivered to Cursor (the PDFs were not attached),
so the committed PDFs are freshly regenerated from live repo sources by definition. The
handbook builder reads the live `config.json` + `docs/VYVAR_CONFIG_GUIDE_CZ.md`; the
target counts (41 pages / 250 entries) match the spec exactly, so live config equals the
snapshot Claude built against for the purpose of key/value content. The handbook's
authored title-footer constant still reads "VYVAR HEAD d437bcd" (authored data inside the
builder) - kept verbatim per the commit-verbatim instruction; it is cosmetic and does not
affect page/entry counts.

### reportlab in requirements - DEVIATION from STEP 2.4 (stated, on purpose)
STEP 2.4 asked to add `reportlab` as a dev/docs extra that "must not become a runtime
dependency of the pipeline." Investigation shows `reportlab` is ALREADY a legitimate
RUNTIME dependency of the pipeline - it renders the SUMMARY MEASURE REPORT PDFs:
`src_py/pdf_report.py`, `src_py/photometry_report.py`, `src_py/ui_aperture_photometry.py`.
It has been a core line in `requirements.txt` since before this arc. Demoting it to a
dev/docs-only extra would break the pipeline's report generation on a fresh install.
Correct handling: leave it as a core requirement and annotate the line to state it is
runtime (reports) AND reused by the docs builders. No new dependency was introduced.

## STEP 3 - cross-links added

- `README.md` Documentation table: two rows - Parameter handbook (Czech, PDF) and
  Install & first-run guide (Czech, PDF).
- `README_CZ.md` Dokumentace table: two rows (ASCII, byte-safe latin-1 insert):
  Prirucka parametru (cesky, PDF) and Instalacni a spousteci prirucka (cesky, PDF).
- `dev/tools/gen_params_md.py`: generated header now carries an "In-depth handbook"
  pointer line to `VYVAR_PARAMETER_HANDBOOK_CZ.pdf` with its regenerate command;
  `docs/VYVAR_PARAMS.md` regenerated (freshness guard green).
- `INSTALL.md`: fixed the guide pointer to `docs/VYVAR_INSTALL_GUIDE_CZ.pdf` (it now
  lives in docs/) and added a pointer to `docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf`.

## STEP 4 - gates

```
pytest:  971 passed, 19 skipped, 31 warnings in 313.01s   (docs guard passes with PDFs)
--fast:  OVERALL PASS
         (WARN lines are expected: new untracked paths not yet committed at check time,
          branch ahead of origin/main 7e88c3b, ledger TODOs, deps-outdated informational)
```

No `src_py/` science-path files were touched, so no `--full` anchor gate is required.

## Maintenance note (recorded)
The handbook's detail texts and deep-dive boxes are authored data INSIDE
`build_parameter_handbook.py`. Values/keys regenerate automatically from live
`config.json` + `VYVAR_CONFIG_GUIDE_CZ.md`, but authored prose for NEW parameter keys
must be added by Claude/Milan. The registry freshness guard covers `VYVAR_PARAMS.md`
only, NOT the PDFs - PDF regeneration belongs to the docs-revision ritual.

## Files changed
- dev/tests/test_docs_layout.py            (guard relaxed + builder-existence check)
- docs/VYVAR_PROCESS.md                     (docs/ rule paragraph)
- dev/tools/docs_pdf/build_parameter_handbook.py   (new, verbatim)
- dev/tools/docs_pdf/build_install_guide.py         (new, verbatim)
- docs/VYVAR_PARAMETER_HANDBOOK_CZ.pdf      (new, regenerated)
- docs/VYVAR_INSTALL_GUIDE_CZ.pdf           (new, regenerated)
- requirements.txt                          (reportlab line annotated; no demotion)
- README.md / README_CZ.md                  (docs index rows)
- dev/tools/gen_params_md.py                (handbook pointer in header)
- docs/VYVAR_PARAMS.md                       (regenerated)
- INSTALL.md                                 (pointer path fix + handbook pointer)

## Push
HELD per STEP 4 - awaiting Milan's PUSH protocol.

## FLOW-DOC addendum (2026-07-18)

Shipped `docs/VYVAR_FLOW_CZ.pdf` (technical pipeline description) via the same
regenerable-builder pattern:

- Builder: `dev/tools/docs_pdf/build_flow_doc.py` (verbatim from Claude)
- Regenerated PDF: 7 pages; contains "10. Faze 2B" and "build_epsf_grid_model"
- README.md / README_CZ.md: one docs-table row each

Function-name spot check (all FOUND at HEAD, no PDF edits needed):

| Symbol | Location |
|--------|----------|
| run_full_photometry_pipeline | src_py/photometry_core.py |
| run_phase2a | src_py/photometry_core.py |
| measure_empty_aperture_sigma_bkg | src_py/photometry_core.py |
| build_epsf_model | src_py/psf_photometry.py |
| build_epsf_grid_model | src_py/psf_photometry.py |
| compute_trust_for_photometry_dir | src_py/trust_flag_core.py |
| build_masterstar_from_detrended | src_py/pipeline.py |
| select_comparison_stars_per_target | src_py/photometry_core.py |

Gates: pytest 973 passed / 19 skipped; `--fast` OVERALL PASS (docs-only; no
`--full`). HOLD with the other stacks for Lenovo + PUSH.
