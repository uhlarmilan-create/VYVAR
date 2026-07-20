CURSOR RESULT - 2026-07-20

What I did
PUBLICATION-PREP in three local commits (A, B, C). Not pushed.

## Part A - field PNG catalog_only removal
Commit: `2a7ff26` fix(report): drop catalog_only (no DAO) category from field PNGs

- Removed cyan catalog_only marker + legend from per-target field maps
  (`save_target_field_map_png` in `src_py/photometry_core.py`).
- Grep across PNG generators: overview map already skipped catalog_only;
  TESS blend path had no catalog_only category.
- Test: `dev/tests/test_field_map_no_catalog_only.py`.
- PNGs are outside photometry SHA set (`dev/tests/photometry_sha.py` patterns
  are lightcurve_*.csv, comp_quality_*.json, comparison_stars_per_target.csv only).
- `--fast` PASS after A.

## Part B - vsx_out_of_scope_types
Commit: `07108b8` feat(targets): vsx_out_of_scope_types filter (token-match, auto
targets only, mask-first skip)

- New module `src_py/vsx_type_scope.py`; config key default `[]`.
- Phase 0 mask-first: auto VSX only; manual immune; skip_reason
  `vsx_type_out_of_scope`; still purged from comp pool.
- INV-CFG-01 amended; registry 271 -> 272; FLOW ch 9 + facts; UI badge;
  tests in `dev/tests/test_vsx_out_of_scope_types.py`.
- Gates: ruff clean; `--fast` PASS (1050 then later suites); P1 golden 7/7 PASS.

### --full outcome for B (required)
OVERALL: FAIL on photometry SHA vs VL-ANCHOR-WCSINV.

| Check | Result |
| --- | --- |
| full-pipeline | PASS (2296s) |
| full-science-compare | PASS (n_lc=166 failures=0) |
| full-photometry-sha-core | FAIL run 03d8fb64... vs snap 3d26f469... (n=333) |
| full-photometry-sha-extended | FAIL |
| vsx_type_out_of_scope markers with [] | 0 (INV-CFG-01 OK) |
| lightcurve_*.csv byte identity vs snap | 166/166 identical |

Root cause of SHA FAIL (NOT Part B): 19 `comp_quality_*.json` files differ only by
ENCODING-POLICY ASCII `sigma` vs snapshot Unicode U+03C3 in slope notes
(+3 bytes each). Science LC set is byte-identical; empty-list no-op for the
filter is proven by science-compare + zero out-of-scope markers.
STOP reported per task; no silent rebaseline.

## Part C - export/report rework
Commit: `c514e7f` feat(export): methods ON/OFF matrix + slim AAVSO/VarAstro
headers; citation fixes (AIJ venue, Jordi volume); OBSCODE warning only when unset

- Methods ON/OFF matrix from `RunCitationContext` (same flags as citations).
- Slim AAVSO/VarAstro headers: matrix + `[METHODS - this run]` ON-only + PDF pointer.
- Full citation blocks + matrix remain in SUMMARY MEASURE REPORT PDF path
  (`emit_pdf_methods_sections`).
- CITATIONS.bib: Collins -> AJ 153, 77 (Stassun & Hessman); Jordi -> A&A 523, A48.
- plain_ascii_citation_text: no LaTeX `\_` / `\Delta` / backslashes in emitted notes.
- OBSCODE warning only when empty; UMIA emits no warning.
- Exports outside photometry SHA; VL-AAVSO-EXPORT is format validation only
  (no SHA refresh mechanism required).
- `validate_aavso_export.py`: 164/164 clean on P1 mini path; EXIT 0.
- Gates: unit tests; `--fast` PASS (1055 passed); P1 golden 7/7 PASS.

### C5 data-line diff summary (P1 mini)
Pre-copy: `tmp/pubprep_export_pre_p1mini` (186 AAVSO+VarAstro files).
Post: re-export with new headers on `draft_000435_p1mini`.

```
C5: files 186  header_diff 175  data_diff 0  missing_post 0
```

Header-only change; ZERO data-line diffs.

## Sample slim AAVSO header (BO_CVn_20260423.txt, verbatim)

```
#
# Pipeline: VYVAR - Automated Differential Photometry Pipeline
#
# METHODS MATRIX (this run):
#   ensemble flux-sum: ON
#   PyTICS: ON
#   iterative comp clip: ON
#   temporal binning: OFF
#   SavGol detrend: OFF
#   Democratic detrender: OFF
#   SysRem: OFF
#   color term: OFF
#   k2: ON (literature)
#   aperture correction Metoda B: ON
#   COG AC: OFF
#   dilution GS11: OFF
#   PSF branch: OFF
#   per-frame saturation: OFF
#   empirical background mode: ON
#   trust gate: ON
#
# [METHODS - this run]
#   Marconi et al. (2026) RASTI (in press) - PyTICS iterative comp weights
#   Gilliland & Brown (1988) PASP 100, 754 - iterative ensemble-relative comp QA
#   Burdanov et al. (2014) - Astrokit sparse-field comp selection (Delta mag matching)
#   Everett & Howell (2001) PASP 113, 649 - ensemble differential photometry QA
#   Smith et al. (2002) AJ 123, 2121 - Sloan u'g'r'i'z' second-order extinction coefficients
#   Jordi et al. (2010) A&A 523, A48 - Gaia BP-RP colour transformations
#   Second-order extinction k'' from literature defaults (BP-RP units; band-aware).
#
# Full algorithm references: SUMMARY MEASURE REPORT (PDF)
#
#TYPE=Extended
#OBSCODE=UMIA
#LATITUDE=50.1122
#LONGITUDE=14.6983
#ELEVATION=275
#SOFTWARE=VYVAR/1.0 (aperture photometry; Broeg 2005 ensemble)
#DELIM=,
#DATE=BJD
#OBSTYPE=CCD
#
```

## Sample slim VarAstro header (citation block through pointer; BO_CVn)

```
#
# Pipeline: VYVAR - Automated Differential Photometry Pipeline
#
# METHODS MATRIX (this run):
#   ensemble flux-sum: ON
#   PyTICS: ON
#   iterative comp clip: ON
#   temporal binning: OFF
#   SavGol detrend: OFF
#   Democratic detrender: OFF
#   SysRem: OFF
#   color term: OFF
#   k2: ON (literature)
#   aperture correction Metoda B: ON
#   COG AC: OFF
#   dilution GS11: OFF
#   PSF branch: OFF
#   per-frame saturation: OFF
#   empirical background mode: ON
#   trust gate: ON
#
# [METHODS - this run]
#   Marconi et al. (2026) RASTI (in press) - PyTICS iterative comp weights
#   Gilliland & Brown (1988) PASP 100, 754 - iterative ensemble-relative comp QA
#   Burdanov et al. (2014) - Astrokit sparse-field comp selection (Delta mag matching)
#   Everett & Howell (2001) PASP 113, 649 - ensemble differential photometry QA
#   Smith et al. (2002) AJ 123, 2121 - Sloan u'g'r'i'z' second-order extinction coefficients
#   Jordi et al. (2010) A&A 523, A48 - Gaia BP-RP colour transformations
#   Second-order extinction k'' from literature defaults (BP-RP units; band-aware).
#
# Full algorithm references: SUMMARY MEASURE REPORT (PDF)
#
```

(VarAstro continues with site/software/comp table body as before; Collins AIJ line in
PHOTOMETRY body now shows AJ 153, 77 / Stassun & Hessman.)

## Citation cross-check (export-emitted vs FLOW ch 19)

| Entry | CITATIONS.bib (post-C) | FLOW ch 19 | Notes |
| --- | --- | --- | --- |
| collins2017 | AJ 153, 77; Collins/Kielkopf/Stassun/Hessman | Not listed as a numbered ch-19 row (AIJ mentioned in prose only) | FIXED bib to ADS 2017AJ....153...77C; report: ch 19 still omits explicit AIJ venue line |
| jordi2010 | A&A 523, A48 | A&A 523, A48 | FIXED bib; now matches FLOW |
| smith2002 | AJ 123, 2121 | AJ 123, 2121 | Match |
| broeg2005 | AN 326, 134 | AN 326, 134 | Match (PDF/full + VarAstro body) |
| honeycutt1992 | PASP 104, 435 | PASP 104, 435 | Match |
| marconi2026 | RASTI (in press) | RASTI / PyTICS | Match (in-press wording) |
| gilliland1988 | PASP 100, 754 | Not in ch 19 list | Report only (pre-existing; not silently edited) |
| burdanov2014 | Astrokit note | Not in ch 19 list | Report only |
| everett2001 | PASP 113, 649 | Not in ch 19 list | Report only |

No further silent edits to FLOW or bib beyond the two tasked corrections.

## Files changed (commits)
- A `2a7ff26`: field PNG + test
- B `07108b8`: vsx_type_scope, config/registry/docs/FLOW/UI/tests/invariants
- C `c514e7f`: citations/export/bib/validator/tests/FLOW/DECISIONS/STATE/PARAMS/handbook

## Errors (if any)
- Part B `--full` SHA FAIL (encoding sigma in comp_quality notes); science no-op OK.
- Concurrent `--fast` can report pytest FAIL with all-passed summary; re-run alone PASS.

## Docs impact
- FLOW ch 9 (B filter); ch 15.2/15.4 (slim headers + matrix); FLOW PDF regen.
- PARAMS.md + parameter handbook (registry 272; aavso_observer_code help).
- DECISIONS: EXPORT-HEADER-SLIM.
- STATE one-liner + facts count 43.
- INVARIANTS INV-CFG-01 (B).
- config.json comment for vsx_out_of_scope_types (B).

## Recurrence
- Any new gated method in exports must extend `RunCitationContext` /
  `build_methods_matrix_lines` together (single-source matrix rule).
- Empty `vsx_out_of_scope_types` must keep science byte-identity; if SHA drifts,
  diagnose encoding/non-science sidecars before blaming the filter.
- OBSCODE: never warn on a configured non-empty code (UMIA is real).
- Re-run `validate_aavso_export.py` after any export-header change; prove
  data-line identity on P1 mini when touching headers.
