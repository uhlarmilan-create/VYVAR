CURSOR RESULT - 2026-07-17

# DOCS-CLEANUP - docs/ holds living project documentation only

Arc: repository hygiene, follow-up to REPO-REORG. Baseline origin/main `6d63441`.

## What I did
Classified every `docs/` entry by Milan's rule (living design/setup/operation docs +
specs stay; audit/investigation/result/target-specific artifacts -> `dev/results/`;
figure/sample subdirectories deleted). Moved 15 artifacts, removed 2 subdirectories,
fixed functional references, and added a guard test + PROCESS rule so the layout is a
property. Full pytest green after each commit; `--fast` anchor gate PASS.

Commits (on top of `6d63441`):
- `3182593 chore(docs): move audit/result artifacts to dev/results`
- `05db1d2 chore(docs): remove round2_figs and sample_reports`
- `e5f376a test(docs): docs-layout guard + PROCESS rule`

## STEP 1 - disposition table (every docs/ entry)

KEEP = living documentation or spec/design/policy/rule. MOVE = audit/investigation/
result/target-specific artifact -> `dev/results/`. REMOVE = figure/sample subdir.

| File | Disposition | Reason |
| --- | --- | --- |
| config_schema.md | KEEP | schema |
| K2_BAND_AWARE_SPEC.md | KEEP | spec |
| VYVAR_CALIBRATION.md | KEEP | core |
| VYVAR_CAL_DIAG_SPEC.md | KEEP | spec |
| VYVAR_CANONICAL_COMBINATION_LOGIC.md | KEEP | design |
| VYVAR_CHECKSTAR_SELECTION_SPEC.md | KEEP | spec |
| VYVAR_CLAUDE_OPERATING_PRINCIPLES.md | KEEP | charter |
| VYVAR_CODE_MAP.md | KEEP | map |
| VYVAR_COMP_DEGRADATION_SPEC.md | KEEP | spec |
| VYVAR_COMP_FLOOR_POLICY_SPEC.md | KEEP | spec |
| VYVAR_DECISIONS.md | KEEP | core |
| VYVAR_DECISION_GROUNDING_RULE.md | KEEP | rule |
| VYVAR_JOURNAL.md | KEEP | core |
| VYVAR_K2_DESIGN_SPEC.md | KEEP | spec |
| VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md | KEEP | spec |
| VYVAR_NEIGHBOR_SUB_DESIGN.md | KEEP | design |
| VYVAR_PARAMS.md | KEEP | generated-spec |
| VYVAR_PIPELINE_CZ.md | KEEP | core |
| VYVAR_PROCESS.md | KEEP | process |
| VYVAR_ROADMAP.md | KEEP | core |
| VYVAR_RUNBOOK.md | KEEP | runbook |
| VYVAR_SIGMA_BUDGET_SPEC.md | KEEP | spec |
| VYVAR_SIGMA_FLOOR_SPEC.md | KEEP | spec |
| VYVAR_SIMPLE_DIFFERENTIAL_SPEC.md | KEEP | spec |
| VYVAR_SPARSE_TRUST_SPEC.md | KEEP | spec |
| VYVAR_STATE.md | KEEP | core |
| VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md | KEEP | spec |
| VYVAR_VALIDATION.md | KEEP | core |
| VYVAR_WIDE_SLOPE_NOISE_SPEC.md | KEEP | spec |
| VYVAR_AUDIT_FINDINGS.md | MOVE | audit |
| VYVAR_AUDIT_LEDGER.md | MOVE | audit |
| VYVAR_FULL_AUDIT_LEDGER.md | MOVE | audit |
| VYVAR_CODE_AUDIT.md | MOVE | audit (was untracked) |
| VYVAR_MATH_PHYS_AUDIT.md | MOVE | audit |
| VYVAR_EPSF_AUDIT.md | MOVE | audit |
| VYVAR_EPSF_FWHM_TEST.md | MOVE | investigation |
| VYVAR_EXCEPT_CENSUS.md | MOVE | census |
| VYVAR_GAIA_DR3_AUDIT.md | MOVE | audit |
| VYVAR_DRAFT367_CROWDING.md | MOVE | draft-specific |
| VYVAR_HCHIPER_CROWDING_RECOMPUTE.md | MOVE | target-specific |
| VYVAR_HCHIPER_PSF_PROBE.md | MOVE | target-specific |
| VYVAR_CHIANDH_BASELINE_RUNBOOK.md | MOVE | target-specific |
| VYVAR_PIPELINE_CZ_rework.md | MOVE | rework |
| VYVAR_REPORTING_COLUMN_GROUNDED_DECISION.md | MOVE | investigation |
| round2_figs/ (4 png tracked + 1 untracked) | REMOVE | figures |
| sample_reports/ (1 pdf tracked) | REMOVE | sample |

No genuinely ambiguous cases: everything not on Milan's explicit lists classified
cleanly by the rule. All 15 moves via `git mv` except `VYVAR_CODE_AUDIT.md` (untracked
-> filesystem move + `git add`). Nothing flagged for KEEP/MOVE re-review.

## STEP 1 - fixed references (functional: code/tests/operational pointers)
- `dev/scripts/_except_bulk_apply.py` - module docstring + `CENSUS_PATH` -> `dev/results/VYVAR_EXCEPT_CENSUS.md`.
- `dev/scripts/_except_census_scan.py` - census write path `out` -> `dev/results/VYVAR_EXCEPT_CENSUS.md`.
- `dev/tests/test_git_dirty_code_classifier.py` - fixture porcelain + files entry `docs/VYVAR_CODE_AUDIT.md` -> `dev/results/VYVAR_CODE_AUDIT.md` (still classifies as scratch).
- `dev/scripts/session_baseline_check.py` - dropped `docs/VYVAR_CODE_AUDIT.md` (STEP 1) and `docs/round2_figs/` (STEP 2) from `KNOWN_UNTRACKED_PREFIXES` (now tracked / deleted).
- `docs/VYVAR_RUNBOOK.md` - "Canonical procedure" pointer -> `dev/results/VYVAR_CHIANDH_BASELINE_RUNBOOK.md`.

## Left as-is (prose / historical) - FLAGGED for Milan's optional follow-up
Per the task's fix-scope (code, tests, PROCESS/CLAUDE pointers) and the explicit
JOURNAL/DECISIONS carve-out, prose cross-references were NOT rewritten. These now
point at the old `docs/...` path for a file that lives in `dev/results/`. None is
functional; all are navigational prose. Listed here so nothing is lost:
- `VYVAR_JOURNAL.md`, `VYVAR_DECISIONS.md`: many `docs/VYVAR_*` mentions - left as-is per task.
- Living-doc prose links to moved files (candidates for a future prose-refresh commit):
  - `VYVAR_STATE.md`: EXCEPT_CENSUS, REPORTING_COLUMN_GROUNDED_DECISION, GAIA_DR3_AUDIT, CHIANDH_BASELINE_RUNBOOK (x2), MATH_PHYS_AUDIT.
  - `VYVAR_ROADMAP.md`: AUDIT_LEDGER, EXCEPT_CENSUS (x4), REPORTING_COLUMN_GROUNDED_DECISION, DRAFT367_CROWDING, HCHIPER_CROWDING_RECOMPUTE, EPSF_AUDIT, GAIA_DR3_AUDIT.
  - `VYVAR_NEIGHBOR_SUB_DESIGN.md`: HCHIPER_CROWDING_RECOMPUTE, EPSF_FWHM_TEST.
  - `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`: CHIANDH_BASELINE_RUNBOOK.
  - `VYVAR_CALIBRATION.md`: FULL_AUDIT_LEDGER (x3). `VYVAR_VALIDATION.md`: DRAFT367_CROWDING.
- `src_py` code comments naming the census (navigational; EXC-#### id is the real anchor):
  `importer.py` (EXC-0089, EXC-0090), `vyvar_platesolver.py` (EXC-0605) -> `docs/VYVAR_EXCEPT_CENSUS.md`.
- Moved audit docs cross-reference each other with `docs/...` prefixes; now co-located in
  `dev/results/`, left as historical content.

## STEP 2 - deleted content
- `docs/round2_figs/`: b1_aperture_skirt.png, b2_lc_beforeafter.png, b2_transparency.png,
  v0454_flip_diag.png (tracked, `git rm`) + v0454_lc_vyvar.png (untracked, deleted).
- `docs/sample_reports/`: summary_report_291.pdf (tracked, `git rm`).
- Git history preserves all six. `VYVAR_CODE_AUDIT.md` (moved to `dev/results/`) references
  round2_figs images; those links are intentionally dead by Milan's decision - recorded,
  not resurrected.

## STEP 3 - rule as a property
- `dev/tests/test_docs_layout.py` (4 tests): docs/ dir exists; no subdirectories; only
  `*.md`; no `CURSOR_*` files. Each failure message names the offenders.
  ```
  dev/tests/test_docs_layout.py::test_docs_dir_exists PASSED
  dev/tests/test_docs_layout.py::test_docs_has_no_subdirectories PASSED
  dev/tests/test_docs_layout.py::test_docs_contains_only_markdown PASSED
  dev/tests/test_docs_layout.py::test_docs_has_no_cursor_prefixed_files PASSED
  4 passed in 0.04s
  ```
- `docs/VYVAR_PROCESS.md`: added the docs/ rule paragraph under "Repository layout &
  ritual paths".
- Hot spot verified: `docs/VYVAR_PARAMS.md` STAYS; `gen_params_md.py` and the params
  freshness test are untouched and pass.

## STEP 4 - gate status
- Full pytest after each commit: STEP 1 `924 passed, 19 skipped`; STEP 2 same tree;
  STEP 3 `928 passed, 19 skipped` (+4 guard tests).
- `session_baseline_check.py --fast`: **OVERALL PASS** (HEAD `e5f376a`,
  `pytest PASS 928 passed, 19 skipped`). WARNs pre-existing/expected: 2 known untracked
  scratch dev-scripts (dy_peg, qatar8), 1 untracked (forensic_disc_ui_match2.py),
  origin differs (unpushed), ledger-todo VL-ANCHOR-424 / VL-ANCHOR-DQ-430.
- Docs moves did not disturb the anchor gate (docs are outside the photometry SHA set).

## STEP 4 - push
GATED. Stack ready to push `git log --oneline 6d63441..HEAD`:
```
3182593 chore(docs): move audit/result artifacts to dev/results
05db1d2 chore(docs): remove round2_figs and sample_reports
e5f376a test(docs): docs-layout guard + PROCESS rule
<this docs(result) commit>
```
Pushed HEAD: **AWAITING Milan's explicit "push".** On approval I push all four to
origin/main and report the pushed HEAD + clean-tree confirmation.

## Errors (if any)
None.

## Files changed
15 docs artifacts relocated to `dev/results/`; 6 figure/sample files deleted;
`_except_bulk_apply.py`, `_except_census_scan.py`, `session_baseline_check.py`,
`test_git_dirty_code_classifier.py`, `VYVAR_RUNBOOK.md`, `VYVAR_PROCESS.md` edited;
`dev/tests/test_docs_layout.py` added. No science/numeric change (docs + dev tooling only).
