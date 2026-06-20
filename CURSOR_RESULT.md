CURSOR RESULT — 2026-06-19 (Fix Step 6: G4-F001 Option B)

What I did
Corrected trust-gate documentation to match production reality (comp QA + check-star + `lc_quality_flag`; no SEP axis). Reclassified SEP/xval as offline-only harness. Added module headers on `xval_*`, guard test `tests/test_no_xval_in_production.py`, ledger G4-F001 → RESOLVED (Option B). No production logic changes.

## 6a — Inventory (docs/strings describing 3-axis or production SEP trust axis)

| File | Line(s) | Issue |
|------|---------|-------|
| `docs/VYVAR_DECISIONS.md` | 412–413 | Listed SEP cross-val + "three-axis" trust gate as pipeline capabilities |
| `docs/VYVAR_DECISIONS.md` | 505–515 | "### 3-axis trust gate" with `sep_confidence` as production input |
| `docs/VYVAR_JOURNAL.md` | 3645 | Session title "3-axis trust flag" |
| `docs/VYVAR_JOURNAL.md` | 3674–3691 | Trust flag section: 3-axis inputs including `xval_results.csv` / sep warnings |
| `docs/VYVAR_PIPELINE_CZ.md` | 225 | SEP cross-val table row without offline-only label |
| `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md` | 13 | "cross-validates with SEP" (implied in-pipeline) |
| `docs/VYVAR_FULL_AUDIT_LEDGER.md` | 1319, 1366 | G4-F001 finding + test-gap referencing 3-axis docs |
| `CURSOR_RESULT.md` | 34 | Prior audit summary (3-axis vs 2-axis) |

**Already correct (no edit required):**
- `docs/VYVAR_STATE.md` ~95–96 — xval offline, trust uses comp_qa + check-star + lc_quality
- `docs/VYVAR_PROCESS.md` ~169–171 — cross-val storage offline; trust gate without SEP
- `docs/VYVAR_DECISIONS.md` ~526–536 — `sep_xval` retired / trust gate v2 (pre-existing)
- `docs/VYVAR_JOURNAL.md` ~3766–3780 — Session 2026-06-03 trust gate v2 (pre-existing)
- `trust_flag_core.py` — implementation already SEP-free; docstring clarified in 6c

**Production code scan:** no production module imports `xval_run` / `xval_harness_core` / `assign_sep_confidence` (confirmed by guard test).

## 6b — Doc corrections

- `VYVAR_DECISIONS.md`: production trust gate section rewritten (comp QA + check-star + `lc_quality_flag`; unevaluated → RED; SEP offline); intro bullet separated SEP study from pipeline trust.
- `VYVAR_JOURNAL.md`: harness-era trust flag marked historical; production since 2026-06-03 = trust gate v2 without SEP.
- `VYVAR_PIPELINE_CZ.md`: SEP row labeled offline harness, not pipeline.
- `VYVAR_NEIGHBOR_SUB_DESIGN.md`: aperture trust wording aligned with production gate + offline SEP.
- `VYVAR_FULL_AUDIT_LEDGER.md`: G4-F001 RESOLVED (Option B); test-gap row updated.

## 6c — Module headers

- `xval_run.py` — standalone OFFLINE cross-validation docstring prefix.
- `xval_harness_core.py` — same prefix.
- `trust_flag_core.py` — production inputs + explicit no SEP/xval axis.

## 6d — Guard test

- `tests/test_no_xval_in_production.py` — scans production `.py` (excludes `tests/`, `scripts/`, `sandbox/`, `xval_*`, `validate_lc_crossval.py`) for forbidden imports.

## Output / findings

- `pytest tests/test_no_xval_in_production.py tests/test_trust_flag.py tests/test_trust_checkstar_hardening.py`: **33 passed**
- `ruff check` on new/changed modules: **clean**

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_JOURNAL.md`
- `docs/VYVAR_PIPELINE_CZ.md`
- `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`
- `trust_flag_core.py` (docstring only)
- `xval_run.py` (docstring only)
- `xval_harness_core.py` (docstring only)
- `tests/test_no_xval_in_production.py` (new)
- `CURSOR_RESULT.md`
