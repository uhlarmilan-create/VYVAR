CURSOR RESULT — 2026-06-19 (G5-F006, G5-F008, G5-F004)

What I did
Completed three isolated Group 5 publication-layer fixes in order: F006 (PDF BJD label) → F008 (comp count clarity) → F004 (export failure visibility). Each: diagnose → minimal fix → test → commit. Updated audit ledger. No push.

## G5-F006 — PDF time axis BJD(TDB)

**Diagnosis:** PDF LC plots and glossary used raw column name `bjd` / label “BJD”; pipeline LC column is `bjd_tdb_mid` stored as `bjd`. VarAstro already had `# TIME SYSTEM: BJD(TDB)`.

**Fix:** `photometry_report.py` — `_pdf_time_axis_label()` maps `bjd` / `bjd_tdb` / `bjd_tdb_mid` → `BJD(TDB)`; `hjd` → `HJD`; `jd` → `JD`. Main LC axes, overlay, candidate LC PNG, session cover kv, glossary.

**Tests:** `tests/test_export_g5_f006.py` — 3 passed.

**Ledger:** G5-F006 → **FIXED** (`b74c301`).

**Commit:** `fix(report): PDF time axis labeled BJD(TDB) to match column (G5-F006)` — **`b74c301`**.

---

## G5-F008 — VarAstro comp count vs trust `n_clean`

**Verify verdict: DISTINCT metrics (no number reconciliation).**

| Metric | Source | Meaning |
|--------|--------|---------|
| `n_good_comp` (summary) | `photometry_core.py:7827-7829` | Ensemble pool: comps with Phase-2A stability `good` or `suspect` |
| `n_clean` (trust) | `comp_qa_core.py:465-470` | comp_qa pool minus Sokolovsky LOO-flagged comps per target |

**Fix (clarity only):**
- `export_reports.py` VarAstro header: label **`n_ensemble_comp`** with comment `(stability good+suspect; not comp_qa n_clean)`.
- `photometry_report.py` glossary: clarified `n_good_comp`; added `n_clean` entry.
- `docs/VYVAR_CALIBRATION.md` consumer table + short note.

**Tests:** `tests/test_export_g5_f008.py` — 1 passed.

**Ledger:** G5-F008 → **FIXED** (`07e6f69`).

**Commit:** `fix(export): clarify/reconcile VarAstro comp count vs trust n_clean (G5-F008)` — **`07e6f69`**.

---

## G5-F004 — Surface silent export failures

**Diagnosis:** `export_reports.py` and `photometry_core.py` Phase-2A batch used `continue` on LC read errors, `logging.info` on empty exportable LC, and `logging.warning` on per-method exceptions — batch could finish with missing AAVSO/VarAstro files and no operator-visible ERROR trail.

**Fix (fail-visible, batch completes):**
- `export_reports.py`: `ExportFailure` typed dict; `record_export_failure()` (ERROR log + append); `log_export_batch_summary()` (ERROR summary at batch end).
- `export_lightcurve_reports`: `export_failures` collector; empty LC → ERROR + record; AAVSO/VarAstro write errors → ERROR + record (partial paths if one write fails).
- `export_all_method_lightcurve_reports`: missing LC / read error / empty LC / export exception → record; passes collector to single-target export.
- `photometry_core.py` Phase-2A loop: `_export_failures` list; gate failures recorded; `export_failures=` passed; `log_export_batch_summary()` after loop.

**Tests:** `tests/test_export_g5_f004.py` — 5 passed. Regression: clean export byte-identical when `export_failures=[]` vs omitted (`test_clean_export_byte_identical_without_failure_collector`). Related G5 tests (F006/F007/F008) — 16 passed total in combined run.

**Ledger:** G5-F004 → **FIXED** (`efbb4de`).

**Commit:** `fix(export): surface export read/write failures (batch summary, no silent skip) (G5-F004)` — **`efbb4de`**.

---

## Errors (if any)

None.

## Files changed (this task)

| Commit | Files |
|--------|-------|
| `b74c301` | `photometry_report.py`, `tests/test_export_g5_f006.py` |
| `07e6f69` | `export_reports.py`, `photometry_report.py`, `tests/test_export_g5_f008.py`, `docs/VYVAR_CALIBRATION.md` |
| `efbb4de` | `export_reports.py`, `photometry_core.py`, `tests/test_export_g5_f004.py` |
| `ec38807` | `docs/VYVAR_FULL_AUDIT_LEDGER.md` |

**Not pushed** — stop for Claude review.
