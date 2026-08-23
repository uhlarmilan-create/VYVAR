CURSOR RESULT - EPSF-VALID-02 R5 aperture audit + F5 (STOP at R5)

Date: 2026-08-22. HEAD: `93b3194` (unchanged — no F5 commit). Parent: CURSOR_TASK EPSF-VALID-02 R1R4.
**STOP:** MAG-relevant diffs confirmed on live draft 516. Restore requested; **awaiting Milan authorization.** F5 deferred.

---

## R5 — Aperture diff audit (BLOCKING)

### a) Per-column summary (134 Light proc CSVs: backup vs live)

Artifact: `dev/results/context/session_20260822_epsf_valid_02_r5f5/r5_aperture_diff_summary.csv`  
Per-file detail: `r5_per_file_diffs.csv` (134 rows)  
Sandbox: `dev/sandbox/epsf_valid_02_r5_aperture_diff_audit.py`

| column | n_rows compared | n_numeric_diff | n_files w/ numeric diff | max \|?\| | median \|?\| (diff rows) | repr-only rows |
|--------|----------------:|---------------:|------------------------:|----------:|-------------------------:|---------------:|
| **dao_flux** | 426,486 | **51,348** | **99** | 812,800.7 ADU | 1,244.8 ADU | 0 |
| **flux** | 426,486 | **51,348** | **99** | 812,800.7 ADU | 1,244.8 ADU | 0 |
| **mag** | 333,072 (both finite) | 0 (finite pairs) | 0 | 0 | 0 | 0 |
| **mag_inst** | — | — | — | — | — | column absent both sides |
| **mag_err** | — | — | — | — | — | column absent both sides |

**MAG column — value-loss (not captured as numeric ? on finite pairs):**

| metric | count |
|--------|------:|
| rows: mag finite in backup ? NaN in live | **41,381** |
| rows: mag NaN in backup ? finite in live | **2,513** |
| **mag-relevant row transitions total** | **43,894** |

**Representation-only diffs:** none (0 repr rows on all columns). Diffs are **numeric** (flux) or **value loss/gain** (mag), not float formatting or column order alone.

**Row-set changes:** 99/134 files differ in row count and/or catalog_id set (export re-ran full catalog match). 35 files byte-identical on aperture columns (Lights 001–038 subset).

**Breakpoint:** aperture columns unchanged for **35** files (max Light_038); **99** files from **Light_039** onward show flux diffs and mag loss. Aligns with P1 “per-frame swallow from frame ~36” — accept rerun rewrote post-breakpoint frames via full export, not PSF-only merge.

### b) Mechanism hypotheses (pre-registered order)

| Hyp | Verdict | Evidence |
|-----|---------|----------|
| **H1** config-source split (UI runtime vs draft/snapshot) | **Inconclusive / secondary** | Accept rerun used `AppConfig()` at HEAD (`epsf_valid_02_accept_rerun.py`). No draft-local config snapshot on disk for 516 to diff. Flux/mag changes correlate with **full re-export** (H2), not an isolated param mismatch on the unchanged first 35 frames. |
| **H2** code-path split: `export_per_frame_catalogs` vs full-pipeline photometry | **CONFIRMED (primary)** | Export path builds a fresh catalog DataFrame and writes the entire sidecar CSV (`_vyvar_df_to_csv(df2, sidecar)` in `_export_per_frame_run_catalog_core`). Accept script intended “PSF columns only” but invoked full export. Post-breakpoint files: row-count shrink, flux recomputed, mag cleared on ~430 rows/frame. First 35 frames untouched (prior partial PSF pass or earlier identical export). |
| **H3** input split: MASTERSTAR / FITS set contamination | **REJECTED for Light diffs** | MASTERSTAR added a 135th proc file (R1); Light_* diffs are per-file sidecar overwrites on the same FITS names, not MASTERSTAR cross-contamination. |
| **H4** era drift (live CSVs predate snapshot era) | **REJECTED as primary** | Backup mtimes Aug 18–21; live Light_* rewritten Aug 22 08:21–09:39 UTC (accept rerun). `--full` byte-identity PASS on frozen snapshot at HEAD does not apply to live Archive draft. Live diffs are **accept-rerun overwrite**, not silent era drift on untouched files. |

**Secondary finding — vacuous ACCEPT guard root cause:** `_aperture_columns_hash` in accept script uses naive `line.split(",")` on **quoted** VYVAR CSV headers (`"mag"` ? `mag`), producing **empty** `pre_aperture_column_hashes.json` and `post_aperture_column_hashes.json`. Reported `aperture_hash_mismatches=0` was structurally vacuous, not evidence of aperture stability.

**Iron constraint violation:** PSF columns were intended additive; **dao_flux/flux recomputed** and **mag lost** on 99 frames — aperture path moved on live draft 516.

### c) Restore decision (pre-registered)

| Criterion | Result |
|-----------|--------|
| MAG-relevant numeric diffs on live draft? | **Yes** — 51,348 flux/dao_flux numeric diffs; 43,894 mag row transitions |
| Representation-only? | **No** |
| Restore recommended? | **Yes** |
| Restore executed? | **No — awaiting Milan authorization** |

**Proposed restore (not executed):** For each of 134 `proc_BO_CVn_Light_*.csv`, restore aperture columns (`dao_flux`, `flux`, `mag`, and any other non-PSF columns) from `proc_backup_pre_accept/`, retain/re-merge PSF columns from accept rerun. Mechanism fix (PSF-only merge or export guard) ? separate follow-up task.

---

## F5 — Enumerator fix

**Status: DEFERRED** per task STOP rule (MAG-relevant diffs ? restore gate before F5).

Planned scope when authorized to continue:

- Shared science-light frame list (exclude MASTERSTAR); `export_per_frame_catalogs` + INV-PSF-FRAME-01 denominator
- Remove stray `proc_MASTERSTAR.csv` on draft 516
- Tests + `--fast` gate

No F5 commit hash. No `--fast` re-run (no code change).

---

## Gate status

| Gate | Result | Note |
|------|--------|------|
| `--fast` | not re-run | no F5 commit |
| `--full` | not required | R5 restore not executed; Archive draft 516 outside snapshot recut scope |

---

## Files changed

| File | Action |
|------|--------|
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_R5F5.md` | created (this deliverable) |
| `dev/sandbox/epsf_valid_02_r5_aperture_diff_audit.py` | created (audit sandbox) |
| `dev/results/context/session_20260822_epsf_valid_02_r5f5/*` | R5 artifacts |

No production code changes. No restore. **STOP.**

**Milan:** authorize restore on draft 516, then F5 enumerator fix can proceed on a follow-up task.
