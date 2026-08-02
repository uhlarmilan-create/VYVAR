# VYVAR -- Audit closure (final register)

**Date:** 2026-08-02
**Commit:** see `git log` at closure stamp (batch B-revised push)
**Purpose:** Single document for referees and maintainers. Every audit item, final disposition,
evidence pointer, and implementation batch. Limitations: `docs/VYVAR_LIMITATIONS.md`.

---

## Audit status

The **science audit is closed** after batch B-revised (mechanism) and pending implementation
batches **D** (numeric fixes, first re-cut) and **E** (remaining fixes, final re-cut). The only
**ongoing thread** is the **MASTERSTAR architecture queue** (register Steps 1-6, C-1/C-2
enhancements beyond saturation gate, TODO-B coaddition) -- an **enhancement**, not a
correctness blocker. The pipeline is scientifically sound with a single-frame MASTERSTAR.

---

## Decisions (Milan, 2026-08-02)

| item | one-line decision |
|------|-------------------|
| **I-11** | Pre-subtraction `sky_surface_bg_median_adu` in Howell sky term (Option 1) |
| **I-04** | NaN + exclude when ensemble scatter unmatched (Option 1) |
| **P-02 / A-6** | Wire scintillation, then per-rig `sigma_sys` floor if chi2_red > 1.2 (Option 3) |
| **T4-1** | N_equiv correction on resampled detection (Option B; confirm 3.78 vs 4.71) |
| **D5-2** | Saturation admission gate at 70% full well (C-1/C-2); no fabricated linearity curve |

Full rationale: `docs/VYVAR_DECISIONS.md` entries 5-9. Brief: `docs/VYVAR_DECISION_BRIEF.md`.

---

## D5-2 mechanism (closed)

Production `flux` does not scale as 10^(-0.4 G) at the bright end. **Mechanism:** saturation /
detector non-linearity for **G 8-9** stars reaching **~97%** of full well (`peak_max_adu` 54231 on
`saturate_limit_adu_85pct` 55705). **Not** aperture (fixed `flux_large` same bin); **not** sky wings.
**Fix:** admission gate excluding or flagging stars above **70%** full well on significant frame
fraction. Evidence: `dev/results/CURSOR_RESULT_batch_B_revised.md`, Step 1n N1 production columns.

---

## Closure register

| ID | Item | Final state | Evidence | Batch |
|----|------|-------------|----------|-------|
| P-10 | Sky-surface sign error | **FIXED** | `test_preprocess_sky_surface.py` | pre-closure |
| SKYSF | In-place guard | **FIXED** | pipeline | pre-closure |
| I-12 | PM logging | **FIXED** | audit t2 | pre-closure |
| T1 | Export time_base | **FIXED** | audit | pre-closure |
| D10-2 | Gaia-Johnson guard | **FIXED** | Stage 1 | pre-closure |
| D5-1 | Aperture provenance | **DOCUMENTED** | Step 1g; A-1 DOCUMENTED | D optional COG |
| **D5-2** | Flux vs G compression | **CONFIRMED** | B-revised M1; gate C-1/C-2 | E |
| A-1 | SNR-table differential | **DOCUMENTED** | Step 1d-1g; ~144 mmag fixture | D optional |
| A-9 | PSF scale | **DOCUMENTED** | Step 1f | -- |
| D1-1 / CR-1 | Cosmic-ray rejection | **QUEUED** | grep absent | E |
| D1-2 | Linearity curve | **DEFERRED** | needs dome-flat ramp | observing plan |
| D1-3 | Master flat docs | **CLOSED** | DECISIONS | -- |
| D10-1 | CV->CR band | **FIXED** | Milan decision | pre-closure |
| sigma_pp | Estimator | **FIXED** | Milan decision | pre-closure |
| threshold 3.8 | DAO threshold | **FIXED** | P-10 bundle | pre-closure |
| **I-11** | Howell sky term | **DECISION -> D** | DECISIONS #5 | D |
| **I-04** | Ensemble scatter | **DECISION -> D** | DECISIONS #6 | D |
| I-03 | Omitted Howell terms | **QUEUED** | after I-11 | D |
| **P-02** | Scintillation | **DECISION -> D** | DECISIONS #7 | D |
| **A-6** | sigma_sys floor | **DECISION -> D** | DECISIONS #7 | D |
| U-09 | DATE-OBS per rig | **MEASURED/DOCUMENTED** | stage 2 | -- |
| Part 0c | Delta pairing | **QUEUED** | stage 3 | E |
| DAO centroid | Aperture placement | **QUEUED** | stage 3 0e | E |
| **T4-1** | Detection on resampled | **DECISION -> E** | DECISIONS #8 | E |
| Anchor re-cut | VL-ANCHOR-WCSINV | **BLOCKED -> E** | after D auth | E |
| **C-1/C-2** | Admission / saturation gate | **QUEUED** | D5-2 DECISIONS #9 | E |
| TODO-B | Coaddition | **QUEUED** | MASTERSTAR spec | post-audit |
| Steps 1-6 | MASTERSTAR stack | **QUEUED** | enhancement thread | post-audit |

---

## Deferred (paper limitations)

See `docs/VYVAR_LIMITATIONS.md`: D1-2 linearity curve, Gaia PM (DR4), non-home rig timing (U-09),
site-specific scintillation validation, A-1 COG fix if deferred from batch D.

---

## Execution order remaining

1. **Batch D** -- I-04, I-11, I-03, P-02/A-6; code push; `--full` re-cut; Milan authorizes fingerprints.
2. **Batch E** -- Part 0c, DAO guard, CR-1, T4-1, C-1/C-2 saturation gate; final `--full`; register 29 FIXED.

Reports: `dev/results/CURSOR_RESULT_batch_{D,E}.md` (D/E to be re-issued).

---

## Evidence index

| Report | Content |
|--------|---------|
| `dev/results/CURSOR_RESULT_batch_B_revised.md` | D5-2 mechanism (production columns) |
| `dev/results/CURSOR_RESULT_batch_{A,B,C}.md` | Doc close, B-open, decision brief |
| `dev/results/CURSOR_RESULT_closure_step1{n,k,l,m}.md` | A-1 / D5-2 diagnosis arc |
| `dev/results/CURSOR_RESULT_audit_stage3_part1c.md` | chi2_red ~4.7 |
| `dev/results/CURSOR_RESULT_audit_stage3_part2b.md` | T4-1 N_equiv 3.78 vs 4.71 |
| `docs/VYVAR_AUDIT_FINAL.md` | Domain synthesis |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | Live register (mirrors this table) |
