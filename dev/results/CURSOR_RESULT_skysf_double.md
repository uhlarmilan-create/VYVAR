CURSOR RESULT - SKYSF-DOUBLE (2026-07-30)

What I did
Part 1 forensics (read-only) on anchor draft_000435_snapshot_skysurface_20260716 and on-disk draft sweep.
STOP GATE 1 ù awaiting Milan clearance before Part 2 (guard implementation).

Evidence outputs: `dev/results/context/session_20260730_skysf_double/`

---

## PART 1 ù Forensics

### 1.1 ù Header detector (calibrated lights)

**CSV:** `anchor_435_snapshot_headers.csv` (150 frames, setup `NoFilter_60_2`)

| Field | Result |
|-------|--------|
| `VY_SKYSF` | **absent on all 150 calibrated frames** |
| `VYSKYORD` | **absent** |
| `VYSKYP2P` | **absent** |
| `VYVARPR` | **absent** |

**VYSKYP2P distribution on calibrated FITS:** N/A (no values).

**Important context:** this anchor (2026-07-16) predates in-place preprocess with `VY_*` headers on calibrated FITS. Sky subtraction ran on the **`processed/lights/` copy tree** (`calibrated ? proc_*.fits`), recorded in `processed/lights/qc_metrics.csv` with `src?dst`.

**Supplementary ù science-path headers on `proc_*.fits` (10-frame sample):**

| Stat | VYSKYP2P (ADU) on proc_ files |
|------|------------------------------:|
| min | 141.0 |
| median | 158.7 |
| max | 177.9 |
| VYSKYORD | 2 on all sampled frames |

Distribution is **unimodal** in the **115ù178 ADU** band (matches `qc_metrics.csv` / `pipeline_meta.json` median **136.84 ADU**). There is **no low-value tail** (<30 ADU) that would indicate a second-pass overwrite on already-flat data.

**Plain reading:** absent markers on calibrated FITS are expected for this era; proc-tree markers show **first-pass gradient scale**, not second-pass residual scale. This is evidence **against** double subtraction on the science path.

---

### 1.2 ù Independent cross-check (stored pixels)

**CSV:** `anchor_435_snapshot_residual_refit.csv`

Refit: source-masked order-2 `_fit_subtract_preprocess_sky_surface` on **stored** pixel arrays.

| Layer | refit surface p2p (ADU) | Interpretation |
|-------|------------------------:|----------------|
| **calibrated** (150 frames) | min 114.3, p05 118.8, **median 136.8**, p95 157.6, max 177.9 | Full sky gradient still present ù **not** in-place subtracted |
| **processed proc_** (frame001 example) | refit p2p **283.0** | **Misleading high** ù see below |
| **Identity check** (frame001) | `max_abs(recompute_single_subtract(cal) ? proc_) = **0.0**` | proc_ is **exactly one** sky subtract of calibrated |

**10/10 proc frames (sample):** byte-identical to a **single** recompute from calibrated (`single_subtract_identity 10/10`).

**Argument from numbers (not expectation):**

- **One subtraction:** proc_ = cal ? S? with ?S??_p2p ? 136ù178 ADU (recorded in proc header / qc_metrics). Confirmed by **0.0 ADU** identity vs recompute.
- **Two subtractions:** proc_ would equal cal ? S? ? S?; header `VYSKYP2P` would reflect **S?** (small, ?30 ADU on flat data). Observed header values are **S?-scale** (136ù178 ADU), not S?-scale.
- **Refit p2p on proc_** can exceed calibrated refit p2p (283 vs 178 on frame001) because star masking is **non-idempotent** after subtract ù different DAOStarFinder detections change the fit domain. This is **not** a double-subtract signature; it coexists with **0.0 identity** to single subtract.

**Conclusion:** science-path pixels received **one** sky-surface subtraction. Calibrated FITS were **not** subtracted in place (gradient intact).

---

### 1.3 ù Provenance

| Source | Finding |
|--------|---------|
| `infolog_20260716_123126.txt` | **No** `[PREPROCESS]`, `QC in-place`, or `MAKE MASTERSTAR` milestone lines (logging predates POST-453 infolog milestones). Infolog confirms **`proc_BO_CVn_Light_001.fits`** used in photometry debug (line ~7204). |
| `processed/lights/qc_metrics.csv` | **139 rows**, all `sky_surface_applied=True`, **calibrated src ? proc_ dst** (copy-tree, one row per frame). `sky_surface_p2p_adu` median **136.84 ADU**. |
| `platesolve/.../pipeline_meta.json` | `sky_surface_n_applied: 139`, `sky_surface_p2p_median_adu: 136.84`, order 2. |
| Repeated MAKE MASTERSTAR | **Not countable** from infolog. qc_metrics row structure (unique src?dst pairs) supports **one** preprocess pass producing proc_ outputs. |

**Finding:** logs cannot prove run-count to modern standard; **qc_metrics + pixel identity** are the authoritative count: **one subtract pass** on the proc tree.

---

### 1.4 ù Scope sweep (calibrated headers only)

**CSV:** `all_drafts_header_sweep.csv`

| Draft on disk | Calibrated frames | VYSKYP2P present | Low-VYSKYP2P signature |
|---------------|------------------:|------------------|------------------------|
| `draft_000435_snapshot_skysurface_20260716` | 150 | 0 | no |
| `draft_000435` | 150 | 0 | no |
| `draft_000435_p1mini` | 18 | 0 | no |

**Scope sweep (calibrated headers only):** no contamination on drafts checkable; in-place-era drafts unavailable locally, status unknown. On-disk drafts are copy-tree era (435 family only); no low-VYSKYP2P on calibrated FITS among those checked.

---

### 1.5 ù VERDICT: **A1 ù ANCHOR CLEAN**

**Evidence:**

1. Science subtract happened once on **`processed/lights/proc_*.fits`**, byte-identical to single recompute (0.0 ADU).
2. proc_ `VYSKYP2P` values are unimodal **first-pass gradient scale** (median ~137 ADU), not second-pass residual scale.
3. No low-VYSKYP2P population on any on-disk draft.
4. Calibrated FITS lack markers and retain full gradient ù consistent with **pre-in-place** architecture, **not** with double in-place subtract.

**Caveat (not contamination):** the R2 double-subtract defect applies to **post-SKIPPROC in-place** path on drafts re-run via MAKE MASTERSTAR today. The July-16 anchor used the **copy-tree** path; contamination check is for **single subtract on science pixels**, which passes.

**VL-ANCHOR-WCSINV:** not invalidated by this finding. Anchor fingerprints remain valid for the frozen proc-tree science.

---

## STOP GATE 1

**Posted. No code changes. Part 2 not started.**

---

## PART 2 -- Guard (Gate 1 cleared)

### 2.3 `apply_sky_surface` decision: **DELETED**

Removed the `apply_sky_surface` parameter from `_qc_enrich_calibrated_in_place` and the OSC caller (`pipeline.py` ~16893). It was dead code after T3 restore (2026-07-16): mono and OSC channel paths gate sky subtract on `preprocess_sky_surface_order > 0` only; mosaics remain excluded via `BAYERPAT` without `VY_CHANNEL`. The parameter was passed as `_ = apply_sky_surface` and never consulted.

### Guard test results (T1-T6)

All tests in `dev/tests/test_skysf_double_guard.py`. **Before guard:** T2 would double-subtract (508.97 ADU class defect); T3 would silently re-fit; T5/T6 N/A.

| Test | Before guard | After guard |
|------|--------------|-------------|
| T1 no marker -> subtract + headers | subtract (no guard) | **PASS** |
| T2 second pass same order -> skip, byte-identical | would re-subtract | **PASS** (skip_count=1) |
| T3 order mismatch -> abort | would re-subtract | **PASS** (`SkySurfaceOrderConflictError`, recal from raw) |
| T4 force_reapply -> second subtract + provenance | N/A | **PASS** |
| T5 legacy cal no markers -> subtract | subtract (correct) | **PASS** |
| T6 skip counter in run summary | log only | **PASS** (`preprocess_sky_summary_from_df` / job summary field) |

**Pytest:** `8 passed` in `dev/tests/test_skysf_double_guard.py` (+ related sky/OSC tests).

### Commits (pushed)

| Hash | Message |
|------|---------|
| `84174ae` | feat(skysf): guard in-place sky-surface subtract with VY_SKYSF headers |
| `46c075e` | fix(bench): post453 preprocess bench raises on zero frames |
| `671f0b2` | docs: SKYSF-DOUBLE guard decision and 508.97 ADU defect cost |
| `c5cf7c5` | refactor(skysf): wire skip counter to job summary; drop apply_sky_surface test arg |

**Push:** `git pull --rebase && git push` -> `main` at `c5cf7c5` (2026-07-30).

### Implementation notes

- Guard reads `VY_SKYSF`/`VYSKYORD` from the **frame being modified** (not a sibling proc_ copy).
- `508.97 ADU` bench metric = **one** double-subtract pass cost (star-mask non-idempotency); documented in JOURNAL + DECISIONS.
- Config: `preprocess_sky_surface_force_reapply` (default False).

---

## PART 3 -- Scope ledger (read-only)

### 3.1 In-place architecture commit

| Item | Value |
|------|-------|
| Commit | **`013cb0c`** |
| Date | **2026-07-22 14:39:13 +0200** |
| Subject | feat(skipproc)!: skip-only preprocess; `_qc_enrich_calibrated_in_place` writes VY_* on calibrated FITS |

**Drafts produced after `013cb0c`** (from `dev/results/context/deleted_drafts.md` + session close; not on this machine):

| Draft | Date | Field | On disk | In `C:\ASTRO\backups\` zips |
|-------|------|-------|---------|----------------------------|
| draft_000449 | 2026-07-22 | MN Boo | gone | no |
| draft_000450 | 2026-07-25 | BO CVn | gone | no |
| draft_000451 | 2026-07-27 | BO CVn | gone | no |
| draft_000452 | 2026-07-27 | BO CVn | gone | no |
| draft_000453 | 2026-07-27 | BO CVn | gone | no |
| draft_000454 | 2026-07-28 | BO CVn | gone | no |
| draft_000455 | 2026-07-28 | BO CVn | gone | no |
| draft_000448 | 2026-07-21 | MN Boo | gone | no (predates commit; transitional) |

On disk today: only **435-era copy-tree** drafts (`draft_000435`, `_snapshot_skysurface_20260716`, `_p1mini`). Backups contain **435 + 424 only** (no 450-455).

### 3.2 Header scan on backups

Partial read (20 calibrated FITS per zip, no full unpack):

| Backup zip | VYSKYP2P on calibrated | Low-VYSKYP2P population |
|------------|------------------------|-------------------------|
| `draft_000435_anchor_live_20260716.zip` | 0/20 | **no** |
| `draft_000435_snapshot_skysurface_20260716.zip` | 0/20 | **no** |

In-place-era drafts not in backups -> **not scanned**.

### 3.3 Cross-reference (ledger + July CURSOR_RESULT)

**`VYVAR_VALIDATION_LEDGER.json`:** no entries cite drafts 449-455. Anchor/science entries point at **draft_435** / **435_p1mini** (copy-tree era, checked clean in Part 1).

**July CURSOR_RESULT files referencing in-place-era drafts:**

| Source file | Draft(s) | Contamination risk |
|-------------|----------|-------------------|
| `CURSOR_RESULT_post451_remediation.md` | 451, 452 | C.4 reprocess metrics; 452 is bench reference for idempotency |
| `CURSOR_RESULT_draft451_analysis.md` | 451 | science tables if preprocess re-run |
| `CURSOR_RESULT_draft453_consolidation.md` | 452, 453 | timing/equivalence |
| `CURSOR_RESULT_draft454.md` | 454 | infolog / equivalence |
| `CURSOR_RESULT_post453.md`, `post453_fixes.md` | 452 | preprocess profile (508.97 ADU = **defect measurement**, not science output) |
| `CURSOR_RESULT_skysurface_regression.md` | 450 | DAO_ONLY % diagnostic |
| `CURSOR_RESULT_session_close.md` | 448-455 | cleanup manifest |
| `CURSOR_RESULT_phase0_*`, `dao_only_verify.md`, `masterstar_count_diag.md` | 450 | Phase-0 / DAO forensics |

Numbers explicitly from **double-preprocess defect path:** `508.97 ADU max_abs_diff` (452 cal vs re-preprocess in place) -- diagnostic only. Science photometry/LC SHA entries for 450-455: **status unknown** (drafts unavailable to re-check headers).

### 3.4 Status table

| draft | era | available? | checked? | verdict |
|-------|-----|------------|----------|---------|
| draft_000435 | copy-tree (pre-013cb0c) | disk + backup | yes (Part 1) | **clean** |
| draft_000435_snapshot_skysurface_20260716 | copy-tree | disk + backup | yes | **clean** |
| draft_000435_p1mini | copy-tree | disk | yes | **clean** |
| draft_000448 | transitional | gone | no | **unknown** |
| draft_000449 | in-place | gone | no | **unknown** |
| draft_000450 | in-place | gone | no | **unknown** |
| draft_000451 | in-place | gone | no | **unknown** |
| draft_000452 | in-place | gone | no | **unknown** |
| draft_000453 | in-place | gone | no | **unknown** |
| draft_000454 | in-place | gone | no | **unknown** |
| draft_000455 | in-place | gone | no | **unknown** |

**Ledger statement:** no contamination on drafts checkable; in-place-era drafts unavailable, status unknown.

---

## Files changed (Part 2)

- `src_py/pipeline.py`, `src_py/config.py`, `src_py/app.py`
- `dev/tests/test_skysf_double_guard.py` (new), `dev/tests/test_osc1_extraction.py`
- `dev/scripts/post453_preprocess_bench.py`
- `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_DECISIONS.md`
