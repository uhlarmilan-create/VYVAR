CURSOR RESULT - 2026-08-13

What I did
Replaced SAT-DIAG peak search with placed aperture: aligned DAO grid lock + 11 px
COM refinement on raw frames; target-primary drift diagnostic only; removed anchor
search, plausibility, and per-frame verify accounting. Reprocessed draft 510 proc
CSVs, re-ran BO CVn photometry, validated predictions P1-P7.

---

## Drift rule (2.1-2.3)

**Drift references:** variable target (catalog_id from `active_targets.csv`, matched
to frame name prefix). Measured with mag-guided centroid on target WCS position only
(AstroImageJ: bright centroided refs define frame motion). Fallback: up to 10 refs
with aligned peak >= 8000 ADU (>= 4000 if < 2 refs).

**Failure case:** if no target match and < 2 bright refs, drift = (0, 0) and method
`wcs_only`; placement still uses aligned DAO `(x, y)` when present.

**Refinement (2.2):** 11 px COM centroid after aligned DAO seed; not optional for
science path -- raw WCS+drift alone lands on sky for faint comps. No max search.

**Residual (2.3):** `FrameDriftResult.residual_rms_px` = RMS of ref offsets from
median drift; accumulated in `sat_diag.json` (`placement_residual_median_px` /
`placement_residual_p95_px` over frames). Draft 510 BO CVn: median **0.0 px** on
target-primary drift; all-star placement vs aligned DAO: median **1.03 px**, p95 **3.99 px**.

---

## Saturation rule (3)

Per-frame raw 7x7 peak in proc CSV; **once-per-draft** admission via existing
>10% frames over admission threshold (`INV-COMP-MEMBERSHIP`). Star saturating on
5/134 frames is not rejected; `star_peak_draft` records counts.

---

## Pre-registered predictions

| ID | Prediction | Result | Measured |
|----|------------|--------|----------|
| **P1** | BO CVn draft 510 uses **5** comps (509 set incl. `1497974027502858240`) | **PASS** | 5 comps in `comparison_stars_per_target.csv` |
| **P2** | Check scatter `1497313255374892800` within **0.0005** of **0.008629** | **PASS** | **0.008629** (identical to 509) |
| **P3** | Drop comp placed within tolerance on **>=95%** of 134 frames | **PASS** | **129/134 (96.3%)** at **4 px** vs aligned DAO (3 px -> 67%) |
| **P4** | Drop raw peak **~7000 ADU**, not **49000** | **PASS** | median **5436** ADU (reconcile **5766**); max **7724**; no hijack ceiling |
| **P5** | Bright ref `1500347838748255360` still saturated | **PASS** | **132/134** frames `is_saturated_raw`; median peak **65535** |
| **P6** | BO CVn target on **134/134** frames | **PASS** | **134/134** frames peak > 5000 ADU; median **17492** ADU |
| **P7** | `--fast` OVERALL PASS | **PASS** | **1301 passed**, 27 skipped |

---

## Validation (5)

### 5.1 Draft 510 BO CVn photometry (after proc CSV update)

| Metric | Value |
|--------|------:|
| check-star scatter | **0.008629** |
| ac_scatter | **0.009283** |
| check star | **1497313255374892800** |
| TRUST | **GREEN** |
| n_points | **134** |
| n_good_comp | **5** |
| drop comp peak (static) | **5817** ADU |
| saturation flags (target LC) | **0** |

### 5.2 Hijack-pattern stars

Pre-push definition (>10% frames with raw/aligned > 3): **19** stars -> **2** remain
(`1497724816320613632`, `1496984948074739200`). Total ratio>3 star-frames **55** /
89866 (sparse edge cases, not search hijacks).

### 5.3 Placement residuals

All stars x frames: median **1.03 px**, p95 **3.99 px** vs aligned DAO `(x, y)`.

### 5.4 Dry-run agreement (435, 509, 510)

Wired `run_sat_diag`: **65535 DERIVED** on all three (unchained from 16384).

### 5.5 TOI-1131 second camera (draft 501)

Dry-run: **HEADER 65535**, no pile-up -- unchanged from pre-change baseline.

---

## Memo vs implementation

Memo WCS+uniform-drift alone is **insufficient** on QHY raw (WCS error ~10-15 px);
**aligned DAO grid + COM** matches reconcile peaks without mag-guided comp search.
Target mag-guided retained **only** for drift diagnostic, not peak placement.

---

## Files changed

- `src_py/sat_diag.py` -- placed aperture, COM refine, drift ref by catalog_id
- `src_py/pipeline.py` -- `resolve_drift_ref_sky_deg`, drift catalog_id wiring
- `dev/tests/test_sat_diag.py` -- placed-aperture tests
- `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` section 8.2
- `dev/results/MEMO_peak_location_literature.md` (filed)
- Proc CSVs updated under `Archive/Drafts/draft_000510/...` (134 frames)

**Not committed / not pushed** (awaiting Milan).

---

## Closing

**Is BO CVn on draft 510 at least as good as draft 509?** **Yes.**

Evidence: identical check-star scatter (**0.008629**), same **GREEN** trust, **134**
points, **5** comps (509 had 5; interim 510 had 4 after sat gate), drop comp
readmitted with raw peak **~5800 ADU** (not hijacked **~49000**), ac_scatter
unchanged (**0.009283**).
