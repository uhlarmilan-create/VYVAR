CURSOR RESULT - 2026-08-13 INV-CAL-01 P2 fix and push

What I did
Corrected INV-CAL-01 P2 to compare like-with-like processing stages on `calibrated/`.
Added stage helpers in `cal_diag.py`, validation harness `dev/tools/inv_cal01_validate.py`,
unit test, docs (spec S12.1-12.2, DECISIONS 11.3, ROADMAP INV-CAL-02). Revalidated P1-P3,
confirmed draft 510 photometry unchanged (read-only), `--fast` PASS, committed and pushed.

---

## 1. P2 rewrite

### 1.1 Choice

**Apply archived sky order to fresh pure cal before pixel compare.**

Alternatives rejected:
- Compare only `VY_SKYSF` absent frames - excludes 509/510 science products.
- Separate pure-cal reference tree - duplicates storage; archive is authoritative when staged.

Helper: `cal_diag.apply_calibrated_stage_for_compare` reads `VY_SKYSF`/`VYSKYORD` and applies
`_fit_subtract_preprocess_sky_surface` to recalibrated pure `(L-D)/F`.

**Restated prediction:** INV-CAL-01 does not change calibrated products; recalibration at the
**same recorded stage** as archive must be pixel-identical.

### 1.2 Predicate and result (pre-declared)

| Draft | Stage | Frames | Identical | max \|diff\| | Result |
|-------|-------|--------|-----------|------------|--------|
| 509 | SKYSF_2 | 150 | 150 | **0.0** | **PASS** |
| 510 | SKYSF_2 | 150 | 150 | **0.0** | **PASS** |

### 1.3 P3 re-report

| Quantity | Gate (master intercept) | Pixel bootstrap (physics memo) |
|----------|-------------------------|--------------------------------|
| P | **24.470589** | **24.547984 +- 0.011045** |
| sigma_p | **0.0** (degenerate: k?0, 2 master medians identical) | **0.011045** |
| Delta_meas | **73.5882** | - |
| Delta_pred | **73.4118** (gate) / **73.6440** (pixel P) | - |
| R | **101.7554** | - |
| Convention | SUM / DERIVED / PASS | - |

**Verdict:** **PASS** on gate criteria (Delta within 5%; SUM/DERIVED/PASS).

**Seven-sigma on P:** Spec anchor `24.548 +- 0.011` is the **pixel subsample bootstrap intercept**
(`tmp/_dark_binning_physics_measure.py` method). Gate Check P uses **stacked master medians**
(24.4706 at both 60 s and 120 s ? k=0 ? P=24.471). Difference **0.078 ADU** is estimator
method, not gate failure. **`+- 0.011` is not honest `sigma_p` for the gate path** - gate
reports `sigma_p=0` when exptime medians are identical. Honest gate uncertainty would require
pixel-level bootstrap in Check P (future work); convention resolution is unaffected.

---

## 2. Nothing moved

### 2.1 Draft 435

P1: **150/150 pixel-identical**, max diff **0.0** (pure stage; no `VY_SKYSF`).

### 2.2 Draft 510 photometry (read-only)

| Metric | Value |
|--------|------:|
| check_scatter | **0.008629278** |
| TRUST | **GREEN** |
| n_points | **134** |
| n_good_comp | **5** |

Source: `trust_1498613634033133184.json`, `photometry_summary.csv`. No re-run; archive untouched.

### 2.3 `--fast` (raw)

```
SESSION BASELINE CHECK (fast)
pytest                       PASS   1313 passed, 27 skipped
OVERALL: PASS
```

### 2.4 Archive FITS

No overwrite of `Archive/Drafts/draft_000509|510/calibrated/`.

---

## 3. Recorded learnings

- Spec S12.1-12.2, DECISIONS S11.3: mutable two-stage `calibrated/` hazard.
- ROADMAP **INV-CAL-02** opened (VY_CALSTAGE, pixel hash, QC alert, directory question).

---

## 4. Push

| Commit | Message |
|--------|---------|
| `f7eaadc` | feat(cal-diag): INV-CAL-01 v2 gate with stage-aware P2 validation |

`origin/main` HEAD: **`f7eaadc`**

`--fast` on final tree: **1313 passed**, 27 skipped, **OVERALL PASS**.

---

## End state

| Item | Status |
|------|--------|
| **P2** | **PASS** (stage-aware) |
| **P3** | **PASS** (Delta 5%; P methods differ by design) |
| **Draft 510 validated result** | **Untouched** |
| **INV-CAL-02 priority** | **HIGH** - provenance/stage integrity; prevents repeat P2/P-10 cost. Above generic checksum (broader, less urgent) and INV-DAG-01 (workflow friction). Below exposure-ramp (science ceiling for SAT-DIAG Tier 2/3). |

## Files changed

- `src_py/cal_diag.py` - stage helpers
- `dev/tools/inv_cal01_validate.py` - P1-P3 harness
- `dev/tests/test_cal_diag_v2_gate.py` - stage test
- `dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md`
- `docs/VYVAR_DECISIONS.md`, `ROADMAP.md`, `STATE.md`, `JOURNAL.md`
- `dev/results/CURSOR_RESULT_inv_cal_01_p2_push.md`
