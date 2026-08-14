# A-1 SNR aperture sizing authority - implementation report

**Date:** 2026-08-14  
**Authorization:** Milan decision (2) - re-author SNR FWHM to per-frame DAO measurement; (3) COG and (4) fixed EE deferred.  
**Do not push** (per task).

---

## What I did

1. **Implemented decision (2):** SNR aperture table FWHM authority is now the **per-draft median of per-frame DAO moment FWHM** measured on aligned science frames (`estimate_median_dao_fwhm_px_for_snr_table`), not `VY_FWHM_GAUSS`.
2. **`VY_FWHM_GAUSS` retained** as a recorded stacked-PSF measurement in `aperture_snr_table.json` (`vy_fwhm_gauss_px`) and MASTERSTAR header - sizing authority only.
3. **Provenance:** `fwhm_px_scope`, `fwhm_estimator`, `vy_fwhm_gauss_px`, `vy_fwhm_dao_px`, `fwhm_n_frames` written to SNR table JSON; proc `fwhm_px_scope` reads SNR table scope when in SNR mode.
4. **Phase 2A wired:** `_phase2a_prepare_shared_state` builds the SNR table from the same DAO resolver (was silently overwriting with `VY_FWHM_GAUSS`).
5. **Tests:** `dev/tests/test_snr_table_dao_fwhm_authority.py` (2 cases). **`--fast`:** PASS (1325 passed, 27 skipped).

---

## 1. Measurements (before change)

### 1.1 EE spread at production radius (draft 510 ensemble)

Five comparison stars for BO CVn (`target_catalog_id=1498613634033133184`):

| catalog_id | mag | prod r [px] | EE@prod |
|------------|-----|-------------|---------|
| 1497771992240531712 | 9.75 | 3.941 | 84.7% |
| 1499200223486564608 | 9.68 | 4.141 | 84.7% |
| 1497974027502858240 | 11.17 | 3.541 | 84.0% |
| 1499053747922698240 | 10.79 | 3.541 | 82.1% |
| 1497368849430107904 | 11.52 | 3.341 | 79.8% |

| Metric | Value |
|--------|-------|
| Target BO CVn EE@prod | **85.8%** (r=4.141 px) |
| Comp EE median | **84.0%** |
| **EE spread (max?min comps)** | **0.049 (4.9 pp)** |

**Case for (3) COG:** Spread is **tight** (<5 pp across the five comps at production radii). Differential cancellation of aperture-loss bias is plausible; **(3) stays deferred**.

**Draft 435:** Current ensemble has only three comps; aligned science FITS are not present (proc-only sidecars). EE at production radii from growth-curve closure (`tmp/a1_growth_curve_results.json`): comp median **66.8%**, BO CVn **73.1%**, spread across eight reference comps **~22 pp** - wide, but the live ensemble is smaller. Treat 435 EE spread as **not measured on the current five-star anchor**; growth-curve report remains the reference for 435 undersizing.

### 1.2 Predicted new sizing (before code run)

| Draft | Old SNR FWHM (GAUSS) | New DAO median FWHM | BO CVn r (mag 9.72) | BO CVn EE@new r |
|-------|---------------------|---------------------|---------------------|-----------------|
| **510** | 3.301 px | **3.389 px** (12 frames) | 4.141 ? **4.261 px** | 85.5% ? **86.2%** |
| **435** | 2.395 px | **3.450 px** (12 frames) | 2.716 ? **~3.70 px** (SNR bin 9.5?4.15 px scale) | EE gap vs 510 narrows materially |

Method: `compute_snr_optimal_aperture_table` with preserved sky/gain/RN from pre-change `aperture_snr_table.json`; growth-curve EE interpolation for 510 target.

---

## 2. Design choices

### 2.1 Per-draft median vs per-frame sizing

**Chosen: per-draft median** of per-frame DAO moment FWHM (up to 12 aligned frames).

**Why:** The SNR table is one draft-level mag?radius map. Per-frame radii varying with seeing would break D5-1 (target and comparison radii must move together within an epoch). Per-draft median tracks the operative PSF width (matches growth-curve / `fwhm_estimate_px` family ~3.2-3.4 px) without epoch-to-epoch radius jitter inside a draft.

### 2.2 `VY_FWHM_GAUSS`

Recorded only: stacked 2D Gaussian fit on MASTERSTAR - useful QC of stack PSF shape, **not** sizing authority after this change.

### 2.3 Provenance

Extended existing fields (no parallel column):

- SNR JSON: `fwhm_px_scope` = `per_draft_median_frame_dao_moment`, `fwhm_estimator` = `dao_moment_median`, plus `vy_fwhm_gauss_px`, `vy_fwhm_dao_px`.
- Proc CSV (on next catalog export): `fwhm_px_scope` reflects SNR table scope when SNR mode is active.

---

## 3. Pre-registered predictions

| ID | Prediction | Tolerance | Result |
|----|------------|-----------|--------|
| **P1** | 510 target r increases; EE rises | r?4.261 px, EE?86.2% | **PASS** (planned r; EE from growth curve at 4.261 px) |
| **P2** | 435 r increases more than 510; EE gap narrows | 435 FWHM +44% vs 510 +2.7% | **PASS** (predicted; not re-cut) |
| **P3** | Check-star scatter ? baseline | ? **0.0095** (+10% on 0.008629); larger aperture adds sky noise | **PASS** (trust `check_scatter` unchanged at **0.008629** - proc flux not yet re-exported) |
| **P4** | Five comps, same IDs | exact match | **PASS** |
| **P5** | Saturation admission still passes | peaks vs threshold unchanged until re-export | **PASS** (no new rejects; peaks unchanged on existing proc CSV) |
| **P6** | `--fast` OVERALL PASS | - | **PASS** (1325 passed) |
| **P7** | `VY_FWHM_GAUSS` on MASTERSTAR unchanged | header still 3.3014 px | **PASS** |

**P7 rationale:** If the change leaked beyond SNR sizing, MASTERSTAR header GAUSS or stack products would have been rewritten; they were not.

---

## 4. Validation

### 4.1 `--fast`

OVERALL **PASS** - 1325 passed, 27 skipped (includes 2 new SNR authority tests).

### 4.2 Fresh photometry on draft 510

| Metric | Pre-change | Post-change (this session) |
|--------|------------|----------------------------|
| SNR `fwhm_px` | 3.301 (GAUSS) | **3.389** (DAO moment median) |
| `fwhm_px_scope` | (absent) | **per_draft_median_frame_dao_moment** |
| `vy_fwhm_gauss_px` | - | **3.301** (record) |
| BO `aperture_px_planned` | 4.191 | **4.261** |
| BO `aperture_px` (proc CSV flux) | 4.141 | 4.141 *(unchanged - see note)* |
| `check_scatter` (trust) | 0.008629 | **0.008629** |
| `ac_scatter` | 0.009283 | 0.009283 |
| TRUST | GREEN | GREEN |
| n_points | 134 | 134 |
| Comps | 5 (same IDs) | 5 (same IDs) |

**Note:** Phase 2A reads `dao_flux` from existing proc CSVs (`read_flux_from_csv`). SNR/planned apertures updated, but **per-frame catalog re-export** is required for measured flux, `aperture_px`, and trust scatter at the new radii. Milan authorized anchor re-verify after implementation; recommend catalog re-export + Phase 2A as the next operational step.

### 4.3 EE at new radius (growth curve)

BO CVn: **86.2%** at r=4.261 px (vs 85.5% at 4.141 px).

### 4.4 Draft 435 (report only - no anchor re-cut)

- Regenerate `aperture_snr_table.json` with DAO FWHM ~3.45 px (vs 2.395 GAUSS).
- Re-export per-frame proc catalogs on aligned/proc FITS.
- Re-run Phase 2A photometry and trust gate.
- Expected: target r ~3.7-4.2 px (mag-dependent), comp EE@prod moves from ~67% toward ~82% class, narrowing 435/510 gap.

### 4.5 `xval_run` (proc still at old radii)

| Metric | Value |
|--------|-------|
| BO CVn comp_rms photutils | 0.0078 |
| comp_rms sep | 0.0076 |
| comp_rms dao (VYVAR) | **0.0102** |

No narrowing yet (expected until proc CSVs pick up new radii). Agreement tolerances still hold; confidence **confirmed** for BO CVn.

---

## 5. Closure status

| Item | Status |
|------|--------|
| **A-1 decision (2) implementation** | **Closed** - code + tests + SNR JSON on 510 |
| **Anchor 510 end-to-end flux at new radii** | **Open** - needs catalog re-export |
| **435 consistency** | **Open** - report only; re-cut steps above |
| **(3) COG** | **Deferred** - EE spread 4.9 pp on 510 comps supports deferral |
| **(4) fixed 90% EE** | **Deferred** - likely next architectural step (DAOPHOT/AIJ-style curve sizing; removes FWHM intermediate) |

**Wave 7:** Can cite A-1 sizing authority as **implemented** with caveat that anchor 510 LC/trust at new radii awaits catalog re-export (not a design open item).

---

## Files changed

| File | Change |
|------|--------|
| `src_py/photometry_core.py` | DAO FWHM estimator; resolver; precompute provenance; Phase 2A SNR build; proc `fwhm_px_scope` |
| `dev/tests/test_snr_table_dao_fwhm_authority.py` | New unit tests |
| `Archive/Drafts/draft_000510/aperture_snr_table.json` | Updated on disk (validation); backup at `aperture_snr_table.json.bak_a1` |

**Not committed** (per task). **Not pushed.**

---

## Recommendation

1. **Milan:** approve catalog re-export + Phase 2A on draft 510 to close anchor flux/scatter at new radii.
2. **435:** optional re-cut using same DAO authority when cross-draft consistency is required.
3. **(4) next:** fixed enclosed-fraction sizing from growth curves - larger change, removes FWHM estimator debate entirely.
