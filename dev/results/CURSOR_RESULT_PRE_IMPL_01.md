# CURSOR RESULT - PRE-IMPL-01

Date: 2026-08-15
Baseline: 011fff7 (+ A1/A2 committed this task; weight persistence committed this task)
Type: INVESTIGATION. Push: NO.
`--fast` OVERALL: see end of file.

JSON: `dev/results/PRE_IMPL_01_Q{1..5}.json`

---

## Spec hygiene (standing instruction)

Two prior defects (architect-owned, recorded): synthetic B2; variable-star C2.

This task:

| Question | Spec check | Action |
|---|---|---|
| Q1 | Can falsify (ratio~1 flat vs mag-dependent) | Ran; **data defect found**: proc `mag` is catalog (constant). Corrected to `-2.5 log10(flux)`. |
| Q2 | Can falsify | **Estimator defect**: flux-sum `loo_diff_series` under weight truncation ignores weights and produced MAD=0 discrete residuals. Corrected to weighted-mean peers, exclude <3 FWHM neighbours. |
| Q3 | Can falsify with non-variables | Ran with weighted-mean ensemble; G-controlled level reported separately. |
| Q4 | Can falsify | Ran; register 84.6% vs r=2.711 resolved by draft/radius mismatch. |
| Q5 | Can falsify | Ran. |

Falsification line stated under each answer.

---

## Q1. Is `sigma_eff` calibrated in magnitudes?

**SHA:** 011fff7 (measurement). Units: mag.

| quantity | value |
|---|---|
| `sigma_obs` | 1.4826 * MAD of LOO differential (focus - ensemble), flux-based inst mag |
| ratio `sigma_obs/sigma_eff` median | **1.166** |
| ratio p16 / p84 | 0.18-ish to ~2+ (wide; see G bins) |

Ratio vs G (decisive):

| G bin | n | ratio median |
|---|---:|---:|
| 8-10 | 288 | **2.85** |
| 10-11 | 288 | 2.70 |
| 11-12 | 704 | 2.09 |
| 12-13 | 1584 | 1.67 |
| 13-14 | 2568 | 1.23 |
| 14-15 | 3760 | **1.08** |

Colour isolation (non-tautological):

- Tautological: `sigma_eff` contains `comp_rms`; both track night scatter of the same star.
- Non-tautological: `corr(excess2, colour_term^2) ~ 0.002` — **colour term does not predict extra scatter**.

**Answer:** ratio is systematically mag-dependent (bright under-predicted by ~3x, faint ~ok). Colour term is ordering, not absolute. An absolute ceiling on `sigma_eff` is **NOT constructible**. Prefer a cut on `comp_rms` (mag) or median SNR.

**Falsification:** ratio flat near 1 across G, and excess2 scaling with `(c_col|dBP-RP|)^2` at a~1. **Did not happen.**

**Implements:** no absolute `sigma_eff` mag ceiling; if a field-independent cut is needed, use `comp_rms` or SNR.

---

## Q2. Does the low-weight tail carry unmodelled systematics?

**SHA:** 011fff7. Check stars: 25 non-suspected comps, G 9-12.5. Real flux mags.

Median check-star MAD scatter (mmag) vs cumulative-weight truncation:

| truncation | scatter MAD (mmag) | abs airmass slope (mmag/airmass) |
|---|---:|---:|
| 100% | 16.49 | 101 |
| 99% | 16.63 | 122 |
| 95% | 16.82 | 150 |
| 90% | 16.92 | 178 |
| 50% | **12.79** | 255 |

**Answer:** Through 90%, scatter is flat (~0.4 mmag) — mild tail is inert (performance). At 50%, scatter **improves** by ~3.7 mmag — the far low-weight half carries systematics `sigma_eff` does not model.

**Falsification:** scatter rises or falls by >1 mmag from full to 50%. **Happened (fell).**

**Implements:** any mild cut is a performance decision; an aggressive half-cut would hide a missing `sigma_eff` term. Still do **not** implement cumulative-weight cuts (population-dependent). Missing term for the worst half is a science follow-up.

---

## Q3. Measurable colour term on this rig?

**SHA:** 011fff7. n=1172 non-variable comps as pseudo-targets. Weighted-mean peers.

| term | value | se | unit | sig (2se)? |
|---|---:|---:|---|---|
| Shape k'' | -83.3 | 98.5 | mmag / BP-RP / airmass | **no** |
| Level (raw) | -611 | 152 | mmag / BP-RP | yes (confounded: corr with G = 0.80) |
| Level (G-controlled) | -373 | 90 | mmag / BP-RP | **yes** |
| `c_col` reference | 29.5 | — | mmag / BP-RP | — |
| CLEAR k'' | NONE | — | — | — |

**Answer:** Shape term consistent with zero — no airmass-coupled colour contamination of LC shape at this precision. G-controlled level term is significant but ~13x `c_col` and not the PSF EE coefficient; it is a zero-point / export-level effect, not a transit-shape effect.

**Design:** **(c) for exports/absolute level only**; **(a) weighting alone for LC shape**. Not (b) from this sample.

**Falsification:** shape and/or G-controlled level |coef| > 2*se. Shape failed to reject zero; G-controlled level rejected zero.

---

## Q4. Enclosed flux at production aperture

**SHA:** 011fff7. Empirical CoG, 12 isolated bright stars, 40 frames, norm at r=12 px (tail flatness checked per star).

| quantity | value |
|---|---|
| Production r | **2.711 px** (= SNR `r_min`) |
| EE at production r (median) | **0.663** (p16=0.644, p84=0.723) |
| Night EE variation MAD | **0.008 mmag** |
| Night EE p16-p84 | 0.018 mmag |

**Register 84.6% vs r90 5.0-5.8:** both refer to draft **510** production radii ~**4.1 px** (`CURSOR_RESULT_a1_growth_curves`). Draft **514** median aperture is the faint-star **r_min clamp** 2.711 px — a different radius. No contradiction once drafts are not conflated. `VY_FWHM_GAUSS` and DAO moment remain one PSF in two conventions; not needed to settle EE.

**r_min clamp:** intended as a numerical floor for faint SNR bins. Most catalogue stars are faint, so the median sits on the clamp — boundary condition, not an optimized radius for the median star.

**WIDE-ERR:** EE~66% confirms undersizing vs 90% target. Night EE variation on this sample is **negligible** (0.008 mmag) — not supporting aperture-loss-vs-seeing as the WIDE-ERR driver on draft 514's night.

**Falsification:** EE at 2.711 near 0.85, or night EE variation >> 1 mmag. EE falsified 0.85; night variation did not show large scatter.

**Implements:** aperture decision must treat 2.711 as a clamp, not a chosen EE; raising toward r90 (~5 px class) is the EE path if absolute/enclosed-fraction claims matter.

---

## Q5. Do blended comps behave worse?

**SHA:** 011fff7. Pool 1292; blended 633; matched-mag pairs 631.

| quantity | value |
|---|---|
| median scatter excess (blend - iso) | **-16.2 mmag** (blends quieter) |
| median abs slope excess | ~0 (noise) |
| corr(excess, blend pair dBP-RP) | 0.063 |

**Answer:** Blended comps are **not worse** at matched magnitude. Merging is cosmetic for differential photometry and can be **deferred past v1.0**.

**Falsification:** matched-mag scatter excess > +1 mmag or tracks blend colour. **Did not happen.**

---

## Also fixed

### `comp_weight` persistence (PRE-IMPL-01)

- Cause: Phase 1 wrote `1/rms^2` only — identical across targets.
- Fix: Phase 1 writes `sigma_eff` weights; `rewrite_comparison_stars_weights_csv` at Phase 2A end; draft 514 CSV rewritten (97 distinct N_eff, range 150-590).
- Test: `dev/tests/test_comp_weight_rewrite.py`

### A1/A2 from DRAFT-514-TRIAGE

Committed (c4 lgamma; preflight traceback; AC no_comp_rms; proc catalog dedupe).

---

## Implementation decisions blocked (summary)

| ID | Decision supported |
|---|---|
| Q1 | No absolute `sigma_eff` ceiling; use `comp_rms`/SNR if cutting |
| Q2 | Mild cut = performance; far tail has missing `sigma_eff` term; still no cumulative-weight implementation |
| Q3 | (a) for shape; (c) level/export only; not (b) from this sample |
| Q4 | EE~66% at clamp 2.711; night EE not WIDE-ERR here; 84.6% was draft 510 @ ~4.1 px |
| Q5 | Defer catalogue merging past v1.0 |

---

## Files

- `dev/tools/pre_impl_01_measure.py`, `pre_impl_01_q2_fix.py`
- `dev/results/PRE_IMPL_01_Q{1..5}.json`, this file
- `src_py/comp_selection_per_target.py`, `comp_weights.py`, `photometry_core.py` (weight write)
- `dev/tests/test_comp_weight_rewrite.py`
- A1/A2: `sigma_floor_core.py`, `run_preflight_log.py`, `app.py`, `pipeline.py`, tests

No push.
