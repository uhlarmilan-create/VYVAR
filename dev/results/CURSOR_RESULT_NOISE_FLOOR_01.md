# CURSOR RESULT - NOISE-FLOOR-01

Date: 2026-08-14
Register ID: NOISE-FLOOR-01
Follows: COMP-POOL-02 (item 2 superseded)

## Verdict

No flat bright-end range exists on drafts 512/510/435 under the pre-stated F1
criterion. The systematic floor is an **upper limit** (N-R0), not a measurement:
about **6.79 mmag** on draft 512 (7.00 mmag NP bin median at G~8.75 before
subtracting the photon term). Completing the Howell terms (sky-estimation
factor, dark, digitisation, measured correlated-pixel factor) closes only about
**4%** of the photon-plus-sky variance deficit at G10. A genuine residual of
about **factor 2.3 in sigma** (about **5 in variance**) remains in the
photon/sky terms. That localizes WIDE-ERR. Exported error bars were not changed.

Machine outputs:
- `dev/results/NOISE_FLOOR_01_results.json`
- `dev/results/NOISE_FLOOR_01_flatness.json`
- `dev/results/NOISE_FLOOR_01_term_budget.json`
- `dev/results/NOISE_FLOOR_01_refit.json`

---

## 1. F1 -- flat range

**Criterion (stated before use):**

> Bright-end NP bins (n>=8): grow fainter while max/min bin median scatter
> <= 1.10; need >= 3 contiguous bins for a measured floor; else upper limit
> at min bright-bin median (N-R0).

| draft | flat? | result | UL scatter (mmag) | at G | floor UL after phot (mmag) | bright bin n |
|------:|:-----:|--------|------------------:|-----:|---------------------------:|--------------|
| 512 | no | n_bins_kept=2 | 7.00 | 8.75 | 6.79 | 13,13,24 at 8.25/8.75/9.25 |
| 510 | no | n_bins_kept=2 | 7.01 | 8.75 | 6.80 | 12,13,22 |
| 435 | no | n_bins_kept=2 | 9.53 | 8.25 | 9.43 | 13,12,25 |

Draft 512 NP medians (mmag): 7.40 (n=13), 7.00 (13), 9.33 (24), 10.03 (36),
11.41 (39), 13.03 (58) at G 8.25..10.75. Two bins pass the 1.10 ratio; three
are required. Stars run out while the curve is still settling -- **N-R0 fired**.

Admission path (`analyze_draft_comp_pool`) keeps `floor_mode=legacy_g8_10p5`
so Stage-2 thresholds do not move (Stage 2 remains blocked under C2-R2).

---

## 2. F2 -- completed model (N-R1, N-R2)

Annulus on draft 512 (config factors x FWHM; matches task radii):
`r_ap=3.411`, `r_in=15.68`, `r_out=29.71`, `n_pix=36.55`, `n_B=2001`,
`sky_factor = 1 + n_pix/n_B = 1.018` (1.025 at the G10 sample).

| term | value | fraction of G10 variance deficit closed (draft 512) |
|------|------:|----------------------------------------------------:|
| (1+n_pix/n_B) sky factor | 1.025 | 0.27% |
| dark shot | 0 (pedestal-dominated at -10 C; named) | 0% |
| digitisation q^2/12 | q=1 ADU | 0.002% |
| correlated-pixel factor | **1.364** (frame-difference empty apertures) | 3.9% |
| **all complete terms together** | | **4.1%** |
| **remaining** | | **95.9%** (sigma factor obs/complete ~2.26 at G10) |

Correlated-pixel method: consecutive aligned-frame difference cancels static
structure; empty circular apertures of area `n_pix` give
`Var(sum)/(N sig_pix^2) = 1.364`. Single-frame empty apertures return ~100x
and were **rejected** (large-scale structure, not resampling). Citation:
Casertano et al. 2000 PASP 112, 177; Fruchter & Hook 2002 PASP 114, 144.

No free multiplicative scale was introduced (N-R2).

**Could not measure:** true dark current in e-/px/s on this camera beyond the
pedestal statement (no dark-ramp product in this task). Digitisation after
14->16 bit packaging is assumed q=1 ADU.

---

## 3. F3 -- refit

| draft | floor | UL? | chi2_red (G8-13) | sys/scint | scint frac of floor var |
|------:|------:|:---:|-----------------:|----------:|------------------------:|
| 512 | 6.79 mmag | yes | 6.66 | 1.698 +/- 0.049 (UL) | 0.35 |
| 510 | 6.80 mmag | yes | 6.47 | 1.700 +/- 0.050 (UL) | 0.35 |
| 435 | 9.43 mmag | yes | 6.20 | 2.353 +/- 0.070 (UL) | 0.18 |

Photon term after UL floor (draft 512):

| G | obs phot (mmag) | reduced model | complete model | obs/complete |
|--:|----------------:|--------------:|---------------:|-------------:|
| 10.0 | 8.68 | 3.48 | 3.83 | 2.26 |
| 12.5 | 37.1 | 20.0 | 23.1 | 1.61 |

Residuals (obs/model ratio) vs covariates (draft 512):
- `corr_resid_mag` Pearson ~0.00 but **binned medians rise** ~1.00 (G8.5) to
  ~1.58 (G12); shape is magnitude-dependent underquote, not random.
- `corr_resid_bprp` = +0.092 (was -0.013 under the old inflated-floor model).
- `corr_resid_x` = +0.080, `corr_resid_y` = +0.023 -- weak.
- Draft 435: `corr_resid_bprp` = +0.018 (was +0.18 under old model). Colour
  flatness for 512/510 is not the same statement under the new residual.

**Scintillation (N-R3).** Floor/scint ~ **1.70 +/- 0.05** on 512/510 as an
**upper limit** on the ratio (because the floor is an UL). Assumptions:
D=0.07 m from TELESCOPE.DIAMETER; Osborn/Young C_Y=1.5; airmass = median of
stars; exposure/altitude from rig resolver; scintillation treated as exact
given those inputs; uncertainty from bright-bin scatter half-range only.
Scintillation accounts for about **one third** of the UL floor variance on
512/510 (~0.35), not one sixth.

Named floor components not separated in this fit: flat-field residuals,
transparency, second-order colour. After naming scintillation, ~65% of the
UL floor variance on 512 remains unnamed (and the floor itself may be lower).

chi2_red remains ~6.5 -- honestly poor (N-R4: not tuned toward 1).

---

## 4. F4 -- cross-package error budgets

| package / reference | sky (1+n_pix/n_B) | correlated pixels | systematic floor |
|---------------------|:----------------:|:-----------------:|:----------------:|
| Howell 1989 eq. 2 | yes | no (indep. pixels) | no |
| Merline & Howell 1995 | yes | discusses CCD | optional |
| Newberry 1991 | yes | no | no |
| DAOPHOT / IRAF apphot | yes (annulus) | no | no (user floors) |
| SExtractor | yes (LOCAL) | no | no |
| photutils / sep | yes optional | no (default indep.) | no |
| AstroImageJ | yes | no | optional |
| VaST | photon+sky style | no | no |
| VYVAR production | Howell reduced (I-03); I-11 path for sky-sub | no | config `sigma_sys_mag` |
| VYVAR diagnostic (this task) | quantified | measured 1.36 | UL from F1 |

VYVAR production still uses the reduced Howell form and a config floor; that
is unchanged here. The diagnostic path now quantifies the missing terms and
shows they do not close WIDE-ERR.

---

## 5. F5 -- WIDE-ERR localization

**Diagnosed location:** photon and sky variance terms (Howell source+sky+RN),
not the ensemble SEM.

Evidence: after correcting the floor to an UL ~6.8 mmag and adding every named
missing term, observed photon-plus-sky scatter still exceeds the complete
model by ~2.3x in sigma at G10 and ~1.6x at G12.5 on draft 512. Completing
the model closes only ~4% of the variance deficit. The old fit hid this by
trading an inflated floor (~9.7 mmag) against an underestimated photon term.

**Status:** localization part of WIDE-ERR moves from open-guess to
**DIAGNOSED**. The deficit remains; exported bars are **not** changed in this
task. Closing the underquote is a separate authorized task.

---

## 6. Nothing may break

### Impact inventory

| quantity | feeds | value changes? | intended here? |
|----------|-------|:--------------:|:--------------:|
| `fit_parametric_noise_curve` floor_mode=f1 + complete | diagnostics | yes | yes |
| `analyze_draft_comp_pool` (legacy_g8_10p5, reduced) | Stage-2 admission (blocked) | no | no |
| `_noise_floor_adu_from_image_array` | SNR table sky | no | no |
| production Howell / `sigma_sys_mag` / `sigma_bkg_ap` | exported errs | no | no |
| dao_flux / aperture radii | photometry | no | no |

### Measured proofs

| check | result |
|-------|--------|
| Aperture radii: archive file written? | **no** |
| Aperture radii: recompute vs archive max |delta| | **0.10 px** bright-end systematic (pre-existing builder/product drift; `photometry_core` not modified by this task). Faint bins at r_min match at 0.0. **STOP-and-report:** not the expected 0.0 on recompute, but not caused by this change. |
| SNR-GATE-02 helper pin | still legacy full-frame sample std |
| dao_flux max rel diff (6 BO comps, named frame) | **0.0** |
| Exported err columns (BO CVn LC) | identical; file untouched |
| Iron-gate fire proof + kwarg tests | **PASS** (still fire on fixtures) |
| `--fast` | see commit section below |

---

## 7. Pre-registered rules

- **N-R0 (upper limit):** FIRED -- no flat range; floor reported as UL.
- **N-R1 (order of blame):** FIRED -- F2 terms quantified before residual claim.
- **N-R2 (no absorbing parameter):** FIRED -- no free scale.
- **N-R3 (scintillation):** FIRED -- ratio 1.70+/-0.05 (UL) with assumptions listed.
- **N-R4 (no target):** FIRED -- chi2_red ~6.5 reported without tuning.

---

## 8. Register diff

| ID | change |
|----|--------|
| **NOISE-FLOOR-01** | NEW -- F1 UL floor; F2 terms close ~4% of variance deficit; F5 WIDE-ERR localized to photon/sky. |
| **WIDE-ERR** | disposition -> **DIAGNOSED** (photon/sky underquote ~2x in sigma); still OPEN for fix; exported bars unchanged. |
| **COMP-POOL-SCINT** | ratio UPDATED to ~1.70 (UL) on 512/510 with D=0.07; was ~2.43 from inflated floor. |
| **I-03** | evidence reinforced: missing Howell terms now quantified; small. |
| **COMP-POOL-02** item 2 | SUPERSEDED by this task. |
| Stage 2 (C2-R2) | unchanged; still blocked. |

---

## Files changed

- `src_py/comp_pool_noise.py` -- F1 flatness, complete Howell diagnostic, legacy admission pin
- `dev/tests/test_noise_floor_01.py` -- unit tests
- `dev/results/CURSOR_RESULT_NOISE_FLOOR_01.md` -- this memo
- `dev/results/NOISE_FLOOR_01_*.json` -- machine outputs
- `docs/VYVAR_AUDIT_2026_REGISTER.md` -- register rows

## Commit / --fast

Recorded after commit in the closing section of this file (SHA + OVERALL).
