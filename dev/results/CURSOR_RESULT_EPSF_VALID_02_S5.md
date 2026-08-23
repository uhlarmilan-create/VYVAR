CURSOR RESULT - 2026-08-22T17:15:00Z (EPSF-VALID-02 S5 Part B + Part D minimum)

What I did
Ran sandbox-only measurements on draft 516 (read proc CSVs allowed; no production writes, no model
swap, no science-path code changes). Harness: `dev/sandbox/epsf_valid_02_s5_measure.py`. Artifacts
under `dev/results/context/session_20260822_epsf_valid_02_s5/`.

HEAD: `2ba3d58`. Gate: no code change in this task; `--fast` not re-run (unchanged from S1ùS4 PASS).

---

## Part B ù written verdicts

### B1 ù Contamination mechanism (production 1475-star model)

**Verdict: defect (correct root cause identified)**

| Model | n_stars_used | psf_dao_ratio (median) | chi2 (median) | Source |
|-------|--------------|------------------------|---------------|--------|
| Production (platesolve) | 1475 | ~1.33 | ~22 | R4 + S5 proc CSV aggregate |
| Gated sandbox (67) | 67 | ~0.997 | ~1.5 | R4 `r4_split_half_summary.json` |

**Production build funnel (reconstructed, Part C gates disabled):**

| Stage | n |
|-------|---|
| csv quality pool | 2264 |
| after CSV join | 2243 |
| after isolation | 1506 |
| extract_stars (final) | **1475** |

**Gated funnel (Part C active, F4):** 2264 ? 68 science-scope ? 67 isolated ? 67 built.

**Violation census ù 1408 production-only stars (1475 ? 67 gated overlap):**

| Part C gate (primary) | count |
|-----------------------|------:|
| out_of_science_scope | 1408 |
| bad_source_state (also flagged) | 1 |
| variable / saturated / edge | 0 |

All 1408 excess stars are field comparison stars outside the 68-member ePSF science set
(`build_epsf_science_set`: 265 targets + 68 per-target comps). They passed legacy quality +
isolation but were never science-scoped. The production build (preùPart C, MS-SOURCES-RETIRE pool)
stacked ~1408 heterogeneous field PSFs into the ePSF.

**Causal chain ? ratio ~1.33 and chi2 ~22**

1. **Biased PSF shape/flux scale:** Anderson & King (2000) and Stetson (1987) require a homogeneous,
   isolated PSF-star sample matched to science stars. Field-wide comps include slightly different
   atmospheric focus, centroid noise, and subtle profile mismatch ? ePSF sum/normalization biased low
   (PSF flux systematically low vs DAO aperture).
2. **Poor fits on science stars:** Misfit wings/background in the spatially variable PSF inflate
   reduced chi2 (median ~22 vs ~1.5 gated).
3. **AC cannot fully fix model error:** Frame-level AC (`psf_ac_factor` ~0.53 on production proc
   CSVs) scales PSF flux toward DAO after the fit, but a wrong PSF shape leaves per-star residuals
   high even after median ratio correction.

**Cutout PNGs (gate labelled):**

- `dev/results/context/session_20260822_epsf_valid_02_s5/b1_cutout_1_out_of_science_scope.png`
- `dev/results/context/session_20260822_epsf_valid_02_s5/b1_cutout_2_out_of_science_scope.png`
- `dev/results/context/session_20260822_epsf_valid_02_s5/b1_cutout_3_out_of_science_scope.png`
- `dev/results/context/session_20260822_epsf_valid_02_s5/b1_cutout_4_out_of_science_scope.png`
- `dev/results/context/session_20260822_epsf_valid_02_s5/b1_cutout_5_bad_source_state.png`

Supporting CSV: `b1_production_only_violations.csv`, summary `b1_violation_summary.json`.

---

### B2 ù Cutout background vs I-11

**Verdict: correct (with documented double-subtraction caveat on already sky-subtracted frames)**

**What is subtracted:** In `_epsf_prepare_stars`, each 17ù17 cutout uses a **2-pixel border median**
(`border[2:-2,2:-2]` excluded); median subtracted before `EPSFBuilder` (`psf_photometry.py` ~1283ù1323).
Comment cites Anderson & King (2000) practice: local sky must be removed before PSF stacking because
**photutils `EPSFBuilder` does not subtract sky** (confirmed photutils 3.0.0 ù no sky step in
`build_epsf` / `_build_epsf_step`).

**photutils normalization (3.0.0 `epsf_builder.py` `_normalize_epsf`):** ePSF array scaled so
`sum(epsf_data) = prod(oversampling)` (for osamp=2, sum=4). Ensures `ImagePSF` with flux=1.0 is a
proper unit-normalized PSF (documented in method Notes, lines 1360ù1394). VYVAR meta records
`epsf_sum_native=1.0`, `epsf_norm_factor=1.0`.

**Validity on sky-subtracted MASTERSTAR (I-11):** MASTERSTAR / light frames are already sky-surface
subtracted. Border median on such frames estimates **residual local bias** (detrending residuals,
flat errors, faint extended wings), not full night sky. This is consistent with Anderson & King (2000)
ùevaluate and subtract sky from each PSF star cutoutù when the frame is already near zero mean ù the
border median targets **local pedestal**, not re-estimating the full sky surface. Residual risk: on
crowded border pixels, median can be biased high/low (documented JOURNAL crowded-field note).

**Zero-background assumption:** After border subtraction, cutouts are nominally zero-centered;
`EPSFBuilder` normalization forces unit flux sum ù consistent with AK00 zero-sky PSF stars.

---

### B3 ù Pixel weighting / variance model in fit

**Verdict: needs-measurement for CCD-equivalent full variance; current model is deliberate sky-only**

**EPSFBuilder (build):** `NDData(data)` only ù **no uncertainty** passed to `extract_stars`
(`psf_photometry.py` ~1268ù1298). photutils 3.0.0 `epsf_stars.py`: when uncertainty absent,
`EPSFStar.weights = ones_like(data)` ù **uniform weighting** in ePSF construction iterations.

**EPSFFitter / PSFPhotometry (science photometry):** VYVAR passes `error=err_cut` from
`_psf_fit_error_cutout`: uniform map with ? = `sqrt(sky/g + (RN/g)ù)` using annulus/border **sky
per px** (`_psf_sky_only_sigma_per_px`, Howell 1989 Poisson-on-sky + read noise). Recorded as
`psf_weight_mode=sky_only`, `psf_err_mode=sandwich_skyonly`. Flux uncertainty via sandwich estimator
(`_psf_sandwich_flux_err`) with the same sky-only ? ù **not** source Poisson.

**Pedestal / XVAL lesson (Howell 2006):** On sky-subtracted float32 frames, using `sqrt(data/g)` or
pedestal-as-photons weights is wrong (AIJ XVAL). VYVAR **does not** feed NDData uncertainty from
frame values into the ePSF build; fit weights use explicit sky estimate, not raw pixel values ù **avoids
the classic pedestal corruption**. Trade-off: weights ignore source Poisson term at the fit stage
(homogeneous ? map); chi2 is therefore ùsky-only weightedù not full CCD-likelihood.

**Effective variance model today:** constant ? per pixel in fit window driven by local sky ADU/px +
gain/RN; ePSF build unweighted.

---

### B4 ù Aperture-correction chain vs production DAO

**Verdict: correct chain; production proc CSVs still carry production-model PSF (ratio ~1.33); gated
model reconciles to ~0.997 after AC**

**Chain (code):**

1. `psf_photometry_stars` ? raw `psf_flux` (unit-normalized ePSF ù fit flux).
2. `_compute_aperture_correction`: among stars with finite flux, `chi2 < 5`, `psf_flux>0`, `dao_flux>0`,
   compute `median(dao_flux / psf_flux)` (MAD-trimmed) ? `psf_ac_factor`; requires ?5 stars.
3. If factor ? 1: multiply **all** `psf_flux` and `psf_flux_err` by `psf_ac_factor`; set
   `psf_ac_applied=True`.
4. Export: `photometry_core` ? `psf_mag = -2.5*log10(psf_flux)` when PSF branch active.

**Same aperture system:** `dao_flux` in proc CSVs is the production DAO aperture photometry on the
same detrended frames; AC explicitly ties PSF flux to that system per frame.

**S4 proc CSV aggregate (134 frames, production 1475-star model still in platesolve):**

| Metric | Value | File |
|--------|------:|------|
| median dao/psf (pre-AC implicit in stored flux) | 1.325 | `b4_proc_csv_summary.json` |
| median chi2 | 11.6 | |
| median psf_ac_factor | 0.529 | |
| median psf_ac_n_used | 83 | |

AC ~0.53 corrects biased low PSF flux toward DAO; stored `psf_flux` is **post-AC**. Residual
frame-level mismatch vs DAO remains (ratio ~1.33 on stored columns reflects production-model bias
before/full AC effectiveness).

**Gated 67-star model (R4-aligned sandbox, 10 frames):**

| Metric | Production | Gated |
|--------|------------|-------|
| psf_dao_ratio median | 1.327 | **0.997** |
| chi2 median | 21.7 | **1.48** |

(R4: `dev/results/context/session_20260822_epsf_valid_02_r1r4/r4_split_half_summary.json`)

With gated model, raw PSF flux already matches DAO within ~0.3% median; AC factor ?1.0 (minimal
correction). **Reconciliation:** ratio ~0.997 = AC factor ?1 because PSF shape/scale is calibrated on
the same comp stars that define DAO reference.

---

## Part D ó minimum certificate (gated 67-star pool, sandbox)

**SUPERSEDED by `CURSOR_RESULT_EPSF_VALID_02_S5B.md` (metric defect: raw-flux offsets).**

### D1 ó Split-half

**Verdict: CONCERN**

| Item | Value |
|------|------:|
| Odd-indexed half (34 stars) build | **FAILED** (`ValueError: All elements of input data must be finite`) |
| Even-indexed half (33 stars) build | OK (`build_d1_even_33/`) |
| Compare mode (fallback) | even-33 vs full gated-67 reference |
| Frames | 12 |
| Matched starùframe pairs | 5 |
| Per-star median ? mmag | ?32.4 |
| Per-star RMS ? mmag | 41.8 |
| Night error budget (median) | **151.2 mmag** |
| Budget source | `dev/results/context/session_20260818_anchor_516_02/err_term_picks.csv` column `err_median_516_mmag` (median across science targets, ERR path) |

Even-half subsample vs full 67-star model: median |?| ? 32 mmag < 151 mmag budget ? **within budget**,
but **CONCERN** because: (1) odd half cannot build an independent ePSF; (2) only 5 matched pairs in
fallback; (3) true odd-vs-even split-half not completed.

Artifacts: `d1_summary.json`, `d1_per_star_median_delta_mmag.csv`.

---

### D2 ù Convergence (N = 15, 30, 50, 67 vs 67-star reference)

**Verdict: CONCERN ù curve does not flatten until full gated pool**

| N_build | n_matched | median ? mmag vs N=67 | RMS ? mmag |
|--------:|----------:|----------------------:|-----------:|
| 15 | 59 | ?324.8 | 327.6 |
| 30 | 56 | ?427.3 | 430.5 |
| 50 | 59 | ?300.4 | 299.3 |
| 67 | 74 | 0.0 | 0.0 |

**Break point:** n=15 included by design ù largest deviation (~325 mmag). n=30 **worse** than n=15 in
this sample. n=50 still ~300 mmag off full model. **Only N=67 (full gated science comps) matches
reference.**

Artifacts:

- `d2_convergence_curve.csv`
- `d2_convergence_curve.png`
- `d2_summary.json`
- Sandbox builds: `build_d2_n15/`, `build_d2_n30/`, `build_d2_n50/`

**Proposed N policy (for meta/docs ù architect + Milan confirm at STOP-B):**

> For draft-516-class fields, use the **full Part C gated science-comp pool** (~67 stars for 516),
> not INTERIM top-N=200 and not partial-N subset builds for production. Empirical D2: N<67 sandbox
> models remain hundreds of mmag from the converged solution; INTERIM N=200 cap should remain
> **disabled** until a field shows N<full converges within the D1 error budget.

---

## Gate status

| Check | Status |
|-------|--------|
| Production writes | None |
| Model swap | None (STOP-B holds) |
| `--fast` | Not required (no code change); last PASS at S1ùS4 HEAD `2ba3d58` |
| HEAD | `2ba3d58` |

---

## Errors (if any)

- Odd-indexed 34-star ePSF build fails (non-finite ePSF after iterations) ù D1 partial.
- `err_bkg_source` empty on proc CSV comp rows; D1 budget taken from anchor ERR picks table instead.
- DB malformed warnings (pre-existing; no impact on sandbox measurements).

---

## Files changed

| File | Role |
|------|------|
| `dev/sandbox/epsf_valid_02_s5_measure.py` | Sandbox harness (new) |
| `dev/results/CURSOR_RESULT_EPSF_VALID_02_S5.md` | This deliverable |
| `dev/results/context/session_20260822_epsf_valid_02_s5/*` | Measurements, PNGs, CSVs, JSON |

No commits (measurement-only task).

---

## STOP-B

**Architect review + Milan swap decision required.**

S5 completes written verdicts (B1ùB4) and minimum certificate (D1/D2). **Do not proceed to S6**
(model swap to gated platesolve ePSF) until architect signs STOP-B and Milan authorizes swap.

Recommended STOP-B questions:

1. Accept gated 67-star model replacement for production `masterstar_epsf.fits` on 516?
2. Adopt proposed N policy (full gated pool, retire INTERIM N=200 for this field class)?
3. D1 CONCERN ù require odd-half build fix (centroid/edge stars) before swap, or accept even-vs-full stability?
