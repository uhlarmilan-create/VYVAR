# VYVAR -- Decision brief for Milan (audit closure batch C)

**Date:** 2026-08-02
**Purpose:** Four open **DECISION** items from the science audit, with physics, options,
literature, recommendation, and the confirming measurement after implementation (batch D).
**No choices are recorded here** -- Milan selects; entries go to `docs/VYVAR_DECISIONS.md`
after choice. Evidence: `dev/results/CURSOR_RESULT_audit_stage{2,3}_*.md`,
`dev/results/CURSOR_RESULT_audit_t4.md`, register items 10, 21, 22, 25.

---

## C.1 -- I-11: Howell sky term on sky-subtracted frames (register 21)

### Physics

The CCD noise equation (Howell 1989, 2006 Handbook sec 2.3) includes a sky Poisson term
proportional to `n_pix * (sky electrons per pixel)`. VYVAR subtracts a 2D sky surface in
preprocess (`pipeline.py` `_fit_subtract_preprocess_sky_surface`); per-frame metadata records
`sky_surface_bg_median_adu` and related QC columns. On the **legacy Howell error path**, the
annulus sky used for the sky term is measured on the **already subtracted** frame, so it
collapses toward zero and the sky Poisson contribution vanishes -- the exported error is
**under-quoted**. The hybrid clamp `BKG_SCALE_R_CLAMP_HI = 2.0` can still under-quote by ~2x
when it engages (audit Tranche 2).

### Current state

**0 production epochs** on anchor draft_435 use this legacy path; it engages in **crowded fields**
where empirical `sigma_bkg_ap` (Labbe et al. 2003 empty-aperture method) is unavailable or fails.
Risk is latent, not absent.

### Options

| # | Option | Summary |
|---|--------|---------|
| **1** | Pre-subtraction sky in Howell term | Use `sky_surface_bg_median_adu` (or equivalent pre-subtract sky) for `n_pix * sky_e`. |
| **2** | Raise `BKG_SCALE_R_CLAMP_HI` | Document and increase the hybrid clamp; keeps subtracted-frame sky read. |
| **3** | Refuse legacy path on subtracted frames | Return NaN / exclude epoch when sky term cannot be formed honestly. |

### Literature

- Howell (2006), *Handbook of CCD Astronomy*, CCD equation: sky term = `n_pix * sqrt(sky_e)`.
- Labbe et al. (2003): empirical background sigma on resampled frames (photometry path already uses this where valid).

### Recommendation

**Option 1.** Sky Poisson noise is set by the photons that **arrived**, not by the post-subtraction
residual. `sky_surface_bg_median_adu` is the physically honest quantity already written at
preprocess time. Options 2 and 3 are fallbacks if Option 1 cannot be wired cleanly (2 = partial
fix; 3 = conservative drop).

### Confirming number after implementation

Check-star **chi2_red** on anchor should remain unchanged (**0 epochs** on legacy path today).
On crowded-field drafts where the path engages, chi2_red should **rise toward correctness**
(less under-quoted error), not toward 649-style artifacts. Report before/after on at least one
crowded draft if available.

---

## C.2 -- I-04: ensemble scatter on unmatched epochs (register 22)

### Physics

Differential photometry error combines photon noise, ensemble scatter, scintillation, and
systematic terms. When the comparison ensemble cannot be matched for an epoch
(`err_scatter_unmatched`), substituting **0.0** for the ensemble scatter term makes the exported
error **optimistic** -- the pipeline claims precision it does not have.

### Current state

**0 epochs** on anchor draft_435 have `err_scatter_unmatched = True` (Stage 2 measurement).
Behaviour is **fail-optimistic by design** -- wrong direction for a publication-grade tool.

### Options

| # | Option | Summary |
|---|--------|---------|
| **1** | NaN + exclude epoch | No measurement exported when scatter cannot be justified. |
| **2** | Flagged inflation | Keep epoch; inflate error by a documented factor; set flag. |

### Literature

Honeycutt (1992, PASP 104, 436): ensemble differential photometry treats missing comparison
data as **missing**, not as zero-variance. Standard rule: propagate when justified, otherwise drop.

### Recommendation

**Option 1 (NaN + exclude).** A tool must not export an error bar it cannot defend. Inflation
(Option 2) introduces a second free parameter with no anchor measurement to tune it.

### Confirming number after implementation

**0 epochs affected** on anchor -- export should remain **byte-identical** on check stars.
Change is a **correctness guarantee**, not an expected numeric shift on draft_435.

---

## C.3 -- P-02 / A-5 / A-6: error budget and scintillation (registers 25, 5, 6)

**These three items are one decision and must be taken together.**

### Physics

Check-star chi2_red on the anchor is **~4.0--4.7** (Part 1c corrected; Part 1b "649" was total
chi2 mis-indexed). Quoted errors are ~**2x too small** vs check-star scatter. Differential variance:

```
sigma_diff^2 = sigma_photon^2 + sigma_ensemble^2 + sigma_scint^2 + sigma_sys^2
```

**P-02:** Scintillation formula implemented (Young 1967; Osborn et al. 2015 MNRAS 452, 1707) but
**not wired** into production error export.

**A-6:** Per-rig systematic floor `sigma_sys_mag = 0` on the wide rig (Part 1c audit).

### Scintillation (computed, C=1.56 median, airmass X=1.2, t=60 s)

| rig | aperture | h_obs | sigma_scint |
|-----|----------|-------|-------------|
| Wide 200 mm (Jirny) | 20 cm | 200 m | **2.39 mmag** |
| Newton 300 mm | 30 cm | 200 m | **1.82 mmag** |
| Brno 800 mm | 80 cm | 500 m | **0.91 mmag** |

### Does scintillation alone close the chi2 gap?

Only if current quoted error ~ **1.4 mmag**. With chi2_red ~ 4, true scatter ~ **2x** quoted;
reaching chi2_red = 1 needs `sigma_total ~ sqrt(4) * sigma_quoted = 2 * sigma_quoted`.
Scintillation adds ~2.4 mmag in quadrature:

- if `sigma_quoted ~ 1.4 mmag`, scint (**2.39 mmag**) can close the gap;
- if `sigma_quoted ~ 3 mmag`, scint alone is **not enough** and a **`sigma_sys` floor** is also required.

Median check-star err on bright epochs (Part 1c): **~0.058 mag (~58 mmag)** per epoch; differential
export is smaller after comp subtraction -- the ~2x chi2 gap is on the **combined** budget, not
the raw epoch err alone.

### Systematic floor (A-6)

Everett & Howell (2001, PASP 113, 1428) and Honeycutt (1992): irreducible **2--5 mmag** floor for
unfiltered wide-field differential photometry (flat residuals, second-order extinction, PSF-shape
variation).

### Options

| # | Option | Summary |
|---|--------|---------|
| **1** | Wire scintillation only | Add `sigma_scint` per rig to exported err. |
| **2** | Per-rig `sigma_sys` floor only | Add documented constant in mmag. |
| **3** | Both (recommended order) | (a) wire scintillation, (b) re-measure chi2_red, (c) add floor if still > ~1.2. |

### Recommendation

**Option 3**, in order:

1. Wire scintillation per rig using computed values above.
2. Re-measure median check-star chi2_red (Part 1c harness).
3. If median chi2_red still **> ~1.2**, add per-rig `sigma_sys` floor tuned to bring median
   toward **~1.0**.

**R8 constraint:** Do **not** tune the floor blindly to force chi2_red = 1. Report **achieved
chi2_red** and **floor value separately** so a referee sees both. The floor is a **measured rig
property**, not a hidden fudge.

**A-5 note:** Recalibrating `masterstar_dao_threshold_sigma` (register 5) depends on T4-1 (batch C.4)
and stack noise; execute after detection-noise decision in batch D/E.

### Confirming number after implementation

Median check-star **chi2_red within ~20% of 1.0** on at least **two rigs**, reported **before/after**
with scintillation and floor contributions listed separately.

---

## C.4 -- T4-1: detection noise on resampled frames (register 10; BLOCKS re-cut)

### Physics

DAOFIND assumes **uncorrelated** pixel noise (Stetson 1987; photutils `scale_threshold=True`).
After `astroalign` resampling onto `detrended_aligned` frames, neighbouring pixels are **correlated**.
Effective noise per resolution element is lower than per-pixel noise implies. Nominal
`masterstar_dao_threshold_sigma = 3.8` becomes **~3.3--3.58 effective**, varying with dither
(Tranche 4 simulation). This changes detection depth and which stars enter photometry.

Part 2b threshold sweep: log-log slope **-1.58** (valid); legacy `N_equiv` arithmetic depends on
measured kernel `rel_err`:

| rel_err source | N_equiv (example frame) |
|----------------|-------------------------|
| Legacy assumed 1.36 | **~4.71** |
| Measured on rebuild (Part 2b) | **~3.78** |

**This decision blocks anchor re-cut** (register 29) because it changes detections and catalogue
membership.

### Options (Tranche 4)

| Option | Description |
|--------|-------------|
| **A** | Leave nominal threshold; document effective ~3.3--3.58 sigma. |
| **B** | Correct threshold for measured pixel correlation (raise nominal so effective = target). |
| **C** | Measure correlation per frame; apply per-frame correction. |
| **D** | Detect on pre-resample (pre-align) frame. |

### Literature

- Casertano et al. (2000, AJ 120, 2747): noise correlation after drizzling/resampling.
- Fruchter & Hook (2002, PASP 114, 144): correlated noise in combined images.
- Stetson (1987, PASP 99, 191): FIND/DAOFIND threshold semantics.

Photometry already handles correlation via Labbe empty apertures on resampled images; **detection**
should threshold the **same noise field** that detection uses, not assume white noise.

### Recommendation

**Option B** with a **single measured correction factor** derived from Part 2b `rel_err`
(**N_equiv ~ 3.78** on measured rebuild vs **~4.71** on legacy 1.36 assumption). Correlation is
a property of the **fixed resampling kernel**; one rig-level factor restores intended effective
threshold. **Option D** (detect pre-resample) is cleaner architecturally -- note as **long-term
direction**; larger change than batch D scope.

### Decision required from Milan

1. Which option (A/B/C/D)?
2. If **B**: confirm `N_equiv` from **3.78** (measured rel_err) vs **4.71** (legacy).

Must be settled **before anchor re-cut** (batch E).

### Confirming number after implementation

Stable **effective detection threshold** at target sigma on resampled frames; detection count on
anchor-class frames documented before/after. Paired with A-5 threshold recalibration where applicable.

---

## How to record choices

After Milan decides, append to `docs/VYVAR_DECISIONS.md` (one entry per item or one combined
P-02/A-6 entry). Batch D implements; batch E re-cut requires T4-1 + Milan fingerprint authorization.

**Register pointers:** items 10, 21, 22, 25 annotated **decision brief ready** (2026-08-02).
