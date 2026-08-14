# CURSOR RESULT - COMP-POOL-02

Date: 2026-08-14
Register ID: COMP-POOL-02
Follows: COMP-POOL-01 Stages 1-3 (local, not pushed)

Report only, except Item 1 diameter override revert (C2-R0 / C2-R4).

---

## Item 1 -- Telescope diameter override was wrong

### Verification

| Source | Value |
|--------|------:|
| `TELESCOPE` row Carl-Zeiss 200mm | `DIAMETER=70.0` mm, `FOCAL=200.0` mm |
| `EQUIPMENTS` QHY294MM `PIXELSIZE` | **4.63** um |
| MASTERSTAR `PIXSIZE1` | 4.63 um |
| MASTERSTAR `XPIXSZ` / `XBINNING` | 9.26 um / 2 |
| MASTERSTAR `SCALE` | 9.55169 arcsec/px |
| Focal from scale: `206265 * 9.26e-6 / 9.55169` | **0.2000 m** |
| f/D from DB | 200/70 = **f/2.86** |
| FITS `TELESCOP` | `Sample Primary 200@F\2.9` |

**Conclusion:** "200 mm" is the **focal length**. Aperture is **70 mm**. The database was right. The Stage-1 override `D=0.2 m` was wrong (C2-R0).

Lens match: Carl Zeiss Jena Sonnar 200 mm f/2.8 has entrance pupil ~71 mm (200/2.8); DB 70 mm is consistent (allphotolenses.com / MFlenses Sonnar 200/2.8 specs).

### Revert performed

- Removed auto-override in `comp_pool_noise.fit_parametric_noise_curve` for drafts 435/509/510/512.
- Corrected `sigma_budget` missing-DB fallbacks from D=0.2 to D=0.07 (same misreading of focal as aperture).
- Production photometry already read `TELESCOPE.DIAMETER=70 mm` when the DB row is present; LC scintillation terms were not using the bad override. Only COMP-POOL diagnostics overrode.

### Recomputed scintillation (Osborn/Young)

| draft | sigma_sys (mag) | scint D=0.07 | ratio sys/scint | was (D=0.2) |
|------:|----------------:|-------------:|----------------:|------------:|
| 512 | 0.00974 | **0.00400** | **2.43** | 4.90 |
| 510 | 0.00973 | **0.00400** | **2.43** | 4.90 |
| 435 | 0.01184 | **0.00401** | **2.95** | 5.94 |

Factor `(0.2/0.07)^(2/3) ~ 2.01` recovers the change. **P-R2 still fires** (ratio != 1); the excess over scintillation is ~2.4-3.0x, not ~5x. P-02 remains "scintillation wired but does not exhaust the bright floor" (WIDE-ERR adjacent).

### Other consumers of `TELESCOPE.DIAMETER`

- `sigma_budget.resolve_rig_scintillation_params` (LC error budget) -- uses DB; already 0.07 when linked.
- UI telescope editor / DB explorer -- display only.
- Plate scale uses `FOCAL` + pixel pitch, not diameter -- unaffected.

Machine: `dev/results/COMP_POOL_02_item1.json`

---

## Item 2 -- chi2_red rejects the model as a description

| draft | chi2_red | interpretation |
|------:|---------:|----------------|
| 512 | 4.17 | rejected as description; usable only as approximation |
| 510 | 3.98 | same |
| 435 | 6.40 | worse |

chi2 definition in code: mean( ((obs - pred) / pred)^2 ) on G8-13 non-catalogue-variables. Values of 4-6 mean residuals are ~2x the model prediction -- **not a successful fit** (C2-R1).

### Residual shape (draft 512; 510 similar)

| Axis | Finding |
|------|---------|
| Magnitude | **Strong trend.** Ratio obs/model: ~0.75 at G8 to ~1.66 at G12.5. Bright end slightly overpredicted; faint end underpredicted. |
| Colour (G9-12) | **Weak.** Blue/mid/red ratio medians 1.30 / 1.38 / 1.22 -- no red excess in residual. |
| Chip position | **Flat.** Four quadrants agree; inner vs outer resid medians equal to ~0.5 mmag. |

**Diagnosis:** the model is **incomplete as a magnitude-dependent noise law**, not a colour or field-position failure, and not primarily catalogue-variable contamination at the bright end (bright asymptote is close). Missing (or mis-scaled) photon/sky term toward faint magnitudes is the leading explanation; that connects to **WIDE-ERR** (quoted errors underpredicted on the wide rig; SNR-GATE implied gain ~2.94 vs equipment 3.17 also pushes predicted phot noise low). Real variables contribute to the high-ratio tail but do not produce the smooth faint-end rise.

**Claim for the memo:** parametric curve is an **approximation for pool admission**, not a statistically accepted description of the scatter (C2-R1).

Machine: `dev/results/COMP_POOL_02_item2.json`, `COMP_POOL_02_residuals_*.csv`

---

## Item 3 -- Draft 435 failure mode

### Measured differences

| quantity | 512 | 510 | 435 |
|----------|----:|----:|----:|
| aperture_r median (px) | 3.41 | 3.46 | **1.92** |
| aperture area (px^2) | 36.6 | 37.6 | **11.5** |
| FWHM mid-night bright (px) | ~3.2 | ~3.2 | ~3.2 |
| r / FWHM | ~1.06 | ~1.08 | **~0.60** |
| NP/param usable median | 1.12 | 1.13 | **1.52** |
| chi2_red | 4.17 | 3.98 | **6.40** |
| admit fraction | 27.5% | 27.3% | **4.2%** |
| BO CVn comps in pool | 4/5 | 4/5 | **0/3** |

Cause (measured, not assumed):

1. Draft 435 uses a **much smaller photometric aperture** (r=1.916 px = A-1 value) while the PSF FWHM is still ~3.2 px (same order as 512).
2. Howell `n_pix` in the parametric model therefore understates sky contribution relative to the actual light-loss / seeing-coupled scatter (A-1: EE@prod ~67-73% on 435 vs ~82-86% on 510).
3. Parametric sigma is too low -> NP/param rises to ~1.52 -> stability ratios inflate -> **inv_eta p84 collapses to 0.657** -> all three archived BO comps fail `inv_eta>0.657` only.

At G10 alone NP/param is ~1.02 even on 435; the 1.52 median is driven by the faint, deep MASTERSTAR population that 512/510 lack. The aperture mismatch still poisons the absolute parametric scale used in stability.

### Is Stage 2 safe when NP/param is far from 1?

**No.** On 435 the method empties BO CVn's ensemble. That is a failure mode, not a threshold difference (C2-R2).

### Recommended guard (do not implement here)

Refuse (or fall back to legacy RMS pool / NP-only validation) when **any** of:

1. Median usable NP/param ratio **> 1.25** (named; or draft p84 of bin ratios), or
2. Median `aperture_r_px / fwhm_estimate_px` **< 0.85** (named EE-safety), or
3. After admission, any science target that had a prior ensemble has **zero** admitted comps in the spatial/colour search radius -- hard refuse sparse-empty.

Record the guard trip in provenance; do not silently return an empty ensemble.

Machine: `dev/results/COMP_POOL_02_item3.json`

---

## Item 4 -- Colour-dependent PSF width

### 4.0 Re-verification

Present on all three drafts. Example draft 512 mid-frame median `fwhm_estimate_px`:

| G | BP-RP 0.5 | 1.0 | 1.5 | red/blue |
|--:|----------:|----:|----:|---------:|
| 9 | 3.14 | 3.25 | 3.38 | +7.6% |
| 10 | 3.12 | 3.31 | 3.41 | +9.3% |
| 11 | 3.27 | 3.35 | 3.60 | +10.1% |

Draft 435 mid-frame at G10: 3.18 -> 3.56 (**+12%**); at G14: 6.62 -> 7.21 (+9%). Architect's deeper table is reproduced in sense and sign. Confound noted: at fixed G a red star is brighter on a red CMOS, which should bias moment-FWHM **narrower**; the observed wider red PSF is therefore not an SNR artifact.

### 4.1 Chromatic aberration / focus

- Focus proxy: median FWHM of G9-11 mid-colour stars per frame.
- Draft 512: corr(delta_FWHM_red-blue, focus_proxy) = **-0.16**; focus range only 3.21-3.41 px.
- Draft 435: corr = **-0.22**; focus range 3.20-3.40 px.
- Temperature in proc CSVs: **not present** (`temp_finite_n=0`).
- Mirror comparison: Archive drafts with MASTERSTAR are all `TELESCOP=Sample Primary 200@F\2.9` (same refractive rig). **No Newton/C9.25 proc products** in Archive to settle the test.

Interpretation: a weak anti-correlation with the small focus swing does not prove longitudinal chromatic aberration, but does not rule it out. The effect is **stable through the night** at the ~0.05 px level of delta scatter. C2-R3: this is an instrument property for methods docs.

### 4.2 Photometry (mmag)

COG star set on draft 512 is all BP-RP < 0.7 -- cannot difference blue/red EE directly from those curves.

**Primary estimate:** take the measured median growth curve (`draft512_cog_curves.csv`), rescale radius by k = FWHM_red/FWHM_blue (~1.082 at G9-11), compare EE at production aperture.

| estimator | mmag (red minus blue) |
|-----------|----------------------:|
| FWHM-rescaled measured COG | **~29.5** |
| Gaussian EE cross-check only | ~12.3 |
| Architect Gaussian prior (not used to validate) | ~60 |
| inst-G colour (G9-11), red-blue | ~275 (dominated by bandpass; not EE) |

Night stability of inst-G red-blue: median ~278 mmag, **std ~6.6 mmag** over 23 samples -- nearly **constant**. A constant offset is absorbed by the zero point for a fixed ensemble; a colour-mismatched ensemble still carries a static colour term. Focus-driven EE change is not detected above that floor on this night.

### 4.3 Pool colour blindness

| draft | admit BP-RP med | reject (in mag) med | admit frac BP-RP>1.5 | reject frac >1.5 |
|------:|----------------:|--------------------:|---------------------:|-----------------:|
| 512 | 0.91 | 0.93 | 3.5% | 8.2% |
| 510 | 0.92 | 0.93 | 4.8% | 7.7% |
| 435 | 1.05 | 0.88 | 2.0% | 5.2% |

On 512/510, red stars are **mildly over-represented among rejects** in the bright pool window -- consistent with wider PSF -> higher scatter -> stability cut. Effect size is small (few percent). On 435 the median colours reverse (small-n admit set). **Admission is weakly colour-dependent on this refractive rig**; it is not colour-blind in practice. Record as rig-dependent behaviour: worse-measured (redder, wider) stars are preferentially excluded -- scientifically defensible, but must not be sold as colour-blind.

### 4.4 Reach (note only)

Touches D10-1 (CV vs CR), D11-1 (Gaia G dilution proxy), unfiltered colour-matching policy, D5-1. Not resolved here.

Machine: `COMP_POOL_02_item4_fwhm.json`, `COMP_POOL_02_item4_phot_pool.json`, `COMP_POOL_02_colour_focus_*.csv`

---

## Pre-registered rules

| Rule | Result |
|------|--------|
| C2-R0 | **Fired.** DB confirmed; override reverted; ratios recomputed (~2.43 on 512). |
| C2-R1 | **Fired.** chi2_red~4-6 = model rejected as description; kept as approximation only. |
| C2-R2 | **Fired.** Stage 2 not safe by default when NP/param >> 1 (435 empties BO). Guard recommended, not implemented. |
| C2-R3 | **Fired.** Colour-width is a real instrument property. |
| C2-R4 | Observed: only Item 1 value correction; no threshold tuning. |

---

## Named gaps

- No Newton/C9.25 frames in Archive for chromatic-aberration control.
- No temperature in proc CSVs for focus/thermal proxies.
- COG growth-curve star set lacks red stars; EE colour mmag uses FWHM-rescaled COG (stated).
- Item 1 f_ratio_from_scale field in JSON mis-typed as D/f once; DB f/D=2.86 is authoritative.

---

## Register diff

- COMP-POOL-02 opened with four findings.
- COMP-POOL-SCINT updated: sys/scint ~2.4 (was ~4.9) after D correction; still open vs P-02/WIDE-ERR.
- COMP-POOL-01 Stage 2 push: **blocked** pending C2-R2 guard decision.

## Files changed (Item 1 only)

- `src_py/comp_pool_noise.py` -- remove D=0.2 auto-override
- `src_py/sigma_budget.py` -- fallback D=0.07 for Carl-Zeiss / generic
