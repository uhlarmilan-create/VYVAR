# F-431 lost ADU transform - T1 characterization

**Definition:** `R = MASTERSTAR(draft_429) - calibrated BO_CVn_Light_008` (same draft).
MASTERSTAR == processed Light_008: **True**.

## Operation class (revised): **SMOOTH_ORDER2_BACKGROUND_SURFACE**

Bulk residual ~ **spatially varying low-order ADU surface** (pedestal ~-95 ADU + mild
quadratic gradient). Not pure constant; not dominant alignment shift (phase-corr peak at 0,0);
not sparse hot-pixel cleaning (impulsive frac ~3e-4). Extremal maxabs~1.7e4 are **sparse
pathological pixels** (already negative cosmics/bleed in calibrated) - not the census driver.

**Census ablation (decisive):**

| Image | DAO pass-1 (sigma=2.1, FWHM=2.5) |
|-------|------------------------------|
| 429/431 calibrated Light_008 | **9115** |
| 429 MASTERSTAR / processed | **2816** |
| cal - constant 95.65 ADU | **9115** (no change) |
| cal + order-2 fit to R | **2626** ~ healthy |
| cal with forced thr=MS | **2572** |

-> The transform that matters is the **order-2 surface**, which raises sigma-clipped sky std
62->84 ADU and thus DAO threshold 131->176, collapsing pass-1 ~9115->~2800.

---

## 1. Amplitude statistics

- min/max = `-16660.669` / `3677.094`
- mean/median/std = `-91.313` / `-98.458` / `56.023`
- maxabs = `16660.669`
- signed percentiles: p0.1/p1/p5/p50/p95/p99/p99.9 =
  `-275.1 / -165.0 / -152.8 / -98.5 / -16.9 / -1.9 / +14.7`
- abs percentiles: p50~98.5, p99~165, p99.9~283.5 (maxabs is far outlier)

### Maxabs localization

- Top |R| peaks are **not** bright star cores (fraction 0.10 of top-20).
- Worst pixel (y=557,x=1204): cal already `-5210`, MS `-21871` - pathology / cosmics amplified.
- After `|R|<500` mask (~99.98% of pixels): mean~-91, std~43.

Figures: `tmp/f431_lost_transform/residual_image.png`, `residual_histogram.png`.

---

## 2. Spatial character

| Model | var explained | resid std (raw 56.0) |
|-------|---------------|----------------------|
| order-1 plane | 0.292 | 47.1 |
| order-2 surface | **0.487** | **40.1** |

Order-2 coefficients `(1, y, y^2, x, xy, x^2)` ~
`[-47.46, -0.0790, 1.05e-4, -0.1400, 4.02e-6, 5.92e-5]` (full-field fit; see `stats.json`).

- Impulsive `|R|>8.MAD`: fraction **0.0003** (n=790) - not a median-filter / hot-pixel recipe.
- Phase correlation (256^2 center): shift **(0,0)**, peak 0.89 - **not** a large rigid offset.
- Subpixel roll search does not collapse MAD (~94-98 stays ~|median|).
- Early 'dipole' star check (30/80) is a **false lead relative to census**: ablations show
  surface subtraction alone reproduces the DAO band; residual dipoles are second-order.

Figures: `surface_fit_order2.png`, `residual_after_surface.png`.

---

## 3. Sky-level + DAO accounting

| Quantity | Value |
|----------|-------|
| cal median | 1955.4 |
| MS / proc median | 1859.7 |
| Delta median | **-95.7 ADU** |
| Meta sky (431->429) | ~1565->1478 (~-87) - same direction/scale |

DAO pass-1 simulation matches live draft headers (VY_NDAO / validate table): **sick ~9k, healthy =2816**.

---

## 4. Re-implementation sketch (for T3, gated)

**Intentional op:** subtract a fitted/evaluated order-2 (or equivalent large-scale) sky surface
from each calibrated frame during shared preprocess, leaving star flux; write result to
`processed/`. Default **ON** only after Milan sign-off (science-affecting) **or** after T2 shows
UI-HEALTHY and the exact call site is extracted.

**Must-have acceptance:** on BO_CVn Light_008 reference, raw DAO pass-1 band **~2500-3200**
(not ~9000) after transform; unit test `cal!=proc` on synthetic bowl+pedestal frame.

**Not:** constant-only pedestal; integer-shift alignment; aggressive 8xMAD clip.

## Artifacts

- `tmp/f431_lost_transform.md` (this file)
- `tmp/f431_lost_transform/stats.json`
- PNGs: residual, surface fit, residual-after-surface, histogram
- Script: `scripts/f431_characterize_lost_transform.py`
