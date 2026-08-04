CURSOR RESULT - 2026-08-04 (WIDE-ERR A)

What I did
Identified whether the ~15 mmag white excess is level-dependent (detector non-linearity)
or a constant rig floor. Read-only on Archive; reused tmp/wide_err_w1w2/ cached check-star
LCs; all new output under tmp/wide_err_a/. Harness: dev/tools/wide_err_a.py.

## Scope

Established from W1+W2: WHITE underquote (p2p_robust/err 1.69, total_robust/err 1.83);
excess magnitude sqrt(17.8^2 - 9.4^2) = 15.1 mmag matches batch D fitted 14.7 / 15.7 mmag.
N_eff mismatch rejected as mechanism.

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

## A1 -- Within-star test (check star 1499906247391001088, 163 fields)

**Peak column:** `peak_max_adu` from per-frame proc CSVs (also present: `peak_dao`).
Across all pooled epochs: p05 = 25774, p50 = 28915, p95 = 32452 ADU.

**Per-field regression** of residual r_i (mmag about weighted mean) on peak ADU:

| metric | value |
|--------|-------|
| n fields | 163 |
| slope median (mmag/ADU) | 0.00347 |
| slope IQR | 0.00231 / 0.00347 / 0.00490 |
| fraction fields p < 0.05 | 0.939 |
| fraction p < 0.05, consistent positive sign | 0.939 |

**With FWHM covariate** (partial peak ADU effect in r ~ peak + FWHM):

| metric | value |
|--------|-------|
| partial peak p median | 0.119 |
| fraction partial p < 0.05 | 0.393 |

Peak and FWHM are **NOT SEPARABLE** on this data: raw peak slopes are highly significant,
but controlling FWHM drops significance to ~39% of fields.

**All-epoch decile bins** (peak ADU decile vs |r| and robust scatter):

| decile peak median (ADU) | median |r| (mmag) | robust scatter (mmag) | n epochs |
|--------------------------|---------------------|-------------------------|----------|
| 25812 | 16.0 | 20.8 | 2282 |
| 26976 | 11.1 | 16.1 | 2282 |
| 27630 | 10.8 | 15.7 | 2282 |
| 28222 | 10.0 | 15.0 | 2282 |
| 28506 | 10.3 | 15.3 | 2282 |
| 29449 | 11.8 | 17.7 | 2282 |
| 30069 | 10.3 | 15.1 | 2282 |
| 30809 | 10.8 | 15.3 | 2282 |
| 31424 | 11.1 | 14.9 | 2282 |
| 32452 | 14.1 | 18.6 | 2119 |

**Shape:** non-monotonic (U-shaped). Lowest peak decile has the **highest** scatter (20.8 mmag);
middle deciles flat ~15 mmag; highest peak decile elevated to 18.6 mmag but below the lowest
decile. This is inconsistent with a simple "higher peak -> higher residual" non-linearity curve.

### A1 reading

Pre-registered: relation present but **non-monotonic** -> report shape, do not force LEVEL or
FLOOR from A1 alone. FWHM confounding prevents separating peak from seeing. **NOT SEPARABLE.**

## A2 -- Population test (166 variable-target LCs)

| metric | value |
|--------|-------|
| n stars | 166 |
| population median excess (mmag) | 18.3 |

**excess vs G** (median per decile): rises from **7.8 mmag** (G~10.2) to **56.0 mmag** (G~14.4).
Not constant in mmag across magnitude.

**excess vs peak ADU** (median per decile): noisy; faintest peak bin shows 68 mmag (small-n
outlier at low peak); bright-end bin (peak ~7057 ADU) **9.2 mmag**.

**Partial Spearman** excess vs peak ADU controlling G: **rho = 0.073, p = 0.35, n = 166**.
No peak dependence at fixed G.

**Batch D cohort comparison:**

| cohort | excess (mmag) | median peak ADU | G range |
|--------|---------------|-----------------|---------|
| check star 1499906247391001088 | **15.1** (W1+W2) | **28915** | 8.74 |
| constant calibrators (n=12, batch D fit) | **8.33** (batch D chi2 scaling) | **2865** | 9.3 - 13.2 |

The 8.33 vs 15.1 split aligns with **peak ADU** (check star ~10x higher peak) but also with
**G** (check star is brighter). Partial correlation controlling G shows **no** peak signal;
peak ADU alone does **not** explain the split once magnitude is accounted for. The calibrator
value is from batch D chi2 scaling, not the LOO LC excess formula (LOO err path is not
comparable).

### A2 reading

Excess is **not** a constant mmag rig floor (strong G dependence). Excess does **not** rise
with peak at fixed G. Pattern: **brightness-dependent** (G-correlated) mechanism that is **not**
isolated as detector level on this data. Does not match FLOOR or LEVEL pre-registered boxes cleanly.

## A3 -- Telescope aperture (report only)

**1. TELESCOPE query** (`TELESCOPENAME LIKE '%Carl-Zeiss%'`):

| ID | TELESCOPENAME | ALIAS | DIAMETER | FOCAL | ACTIVE | IS_DEFAULT |
|----|---------------|-------|----------|-------|--------|------------|
| 1 | Carl-Zeiss 200mm | Teleobjektiv1 | 200.0 | 200.0 | 1 | 1 |

**2. Sensor / binning** (FITS header, detrended_aligned/lights/NoFilter_60_2, frame 001):

| keyword | value |
|---------|-------|
| XPIXSZ / YPIXSZ | 9.26 um (unbinned pixel) |
| XBINNING / YBINNING | 2 / 2 |
| SCALE | 9.55169 arcsec/pixel (binned) |
| FOCALLEN (header) | 200.0 mm |
| APTDIA (header) | 70.0 mm |

**3. Implied focal length:**

pixel_pitch_binned = 9.26 um x 2 = 0.01852 mm
f_implied = 206265 x 0.01852 / 9.55169 = **399.9 mm**

Against DB: DIAMETER = 200 mm, FOCAL = 200 mm. Implied f ~ 400 mm vs header FOCALLEN 200 mm.
Using DB diameter: **f/D = 400/200 = f/2.0** (not f/1.0). Header APTDIA 70 mm is inconsistent
with DB 200 mm (fix_telescope_diameter.py 72 -> 200 mm per SIGMA-A2).

**4. Scintillation** (Young/Osborn, exposure 60 s, altitude 275 m, airmass p05/p50/p95 =
1.013 / 1.040 / 1.187):

| D (m) | scint p05 (mmag) | scint p50 (mmag) | scint p95 (mmag) |
|-------|------------------|------------------|------------------|
| 0.200 | 1.76 | 1.83 | 2.24 |
| 0.072 | 3.48 | 3.62 | 4.42 |

**5. Effect of D=0.072 m scintillation on check-star ratio** (replace scint_0.2 with scint_0.072
in err quadrature, per-field median):

| metric | value |
|--------|-------|
| median sigma_total_robust / err (D=0.200) | 1.828 |
| median sigma_total_robust / err_corrected (D=0.072 scint) | 1.665 |
| fractional reduction | **8.9%** |

Closes **~9%** of the underquote gap, consistent with the "~10 percent" expectation. Does **not**
fix WIDE-ERR (ratio remains 1.67 >> 1).

## Combined line

**WIDE-ERR-A-UNDECIDED** -- within-star peak-residual relation is FWHM-confounded and
decile-non-monotonic; population excess is G-dependent without peak signal at fixed G; the
15 mmag check-star excess is not demonstrated as a constant rig floor (B cannot assume FLOOR),
nor as clean detector level-dependence (route to D1-2 not established). Missing: separable
peak/FWHM test and/or a brightness-dependent error term beyond scint/N_eff.

## A4 -- Commits

Pending: commit dev/tools/wide_err_w1w2.py + dev/tools/wide_err_a.py (ASCII check).

## Files created

- dev/results/CURSOR_RESULT_wide_err_a.md (this file)
- dev/tools/wide_err_a.py
- tmp/wide_err_a/wide_err_a.json

## Errors

None blocking.
