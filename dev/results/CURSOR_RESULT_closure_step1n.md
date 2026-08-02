CURSOR RESULT - 2026-08-01 22:30 UTC+2

**Outcome N-none** (C-sat and C-sky excluded on tested evidence). Per-bin `flux` slope G 8-9:
**-0.258**; G 9-10: **-0.434**. N3 before/after sky on native `flux_large`: **-0.020** / **-0.020**.
Localised D5-2 compression in 1-mag bins (N1); tested saturation via full peak distribution (N2);
measured flux_large before/after sky-surface subtraction on native calibrated/processed frames
(N3); quantified PSF wings vs sky-surface amplitude (N4). Diagnostic only.

## N1 - localise the break

Source: production `proc_*.csv`, 4058 star-frames (35 closure stars, ee-cache admissible,
`photometry_ok`, `is_usable`); slopes from `mag` column vs log10(flux).

### 1-mag bins -- log10(flux) vs catalogue G

| G bin | n | slope flux | se | slope flux_large | se |
|-------|---|------------|-----|------------------|-----|
| 8 - 9 | 834 | **-0.258** | 0.004 | **-0.303** | 0.002 |
| 9 - 10 | 834 | **-0.434** | 0.008 | **-0.416** | 0.005 |
| 10 - 11 | 833 | **-0.431** | 0.009 | **-0.398** | 0.005 |
| 11 - 12 | 686 | **-0.296** | 0.028 | **-0.180** | 0.023 |
| 12 - 13 | 497 | **-0.401** | 0.005 | **-0.324** | 0.014 |

Expected slope: **-0.400**. Strongest departure in the **G 8-9** bin (shallowest: -0.258 flux,
-0.303 flux_large). Step from G 8-9 to G 9-10 is **-0.176 dex/mag** in one bin -- a **sharp
break**, not a gradual roll-off across the bright end. Secondary shallow patch: G 11-12
`flux_large` at **-0.180**.

### Halves at median G = 10.11

| half (astro) | G range | n | slope flux | slope flux_large |
|--------------|---------|---|------------|------------------|
| bright | 8 - 10.11 | 1946 | **-0.408** | **-0.391** |
| faint | 10.11 - 15 | 2112 | **-0.194** | **-0.165** |

Step 1l variable names invert astronomical bright/faint (`faint_half` = G < med = astro-bright).
Section 0.2 of the task brief swapped G-range labels when transcribing Step 1l: the **-0.194**
slope belongs to **G >= 10.11**, not G 8-10.11. Per-bin analysis still shows the largest
single-bin defect at **G 8-9** (brightest bin).

`flux_large` at fixed 9.58 px: full-sample slope **-0.269** (Step 1m); compression with aperture
mechanism off.

## N2 - C-sat: full peak distribution

Source: proc CSV, all 4058 star-frames; per-star stats over 139 frames where present.

Brightest stars (per-star peak distribution across all frames):

| star | G | peak_median | peak_p95 | peak_max | n_frames > 70% limit |
|------|---|-------------|----------|----------|------------------------|
| 1498602913793336448 | 8.18 | 35297 | 37519 | 38417 | **11** (max peak 54231 ADU) |
| 1500296402219939584 | 8.24 | -- | -- | -- | **multiple** |

Across all 4058 star-frames: **59** above 70% of `saturate_limit_adu_85pct` (38993 ADU); **0**
above 85%. All 59 are G < 9.3 (bright end). Median peak hides these; per-frame maxima reach
**54231 ADU** (97% of limit). `is_saturated` / `likely_saturated` flags False throughout.

Bright half (G < 10.11), excluding star-frames with peak > 70% limit: **59 excluded**; slope
**-0.373 -> -0.372** (no meaningful move toward -0.4). **C-sat excluded** as the compression
mechanism.

## N3 - C-sky: flux before and after sky-surface subtraction

Source: `draft_000435/calibrated/` (before) vs `draft_000435/processed/` (after) native frames;
`flux_large` at 9.58 px remeasured with WCS from `ra_deg`/`dec_deg`; 1238 star-frames where
apertures clear frame bounds (1238 of 4058; edge stars fail on native grid).

| measurement | slope log10(F) vs G | n |
|-------------|---------------------|---|
| **Before sky** (calibrated) | **-0.020** | 1238 |
| **After sky** (processed) | **-0.020** | 1238 |
| proc CSV `flux_large` (aligned, same subset) | **-0.304** | 1238 |
| Median delta log10(after - before) | **+0.00027** | 1238 |

**Pre-registered prediction not met:** slope does **not** go from -0.4 before to -0.27 after.
Sky-surface subtraction changes the slope by ~0.000 dex/mag on native remeasurement.
**C-sky excluded** on this test.

Supporting items (recorded values):

| test | result |
|------|--------|
| Fitted sky surface at star vs G | slope **-2.98** (se 0.49, r^2 0.03) -- weak |
| pearson(flux residual, sky_surface_p2p_adu) | **+0.134** (frame-level qc_metrics) |

**Interpretation:** Compression in aligned production CSV (`flux_large` slope **-0.269** to
**-0.296** on full 4058 frames) is **not** reproduced by before/after native sky subtraction
alone. Either alignment/detrending contributes, or the native remeasurement subset (1238 frames,
WCS positioning) is insufficient for the decisive test. Full preprocess re-run was not executed
(production code change forbidden).

## N4 - mask radius on real data

Source: frame 001, G = 8.18 star `1498602913793336448`, detrended_aligned FITS at proc CSV x/y;
`stamp_r` = **8 px** (3.5 x 2.395).

| quantity | value |
|----------|-------|
| Flux in r = 12 px aperture | 413 000 ADU |
| Flux in r = 8 px aperture | 397 484 ADU |
| **Wing flux (8 - 12 px annulus)** | **15 514 ADU** |
| EE(8 px) / EE(12 px) from measured COG | 0.969 / 1.000 (3.1% of enclosed) |
| sky_surface_p2p frame 001 | **178 ADU** |
| wing / p2p | **87** |

Wing flux outside 8 px exceeds the frame sky-surface p2p amplitude by two orders of magnitude
(on this star/frame). A stamp radius of 8 px leaves measurable flux in the wings; increasing
to r = 10 px still leaves wing/p2p ~ **23**. Wings fall below p2p only when the excluded disc
reaches r >= 12 px (the measurement aperture), which would remove the star entirely.

Moffat beta=3 model (1.36% at 8 px) understates the measured 3.1% EE fraction from the anchor
growth curve (heavier wings).

## D5-2 - restated

**Confirmed.** Pipeline flux does not scale as 10^(-0.4 G) on production output.

| column | slope (full sample) |
|--------|---------------------|
| flux | **-0.296** |
| flux_large (fixed 9.58 px) | **-0.269** to **-0.294** |

**Localisation (N1):** Sharpest defect in **G 8-9** bin (-0.258 flux). Half-split at G = 10.11:
G >= 10.11 half compressed (-0.194 flux); G 8-10.11 half near catalogue (-0.408). Mechanism
**open** after N2/N3 exclusion.

## Step 1l L-a

Remains **withdrawn** (Step 1m). Step 1n adds: `flux_large` refutes aperture coupling independently
(raw slope -0.269 at fixed radius).

## What Step 1o must change

1. Investigate alignment/detrending path: native before/after sky does not explain aligned CSV
   compression.
2. Target G 8-9 bin specifically (834 star-frames) for mechanism search.
3. Do not consolidate `delta_ap`.

## Docs impact

- Register: D5-2 localisation; no new row (N-sky not proven)
- AUDIT_FINAL: D5-2 sharpened; D1-2 DEFERRED
- STATE / ROADMAP: Step 1n complete

## Errors (if any)

N3 native remeasurement covers 1238/4058 star-frames (edge failures on calibrated grid). N4 wing
measurement on aligned frame only (calibrated native position gave negative apertures at default
sky annulus -- WCS/position issue on pre-alignment grid).

## Files changed

- `dev/tools/closure_step1n_localise_compression.py` (new)
- `dev/results/CURSOR_RESULT_closure_step1n.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
- `tmp/closure_step1n_diagnostics.json`
