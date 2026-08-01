CURSOR RESULT - 2026-08-01 22:30 UTC+2

**Outcome N-none** (C-sat and C-sky excluded on tested evidence). Per-bin `flux` slope G 8-9:
**-0.258**; G 9-10: **-0.434**. N3 before/after sky on native `flux_large`: **-0.020** / **-0.020**.

What I did
Localised D5-2 compression in 1-mag bins (N1); tested saturation via full peak distribution (N2);
measured flux_large before/after sky-surface subtraction on native calibrated/processed frames
(N3); quantified PSF wings vs sky-surface amplitude (N4). Diagnostic only.

## N1 - localise the break

Source: `dev/tools/closure_step1n_localise_compression.py`; production `proc_*.csv`, 4058
star-frames (`photometry_ok` and `is_usable`).

### 1-mag bins -- log10(flux) vs catalogue G

| G bin | n | slope flux | slope flux_large |
|-------|---|------------|------------------|
| 8 - 9 | 834 | **-0.258** | **-0.303** |
| 9 - 10 | 834 | **-0.434** | **-0.416** |
| 10 - 11 | 834 | **-0.431** | **-0.398** |
| 11 - 12 | 834 | **-0.417** | **-0.334** |
| 12 - 13 | 834 | **-0.492** | **-0.413** |

Expected slope: **-0.400**. Departure from -0.4 is strongest in the **G 8-9** bin (shallowest)
for both columns. G 9-10 through G 12-13 are near -0.4 for `flux_large` (fixed 9.58 px).
G 11-12 `flux_large` at **-0.334** is a secondary shallow patch. Pattern is **not** a single
sharp threshold; it is bin-local with the largest gap at the brightest bin.

### Halves at G_split = 10.11 (task convention: bright = G < 10.11)

| half | G range | n | slope flux | slope flux_large |
|------|---------|---|------------|------------------|
| bright | 8 - 10.11 | 1946 | **-0.408** | **-0.391** |
| faint | 10.11 - 15 | 2780 | **-0.234** | **-0.201** |

**Note on naming:** Step 1l used `faint_half` = G < med and `bright_half` = G >= med (inverted
astronomical labels). Under task convention (bright = lower G), the **G >= 10.11** half shows
compression on `flux`; the **G < 10.11** half is near -0.4 on `flux_large`. Section 0's table
rows matched Step 1l variable names, not astronomical bright/faint.

`flux_large` at fixed 9.58 px: raw slope **-0.294** (full sample), confirming Step 1m -- compression
with the aperture mechanism off.

## N2 - C-sat: full peak distribution

Source: proc CSV, all 4058 star-frames; per-star stats over 139 frames where present.

Brightest stars (examples):

| star | G | peak_max | peak_p95 | limit_85 | frames > 70% limit |
|------|---|----------|----------|----------|---------------------|
| 1498602913793336448 | 8.18 | 38417 | 37519 | 55705 | **0** |
| 1499906247391001088 | 8.74 | 38110 | 32862 | 55705 | **0** |

Across all 35 stars: **0** star-frames above 70% of `saturate_limit_adu_85pct`; **0** above 85%.
`is_saturated` and `likely_saturated` False throughout.

Bright half (G < 10.11), excluding star-frames with peak > 70% limit: **0 excluded**; slope
unchanged at **-0.408**. **C-sat excluded** -- no hidden saturation in the tail.

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

**Localisation (N1):** Strongest shallow slope in the **G 8-9** bin (-0.258 flux, -0.303
flux_large). Most 1-mag bins at G >= 9 show slopes near **-0.4** on `flux_large`. Compression
is **not** uniform across the bright end; it is localised primarily to the brightest magnitude
bin in the closure set. Mechanism **open** after N2/N3 exclusion.

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
