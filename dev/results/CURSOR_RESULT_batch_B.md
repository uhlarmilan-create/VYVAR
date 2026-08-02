CURSOR RESULT - 2026-08-02 10:15 UTC+2

**Outcome B-open.** B1 before/after slopes **-0.232 / -0.243** (VOID, sanity gate). B2 partial
deficit~peak | r50 = **-0.945** (faint-half reference); **+0.37** with G 9-11 reference (below +0.4).

What I did
Ran batch B diagnostics (B1 valid before/after sky on aligned grid; B2 non-linearity with
`flux_large`, seeing control). Harness: `dev/tools/closure_batch_b_d52_mechanism.py`.
Source: anchor `draft_000435_snapshot_skysurface_20260716`, 4058 ee-cache star-frames.

## B1 - before/after sky on aligned grid

**Method:** Fixed 9.58 px aperture at proc CSV `x, y`. After = `detrended_aligned` FITS. Before =
`calibrated` FITS at same pixel coordinates (aligned grid; same array shape). Before-add =
calibrated flux + sky-surface integral in aperture (pipeline `_fit_subtract_preprocess_sky_surface`,
order 2, once per frame). n = **3204** star-frames (854 void apertures).

| fit | slope | se | n | sanity (-0.48 to -0.32) |
|-----|-------|-----|---|-------------------------|
| **Before** (calibrated) | **-0.232** | 0.004 | 3204 | **FAIL** |
| **After** (aligned) | **-0.243** | 0.002 | 3204 | **FAIL** |
| Before + surface addback | **-0.231** | 0.004 | 3204 | **FAIL** |

| delta | value |
|-------|-------|
| median log10(after - before) | **+0.034 dex** |
| median log10(addback - after) | **-0.044 dex** |

**Verdict:** **VOID.** Neither before nor after slope is near -0.4 (mandatory sanity gate). No
conclusion drawn on sky subtraction. Step 1n N3 (-0.020) is consistent with a broken before
baseline; this B1 attempt on the aligned grid still fails the gate. Likely cause: remeasured
aperture photometry at proc x,y on calibrated native pixels does not reproduce production
`flux_large` calibration (production slope -0.269 on same stars).

## B2 - non-linearity (flux_large, seeing control)

**Reference faint half (G > 10, task spec):**

| quantity | value |
|----------|-------|
| n | 2390 |
| slope | **-0.180** |
| sanity (-0.48 to -0.32) | **FAIL** (expected ~-0.4) |

Faint-half reference is itself compressed; pre-registered extrapolation is confounded.

**Bright G < 9 (834 star-frames), faint-half reference:**

| test | value |
|------|-------|
| mean deficit (dex) | **-0.483** |
| pearson(deficit, peak) | **-0.847** |
| pearson(deficit, r50) | **+0.012** |
| **partial(deficit, peak \| r50)** | **-0.892** |
| pre-registered NL (> +0.4) | **FAIL** |

Deficit vs peak is **negative** (opposite sign to non-linearity hypothesis).

**Sensitivity (not pre-registered):**

| reference | bright G < 9 mean deficit | partial peak \| r50 |
|-----------|----------------------------|---------------------|
| Fixed slope **-0.4** (anchored at median G) | **-0.016 dex** | **+0.32** |
| **G 9-11** fit (slope **-0.414**) | **+0.035 dex** | **+0.37** |

G 9-11 reference gives weak positive partial (+0.37) but **below** +0.4 threshold.

**Deficit binned by peak ADU (G < 9, faint-half reference):**

| peak ADU | n | mean deficit (dex) |
|----------|---|---------------------|
| 20k-25k | 120 | -0.441 |
| 25k-30k | 315 | -0.462 |
| 30k-35k | 237 | -0.497 |
| 35k-40k | 115 | -0.533 |
| 40k-45k | 40 | -0.541 |
| 45k-55k | 6 | -0.553 |

Trend is toward **more negative** deficit at higher peak (opposite NL sign under this reference).

## Outcome

**B-open.** Neither B-sky nor B-nl on pre-registered criteria.

- **B1:** VOID (sanity gate). Sky mechanism neither confirmed nor excluded on valid instrument.
- **B2:** Faint reference invalid (-0.18). Primary test partial **-0.89** (wrong sign). Alternate
  G 9-11 reference partial **+0.37** < **+0.4** threshold.

## D5-2 / D1-2 state

| item | state |
|------|-------|
| **D5-2** compression | **MEASURED** (slopes -0.296 / -0.269; G 8-9 localisation from Step 1n) |
| **D5-2 mechanism** | **DEFERRED** -- valid before/after sky test not yet achieved; NL test inconclusive |
| **D1-2** | **DEFERRED** -- not actionable (B-nl not met) |

**One measurement that would decide:** B1 with production-equivalent aperture photometry on a
stored pre-subtraction **aligned** frame (or invert alignment on calibrated before sky), passing
the -0.4 sanity gate; plus B2 with a reference slope verified near -0.4 (e.g. G 9-12 bins).

## Implementation batch routing

No fix routed. If a future B-nl confirms: per-sensor linearity lookup before aperture sums
(Howell 2006 sec 4.4). If B-sky confirms: mask-radius fix in sky-surface path (P-10 class).

## Files changed

- `dev/tools/closure_batch_b_d52_mechanism.py` (new)
- `dev/results/CURSOR_RESULT_batch_B.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_LIMITATIONS.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
- `tmp/closure_batch_b_diagnostics.json`
