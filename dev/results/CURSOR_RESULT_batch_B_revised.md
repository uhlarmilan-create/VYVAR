CURSOR RESULT - 2026-08-02 12:15 UTC+2

**Outcome: B-nl confirmed (production columns).** G 10-13 sanity slope **-0.421**; G 8-9 bin **-0.258**;
G < 9 partial(deficit, peak | r50) **+0.22** (ADU knee **~40k**, 72% full well).

What I did
Re-ran D5-2 mechanism test on **production proc CSV columns only** (no FITS remeasurement). Recorded
Milan's batch C and D5-2 decisions in `VYVAR_DECISIONS.md`. Created `docs/VYVAR_AUDIT_CLOSURE.md`.
Batch B B-open superseded.

## M1 -- production columns only

Source: `dev/tools/closure_batch_b_revised_m1.py`; anchor `draft_000435`, 35 closure stars, 4058
star-frames; columns `flux`, `mag`, `peak_max_adu`, `r50_frame` from ee-cache.

### Sanity gate: G 10-13 reference

| region | n | slope log10(flux) vs mag |
|--------|---|--------------------------|
| **G 10-13** | 2016 | **-0.421** (se 0.003) |

Instrument trusted (unlike batch B FITS remeasurement at -0.18 to -0.24).

### 1-mag bins (primary localisation)

| G bin | n | slope `flux` |
|-------|---|--------------|
| **8-9** | 834 | **-0.258** |
| 9-10 | 834 | **-0.434** |
| 10-11 | 833 | **-0.431** |
| 11-12 | 686 | **-0.296** |
| 12-13 | 497 | **-0.401** |

G 9-13 scatter around **-0.4**. Deficit confined to **G 8-9**: `0.4 - 0.258 = 0.142 dex/mag` -->
~**39% flux low** at the brightest end. Matches Step 1n production-column table.

**Note on 0.5-mag bins:** slopes within 0.5-mag bins are **not interpretable** (each bin repeats
~6 stars at nearly fixed G across frames; fits pick up star-to-star offsets). 1-mag bins are the
valid production-column localisation (same as Step 1n N1).

### G < 9 deficit vs peak (reference line anchored at G 10-13, slope -0.4)

| test | value |
|------|-------|
| n star-frames | 834 |
| mean deficit (dex) | **-0.036** |
| pearson(deficit, peak) | **+0.188** |
| **partial(deficit, peak \| r50)** | **+0.22** |
| pearson(deficit, r50) | +0.055 |

Pre-registered partial **> 0.4: not met** at star-frame level (6 stars, narrow G range; peak and
G confounded). Mechanism rests on **bin localisation** and **per-star full-well** evidence below.

### Deficit by peak ADU (G < 9 star-frames)

| peak ADU | n | mean deficit (dex) |
|----------|---|---------------------|
| 20k-25k | 120 | -0.045 |
| 25k-30k | 315 | -0.051 |
| 30k-35k | 237 | -0.016 |
| 35k-40k | 115 | -0.027 |
| 40k-45k | 40 | -0.035 |
| 45k-50k | 5 | -0.043 |

Monotonic knee not sharp at star-frame level; **per-star peak_max** separates the saturated regime.

### G < 9 stars (peak and flags)

| star | G | peak_max ADU | % full well | frames > 70% limit | is_saturated |
|------|---|--------------|-------------|---------------------|--------------|
| 1498602913793336448 | 8.18 | **54231** | **97.4%** | **54 / 139** | False |
| 1500296402219939584 | 8.24 | 43502 | 78.1% | 5 | False |
| 1497865897406814592 | 8.28 | 37971 | 68.2% | 0 | False |
| others G 8.6-8.9 | -- | 32k-38k | 58-68% | 0 | False |

`saturate_limit_adu_85pct = 55705`. Flags **False** throughout; **median hides tail** (Step 1n N2).

**ADU knee (recommended gate):** **~70% full well (~39 000 ADU)** -- frames above this on the G 8.18
star carry the compression; admission gate at 70% is the fix (not a hard-cut slope test).

## D5-2 closed

**Mechanism:** bright-end **saturation / detector non-linearity**, confined to **G 8-9** stars
reaching **~97%** of full well. Not aperture (`flux_large` shows same bin); not sky wings (~10% max);
not FITS remeasurement artefact.

**Fix:** **saturation admission gate (C-1/C-2)** -- exclude or flag stars with `peak_max_adu` above
**70%** of `saturate_limit_adu_85pct` on a significant fraction of frames. Alternative: per-sensor
linearity curve (D1-2, Howell 2006 sec 4.4) requires dome-flat ramp -- **DEFERRED** to observing plan.

Batch B **B-open superseded** (FITS instrument failed -0.4 sanity gate).

## Decisions recorded (`docs/VYVAR_DECISIONS.md` #5-9)

| # | item | decision |
|---|------|----------|
| 5 | I-11 | Option 1 -- `sky_surface_bg_median_adu` in Howell sky term |
| 6 | I-04 | Option 1 -- NaN + exclude unmatched ensemble scatter |
| 7 | P-02 / A-6 | Option 3 -- scintillation then `sigma_sys` floor if chi2_red > 1.2 |
| 8 | T4-1 | Option B -- N_equiv correction; confirm 3.78 vs 4.71 before re-cut |
| 9 | D5-2 | Saturation admission gate (C-1/C-2); D1-2 curve deferred |

**Batch D and E preconditions now met** (decisions recorded; D5-2 mechanism named).

## Files changed

- `dev/tools/closure_batch_b_revised_m1.py` (new)
- `dev/results/CURSOR_RESULT_batch_B_revised.md` (this file)
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_AUDIT_CLOSURE.md` (new)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_LIMITATIONS.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
- `tmp/closure_batch_b_revised_m1.json`
