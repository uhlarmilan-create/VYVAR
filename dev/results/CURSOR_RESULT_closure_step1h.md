CURSOR RESULT - 2026-08-01 CLOSURE STEP 1h (diagnose magnitude dependence of delta_ap)

**Outcome: H1 -- cause identified.** Fainter-proxy COG numerator failure: `EE_target(1.916)` scatter
and out-of-band values grow with decreasing proxy SNR. Shared comparison denominator is stable and
does not explain the G trend. Catalogue neighbour contamination excluded.

---

## What I did

Built and ran `dev/tools/closure_step1h_diagnose_magnitude_dependence.py` on the Step 1g cache
(`tmp/closure_step1f_ee_cache.npz`, `tmp/closure_step1g_results.json`). Diagnostics D1-D5 only; no
re-measurement, no config change, no consolidated number. Added contamination sensitivity reference
table to `closure_a1_reference_fixture.py --emit`. Updated register, STATE, ROADMAP.

## H3 -- reproducibility

Step 1g p95-p5 reproduced exactly from cache (delta 0.0 mmag all five proxies):

| proxy G | Step 1g p95-p5 [mmag] | recomputed |
|--------:|----------------------:|-----------:|
| 11.52 | 94.3 | 94.3 |
| 11.53 | 171.8 | 171.8 |
| 12.03 | 264.5 | 264.5 |
| 12.59 | 451.9 | 451.9 |
| 12.68 | 425.8 | 425.8 |

Source: `tmp/closure_step1h_diagnostics.json` key `H3_step1g_reproduction`.

---

## D1 -- frame-level outliers

Not an outlier problem. Each proxy has **7** frames above p95 and **7** below p5 (expected for
percentile tails on 131-139 frames). Trimming the 5 most extreme frames per proxy reduces p95-p5
modestly:

| proxy G | n frames | p95-p5 | p84-p16 | MAD | after trim-5 |
|--------:|---------:|-------:|--------:|----:|-------------:|
| 11.52 | 139 | 94.3 | 48.6 | 16.5 | 89.3 |
| 11.53 | 131 | 171.8 | 97.7 | 36.8 | 151.8 |
| 12.03 | 139 | 264.5 | 190.4 | 60.2 | 252.3 |
| 12.59 | 139 | 451.9 | 243.5 | 82.1 | 389.5 |
| 12.68 | 137 | 425.8 | 293.2 | 91.2 | 404.5 |

Extreme frames differ by proxy (no single bad frame drives all five). Example G 12.68 max:
`proc_BO_CVn_Light_090.csv` (delta_ap **1955 mmag**, r50=1.765 px, sky=1531 ADU/px, airmass=1.014,
31 stars kept). Min: `proc_BO_CVn_Light_037.csv` (delta_ap **-19 mmag**). Range is distributed
across seeing/sky conditions, not 5 rogue frames.

Source: `tmp/closure_step1h_diagnostics.json` D1; frame metadata from cache + Step 1g drops.

---

## D2 -- direct EE(1.916) vs delta_ap

Physics reference (Moffat beta=3, fixture): `EE(1.916)` spans **0.626** at r50=1.97 px to
**0.464** at r50=1.46 px (delta **0.162** in EE units).

| proxy G | EE_target p95-p5 | EE min | EE max | pearson(EE,r50) | delta_ap p95-p5 |
|--------:|-----------------:|-------:|-------:|----------------:|----------------:|
| 11.52 | 0.057 | 0.501 | 0.627 | -0.633 | 94.3 |
| 11.53 | 0.103 | 0.553 | 0.715 | -0.345 | 171.8 |
| 12.03 | 0.122 | 0.399 | 0.600 | -0.255 | 264.5 |
| 12.59 | 0.212 | 0.381 | 0.743 | -0.180 | 451.9 |
| 12.68 | 0.200 | 0.110 | 0.684 | -0.031 | 425.8 |

Per-frame EE std at 1.916 px (cache recomputation): **0.017** (G 11.52) to **0.075** (G 12.68).
Frames outside physics band [0.464, 0.626]: G 11.52 **0**; G 11.53 **72** frames EE>0.65; G 12.03
**33** EE<0.45; G 12.59 **26** EE<0.45 + **3** EE>0.65; G 12.68 **27** EE<0.45 + **5** EE>0.65.

**Verdict:** defect is in the **numerator**. Fainter proxies show EE spread exceeding the physics
band; brightest proxy (G 11.52) is closest to expected behaviour. Median comparison EE denominator
p95-p5 is only **0.013** EE units (shared across proxies).

Source: `tmp/closure_step1h_diagnostics.json` D2; EE band counts from cache recomputation.

---

## D3 -- contamination inside aperture

Among the 35 closure stars: **zero** neighbours within **8 px** and **dG<5** for all five proxies
and all six G 8-9 comparisons (median positions).

Extended check on frame-0 catalogue (**2579** detected sources): still **zero** neighbours within
8 px / dG<5 for every proxy and every G 8-9 comp.

| proxy G | n neighbours (8 px, dG<5) | estimated contam flux fraction |
|--------:|--------------------------:|-------------------------------:|
| 11.52 | 0 | 0 |
| 11.53 | 0 | 0 |
| 12.03 | 0 | 0 |
| 12.59 | 0 | 0 |
| 12.68 | 0 | 0 |

Pearson(range, contam_frac): undefined (zero variance). Spearman(range, G): **0.90**.

**Contamination from catalogue sources within 8 px is excluded.** The leading contamination
candidate from Step 1g preamble does not apply on this anchor. Sensitivity table retained in
fixture for sub-catalogue / DAO-blended sources not in Gaia.

Source: D3 in diagnostics JSON; frame-0 full-catalogue search (2579 sources).

---

## D4 -- shared structure and slope x span / Pearson

### Cross-proxy correlation (129 frames where all five finite)

Mean off-diagonal Pearson: **0.002**. Min: **-0.148**. Proxies do **not** share a common per-frame
shape; each numerator diverges independently.

### Regressions (partial correlation controlling r50)

| proxy G | p95-p5 | pearson(r50) | partial(r50 | denom) | pearson(denom) | fit resid RMS |
|--------:|-------:|-------------:|------------------:|---------------:|--------------:|
| 11.52 | 94.3 | 0.544 | 0.529 | -0.200 | 25.9 |
| 11.53 | 171.8 | 0.280 | 0.281 | -0.079 | 50.8 |
| 12.03 | 264.5 | 0.214 | 0.206 | -0.077 | 85.9 |
| 12.59 | 451.9 | 0.151 | 0.170 | -0.016 | 131.9 |
| 12.68 | 425.8 | -0.034 | 0.070 | 0.171 | 207.0 |

Fit residual RMS grows with G (25.9 -> 207.0 mmag): fainter proxies are **not** well described by
a linear r50 trend; large non-r50 scatter remains.

### slope x span vs p95-p5 / Pearson=0.54

The Step 1g markdown table claimed slope x span = p95-p5 on four proxies and Pearson=0.54 on all
five. **That table was incorrect.** JSON on disk and Step 1h recomputation show:

- slope x span equals p95-p5 only approximately for G 11.52 (96.8 vs 94.3 mmag).
- Pearson vs r50 **decreases** with G (0.54 -> -0.03); it is not constant.
- slope x span and p95-p5 are **different estimators**: slope x span is a linear fit over r50;
  p95-p5 is a percentile range. They coincide only when residuals are small and the relation is
  nearly linear (brightest proxy only).

Source: `tmp/closure_step1h_diagnostics.json` D4; Step 1g JSON `M1.proxies.*.G8_9`.

---

## D5 -- comparison denominator

Per-frame median G 8-9 comparison EE:

- p95-p5: **0.0128** EE units (~29 mmag equivalent)
- mean std across 6 comps: **0.037** EE units
- Pearson vs r50: **-0.548** (shared seeing driver)

Recomputing delta_ap with **fixed** denominator (per-comp median EE over all frames):

| proxy G | p95-p5 original | p95-p5 fixed denom | fraction in numerator |
|--------:|----------------:|-------------------:|----------------------:|
| 11.52 | 94.3 | 107.6 | 1.14 |
| 11.53 | 171.8 | 171.9 | 1.00 |
| 12.03 | 264.5 | 275.0 | 1.04 |
| 12.59 | 451.9 | 447.4 | 0.99 |
| 12.68 | 425.8 | 426.8 | 1.00 |

Fixing the denominator changes ranges by **<=14%**. The magnitude trend **survives** (94 -> 448
mmag). The G6 spread is **not** a denominator artefact.

Source: `tmp/closure_step1h_diagnostics.json` D5.

---

## H1 -- identified cause

**Name:** SNR-driven COG numerator instability on fainter proxies at fixed r=1.916 px.

**Mechanism:** `delta_ap = -2.5*log10(EE_target / EE_comp_med)`. At fixed aperture radius,
`EE_target` should depend only on PSF shape (r50). On fainter stars the photutils COG curve is
noisy; measured `EE(1.916)` wanders outside the physics-permitted band [0.464, 0.626], producing
hundreds of mmag scatter that grows as proxy G increases (lower peak flux, lower COG SNR).

**Quantification:**

- p95-p5 vs G: Spearman **0.90** (94 -> 452 mmag over 1.16 mag)
- EE_target std: **0.017 -> 0.075** (5x)
- Denominator stable: fixed-denom ranges **99-114%** of original
- Contamination: **excluded** (0 neighbours)
- Photon noise / sky bias: already excluded in task preamble (32-57x below measured)

**Brightest proxy G 11.52** (94 mmag) is the only proxy whose EE statistics sit near the physics
band; it is the admissible diagnostic anchor until measurement is repaired.

---

## What Step 1i must change

1. **Add numerator QC gate:** reject or flag frames where `EE_target(1.916)` falls outside the
   Moffat-predicted band for that frame's r50 (fixture: 0.464-0.626 over anchor r50 span), before
   computing delta_ap range.
2. **Proxy admissibility by COG quality:** require per-proxy EE std at 1.916 below a gate tied to
   predicted photon noise (G8), not merely frame count.
3. **Do not narrow the G 11.5-13.0 band to pass G6** (R8) unless evidence shows a narrower band
   has validated COG curves -- Step 1h shows the band spread exposes a measurement defect, not
   physics.
4. **Re-measure only after gates pass** on a single canonical proxy (G ~11.5) or on all proxies
   that pass numerator QC; no consolidated number until then.

Options (i), (iii), (iv) remain unapplied. No production code change in Step 1h.

---

## Register (unchanged verdict)

> **A-1b CONFIRMED.** Seeing-correlated, magnitude-dependent differential aperture systematic,
> magnitude of order 10^2 mmag. Target on the `r_min` clamp at 1.916 px against comparisons at
> larger magnitude-binned radii, with no curve-of-growth correction. Exact value open.
> Physics expectation from `dev/tools/closure_a1_reference_fixture.py`: 144.3 mmag for G 8-9
> comparisons over the anchor's measured r50 span.

**Step 1h diagnosis (2026-08-01):** G6 magnitude spread traced to **fainter-proxy COG numerator
instability**, not denominator, contamination, or outliers. Cause identified (H1); Step 1i
numerator QC required before re-measurement.

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 row: add Step 1h diagnosis line |
| `VYVAR_AUDIT_FINAL.md` | no change (contamination did not implicate isolation rule) |
| `VYVAR_DECISIONS.md` | no entry |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1h complete; Step 1i queued |
| `closure_a1_reference_fixture.py` | contamination sensitivity table in --emit; G0-G5 pass |

---

## Errors (if any)

None. Step 1g markdown table for slope x span / Pearson was wrong vs JSON; corrected in this report.

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1h_diagnose_magnitude_dependence.py` | D1-D5 diagnostic harness |
| `dev/tools/closure_a1_reference_fixture.py` | contamination sensitivity in --emit |
| `dev/results/CURSOR_RESULT_closure_step1h.md` | this report |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | Step 1h diagnosis note |
| `docs/VYVAR_STATE.md` | Step 1h status |
| `docs/VYVAR_ROADMAP.md` | Step 1h status |

## Commands

```bash
python dev/tools/closure_step1h_diagnose_magnitude_dependence.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --step1g-json tmp/closure_step1g_results.json \
  --cache tmp/closure_step1f_ee_cache.npz \
  --out tmp/closure_step1h_diagnostics.json

python dev/tools/closure_a1_reference_fixture.py
```
