CURSOR RESULT - 2026-08-01 14:45 UTC+2

**Outcome K-a** (detector non-linearity / peak-ADU compression). K1 colour-corrected slope (harness, per-star median G+BPRP): **-0.312** (expected -0.4). K3 production slope (mag+BPRP): **-0.339**.

What I did
Decomposed the F(12) slope defect (K1-K2) on 35 closure stars x 139 frames using the Step 1j harness; ran the same slope test on production `proc_*.csv` photometry (K3); updated the A-1 reference fixture annulus to production geometry and re-ran G0-G5 (K4). Diagnostic only; no production code change.

## Output / findings

### K1 -- decompose the slope (harness F(12), geometry 25-45 px annulus)

Source: `dev/tools/closure_step1k_decompose_f12_slope.py` on `tmp/closure_step1f_ee_cache.npz`, draft_435.

| fit | n | slope vs G | se | notes |
|-----|---|------------|-----|-------|
| log10(F12) vs G, star-frames | 4083 | **-0.285** | 0.0021 | Step 1j J1 reproduced |
| + BP-RP joint, star-frames | 3944 | **-0.323** | 0.0022 | BP-RP coeff **-0.163 dex/mag** |
| per-star median F12 vs G | 32 | -0.284 | 0.025 | |
| per-star median + BP-RP | 31 | **-0.312** | 0.025 | BP-RP coeff **-0.219** (se 0.142) |

Colour (BP-RP) moves the slope toward -0.4 but does **not** reach it (K-b partial only).

**Residual vs peak_max_adu** (G-only fit): Pearson **r = +0.359** (source: proc CSV column per star-frame).

| peak ADU bin | n | mean residual (dex) |
|--------------|---|----------------------|
| 0-2000 | 101 | -0.401 |
| 2000-4000 | 786 | -0.126 |
| 4000-8000 | 1110 | -0.022 |
| 8000-20000 | 996 | +0.032 |
| 20000+ | 1090 | +0.122 |

Monotonic trend: faint/low-peak stars sit **below** the G-only line; bright/high-peak stars sit **above** it. Break near **4000-8000 ADU** peak.

**Bright vs faint half** (split at median G ~ 10.0):
- faint half slope **-0.404** (n=1946) -- consistent with -0.4
- bright half slope **-0.185** (n=2137) -- compressed

**Saturation exclusion** (5 brightest stars, G 8.18-8.74): `is_saturated=False`, `likely_saturated=False` for all; `peak_max_adu` median 25k-38k vs `saturate_limit_adu_85pct` **55705** (source: proc CSV). Saturation is not the driver.

### K2 -- two defects quantified separately

Source: same script; additive correction `F12_corr = F12 + (sky_ann - global_median) * A(12)`, `A(12) = 452.4 px^2`.

| quantity | value | source |
|----------|-------|--------|
| slope, per-star median F12 raw | **-0.284** | K2 fit |
| slope after additive correction | **-0.198** | K2 fit |
| G 11.52 / G 11.53 F12 ratio raw | **2.605x** | medians 43222 / 16595 ADU |
| ratio after additive correction | **1.201x** | (task Section 0 predicted ~1.37x) |
| sky_ann difference (11.52 - 11.53) | **-33.4 ADU/px** | harness annulus |
| implied sky from F12 difference | **58.9 ADU/px** | (F12 diff)/A(12) |

Additive sky offset accounts for ~56% of the G 11.52/11.53 pair discrepancy (Step 1j arithmetic confirmed). After global-median correction the ratio falls to **1.20x** (refutes exact 1.37x; close). Slope **does not** stay at -0.285 after additive correction (-0.198): sky offset and magnitude are correlated (Step 1j J2: Pearson sky vs field-centre distance **-0.53**), so the two defects are not independent under a global sky reference.

Same-magnitude pair ratios (dG <= 0.05 mag): median **1.008x** raw, **1.201x** after additive (n_pairs 16 / 15).

### K3 -- production-path test (decisive)

Source: `flux`, `mag`, `bp_rp`, `peak_max_adu` from `proc_BO_CVn_Light_*.csv` (4058 star-frames, `photometry_ok` and `is_usable` true).

| fit | n | slope vs mag | se |
|-----|---|--------------|-----|
| log10(flux) vs mag | 4058 | **-0.296** | 0.0022 |
| + BP-RP joint | 3919 | **-0.339** | 0.0023 |

`dao_flux` vs mag: identical slope **-0.296** (same column path in this draft).

Production shows the **same compression** as the harness (-0.296 vs -0.285). This is **not** a harness-only artefact (K-c excluded).

Residual vs peak_max_adu: Pearson **r = +0.362**; bin pattern matches K1 (mean residual -0.421 at peak 0-2000 ADU to +0.123 at peak 20000+ ADU).

**Proposed finding D5-2:** Production aperture flux on anchor draft_435 does not scale with catalogue magnitude at the expected log flux slope -0.4 mag^-1; measured slope **-0.296** (mag only) and **-0.339** (mag + BP-RP). The defect is present in pipeline output, not only in the closure harness recomputation. It affects any magnitude-dependent correction that assumes flux proportional to 10^(-0.4 G). Mechanism consistent with detector non-linearity at the bright end (see D1-2); distinct finding ID so it is not folded into A-1.

### K4 -- fixture annulus geometry

Changed `dev/tools/closure_a1_reference_fixture.py` annulus from harness-only **25-45 px** to production geometry at `R_TARGET=1.916`, `fw=2.395`:
- `r_in = 11.376 px`, `r_out = 21.555 px`

**G0-G5:** all **PASS** (source: fixture run 2026-08-01).

**delta_ap table change** (mmag, vs prior 25-45 px annulus): sub-0.1 mmag at all r50 rows; `G_gt_11` at r50=1.87 unchanged at **71.66 mmag** (71.66 -> 71.66); range over span unchanged at **14.8 mmag** for G_gt_11. Prior comparisons using 25-45 px remain valid at ~0.05 mmag level; annulus label must be recorded when citing fixture numbers.

**G 12.59 (1499238946911605504) narrow/production failure explained:**

| quantity | value | source |
|----------|-------|--------|
| median position | x=838.2, y=70.7 px | EE cache centroids |
| frame shape | 2082 x 1397 | FITS |
| distance to nearest edge | **70.7 px** (top edge) | min(x, y, nx-x, ny-y) |
| prod annulus r_in / r_out | 11.376 / 21.555 px | `_prod_annulus(1.916)` |
| COG OOB frames | **0 / 139** | bounds check |

Failure is **not** array edge exit. Narrow annulus `r_in=12 px` and production `r_in=11.376 px` both lie **inside** the F(12) measurement aperture (r=12 px), contaminating the sky estimate with stellar flux. Sky is over-estimated (~2379-2394 ADU/px vs harness 2316 ADU/px on frame 001); F(12) goes **negative** and `_measure_cog` returns None. Harness annulus `r_in=25 px` sits outside r=12 and succeeds. G 12.59 sits near the top edge (y=70.7) but has adequate margin for the annulus geometry; the overlap with the flux aperture is the mechanism.

## Outcome rationale

- **K-a:** Residual tracks peak ADU with a break near 4-8k ADU; bright-half slope -0.185 vs faint-half -0.404. Matches D1-2 compression signature. First valid isolated residual-vs-peak trend (Stage 2 measurement was magnitude-dominated).
- **K-b partial:** BP-RP joint fit moves slope from -0.285 toward -0.4 (harness -0.323 star-frames, production -0.339) but does not close the gap. Colour contributes; not sufficient alone.
- **K-c excluded:** Production compressed at -0.296, not -0.4.
- **K-d excluded:** Non-linearity and partial colour explain the data; saturation excluded.

## What Step 1l must change

1. Treat A-1 F(12) path as **two defects**: additive (per-star sky, ~56% of pair anomaly quantified) and multiplicative (flux-vs-G compression, slope defect).
2. Do **not** apply global-median sky correction as a single fix -- it couples defects (slope goes to -0.198).
3. Pursue **D1-2 / D5-2** on production flux: linearity correction or peak-dependent flux calibration before any consolidated `delta_ap`.
4. For harness F(12) recomputation use annulus with `r_in > 12 px` (production formula violates this at r_ap=1.916); G 12.59 is a worked example.
5. Re-run J3 EE-scatter table with a valid sky annulus (or document production r_in overlap as harness limitation).

## Docs impact

- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` -- A-1 mechanism: two defects; Step 1k K-a; D5-2 row
- `docs/VYVAR_AUDIT_FINAL.md` -- D1-2 first valid measurement; D5-2 new
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md` -- Step 1k complete
- `dev/tools/closure_a1_reference_fixture.py` -- production annulus (K4)

## Errors (if any)

None.

## Files changed

- `dev/tools/closure_step1k_decompose_f12_slope.py` (new)
- `dev/tools/closure_a1_reference_fixture.py` (annulus 11.376-21.555 px)
- `dev/results/CURSOR_RESULT_closure_step1k.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
- `tmp/closure_step1k_diagnostics.json` (local output)
