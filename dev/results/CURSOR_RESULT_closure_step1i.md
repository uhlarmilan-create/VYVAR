CURSOR RESULT - 2026-08-01 CLOSURE STEP 1i (locate numerator failure mechanism)

**Outcome: I3 -- both mechanisms, with measured attribution.** E5 excludes placement as the
dominant driver of EE scatter. HIGH excursions (EE > 0.65) are 12 px normalisation failure on
faint stars (collapsed F(12), converged-wrong Gaussian fits). LOW excursions (EE < 0.45) are
rare placement errors (up to 3.2 px fit-to-WCS on identified frames).

**E5 before/after (EE std at 1.916 px):**

| proxy G | EE std (fitted centroid) | EE std (WCS position) | ratio wcs/fit |
|--------:|-------------------------:|----------------------:|--------------:|
| 11.52 | 0.0175 | 0.0176 | 1.008 |
| 11.53 | 0.0320 | 0.0321 | 1.001 |
| 12.03 | 0.0398 | 0.0401 | 1.008 |
| 12.59 | 0.0638 | 0.0644 | 1.009 |
| 12.68 | 0.0750 | 0.0757 | 1.010 |

Source: `tmp/closure_step1i_diagnostics.json` E5; recomputation via
`closure_step1i_diagnose_numerator_mechanism.py`.

---

## What I did

Ran E5 first (WCS-position COG control), then E1-E4 on all five Step 1g proxies against the
existing EE cache. Added placement and sky-bias sensitivity tables to
`closure_a1_reference_fixture.py --emit`. No production change, no consolidated number.

Step 1h H1 naming ("SNR-driven COG numerator instability") is **rejected**: predicted photon
noise std EE is 37-65x below measured, and E5 shows placement is not the fix.

---

## E1 -- centroid actually used

Static Gaia catalog RA/Dec projected through per-frame WCS vs Gaussian-fitted centroid vs DAO
proc position.

| proxy G | sep fit-WCS median [px] | p95 | max | >1 px | >2 px | >3 px | pearson(EE dev, sep) |
|--------:|------------------------:|----:|----:|------:|------:|------:|---------------------:|
| 11.52 | 0.110 | 0.174 | 0.200 | 0 | 0 | 0 | -0.017 |
| 11.53 | 0.078 | 0.119 | 0.141 | 0 | 0 | 0 | -0.024 |
| 12.03 | 0.101 | 0.174 | 0.216 | 0 | 0 | 0 | 0.007 |
| 12.59 | 0.156 | 0.260 | 0.293 | 0 | 0 | 0 | 0.046 |
| 12.68 | 0.092 | 0.196 | **3.174** | **1** | **1** | **1** | 0.536 |

Fit-to-WCS separations are **sub-pixel to 0.3 px** on all but one star-frame (G 12.68,
`proc_BO_CVn_Light_090.csv`, 3.17 px). Pearson vs EE deviation is near zero for four proxies.
**Placement error cannot produce EE = 0.743** (fixture: only decreases EE). It can produce
EE ~ 0.11 at ~3 px offset (fixture table).

Source: `tmp/closure_step1i_diagnostics.json` E1.

---

## E2 -- fit quality (converged but wrong)

685 proxy star-frames with full Gaussian2D+Const2D diagnostics.

| proxy G | chi2_red med | centroid shift med [px] | shift p95 [px] | fit_amp/peak med |
|--------:|-------------:|------------------------:|---------------:|-----------------:|
| 11.52 | 0.04 | 0.061 | 0.097 | **0.813** |
| 11.53 | 0.13 | 0.009 | 0.021 | 0.544 |
| 12.03 | 0.25 | 0.063 | 0.136 | 0.419 |
| 12.59 | **0.99** | 0.026 | 0.057 | **0.093** |
| 12.68 | 0.46 | 0.031 | 0.077 | 0.315 |

All proxies: pearson(EE dev, chi2_red) = **0.001**; pearson(EE dev, centroid shift) = **0.235**.

On G 12.59 the fit converges with chi2 ~ 1 but **amplitude is only 9% of peak** -- a
converged-wrong fit that passes Step 1e's convergence gate. This degrades faster than photon
noise as G increases (amp/peak 0.81 -> 0.09).

Source: `tmp/closure_step1i_diagnostics.json` E2.

---

## E3 -- ten extreme star-frames (visual forensics)

Selected by |EE - EE_expected(r50)| across all proxies. Mechanism visible case-by-case:

| # | G | frame | EE | expected | tail | mechanism visible |
|---|---|-------|---:|---------:|------|-------------------|
| 1 | 12.68 | Light_090 | **0.110** | 0.550 | LOW | **3.17 px fit-to-WCS shift**; F(12)=49 061 ADU (6.6x proxy median) inflates denominator -> EE collapses. Cutout peak 2560 ADU, center 2114 ADU. |
| 2 | 12.68 | Light_003 | 0.227 | 0.578 | LOW | F(12)=17 689 moderate; chi2=0.70, amp/peak=0.26. |
| 3 | 12.59 | Light_023 | **0.743** | 0.551 | HIGH | F(12)=**5 439 ADU** (proxy min); amp/peak=**0.083**; chi2=0.99. Denominator collapse drives EE up. 3 sources in annulus. |
| 4 | 12.68 | Light_006 | 0.362 | 0.548 | LOW | F(12)=10 708; amp/peak=0.27. |
| 5 | 12.59 | Light_037 | 0.387 | 0.572 | mid | F(12)=11 182; amp/peak=0.083. |
| 6 | 12.59 | Light_004 | 0.384 | 0.562 | mid | F(12)=11 806; amp/peak=0.082. |
| 7 | 11.53 | Light_016 | **0.708** | 0.536 | HIGH | F(12)=14 892; F(1.916)=10 543 (both elevated); amp/peak=0.50. Not a pure sky-bias case. |
| 8 | 11.53 | Light_083 | 0.694 | 0.523 | HIGH | Same pattern as #7. |
| 9 | 11.53 | Light_061 | 0.702 | 0.534 | HIGH | Same pattern as #7. |
| 10 | 12.68 | Light_028 | 0.395 | 0.561 | mid | F(12)=9 905; amp/peak=0.26. |

**LOW tail (#1):** placement -- 2.9 px Gaussian shift from DAO, 3.17 px from WCS catalog position.
Matches fixture: ~3 px offset -> EE ~ 0.13.

**HIGH tail (#3, G 12.59):** normalisation -- F(12) at proxy minimum (5 439 ADU vs median 8 433);
fit_amp/peak = 0.08. Matches fixture sky-bias table qualitatively: small F(12) drives EE above
0.65 without needing 100 ADU/px systematic offset.

**HIGH tail (#7-9, G 11.53):** elevated F(1.916) and F(12) together; fit_amp/peak ~ 0.5. Distinct
from #3; may be local PSF/asymmetric profile, not annulus sky alone.

Source: `tmp/closure_step1i_diagnostics.json` E3; cutout stats from 40x40 px regions.

---

## E4 -- sky annulus on real frames

Annulus 25-45 px at fitted centroid; per proxy per frame.

| proxy G | sky_ann - global med [ADU/px] | p95 abs diff | pearson(EE dev, sky diff) | F(12) median [ADU] | F(12) min [ADU] |
|--------:|------------------------------:|-------------:|--------------------------:|-------------------:|----------------:|
| 11.52 | +46 | 50 | 0.020 | 43 222 | 40 809 |
| 11.53 | +79 | 89 | 0.164 | 16 595 | 14 827 |
| 12.03 | -8 | 72 | -0.074 | 12 850 | 10 210 |
| 12.59 | -72 | 87 | **-0.226** | 8 433 | **5 439** |
| 12.68 | -85 | 141 | 0.228 | 7 410 | 5 393 |

All proxies pooled: pearson(EE dev, sky_ann - global) = **0.167**;
pearson(EE dev, F(12)) = **-0.213**; sources in annulus median = **3**.

F(12) drops **6.6x** from G 11.52 to G 12.68. Worst HIGH-EE frame (Light_023): F(12)=5 439 ADU,
EE=0.743. The magnitude trend tracks **denominator flux collapse**, not a uniform 100 ADU/px
sky bias. Occasional annulus sky differences reach **141 ADU/px** (p95 abs, G 12.68) on specific
frames -- enough per fixture table to move EE substantially when F(12) is already small.

**Production note:** the closure harness uses the same 25-45 px annulus as
`closure_a1_reference_fixture.py` L2. F(12) sensitivity at faint G implicates this annulus +
normalisation design, not only the closure harness (see docs impact).

Source: `tmp/closure_step1i_diagnostics.json` E4.

---

## E5 -- control recomputation (decisive)

Recomputed COG at **WCS(catalog RA/Dec)** instead of Gaussian-fitted centroid; all else unchanged.

**Scatter does not collapse.** Ratio wcs/fit = **1.001-1.010** across all five proxies. Placement
is **not** the dominant mechanism for the magnitude-dependent EE std.

Source: `tmp/closure_step1i_diagnostics.json` E5.

---

## Mechanism summary (replaces Step 1h "SNR-driven" label)

| Mechanism | tail | share of magnitude trend | evidence |
|-----------|------|--------------------------|----------|
| **12 px normalisation failure** | HIGH EE, growing std with G | **dominant** | E5 no change; E4 F(12) -6.6x with G; pearson(EE dev,F12)=-0.21; E2 amp/peak 0.81->0.09; E3 #3 |
| **Placement error** | LOW EE, rare | **minor** (1/685 frames >1 px to WCS) | E1; E3 #1 (3.17 px); fixture placement table |
| Photon noise | -- | **excluded** | task table: 37-65x below measured |
| Contamination | -- | excluded (Step 1h D3) | -- |

---

## What Step 1j must change

1. **Do not filter on EE vs Moffat band** (Step 1h item 1 rejected; R8).
2. **Admissibility on measured causes:** e.g. fit_amp/peak floor, chi2_red ceiling, F(12) minimum
   ADU, or fit-to-WCS separation threshold -- each justified on E2/E3/E4, not on EE outcome.
3. **Centroid strategy:** test COG at DAO proc (x,y) or WCS catalog position vs Gaussian fit;
   E5 shows WCS vs fit are equivalent for scatter, but E3 #1 shows Gaussian can diverge 3 px on
   individual frames.
4. **Annulus / normalisation review:** F(12) collapse on G > 12 needs production-path review
   (25-45 px annulus, faint-star handling). This may extend beyond A-1 closure.
5. Proxy band stays G 11.5-13.0. No consolidated delta_ap until cause-based gates pass.

---

## Register (wording update)

Replace Step 1h "SNR-driven COG numerator instability" with:

> **A-1b CONFIRMED.** Numerator excursions from 12 px normalisation failure on faint stars
> (collapsed F(12), converged-wrong Gaussian fits) plus rare placement errors on the low-EE tail.
> Exact delta_ap open.

---

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | Replace Step 1h mechanism wording with I3 finding |
| `VYVAR_AUDIT_FINAL.md` | Note: F(12) annulus normalisation at 25-45 px affects faint-star COG on anchor; production review warranted (beyond A-1 number alone) |
| `VYVAR_DECISIONS.md` | no entry |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1i complete; Step 1j queued |
| `closure_a1_reference_fixture.py` | placement + sky-bias sensitivity in --emit; G0-G5 pass |

---

## Errors (if any)

None.

---

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1i_diagnose_numerator_mechanism.py` | E1-E5 diagnostic harness |
| `dev/tools/closure_a1_reference_fixture.py` | placement + sky-bias sensitivity tables |
| `dev/results/CURSOR_RESULT_closure_step1i.md` | this report |
| `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` | mechanism wording |
| `docs/VYVAR_AUDIT_FINAL.md` | annulus finding note |
| `docs/VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1i status |

## Commands

```bash
python dev/tools/closure_step1i_diagnose_numerator_mechanism.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --step1g-json tmp/closure_step1g_results.json \
  --cache tmp/closure_step1f_ee_cache.npz \
  --out tmp/closure_step1i_diagnostics.json

python dev/tools/closure_a1_reference_fixture.py
```
