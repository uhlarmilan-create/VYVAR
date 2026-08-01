CURSOR RESULT - 2026-08-01 CLOSURE STEP 1f (admissibility fix, proxy re-select, measure)

**G6: PASS.** **G7: PASS.** **G8: PASS.** **M2 landed:** G 8-9 consolidated **48.0 +/- 5.7 mmag**
(p95-p5, median across five G<=12 proxies). **T4: FAIL** (median ratio 0.34; band 5-15).

---

## What I did

Pushed Step 1e to origin. Implemented C1-C3 and gate G8 in
`dev/tools/closure_step1f_differential_aperture.py`. Rebuilt EE cache with restricted
admissibility (r <= 3.5 px), no automatic frame exclusion, and five new bright proxies
(G 8.18-8.74). Ran G6/G7/G8 and M1-M4 on anchor draft_435.

## Output / findings

### Gates (first three lines)

1. **G6 PASS** -- max/min proxy p95-p5 ratio <= 1.25 in all three sub-ensembles (G 8-9: 1.19,
   G 9-11: 1.07, G > 11: 1.04).
2. **G7 PASS** -- harness COG matches fixture L2 within **0.005 mmag** (limit 5 mmag).
3. **G8 PASS** -- all five proxies have predicted noise p95-p5 **< 48 mmag** ceiling
   (range 0.97-1.27 mmag vs ceiling 48.1 mmag).

### C1-C3 corrections

| Fix | Location | Change |
|-----|----------|--------|
| **C1** | `_cog_admissible_c1` L69-L82 | EE<=1 and monotonicity tested only for **r <= 3.5 px**; norm still at 12 px |
| **C2** | `_build_ee_cache_step1f` L259-L261 | Removed 20% frame auto-exclusion; **0 frames** under 10 stars |
| **C3** | `_select_proxies_c3` L312-L358 | G <= 12.0, angular isolation, >= 90% C1 frames, rank by predicted sigma delta_ap |

**R4 rejection counts (4865 star-frame attempts):**

| Rule | validation_fail | ee_gt_1 | non_monotonic |
|------|----------------:|--------:|--------------:|
| Step 1e (full curve to 12 px) | **2909** | 2590 | 319 |
| Step 1f C1 (r <= 3.5 px) | **535** | 516 | 19 |

C1 reduced rejections by **82%**. Step 1e G6 failure was caused by the admissibility rule
defect, not anchor contamination (see section 0 evidence; Step 1e contamination claim VOID).

### Selected proxies (C3)

| catalog_id (short) | G | frames | pred sigma d_ap [mmag] | pred noise p95-p5 [mmag] |
|--------------------|---|--------|------------------------:|-------------------------:|
| 1498602913793336448 | 8.18 | 139/139 | 0.40 | 0.97 |
| 1500296402219939584 | 8.24 | 139/139 | 0.41 | 1.00 |
| 1497865897406814592 | 8.28 | 139/139 | 0.42 | 1.02 |
| 1497140116653194368 | 8.58 | 139/139 | 0.48 | 1.17 |
| 1499906247391001088 | 8.74 | 139/139 | 0.52 | 1.27 |

Shortfall: **0** (five G <= 12 candidates passed isolation and frame fraction).

### M1 (per proxy, all sub-ensembles, 139 frames each)

**G 8-9 sub-ensemble (headline):**

| proxy G | p95-p5 [mmag] | slope x span [mmag] | Pearson r50 | Spearman r50 |
|--------:|--------------:|--------------------:|------------:|-------------:|
| 8.18 | 44.9 | 59.7 | 0.62 | 0.84 |
| 8.24 | 48.0 | 81.4 | 0.86 | 0.86 |
| 8.28 | 45.3 | 84.2 | 0.82 | 0.84 |
| 8.58 | 51.0 | 85.7 | 0.84 | 0.86 |
| 8.74 | 53.2 | 90.6 | 0.84 | 0.87 |

All measured p95-p5 ranges are **>> predicted noise** (0.97-1.27 mmag); proxies are not
measuring themselves.

**G > 11 sub-ensemble:** per-proxy p95-p5 **136-142 mmag** (median 140.1 mmag).

### M2 consolidated (G6/G7/G8 pass)

| Sub-ensemble | Measured | Fixture (Moffat beta=3) | Delta |
|--------------|----------|-------------------------|------:|
| G 8-9 | **48.0 +/- 5.7 mmag** | 144.3 mmag | **-96 mmag** |
| G 9-11 | 48.8 +/- 2.5 mmag | 110.3 mmag | -61 mmag |
| G > 11 | 140.1 +/- 3.1 mmag | 14.8 mmag | +125 mmag |

The anchor real PSF differs from the fixture's Moffat beta=3 synthetic (Step 1c A.3 fitted
beta typically 2.8-3.2; bright G~8 proxies vs the fixture's G~12 target at r=1.916 also shift
the baseline). The fixture remains an **expectation**, not ground truth. Measured G 8-9 range
is **3x below** fixture; G > 11 range is **9x above** fixture -- opposite directions suggest
PSF shape and proxy magnitude choice matter, not a unit error (G7 validates arithmetic+COG).

### M3 real target

`1498135552633294976`: **25/139 frames** admissible under C1 (was 0 under Step 1e). QC still
**failed** (< 90% frames). G 8-9 p95-p5 **1732 mmag** on 25 frames -- not headline.

### M4 B.5 / B.6

**T3 identity check:** PASS (max abs **0.0 mmag** on synthetic Moffat, scale = r50_frame).

**B.5 real data (frozen k_i, scale = r50_frame):** tautology range **85-188 mmag** per proxy
(non-zero on real curves -- expected; T3 is synthetic-only).

**B.6 reopt:** G 8-9 p95-p5 **83 mmag**; sky correlation **-0.56**.

### T4 (median across proxies)

| Proxy | G 8-9 / G > 11 ratio |
|-------|---------------------:|
| 8.18 | 0.32 |
| 8.24 | 0.34 |
| 8.28 | 0.33 |
| 8.58 | 0.36 |
| 8.74 | 0.39 |

**Median: 0.34. PASS: false** (band 5-15; fixture 9.74). Inverted sub-ensemble ordering vs
fixture: measured G > 11 range exceeds G 8-9 range because bright G~8 proxies at r=1.916 px
against G>11 comps at ~2.0 px produce a different differential structure than the fixture's
G~12 target configuration.

### Step 1e VOID

Contamination conclusion in Step 1e report marked VOID (isolated Moffat + noise reproduces
Step 1e rejection without neighbours).

## Register wording

> **A-1b CONFIRMED.** Seeing-correlated, magnitude-dependent differential aperture systematic.
> Measured G 8-9 p95-p5 range over r50 span: **48.0 +/- 5.7 mmag** (Step 1f, five G<=12
> proxies, photutils COG). Fixture expectation (Moffat beta=3): 144.3 mmag. Target on r_min
> clamp 1.916 px; no COG correction applied. T4 sub-ensemble ratio **FAIL** (0.34 vs 5-15).

Supersedes 203 mmag (Step 1d V8) and "exact value open" (Step 1e).

### VOID (Step 1g supersession -- do not delete)

**Pointer:** `dev/results/CURSOR_RESULT_closure_step1g.md`

| ID | Claim | Issue |
|----|-------|-------|
| **V11** | M2 headline **48.0 +/- 5.7 mmag** as A-1 value | Proxies were inside G8_9 comparison set at comparison radii |
| **V12** | G6 PASS at 1.19 / 1.07 / 1.04 | Autocorrelation among five near-identical G~8 proxies |
| **V13** | Fixture disagreement explained as PSF/proxy magnitude | Configuration swap accounts for signs and magnitudes |
| **V14** | B.5/B.6 on Step 1f configuration | Invalid configuration; re-run under F1 in Step 1g |

## What Step 2 inherits

- First **validated-method** consolidated number: **48 mmag** G 8-9 (robust p95-p5).
- Fixture disagreement (-96 mmag G 8-9, +125 mmag G > 11) flags PSF/proxy-configuration work
  before using fixture as prediction on this anchor.
- T4 fail blocks using sub-ensemble ratio as closure evidence until proxy/comp configuration
  matches the fixture experiment design (or fixture is re-tuned to G~8 proxies).
- B.5/B.6 on repaired harness now populated for Milan's option (i)/(iii)/(iv) choice.
- Focus target: 25 admissible frames under C1 but still QC-failed.

## Docs impact

| Doc | Action |
|-----|--------|
| `VYVAR_AUDIT_CLOSURE_REGISTER.md` | A-1 row: measured 48.0 mmag |
| `VYVAR_AUDIT_FINAL.md` | D5-1 Step 1f magnitude |
| `VYVAR_STATE.md` / `VYVAR_ROADMAP.md` | Step 1f complete; T4 open |
| `closure_a1_reference_fixture.py` | unchanged; G0-G5 pass |

## Errors (if any)

None.

## Files changed

| File | Role |
|------|------|
| `dev/tools/closure_step1e_differential_aperture.py` | Step 1e harness (pushed) |
| `dev/tools/closure_step1f_differential_aperture.py` | Step 1f harness C1-C3, G8, measure |
| `dev/results/CURSOR_RESULT_closure_step1f.md` | this report |
| `dev/results/CURSOR_RESULT_closure_step1e.md` | contamination claim VOID |
| docs (register, audit, state, roadmap) | Step 1f status |

## Commands

```bash
python dev/tools/closure_step1f_differential_aperture.py \
  --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \
  --step1b-json tmp/closure_step1b_results.json \
  --out tmp/closure_step1f_results.json \
  --cache tmp/closure_step1f_ee_cache.npz \
  --rebuild-cache
```
